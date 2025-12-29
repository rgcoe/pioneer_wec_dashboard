import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotly
from plotly_calplot import calplot
import requests
import logging
from parse_wec_decimated_log import parse_putty_log
import os
from pathlib import Path

colors = plotly.colors.DEFAULT_PLOTLY_COLORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# directory to cache raw WEC text files
WEC_TEXT_CACHE = Path(".cache/wec")
PWRSYS_TEXT_CACHE = Path(".cache/pwrsys")
DATA_DIR = Path("output/data")


def _ensure_wec_text_cache_dir() -> None:
    WEC_TEXT_CACHE.mkdir(parents=True, exist_ok=True)


def _wec_text_cache_file(date_str: str) -> Path:
    return WEC_TEXT_CACHE / f"{date_str}.wec.dec.10.log"


def _ensure_pwrsys_text_cache_dir() -> None:
    PWRSYS_TEXT_CACHE.mkdir(parents=True, exist_ok=True)


def _pwrsys_text_cache_file(date_str: str) -> Path:
    return PWRSYS_TEXT_CACHE / f"{date_str}.pwrsys.log"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ndbc(
    buoy_id: str = "44014", start_date: datetime = None, max_retries: int = 3
) -> xr.Dataset:

    now = datetime.now().date()
    if start_date is None:
        start_date = now - timedelta(days=7)
    start_year = start_date.year
    num_days = (now - start_date).days + 1
    logger.info(
        f"Fetching NDBC data data starting from {start_date.strftime('%Y-%m-%d')} ({num_days} days)"
    )

    urls = []
    urls += [f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"]

    if num_days > 45:
        for month in range(start_date.month, now.month):
            month_str = (datetime(2000, month, 1)).strftime("%b")
            urls.append(
                f"https://www.ndbc.noaa.gov/data/stdmet/{month_str}/{buoy_id}.txt"
            )
        if start_year < now.year:
            for year in range(start_year, now.year):
                urls.append(
                    f"https://www.ndbc.noaa.gov/data/historical/stdmet/{buoy_id}h{year}.txt.gz"
                )

    logger.info(f"Fetching NDBC data for buoy {buoy_id}")
    logger.info(f"Using URLs: {urls}")

    dsl = []
    for url in urls:
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching data from {url} (attempt {attempt + 1})")
                ds1 = _parse_ndbc_stdmet(url)
                dsl.append(ds1)
                break
            except Exception as e:
                logger.warning(f"Failed to fetch/parse data from {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for {url}, skipping.")
                else:
                    logger.info(f"Retrying...")
    ds = xr.concat(dsl, dim="time").sortby("time").drop_duplicates("time")
    ds = ds.expand_dims("buoy").assign_coords(buoy=[buoy_id])
    return ds


def _parse_ndbc_stdmet(url: str) -> xr.Dataset:

    df = pd.read_csv(
        url,
        sep=r"\s+",
        comment="#",
        na_values=["MM", 99.0, 99.00, 999],
        engine="python",
        header=None,
    )

    colnames = [
        "YY",
        "MM",
        "DD",
        "hh",
        "mm",
        "WDIR",
        "WSPD",
        "GST",
        "WVHT",
        "DPD",
        "APD",
        "MWD",
        "PRES",
        "ATMP",
        "WTMP",
        "DEWP",
        "VIS",
        "TIDE",
    ]

    if url.__contains__("realtime2"):
        colnames.insert(16, "PTDY")  # add PTDY column for realtime2 files

    df.columns = colnames

    # ---- UNITS (from second header line) ----
    units = {
        "YY": "-",
        "MM": "-",
        "DD": "-",
        "hh": "-",
        "mm": "-",
        "WDIR": "°",
        "WSPD": "m/s",
        "GST": "m/s",
        "WVHT": "m",
        "DPD": "s",
        "APD": "s",
        "MWD": "°",
        "PRES": "hPa",
        "ATMP": "°",
        "WTMP": "°",
        "DEWP": "°",
        "VIS": "nmi",
        "PTDY": "hPa",
        "TIDE": "ft",
    }

    df["time"] = pd.to_datetime(
        df[["YY", "MM", "DD", "hh", "mm"]]
        .astype({"YY": "int", "MM": "int", "DD": "int", "hh": "int", "mm": "int"})
        .rename(
            columns={
                "YY": "year",
                "MM": "month",
                "DD": "day",
                "hh": "hour",
                "mm": "minute",
            }
        )
    )

    df = df.set_index("time")

    # Drop the raw time columns if desired
    df = df.drop(columns=["YY", "MM", "DD", "hh", "mm"])

    ds = xr.Dataset(
        {
            var: (["time"], df[var].values, {"units": units.get(var, "")})
            for var in df.columns
        },
        coords={"time": df.index.values},
        attrs={},
    )

    long_names = {
        "WDIR": "Wind Direction",
        "WSPD": "Wind Speed",
        "GST": "Wind Gust",
        "WVHT": "Significant Wave Height",
        "DPD": "Peak Wave Period",
        "APD": "Average Wave Period",
        "MWD": "Mean Wave Direction",
        "PRES": "Atmospheric Pressure",
        "ATMP": "Air Temperature",
        "WTMP": "Water Temperature",
        "DEWP": "Dew Point",
        "VIS": "Visibility",
        "PTDY": "Pressure Tendency",
        "TIDE": "Tide",
        "time": "Time",
        "buoy": "Buoy ID",
    }
    for var in ds.data_vars:
        if var in long_names:
            ds[var].attrs["long_name"] = long_names[var]

    ds = ds.sortby("time")

    ds["dir_diff"] = np.abs(ds["MWD"] - ds["WDIR"]) % 180
    ds["dir_diff"].attrs["units"] = "°"
    ds["dir_diff"].attrs["long_name"] = "Direction Difference"

    return ds


def fetch_wec_data(start_date: datetime = None, max_retries: int = 3) -> xr.Dataset:

    logger.info(f"Fetching WEC data")

    end_time = pd.Timestamp.utcnow()
    current_date = end_time.date()

    if start_date is None:
        start_date = current_date - timedelta(days=7)

    dates_to_try = [
        start_date + timedelta(days=i)
        for i in range((current_date - start_date).days + 1)
    ]

    base_url = "https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/dcl12/wec_decimated"

    dsl = []
    for attempt_date in dates_to_try:
        date_str = attempt_date.strftime("%Y%m%d")
        file_url = f"{base_url}/{date_str}.wec.dec.10.log"
        _ensure_wec_text_cache_dir()
        cache_file = _wec_text_cache_file(date_str)

        # prefer cached text file
        if cache_file.exists():
            try:
                logger.info(f"Loading cached WEC text for {date_str} from {cache_file}")
                text_to_parse = cache_file.read_text()
            except Exception as e:
                logger.warning(f"Failed to read cached WEC file {cache_file}: {e}")
                text_to_parse = None
        else:
            # download and cache
            try:
                logger.info(f"Fetching data for {date_str}")
                resp = requests.get(file_url, timeout=30)
                if resp.status_code == 404:
                    logger.warning(f"File not found for {date_str}")
                    text_to_parse = None
                else:
                    text_to_parse = resp.text
                    try:
                        cache_file.write_text(resp.text)
                        logger.info(f"Cached WEC text to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to write WEC cache {cache_file}: {e}")
            except Exception as e:
                logger.error(f"Error fetching data for {date_str}: {e}")
                text_to_parse = None

        if text_to_parse is not None and len(text_to_parse) == 0:
            logger.warning(f"Empty WEC text file for {date_str}")
            text_to_parse = None

        if text_to_parse is not None:
            logger.info(f"Parsing data for {date_str}")
            ds1, _ = parse_putty_log(text_to_parse)
            dsl.append(ds1)

    ds = xr.concat(dsl, dim="time")
    ds = ds.sortby("time")

    df = pd.read_csv(
        "Deployment1_Schedule.csv",
        header=None,
        names=["time", "Mode", "Gain"],
        index_col=False,
        sep=",",
    )
    df = df.set_index("time")
    dsg = df.to_xarray()
    dsg["time"] = pd.to_datetime(dsg["time"]).values
    dsg = dsg["Gain"].interp_like(ds["DcP"], method="zero")
    dsg = dsg.fillna(0.130)  # Default value for damping gain
    ds = xr.merge([ds, dsg])

    return ds


def fetch_pwrsys_data(start_date: datetime = None) -> xr.Dataset:
    """
    Fetch power system data (solar PV panels and wind turbines) from OOI.
    All device data is contained in a single daily log file.

    Args:
        start_date: Start date for data fetch. Defaults to 7 days ago.

    Returns:
        xr.Dataset: Power system data with variables (status, voltage, current) and dimensions (device, time)
    """

    logger.info(f"Fetching power system data")

    end_time = pd.Timestamp.utcnow()
    current_date = end_time.date()

    if start_date is None:
        start_date = current_date - timedelta(days=7)

    dates_to_try = [
        start_date + timedelta(days=i)
        for i in range((current_date - start_date).days + 1)
    ]

    base_url = (
        "https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/pwrsys"
    )

    all_data = []
    _ensure_pwrsys_text_cache_dir()

    for attempt_date in dates_to_try:
        date_str = attempt_date.strftime("%Y%m%d")
        file_url = f"{base_url}/{date_str}.pwrsys.log"
        cache_file = _pwrsys_text_cache_file(date_str)

        text_to_parse = None

        # prefer cached text file
        if cache_file.exists():
            try:
                logger.info(
                    f"Loading cached power system text for {date_str} from {cache_file}"
                )
                text_to_parse = cache_file.read_text()
            except Exception as e:
                logger.warning(
                    f"Failed to read cached power system file {cache_file}: {e}"
                )
                text_to_parse = None
        else:
            # download and cache
            try:
                logger.info(f"Fetching power system data for {date_str}")
                resp = requests.get(file_url, timeout=30)

                if resp.status_code == 404:
                    logger.debug(f"Power system file not found for {date_str}")
                    text_to_parse = None
                elif resp.status_code != 200:
                    logger.warning(
                        f"Failed to fetch power system data for {date_str}: HTTP {resp.status_code}"
                    )
                    text_to_parse = None
                else:
                    text_to_parse = resp.text
                    try:
                        cache_file.write_text(resp.text)
                        logger.info(f"Cached power system text to {cache_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to write power system cache {cache_file}: {e}"
                        )
            except Exception as e:
                logger.warning(f"Error fetching power system data for {date_str}: {e}")
                text_to_parse = None

        if text_to_parse is not None and len(text_to_parse) == 0:
            logger.debug(f"Empty power system data for {date_str}")
            text_to_parse = None

        if text_to_parse is not None:
            # Parse the data
            df = _parse_pwrsys_log(text_to_parse)
            if df is not None and len(df) > 0:
                all_data.append(df)
                logger.info(f"Successfully parsed power system data for {date_str}")

    if not all_data:
        logger.error("No power system data was successfully fetched")
        return xr.Dataset()

    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=False)
    df_combined = df_combined.sort_index()

    # Extract device names and variable types (include batteries)
    devices = ["pv1", "pv2", "pv3", "pv4", "wt1", "wt2", "bt1", "bt2", "bt3", "bt4"]
    var_map = {
        "status": "status",
        "voltage": "voltage",
        "current": "current",
        "temperature": "temp",
    }

    # Determine which devices have any data in the combined dataframe
    device_coords = [
        d
        for d in devices
        if any(col.startswith(f"{d}_") for col in df_combined.columns)
    ]

    # Build 2D arrays for each variable type: (device, time)
    data_vars = {}
    time_len = len(df_combined.index)

    for var_name, col_suffix in var_map.items():
        var_data = []
        for device in device_coords:
            col_name = f"{device}_{col_suffix}"
            if col_name in df_combined.columns:
                var_data.append(df_combined[col_name].values)
            else:
                # fill missing device variable with NaNs
                var_data.append(np.full(time_len, np.nan))

        if any(~np.isnan(np.array(var_data)).all(axis=1)) or True:
            # Stack into 2D array (device, time)
            data_vars[var_name] = (["device", "time"], np.array(var_data))

    if not data_vars or not device_coords:
        logger.error("No device data found in parsed power system log")
        return xr.Dataset()

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars, coords={"device": device_coords, "time": df_combined.index.values}
    )

    # Assign device types
    gtype = {"pv": "solar", "wt": "wind", "bt": "battery"}
    ds = ds.assign_coords(
        gtype=(
            "device",
            [gtype.get(d[:2], "unknown") for d in device_coords],
            {"long_name": "Generation type"},
        ),
    )

    # Add units and long names
    ds["time"].attrs["long_name"] = "Time"
    ds["status"].attrs["units"] = "-"
    ds["status"].attrs["long_name"] = "Device Status"
    ds["voltage"].attrs["units"] = "V"
    ds["voltage"].attrs["long_name"] = "Voltage"
    ds["current"] = ds["current"] / 1e3
    ds["current"].attrs["units"] = "A"
    ds["current"].attrs["long_name"] = "Current"
    ds["temperature"].attrs["units"] = "°C"
    ds["temperature"].attrs["long_name"] = "Battery temperature"

    ds1 = ds.where(ds["gtype"] == "battery").dropna(dim="device", how="all")
    ds["ocv"] = ds1["voltage"] + 0.368 * ds1["current"]
    ds["soc"] = (ds["ocv"] - 23.16) * 100 / 2.4
    ds["soc"].attrs["units"] = "%"
    ds["soc"].attrs["long_name"] = "State of charge"

    ds = ds.sortby("time")

    return ds


def _parse_pwrsys_log(content: str) -> pd.DataFrame:
    """
    Parse power system log data from OOI containing all devices.
    Log format: YYYY/MM/DD HH:MM:SS.mmm PwrSys ... pv1 status voltage current pv2 status voltage current ... wt1 status voltage current ...

    For each device: status (integer), voltage (float), current (float)

    Args:
        content: Raw text content from the log file

    Returns:
        pd.DataFrame: Parsed data with time index and columns for each device
    """

    lines = content.strip().split("\n")

    # Skip header lines (those starting with #)
    data_lines = [line for line in lines if line.strip() and not line.startswith("#")]

    if not data_lines:
        logger.warning(f"No data lines found in power system log")
        return None

    try:
        data = []
        devices_of_interest = [
            "pv1",
            "pv2",
            "pv3",
            "pv4",
            "wt1",
            "wt2",
            "bt1",
            "bt2",
            "bt3",
            "bt4",
        ]

        for line in data_lines:
            parts = line.split()

            if len(parts) < 3:
                continue

            try:
                # Parse timestamp: first two tokens are date and time
                timestamp_str = f"{parts[0]} {parts[1]}"
                timestamp = pd.to_datetime(timestamp_str)

                # Find each device in the line and extract its values
                row_data = {"time": timestamp}

                for device in devices_of_interest:
                    try:
                        device_idx = parts.index(device)

                        # Batteries report: temp, voltage, current (mA)
                        if device.startswith("bt"):
                            if device_idx + 3 < len(parts):
                                temp = float(parts[device_idx + 1])
                                voltage = float(parts[device_idx + 2])
                                current = float(parts[device_idx + 3])

                                row_data[f"{device}_temp"] = temp
                                row_data[f"{device}_voltage"] = voltage
                                row_data[f"{device}_current"] = current
                        else:
                            # pv and wt report: status, voltage, current
                            if device_idx + 3 < len(parts):
                                status = int(parts[device_idx + 1])
                                voltage = float(parts[device_idx + 2])
                                current = float(parts[device_idx + 3])

                                row_data[f"{device}_status"] = status
                                row_data[f"{device}_voltage"] = voltage
                                row_data[f"{device}_current"] = current
                    except (ValueError, IndexError):
                        # Device not found or malformed in this line
                        pass

                # Only add row if we found at least some device data
                if len(row_data) > 1:
                    data.append(row_data)

            except (ValueError, IndexError):
                logger.debug(f"Could not parse line: {line}")
                continue

        if not data:
            return None

        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(data)
        df.set_index("time", inplace=True)

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        logger.error(f"Error parsing power system log: {e}")
        return None


def resample_and_combine(ds_wec, dsl, freq="1H"):
    ds1 = ds_wec.resample(time=freq).mean()

    dstm = []
    for dsi in dsl:
        dsi = dsi.dropna("time", how="all").resample(time=freq).mean()
        dstm.append(dsi)

    ds0 = xr.merge([ds1] + dstm)
    ds0 = ds0.sel(time=slice(ds1["time"][0], ds1["time"][-1]))

    return ds0


def make_scatter_3d(ds):
    dstp = ds.mean("buoy")[["WVHT", "dir_diff", "DcP"]].dropna(dim="time", how="any")
    fig = px.scatter_3d(
        dstp,
        x="WVHT",
        y="dir_diff",
        z="DcP",
        color="DcP",
        size="DcP",
        color_continuous_scale="reds",
        opacity=0.5,
        labels={
            "WVHT": r"Sig. wave height [m]",
            "DcP": "DC power [W]",
            "dir_diff": "Direction diff. [°]",
        },
    )
    fig.update_layout(
        # template='simple_white',
        xaxis_title="$H_{m0}$ [m]",
        scene_camera=dict(eye=dict(x=2.0, y=2.0, z=0.75)),
        height=1200,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_traces(
        customdata=dstp.time.dt.strftime("%Y-%m-%d %H:%M"),
        hovertemplate="""
        %{customdata}
        <extra></extra>
        """,
    )
    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(tickmode="array", tickvals=[0, 30, 60, 90, 120, 150, 180]),
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def make_time_hist(dstp):

    # no subplot titles; show info in y-axis labels instead
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    vars_to_plot = [
        {"WVHT": "#1f77b4"},
        {"WSPD": "#17becf"},
        {"DPD": "#1f77b4", "APD": "#605aff"},
        {"MWD": "#1f77b4", "WDIR": "#17becf"},
    ]
    for i, vars in enumerate(vars_to_plot):
        for var, mcolor in vars.items():
            dftp = dstp[var].to_pandas()
            for col, color in zip(dftp.columns, colors):
                fig.add_trace(
                    go.Scatter(
                        x=dftp.index,
                        y=dftp[col],
                        mode="lines",
                        name=col,
                        line=dict(color=mcolor, width=0.5),
                        # hovertemplate="%{y:.1f}",
                        hoverinfo="skip",
                    ),
                    row=i + 1,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    x=dftp.index,
                    y=dftp.mean(axis=1),
                    mode="lines",
                    name=f"{dstp[var].attrs['long_name']}",
                    line=dict(color=mcolor, width=2),
                    hovertemplate="%{y:.1f}" + dstp[var].attrs["units"],
                ),
                row=i + 1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=dstp["ExP"].time,
            y=dstp["ExP"].clip(0, np.infty),
            name="Export power",
            mode="lines",
            line=dict(color="#ff0eb3"),
            hovertemplate="%{y:.1f} W",
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dstp["DcP"].time,
            y=dstp["DcP"],
            name="DC bus power",
            mode="lines",
            line=dict(color="black"),
            hovertemplate="%{y:.1f} W",
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dstp["Gain"].time,
            y=dstp["Gain"],
            name="Damping gain",
            mode="lines",
            line_shape="hv",
            line=dict(color="black"),
            hovertemplate="%{y:.3f} As/rad",
        ),
        row=6,
        col=1,
    )

    pow = ds["current"] * ds["voltage"]
    pow = pow.groupby("gtype").sum().sel(gtype=["solar", "wind"])

    fig.add_trace(
        go.Scatter(
            x=pow.sel(gtype="wind").time,
            y=pow.sel(gtype="wind"),
            name="Wind",
            mode="lines",
            line=dict(color="#17becf"),
            hovertemplate="%{y:.1f} W",
        ),
        row=7,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=pow.sel(gtype="solar").time,
            y=pow.sel(gtype="solar"),
            name="Solar",
            mode="lines",
            line=dict(color="#ffb70e"),
            hovertemplate="%{y:.1f} W",
        ),
        row=7,
        col=1,
    )

    # soc = ds["soc"].mean(dim="device").clip(0, 100)

    # fig.add_trace(
    #     go.Scatter(
    #         x=soc.time,
    #         y=soc,
    #         name="State of charge",
    #         mode="lines",
    #         line=dict(color="black"),
    #         hovertemplate="%{y:.1f} W",
    #     ),
    #     row=8,
    #     col=1,
    # )

    ds1 = ds.where(ds["gtype"] == "battery").dropna(dim="device", how="all")
    ds2 = ds1["voltage"].mean(dim="device")

    fig.add_trace(
        go.Scatter(
            x=ds2.time,
            y=ds2,
            name="Battery voltage",
            mode="lines",
            line=dict(color="black"),
            hovertemplate="%{y:.1f} W",
        ),
        row=8,
        col=1,
    )

    fig.update_layout(
        height=1400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Sig. wave<br>height [m]", row=1, col=1)
    fig.update_yaxes(title_text="Wind<br>speed [m/s]", row=2, col=1)
    fig.update_yaxes(title_text="Wave<br>period [s]", row=3, col=1)
    fig.update_yaxes(title_text="Wave & wind<br>dir. [deg]", row=4, col=1)
    fig.update_yaxes(title_text="WEC<br>power [W]", range=[0, np.infty], row=5, col=1)
    fig.update_yaxes(title_text="Damping gain<br>[As/rad]", row=6, col=1)
    fig.update_yaxes(title_text="Power<br>[W]", row=7, col=1)
    # fig.update_yaxes(title_text="State of<br>charge [-]", row=8, col=1)
    fig.update_yaxes(title_text="Battery<br>voltage [V]", row=8, col=1)

    fig.update_layout(
        # title="Pioneer WEC",
        template="simple_white",
        hovermode="x unified",
        height=1000,
        margin=dict(l=60, r=40, t=100, b=50),
        showlegend=False,
        font=dict(size=10),
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=2, label="2d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            )
        )
    )

    return fig


def make_wec_histograms(ds):
    df = ds[["DcP", "ExP"]].to_array(dim="type").to_pandas().transpose()
    df.rename(columns={"DcP": "DC", "ExP": "Export"}, inplace=True)

    fig = px.histogram(
        df,
        marginal="box",
        labels={"value": "Power [W]", "type": "Type"},
        color_discrete_sequence=["black", "#ff0eb3"],
        orientation="h",
        barmode="overlay",
    )

    fig.update_layout(
        xaxis_title="Count[-]",
        yaxis_title="Power [W]",
    )
    fig.update_yaxes(range=[0, np.infty])

    fig.update_layout(
        template="simple_white",
        # hovermode="y unified",
        # height=1000,
        # margin=dict(l=60, r=40, t=100, b=50),
        # showlegend=False,
        # font=dict(size=10),
    )

    return fig


def make_correlation_matrix(ds):
    ds0 = ds.mean("buoy")
    fig = px.scatter_matrix(
        ds0[
            ["WVHT", "WSPD", "DPD", "APD", "dir_diff", "DcP", "ExP", "Vel"]
        ].to_pandas(),
        labels={
            "WVHT": "Wave Height<br>[m]",
            "WSPD": "Wind Speed<br>[m/s]",
            "DPD": "Peak Period<br>[s]",
            "APD": "Average period<br>[s]",
            "dir_diff": "Wave/wind dir.<br>diff.[deg]",
            "DcP": "DC power<br>[W]",
            "ExP": "Export power<br>[W]",
            "Vel": "RMS velocity<br>[deg/s]",
        },
        width=800,
        height=800,
    )
    fig.update_layout(
        font=dict(size=8),
    )
    fig.update_traces(marker=dict(size=5, color="black", opacity=0.25))
    return fig


def make_jpd(ds):
    ds0 = ds.mean("buoy")
    fig = px.density_heatmap(
        ds0[["DPD", "WVHT"]],
        x="DPD",
        y="WVHT",
        #  color_continuous_scale='Viridis',
        labels={"DPD": "Peak wave period [s]", "WVHT": "Sig. wave height [m]"},
        marginal_x="histogram",
        marginal_y="histogram",
    )

    return fig


def make_power_matrix(ds):
    ds0 = ds.mean("buoy")
    fig = px.density_heatmap(
        ds0[["DPD", "WVHT", "DcP"]],
        x="DPD",
        y="WVHT",
        z="DcP",
        color_continuous_scale="Reds",
        histfunc="avg",
        labels={
            "DPD": "Peak wave period [s]",
            "WVHT": "Sig. wave height [m]",
            "DcP": "DC power [W]",
        },
    )

    fig.add_trace(
        go.Scatter(
            x=ds0["DPD"],
            y=ds0["WVHT"],
            mode="markers",
            marker=dict(
                color="black",  # Set point color to black
                size=5,  # Set point size
                opacity=0.25,  # Set point opacity
            ),
            hoverinfo="skip",
        )
    )

    return fig


def make_cw_matrix(ds, tp_to_te=0.9):
    ds0 = ds.mean("buoy")
    ds0["Te"] = ds0["DPD"] * tp_to_te
    ds0["J"] = ds0["Te"] * ds0["WVHT"] ** 2 * 1025 * 9.81**2 / (64 * np.pi)
    ds0["cw"] = ds0["DcP"] / ds0["J"]
    fig = px.density_heatmap(
        ds0[["DPD", "WVHT", "cw"]],
        x="DPD",
        y="WVHT",
        z="cw",
        color_continuous_scale="Reds",
        histfunc="avg",
        labels={
            "DPD": "Peak wave period [s]",
            "WVHT": "Sig. wave height [m]",
            "cw": "Capture width [m]",
        },
    )

    fig.add_trace(
        go.Scatter(
            x=ds0["DPD"],
            y=ds0["WVHT"],
            mode="markers",
            marker=dict(
                color="black",  # Set point color to black
                size=5,  # Set point size
                opacity=0.25,  # Set point opacity
            ),
            hoverinfo="skip",
        )
    )

    return fig


def make_gain_scatter(ds):
    dstp = ds[["Gain", "DcP"]].groupby_bins("Gain", bins=20).mean().dropna("Gain_bins")

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=dstp["Gain"],
            y=dstp["DcP"],
            mode="lines",
            name="DC power",
            line=dict(color="black"),
            hovertemplate="%{y:.1f}W",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=ds["Gain"],
            y=ds["DcP"],
            mode="markers",
            # name="Gain",
            marker=dict(size=4, color="black", opacity=0.25),
            hoverinfo="skip",
        ),
    )

    fig.update_layout(
        template="simple_white",
        showlegend=False,
    )

    fig.update_layout(
        xaxis_title="Damping gain [As/rad]",
        yaxis_title="Power [W]",
    )
    fig.update_yaxes(range=[0, np.infty])

    return fig


def make_calendar(ds):
    fig = calplot(
        ds["DcP"].to_dataframe().resample("D").mean().reset_index(),
        x="time",
        y="DcP",
        start_month=11,
        month_lines_width=5,
        month_lines_color="black",
        colorscale="reds",
        name="DC power",
    )

    return fig


def make_generators_box(ds):
    pow = ds["current"] * ds["voltage"]
    pow = pow.groupby("gtype").sum().sel(gtype=["solar", "wind"])
    tmp1 = ds["DcP"].expand_dims(dim={"gtype": ["WEC"]})
    pow.name = "Power"
    pow = xr.concat([pow, tmp1], dim="gtype")
    pow.attrs = {"units": "W", "long_name": "Power"}
    df = pow.dropna(dim="time").to_pandas()

    fig = px.box(
        df,
        #  log_y=True,
        labels={"value": "Power [W]", "gtype": "Type"},
        points="all",
        notched=False,
        color_discrete_sequence=px.colors.qualitative.Set1,
        color="gtype",
    )

    fig.update_traces(boxmean=True)

    fig.update_layout(
        template="simple_white",
    )
    # fig.update_yaxes(range=[0, np.infty])
    fig.update_xaxes(ticktext=["Solar", "Wind", "WEC"])

    return fig


if __name__ == "__main__":

    _ensure_data_dir()

    start_date = datetime(2025, 11, 3).date()

    ds_pwrsys = fetch_pwrsys_data(start_date=start_date)
    ds_pwrsys.to_netcdf(
        os.path.join(DATA_DIR, "pwrsys_data.h5"), engine="h5netcdf", invalid_netcdf=True
    )

    buoys = ["44014", "44079", "41083", "44095"]
    ds_ndbc = xr.concat(
        [fetch_ndbc(buoy_id=buoy_id, start_date=start_date) for buoy_id in buoys],
        dim="buoy",
    )
    ds_ndbc.to_netcdf(
        os.path.join(DATA_DIR, "ndbc_data.h5"), engine="h5netcdf", invalid_netcdf=True
    )

    ds_wec = fetch_wec_data(start_date=start_date)
    ds_wec.to_netcdf(
        os.path.join(DATA_DIR, "wec_data.h5"), engine="h5netcdf", invalid_netcdf=True
    )

    ds = resample_and_combine(ds_wec, [ds_ndbc, ds_pwrsys])

    logger.info("Generating plots")

    fig1 = make_scatter_3d(ds)
    fig1.write_html("output/scatter_3d.html")

    fig2 = make_time_hist(ds)
    fig2.write_html("output/time_hist.html")

    fig3 = make_correlation_matrix(ds)
    fig3.write_html("output/correlation_matrix.html")

    fig4 = make_jpd(ds)
    fig4.write_html("output/jpd.html")

    fig5 = make_power_matrix(ds)
    fig5.write_html("output/power_matrix.html")

    fig6 = make_cw_matrix(ds)
    fig6.write_html("output/cw_matrix.html")

    fig7 = make_wec_histograms(ds)
    fig7.write_html("output/histograms.html")

    fig8 = make_gain_scatter(ds)
    fig8.write_html("output/gain_scatter.html")

    fig9 = make_calendar(ds)
    fig9.write_html("output/calendar.html")

    fig10 = make_generators_box(ds)
    fig10.write_html("output/generators_box.html")
