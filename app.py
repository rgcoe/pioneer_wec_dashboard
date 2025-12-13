import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotly
import requests
import logging
from parse_wec_decimated_log import parse_putty_log
import os
from pathlib import Path

colors = plotly.colors.DEFAULT_PLOTLY_COLORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# directory to cache raw WEC text files
WEC_TEXT_CACHE = Path(".cache")
DATA_DIR = Path("output/data")


def _ensure_wec_text_cache_dir() -> None:
    WEC_TEXT_CACHE.mkdir(parents=True, exist_ok=True)


def _wec_text_cache_file(date_str: str) -> Path:
    return WEC_TEXT_CACHE / f"{date_str}.wec.dec.10.log"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ndbc(buoy_id: str = "44014", max_retries: int = 3) -> xr.Dataset:

    logger.info(f"Fetching NDBC data for buoy {buoy_id}")

    url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"

    df = pd.read_csv(
        url,
        sep=r"\s+",
        comment="#",
        na_values="MM",
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
        "PTDY",
        "TIDE",
    ]

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
    ds = ds.expand_dims("buoy").assign_coords(buoy=[buoy_id])
    long_names = {
        "WDIR": "Wind Direction",
        "WSPD": "Wind Speed",
        "GST": "Wind Gust",
        "WVHT": "Significant Wave Height",
        "DPD": "Dominant Wave Period",
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
    dsg = dsg.interp_like(ds["DcP"], method="zero")

    ds = xr.merge([ds, dsg["Gain"]])

    return ds


def resample_and_combine(ds_wec, ds_ndbc, freq="1H"):
    ds1 = ds_wec.resample(time=freq).mean()

    ds2 = ds_ndbc.dropna("time", how="all").resample(time=freq).mean()

    dir_diff = np.abs(ds2["MWD"] - ds2["WDIR"]) % 180
    dir_diff.name = "dir_diff"
    dir_diff.attrs["units"] = "°"
    dir_diff.attrs["long_name"] = "Direction Difference"

    ds0 = xr.merge([ds1, ds2, dir_diff])
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
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # vars_to_plot = [[{"WVHT":"#1f77b4"}], [{"WSPD":"#17becf"}], [{"DPD":"#1f77b4"}], [{"MWD":"#1f77b4"}, {"WDIR":"#17becf"}]]
    vars_to_plot = [
        {"WVHT": "#1f77b4"},
        {"WSPD": "#17becf"},
        {"DPD": "#1f77b4"},
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
            y=dstp["ExP"],
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

    fig.update_layout(
        height=1400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # move descriptive info into y-axis labels
    fig.update_yaxes(title_text="Significant wave height (m)", row=1, col=1)
    fig.update_yaxes(title_text="Wind speed (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Dominant wave period (s)", row=3, col=1)
    fig.update_yaxes(title_text="Wave & wind direction (deg)", row=4, col=1)
    fig.update_yaxes(title_text="WEC DC bus power (W)", row=5, col=1)

    # label x-axis on the bottom subplot
    # fig.update_xaxes(title_text='Time', row=5, col=1)

    fig.update_layout(
        # title="Pioneer WEC",
        template="simple_white",
        hovermode="x unified",
        height=1000,
        margin=dict(l=60, r=40, t=100, b=50),
        showlegend=False,
        font=dict(size=10),
    )

    return fig


def make_histograms(ds):
    df = ds[["DcP", "ExP"]].to_array(dim="type").to_pandas().transpose()
    df.rename(columns={"DcP": "DC", "ExP": "Export"}, inplace=True)

    fig = px.histogram(
        df,
        marginal="box",
        labels={"value": "Power (W)", "type": "Type"},
        color_discrete_sequence=["black", "#ff0eb3"],
        orientation="h",
    )

    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Power (W)",
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
            "WVHT": "Wave Height<br>(m)",
            "WSPD": "Wind Speed<br>(m/s)",
            "DPD": "Peak Period<br>(s)",
            "APD": "Average period<br>(s)",
            "dir_diff": "Wave/wind direction<br>diff.(deg)",
            "DcP": "DC power<br>(W)",
            "ExP": "Export power<br>(W)",
            "Vel": "RMS velocity<br>(deg/s)",
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
        labels={"DPD": "Peak Wave Period (s)", "WVHT": "Significant Wave Height (m)"},
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
            "DPD": "Peak Wave Period (s)",
            "WVHT": "Significant Wave Height (m)",
            "DcP": "DC power (w)",
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
            "DPD": "Peak Wave Period (s)",
            "WVHT": "Significant Wave Height (m)",
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
    fig = px.scatter(
        ds[["Gain", "DcP"]],
        x="Gain",
        y="DcP",
        labels={"Gain": "Damping gain [As/rad]", "DcP": "DC bus power [W]"},
    )
    fig.update_layout(
        template="simple_white",
        # hovermode="y unified",
        # height=1000,
        # margin=dict(l=60, r=40, t=100, b=50),
        # showlegend=False,
        # font=dict(size=10),
    )
    fig.update_traces(marker=dict(size=5, color="black", opacity=0.5))

    return fig


if __name__ == "__main__":

    _ensure_data_dir()

    buoys = ["44014", "44079", "41083", "44095"]
    ds_ndbc = xr.concat([fetch_ndbc(buoy_id=buoy_id) for buoy_id in buoys], dim="buoy")
    ds_ndbc.to_netcdf(
        os.path.join(DATA_DIR, "ndbc_data.h5"), engine="h5netcdf", invalid_netcdf=True
    )

    start_date = datetime(2025, 11, 3).date()
    ds_wec = fetch_wec_data(start_date=start_date)
    ds_wec.to_netcdf(
        os.path.join(DATA_DIR, "wec_data.h5"), engine="h5netcdf", invalid_netcdf=True
    )

    ds = resample_and_combine(ds_wec, ds_ndbc)

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

    fig7 = make_histograms(ds)
    fig7.write_html("output/histograms.html")

    fig8 = make_gain_scatter(ds)
    fig8.write_html("output/gain_scatter.html")
