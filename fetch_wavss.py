#!/usr/bin/env python3
"""
Fetch 72 hours of significant wave height data from NDBC Buoy 44014
and generate an interactive Plotly HTML graph.
"""

import re
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xarray as xr
from pathlib import Path as _Path


def fetch_ndbc_wave_data(buoy_id: str = "44014", hours: int = 72, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch significant wave height data from NDBC buoy via HTTP.
    
    NDBC provides real-time data in plain text format.
    
    Args:
        buoy_id: NDBC buoy ID (default: 44014 - Delaware Bay entrance)
        hours: Number of hours back to fetch (default: 72)
        max_retries: Number of retry attempts for network failures
    
    Returns:
        DataFrame with time index and wave height columns
    """
    # Calculate time range (UTC-aware for comparison with DataFrame index)
    end_time = pd.Timestamp.utcnow()
    start_time = end_time - pd.Timedelta(hours=hours)
    
    print(f"Fetching {hours} hours of NDBC data from {start_time} to {end_time}")
    print(f"Buoy: {buoy_id}")
    
    # Retry loop for network issues
    for attempt in range(1, max_retries + 1):
        try:
            # NDBC standard data URL
            url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"
            
            print(f"Fetching from {url}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            
            # Parse NDBC format (space-separated, with header rows)
            lines = resp.text.strip().split('\n')
            
            if len(lines) < 3:
                print(f"No data returned for buoy {buoy_id}", file=sys.stderr)
                return pd.DataFrame()
            
            # NDBC format has two header lines starting with #, then data
            # Parse using pandas: skip comment lines and the units line
            lines = resp.text.strip().split('\n')
            
            # Find first data line (first non-comment line after headers)
            header_line = None
            data_start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('#YY'):
                    header_line = line.lstrip('#').strip()
                    data_start_idx = i + 2  # Skip units line too
                    break
            
            if header_line is None:
                print("Error: Could not find NDBC header line", file=sys.stderr)
                return pd.DataFrame()
            
            # Parse data from data_start_idx onward
            data_lines = '\n'.join(lines[data_start_idx:])
            df = pd.read_csv(
                StringIO(data_lines),
                delim_whitespace=True,
                na_values=['MM'],
                names=header_line.split(),  # Use extracted header
            )
            
            print(f"Parsed DataFrame: {len(df)} rows, columns: {list(df.columns)}")
            
            # Standard NDBC columns: YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD
            
            # Create datetime from date/time columns
            time_cols = ['YY', 'MM', 'DD', 'hh', 'mm']
            found_cols = [c for c in time_cols if c in df.columns]
            print(f"Time columns found: {found_cols}")
            
            if all(c in df.columns for c in time_cols):
                df['time'] = pd.to_datetime(
                    df[time_cols].rename(
                        columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}
                    ),
                    utc=True
                )
                df = df.set_index('time')
            else:
                print(f"Error: Missing time columns. Expected {time_cols}, found {found_cols}", file=sys.stderr)
                return pd.DataFrame()
            
            # Filter to requested time range
            df = df[(df.index >= start_time) & (df.index <= end_time)]
            print(f"After time filter: {len(df)} records, date range: {df.index.min()} to {df.index.max()}")
            
            # NOTE: don't drop non-numeric columns here — some direction fields may be parsed
            # as objects (e.g., containing 'MM' or other markers). We'll select and coerce
            # the specific columns we need after renaming.
            
            # Extract wave height, period (prefer DPD; fallback to APD), and directions
            cols_to_keep = ['DPD', 'WVHT', 'WDIR', 'MWD', 'WSPD']

            df = df[cols_to_keep].copy()
            # Build rename map
            rename_map = {'WVHT': 'significant_wave_height',
                          'DPD': 'dominant_period',
                          'WDIR': 'wind_direction',
                          'MWD': 'mean_wave_direction',
                          'WSPD': 'wind_speed'}

            df = df.rename(columns=rename_map)

            # Only drop rows where ALL columns are NaN (this allows partial data)
            # Coerce key columns to numeric where appropriate (invalid parse -> NaN)
            for col in ('significant_wave_height', 'dominant_period', 'wind_direction', 'mean_wave_direction', 'wind_speed'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(how='all')
            df = df.sort_index()

            
            if df.empty:
                print(f"No valid data after filtering for buoy {buoy_id}", file=sys.stderr)
                return pd.DataFrame()
            
            print(f"Successfully fetched {len(df)} records from NDBC buoy {buoy_id}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)
            if attempt < max_retries:
                wait_time = 5 * attempt
                print(f"Retrying in {wait_time} seconds...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                raise


def fetch_ooi_dc_power(hours: int = 72, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch DC bus power data from OOI CP10CNSM decimated logs.
    
    Fetches multiple daily log files to ensure full coverage of requested time window.
    Data is available in daily log files at:
    https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/dcl12/wec_decimated/
    
    Format: YYYY/MM/DD HH:MM:SS.sss ... DcP: <value> ...
    
    Args:
        hours: Number of hours back to fetch (default: 72)
        max_retries: Number of retry attempts for network failures
    
    Returns:
        DataFrame with time index and dc_bus_power column, or None if unavailable
    """
    # Calculate time range (UTC-aware)
    end_time = pd.Timestamp.utcnow()
    start_time = end_time - pd.Timedelta(hours=hours)
    
    print(f"\nFetching OOI DC power data from {start_time} to {end_time}")
    print(f"Will fetch multiple daily files to cover full {hours}-hour window")
    
    # Generate list of dates to try (most recent first)
    # For 72 hours, we might need up to 4 days of data depending on time boundaries
    current_date = end_time.date()
    dates_to_try = []
    for i in range(hours // 24 + 2):  # Try enough days to cover the window
        date = current_date - timedelta(days=i)
        dates_to_try.append(date)
    
    base_url = "https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/dcl12/wec_decimated"
    all_data = []
    files_fetched = 0
    
    # Fetch from multiple dates to ensure full coverage
    for attempt_date in dates_to_try:
        date_str = attempt_date.strftime("%Y%m%d")
        file_url = f"{base_url}/{date_str}.wec.dec.10.log"
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Fetching {date_str}.wec.dec.10.log")
                resp = requests.get(file_url, timeout=30)
                
                if resp.status_code == 404:
                    # File doesn't exist, try next date
                    print(f"  → File not found")
                    break
                
                resp.raise_for_status()
                
                # Parse the log file
                # Format: YYYY/MM/DD HH:MM:SS.sss ... DcP: <value> ...
                data_count = 0
                for line in resp.text.strip().split('\n'):
                    if not line.strip():
                        continue
                    
                    try:
                        # Extract timestamp (YYYY/MM/DD HH:MM:SS.sss)
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        
                        date_part = parts[0]  # YYYY/MM/DD
                        time_part = parts[1]  # HH:MM:SS.sss
                        
                        # Parse datetime
                        dt = pd.to_datetime(f"{date_part} {time_part}", utc=True)
                        
                        # Extract DcP and ExP values using regex patterns
                        dcp_match = re.search(r'DcP:\s*([-+]?\d+\.?\d*e[+-]?\d+|[-+]?\d+\.?\d*)', line)
                        exp_match = re.search(r'ExP:\s*([-+]?\d+\.?\d*e[+-]?\d+|[-+]?\d+\.?\d*)', line)
                        if dcp_match or exp_match:
                            row_data = {'time': dt}
                            if dcp_match:
                                dcp_value = float(dcp_match.group(1))
                                # Reverse sign by multiplying by -1
                                row_data['dc_bus_power'] = -dcp_value
                            if exp_match:
                                exp_value = float(exp_match.group(1))
                                row_data['export_power'] = exp_value - 8
                            all_data.append(row_data)
                            data_count += 1
                    except (ValueError, IndexError):
                        continue
                
                if data_count > 0:
                    print(f"  → Fetched {data_count} records from {date_str}")
                    files_fetched += 1
                else:
                    print(f"  → No valid data in {date_str}")
                
                # Continue to next date, don't break (we want to fetch multiple files)
                break
                
            except Exception as e:
                print(f"  Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    break
    
    if not all_data:
        print("Warning: Could not fetch OOI DC power data. Proceeding with NDBC data only.")
        return None
    
    # Combine all fetched data into single DataFrame
    df = pd.DataFrame(all_data)
    df = df.set_index('time')
    df = df.sort_index()
    
    # Filter to requested time range
    df = df[(df.index >= start_time) & (df.index <= end_time)]
    
    if df.empty:
        print("Warning: No OOI data within time range. Proceeding with NDBC data only.")
        return None
    
    print(f"Successfully fetched {len(df)} total records from {files_fetched} file(s)")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    # Save full OOI DataFrame to NetCDF for downstream use
    try:
        out_dir = _Path('/app/output')
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = df.to_xarray()
        ds['time'] = pd.to_datetime(ds['time'].values)
        nc_path = out_dir / 'ooi_data.nc'
        ds.to_netcdf(str(nc_path))
        print(f"Saved OOI data to {nc_path}")
    except Exception as e:
        print(f"Warning: failed to write OOI NetCDF: {e}")

    return df


def create_contour_plot(df: pd.DataFrame, dc_power_df: Optional[pd.DataFrame], start_date: Optional[pd.Timestamp] = None) -> Optional[go.Figure]:
    """
    Create a contour plot showing DC bus power as a function of wave period (x) and height (y).
    
    Args:
        df: DataFrame with wave height and period data
        dc_power_df: DataFrame with DC power data
        start_date: Optional start date for filtering data (default: None, uses all data)
    
    Returns:
        Plotly Figure object or None if data is insufficient
    """
    if dc_power_df is None or dc_power_df.empty:
        return None
    
    # Check if we have both wave height and period
    if 'significant_wave_height' not in df.columns or 'dominant_period' not in df.columns:
        return None
    
    # Filter by start date if provided
    if start_date is not None:
        df = df[df.index >= start_date]
        dc_power_df = dc_power_df[dc_power_df.index >= start_date]
    
    # Align the dataframes to same time index using interpolation
    # Create a common index and interpolate both datasets to that index
    min_time = max(df.index.min(), dc_power_df.index.min())
    max_time = min(df.index.max(), dc_power_df.index.max())
    
    # Create hourly index for interpolation
    common_index = pd.date_range(min_time, max_time, freq='1H')
    # Use helper function to resample and merge data
    merged_df = resample_and_merge(df, dc_power_df, freq='1H', start_date=start_date)
    if merged_df is None or merged_df.empty:
        print("Warning: merged data empty for contour plot")
        return None
    
    if len(merged_df) < 20:
        print("Warning: Not enough data points for contour plot")
        return None
    
    # Create 2D histogram bins for period and height
    period_bins = 15
    height_bins = 15
    
    # Create bin edges
    x_edges = np.linspace(merged_df['dominant_period'].min(), merged_df['dominant_period'].max(), period_bins + 1)
    y_edges = np.linspace(merged_df['significant_wave_height'].min(), merged_df['significant_wave_height'].max(), height_bins + 1)
    
    # Create bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create 2D array of average DC power for each bin
    z = np.zeros((len(y_centers), len(x_centers)))
    counts = np.zeros((len(y_centers), len(x_centers)))
    
    for idx, row in merged_df.iterrows():
        x_bin = np.searchsorted(x_edges, row['dominant_period']) - 1
        y_bin = np.searchsorted(y_edges, row['significant_wave_height']) - 1
        
        if 0 <= x_bin < len(x_centers) and 0 <= y_bin < len(y_centers):
            z[y_bin, x_bin] += row['dc_bus_power']
            counts[y_bin, x_bin] += 1
    
    # Average the values in each bin
    z = np.divide(z, counts, where=counts > 0, out=np.full_like(z, np.nan))
    
    fig = go.Figure(data=go.Contour(
        x=x_centers,
        y=y_centers,
        z=z,
        colorscale='Viridis',
        colorbar=dict(title="DC Power<br>(W)"),
        hovertemplate="Period: %{x:.2f} s<br>Height: %{y:.2f} m<br>Avg Power: %{z:.1f} W<extra></extra>",
    ))
    
    fig.update_layout(
        title="DC Bus Power vs Wave Height & Dominant Period (Last 7 Days)",
        xaxis_title="Dominant Wave Period (s)",
        yaxis_title="Significant Wave Height (m)",
        template="plotly_white",
        height=600,
        width=800,
        hovermode="closest",
    )
    
    return fig


def resample_and_merge(ndbc_df: pd.DataFrame, ooi_df: pd.DataFrame, freq: str = '1H', start_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Resample and merge NDBC and OOI data to a common time index.

    Returns a DataFrame indexed by the common resample `freq` with columns from both inputs.
    """
    if ndbc_df is None or ndbc_df.empty:
        raise ValueError("NDBC dataframe is empty")
    if ooi_df is None or ooi_df.empty:
        raise ValueError("OOI dataframe is empty")

    # Apply start_date if provided
    if start_date is not None:
        ndbc_df = ndbc_df[ndbc_df.index >= start_date]
        ooi_df = ooi_df[ooi_df.index >= start_date]

    min_time = max(ndbc_df.index.min(), ooi_df.index.min())
    max_time = min(ndbc_df.index.max(), ooi_df.index.max())
    if min_time >= max_time:
        return pd.DataFrame()

    common_index = pd.date_range(min_time, max_time, freq=freq)

    ndbc_interp = ndbc_df.reindex(ndbc_df.index.union(common_index)).interpolate(method='index').loc[common_index]
    ooi_interp = ooi_df.reindex(ooi_df.index.union(common_index)).interpolate(method='index').loc[common_index]

    # Select common columns if present
    left_cols = [c for c in ['significant_wave_height', 'dominant_period', 'wind_direction', 'mean_wave_direction', 'wind_speed'] if c in ndbc_interp.columns]
    right_cols = [c for c in ['dc_bus_power', 'export_power'] if c in ooi_interp.columns]

    merged = pd.concat([ndbc_interp[left_cols], ooi_interp[right_cols]], axis=1)
    return merged


def create_scatter_plot(df: pd.DataFrame, dc_power_df: Optional[pd.DataFrame], start_date: Optional[pd.Timestamp] = None) -> Optional[go.Figure]:
    """
    Create a scatter plot of DC power vs wave height with 1-hour resampling.
    
    Args:
        df: DataFrame with wave height and period data
        dc_power_df: DataFrame with DC power data
        start_date: Optional start date for filtering data (default: None, uses all data)
    
    Returns:
        Plotly Figure object or None if data is insufficient
    """
    if dc_power_df is None or dc_power_df.empty:
        return None
    
    # Check if we have wave height
    if 'significant_wave_height' not in df.columns:
        return None
    
    # Filter by start date if provided
    if start_date is not None:
        df = df[df.index >= start_date]
        dc_power_df = dc_power_df[dc_power_df.index >= start_date]
    
    # Use helper to resample and merge NDBC + OOI into one hourly DataFrame
    try:
        merged_df = resample_and_merge(df, dc_power_df, freq='1H', start_date=start_date)
    except Exception as e:
        print(f"Error merging data for scatter: {e}")
        return None
    merged_df = merged_df.dropna()
    
    if len(merged_df) < 10:
        print("Warning: Not enough data points for scatter plot")
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_df['significant_wave_height'],
        y=merged_df['dc_bus_power'],
        mode='markers',
        marker=dict(
            size=8,
            color=merged_df['dc_bus_power'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="DC Power<br>(W)"),
            line=dict(width=0.5, color='white'),
        ),
        name='DC Power vs Wave Height',
        hovertemplate="Height: %{x:.2f} m<br>Power: %{y:.1f} W<extra></extra>",
    ))
    
    fig.update_layout(
        title="DC Bus Power vs Wave Height (1-Hour Resampled, Last 7 Days)",
        xaxis_title="Significant Wave Height (m)",
        yaxis_title="DC Bus Power (W)",
        template="plotly_white",
        height=600,
        width=800,
        hovermode="closest",
    )
    
    return fig


def create_plot(df: pd.DataFrame, dc_power_df: Optional[pd.DataFrame], output_path: Path) -> None:
    """
    Create an interactive Plotly graph with wave and DC power data.
    
    Args:
        df: DataFrame with time index and wave data (WVHT and/or APD columns)
        dc_power_df: Optional DataFrame with DC power data
        output_path: Path to save the HTML file
    """
    print(f"Creating plot with columns: {list(df.columns)}")
    
    # Determine number of subplots (wave height, period, dc power, directions)
    num_subplots = 5

    fig = make_subplots(
            rows=num_subplots, cols=1,
            specs=[[{"secondary_y": False}] for _ in range(num_subplots)],
            vertical_spacing=0.025,
            shared_xaxes=True,
        )
    
    fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['significant_wave_height'].rolling("1H", min_periods=1).mean(),
                    mode="lines",
                    name="Sig. wave height [m]",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="Height: %{y:.2f} m<extra></extra>",
                ),
                row=1, col=1
            )
    
    fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['dominant_period'].rolling("1H", min_periods=1).mean(),
                    mode="lines",
                    name="Dominant wave period [s]",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="Period: %{y:.2f} s<extra></extra>",
                ),
                row=2, col=1
            )
    
    fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['wind_speed'].rolling("1H", min_periods=1).mean(),
                    mode="lines",
                    name="Wind speed [m/s]",
                    line=dict(color="#17becf", width=2),
                    hovertemplate="Wind: %{y:.2f} m/s<extra></extra>",
                ),
                row=3, col=1
            )
    
    fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['mean_wave_direction'].rolling("1H", min_periods=1).mean(),
                    mode="lines",
                    name="Wave direction [°]",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="Wave: %{y:.0f}°<extra></extra>",
                ),
                row=4, col=1
            )
    
    fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['wind_direction'].rolling("1H", min_periods=1).mean(),
                    mode="lines",
                    name="Wind direction [°]",
                    line=dict(color="#17becf", width=2),
                    hovertemplate="Wind: %{y:.0f}°<extra></extra>",
                ),
                row=4, col=1
            )

    rolling_mean_dcp = dc_power_df['dc_bus_power'].rolling("1H", min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=rolling_mean_dcp.index,
            y=rolling_mean_dcp,
            mode="lines",
            name="DC bus power [W]",
            line=dict(color="#8c564b", width=2),
            hovertemplate="DC bus: %{y:.1f} W<extra></extra>",
        ),
        row=5, col=1
    )

    rolling_mean_dcp = dc_power_df['export_power'].rolling("1H", min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=rolling_mean_dcp.index,
            y=rolling_mean_dcp,
            mode="lines",
            name="Export power [W]",
            line=dict(color="#d750cc", width=2),
            hovertemplate="Export: %{y:.1f} W<extra></extra>",
        ),
        row=5, col=1
    )

    
    # Update layout
    title = "NDBC Buoy 44014 & OOI CP10CNSM"
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=300 * num_subplots,
        margin=dict(l=60, r=40, t=100, b=50),
    )
    
    fig.update_yaxes(title_text="Sig. Wave Height (m)", row=1, col=1)    
    fig.update_yaxes(title_text="Dominant Wave\nPeriod (s)", row=2, col=1)
    fig.update_yaxes(title_text="Wind Speed\n(m/s)", row=3, col=1)    
    fig.update_yaxes(title_text="Direction\n(°)", row=4, col=1)
    fig.update_yaxes(title_text="Power (W)", row=5, col=1)
    
    # Update x-axis label on bottom subplot
    fig.update_xaxes(title_text="Time (UTC)", row=num_subplots, col=1)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save HTML
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Plot saved to {output_path}")


def main():
    """Main entry point."""
    output_file = Path("/app/output/wavss_plot.html")
    contour_file = Path("/app/output/wavss_contour.html")
    scatter_file = Path("/app/output/wavss_scatter.html")
    
    try:
        # Fetch data from NDBC buoy 44014 (7 days = 168 hours)
        df = fetch_ndbc_wave_data(buoy_id="44014", hours=168)
        df.to_csv("/app/output/wavss_ndbc_data.csv")
        
        if df.empty:
            print("Error: No data returned from NDBC", file=sys.stderr)
            sys.exit(1)
        
        # Try to fetch OOI DC power data (optional, 7 days)
        dc_power_df = fetch_ooi_dc_power(hours=168)
        
        # Create time series plot with both datasets
        create_plot(df, dc_power_df, output_file)
        
        # Create contour and scatter plots using data starting 2025-11-03
        start_date = pd.Timestamp("2025-11-03T00:00:00Z")
        # Create contour plot
        contour_fig = create_contour_plot(df, dc_power_df, start_date=start_date)
        if contour_fig:
            contour_file.parent.mkdir(parents=True, exist_ok=True)
            contour_fig.write_html(str(contour_file), include_plotlyjs="cdn")
            print(f"Contour plot saved to {contour_file}")
        
        # Create scatter plot
        scatter_fig = create_scatter_plot(df, dc_power_df, start_date=start_date)
        if scatter_fig:
            scatter_file.parent.mkdir(parents=True, exist_ok=True)
            scatter_fig.write_html(str(scatter_file), include_plotlyjs="cdn")
            print(f"Scatter plot saved to {scatter_file}")
        
        print("Success!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
