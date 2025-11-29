# NDBC Wave Height Visualization

Dockerized Python application that fetches the most recent 72 hours of significant wave height data from NDBC Buoy 44014 and generates an interactive Plotly graph.

## Features

- **Real-time data fetching** from NDBC (National Buoy Data Center)
- **NDBC Buoy 44014** - Delaware Bay entrance station
- **72-hour rolling window** of significant wave height data
- **Interactive Plotly visualization** with:
  - Raw data trace
  - 6-hour rolling mean for trend analysis
  - Hover tooltips with timestamps and values
- **Docker containerized** for easy, reproducible deployment
- **Automatic HTML output** with minimal dependencies

## Data Source

- **Buoy**: NDBC Station 44014 (Delaware Bay entrance)
- **Format**: Real-time wave observation data via HTTP
- **Endpoint**: `https://www.ndbc.noaa.gov/data/realtime2/44014.txt`
- **Time Coverage**: Latest 72 hours available

## Requirements

- Docker & Docker Compose
- Or Python 3.11+ with dependencies from `requirements.txt`

## Quick Start

### With Docker Compose

```bash
docker-compose up --build
```

The generated plot will be saved to `output/wavss_plot.html`.

### With Docker (manual)

```bash
docker build -t pioneer-wavss .
docker run -v $(pwd)/output:/app/output pioneer-wavss
```

### Local Python (no Docker)

```bash
pip install -r requirements.txt
python fetch_wavss.py
```

## Output

The script generates `output/wavss_plot.html` - an interactive Plotly HTML file showing:

- **Raw data points** (blue line, thinner)
- **6-hour rolling mean** (orange line, thicker) for smoothed trend visualization
- **Hover information** with exact timestamps and wave heights in meters

## Configuration

All settings are configured for NDBC Buoy 44014 with 72-hour fetches. To modify:

Edit `fetch_wavss.py`:

- `buoy_id` parameter in `fetch_ndbc_wave_data()` call (default: "44014")
- `hours` parameter for different time windows (default: 72)
- `output_path` in `main()` for different output location

## Dependencies

- **pandas** 1.5.3 - Data manipulation and datetime handling
- **plotly** 5.17.0 - Interactive visualization
- **requests** 2.31.0 - HTTP client for NDBC data fetching
- **numpy** 1.24.3 - Numeric operations (pandas dependency)

## Notes

- NDBC provides real-time data that is updated every 10-30 minutes
- The HTTP endpoint returns the most recent available observations
- Missing data values in NDBC format are marked as "MM" and automatically converted to NaN
- The 6-hour rolling mean provides smoothing for trend visualization
