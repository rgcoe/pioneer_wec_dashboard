#!/usr/bin/env python3
"""Dash web app to serve generated Plotly HTML plots and data files.

Usage:
  pip install -r requirements.txt
  python webapp.py

Open http://localhost:8050
"""

from pathlib import Path
from flask import send_from_directory
import dash
from dash import html, dcc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

def read_html(fname: Path) -> str:
    if not fname.exists():
        return f"<p>File not found: {fname.name}</p>"
    try:
        return fname.read_text(encoding='utf-8')
    except Exception:
        return f"<p>Unable to read file: {fname.name}</p>"


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


@server.route('/downloads/<path:filename>')
def serve_download(filename):
    # Serve files from the output directory for manual download
    return send_from_directory(str(OUT.resolve()), filename, as_attachment=True)


def make_layout():
    # locate plot HTML files
    files = {
        'Time Series': OUT / 'wavss_plot.html',
        'Contour': OUT / 'wavss_contour.html',
        'Scatter': OUT / 'wavss_scatter.html',
    }

    tabs = []
    for title, path in files.items():
        html_content = read_html(path)
        # embed the full HTML as srcDoc in an iframe
        tabs.append(dcc.Tab(label=title, children=[
            html.Div([
                html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '800px', 'border': 'none'})
            ])
        ]))

    # downloads area: list available data files
    download_links = []
    for fname in ['ndbc_data.nc', 'ooi_data.nc', 'wavss_ndbc_data.csv']:
        p = OUT / fname
        if p.exists():
            download_links.append(html.Li(html.A(fname, href=f"/downloads/{fname}")))
    if not download_links:
        download_links = [html.Li("No data files available yet (")]

    layout = html.Div([
        html.H1('Pioneer WavSS Dashboard'),
        dcc.Tabs(id='tabs-example', children=tabs),
        html.H2('Downloads'),
        html.Ul(download_links),
        html.P('This dashboard embeds the Plotly-generated HTML files from the project `output/` directory.'),
    ])
    return layout


app.layout = make_layout()


if __name__ == '__main__':
    # Run development server
    app.run_server(host='0.0.0.0', port=8050, debug=True)
