"""Build the interactive Mexico maps as a standalone HTML file.

GitHub does not run JavaScript in rendered notebooks, so the
interactive version lives in docs/mexico_rates.html, served by GitHub
Pages; the notebook shows a static view and links there. This script
regenerates the file from the committed data:

    python examples/rates/build_interactive.py

It needs plotly in addition to the package (only for building; readers
of the HTML need nothing). Two map panels, GADM 2.0 left and GADM 4.1
right, with one control for the percentile across the sample's draws and
one for the time window. The values are
deaths per 100,000 people per year from the ratio route in compute.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute  # noqa: E402

DOCS = HERE.parents[1] / "docs"
SIMPLIFY = 0.01


def _panel(version: str, plot_file: str):
    keys = compute.target_keys(version)
    q = compute.window_percentiles(version)
    shapes = gpd.read_parquet(compute.DATA / plot_file)
    shapes = shapes.merge(q[q["window"] == "2080-2099"][keys], on=keys, how="inner")
    shapes = shapes.drop_duplicates(subset=keys).reset_index(drop=True)
    shapes["uid"] = ["|".join(str(v) for v in row) for row in shapes[keys].values]
    shapes["geometry"] = shapes.geometry.simplify(SIMPLIFY)
    geojson = json.loads(shapes.set_index("uid")[["geometry"]].to_json())
    for feat in geojson["features"]:
        feat["id"] = feat["id"]
    q = q.copy()
    q["uid"] = ["|".join(str(v) for v in row) for row in q[keys].values]
    series = {}
    for window in compute.WINDOWS:
        for pct in ("q05", "q50", "q95"):
            sub = q[q["window"] == window].set_index("uid")[pct]
            series[f"{window}|{pct}"] = [
                round(float(sub.get(u, float("nan"))), 2) for u in shapes["uid"]
            ]
    names = (
        shapes["NAME_2"].astype(str) + ", " + shapes["NAME_1"].astype(str)
    ).tolist()
    return list(shapes["uid"]), geojson, series, names


def main() -> int:
    import plotly.graph_objects as go

    locs20, gj20, series20, names20 = _panel("2.0", "mex_adm2_gadm20_plot.parquet")
    locs41, gj41, series41, names41 = _panel("4.1", "mex_adm2_gadm41_plot.parquet")
    zmax = max(
        abs(v) for s in (series20, series41) for arr in s.values() for v in arr
        if v == v
    )

    default = "2080-2099|q50"
    fig = go.Figure()
    for i, (locs, gj, series, names, title) in enumerate((
        (locs20, gj20, series20, names20, "GADM 2.0"),
        (locs41, gj41, series41, names41, "GADM 4.1"),
    )):
        fig.add_trace(go.Choropleth(
            geojson=gj,
            locations=locs,
            z=series[default],
            text=names,
            hovertemplate="%{text}<br>%{z:.1f} per 100,000<extra>" + title + "</extra>",
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-zmax,
            zmax=zmax,
            marker_line_width=0.1,
            marker_line_color="#888",
            geo="geo" if i == 0 else "geo2",
            colorbar=dict(title="deaths per<br>100,000", len=0.7) if i == 1 else None,
            showscale=(i == 1),
        ))
    for g in ("geo", "geo2"):
        fig.update_layout(**{g: dict(
            fitbounds="locations", visible=False, projection_type="mercator",
        )})
    fig.update_layout(
        geo=dict(domain=dict(x=[0.0, 0.49])),
        geo2=dict(domain=dict(x=[0.51, 1.0])),
        title=dict(
            text="Mortality rates by Mexican municipality, GADM 2.0 and GADM 4.1<br>"
                 "<sup>deaths per 100,000 people per year; one Monte Carlo batch, "
                 "13 climate models, rcp85, SSP3; negative is fewer deaths</sup>",
            x=0.5,
        ),
        margin=dict(l=10, r=10, t=90, b=10),
        annotations=[
            dict(text="GADM 2.0", x=0.22, y=1.0, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=14)),
            dict(text="GADM 4.1", x=0.78, y=1.0, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=14)),
        ],
    )

    html = fig.to_html(include_plotlyjs=True, full_html=True, div_id="maps")
    controls = f"""
<div style="font-family: sans-serif; margin: 8px 16px;">
  Percentile:
  <select id="pct">
    <option value="q05">5th</option>
    <option value="q50" selected>50th</option>
    <option value="q95">95th</option>
  </select>
  &nbsp;&nbsp;Window:
  <select id="win">
    <option value="2080-2099" selected>2080-2099</option>
    <option value="2090-2099">2090-2099</option>
  </select>
</div>
<script>
var series20 = {json.dumps(series20)};
var series41 = {json.dumps(series41)};
function refresh() {{
  var key = document.getElementById("win").value + "|" +
            document.getElementById("pct").value;
  Plotly.restyle("maps", {{z: [series20[key], series41[key]]}}, [0, 1]);
}}
document.getElementById("pct").addEventListener("change", refresh);
document.getElementById("win").addEventListener("change", refresh);
</script>
"""
    html = html.replace("</body>", controls + "</body>")
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "mexico_rates.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
