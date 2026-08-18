"""Build the interactive Mexico maps as a standalone HTML file.

GitHub does not run JavaScript in rendered notebooks, so the
interactive version lives in docs/mexico_rates.html, served by GitHub
Pages; the notebook shows a static view and links there. This script
regenerates the file from the committed data:

    python examples/rates/build_interactive.py

It needs plotly in addition to the package (only for building; readers
of the HTML need nothing). GADM 2.0 and GADM 4.1 render as two plots
side by side on one shared geographic extent so they compare directly.
Each panel is its own plot so each carries its own hover label, and
hovering a unit that exists in both versions raises the label on its
counterpart in the other panel, through the committed crosswalk.
Controls above the maps switch the percentile across the sample's
draws and the time window. The values are the effect of climate change
on the mortality rate, deaths per 100,000 people per year, full
adaptation minus histclim, from the ratio route in compute.py.
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
    return list(shapes["uid"]), geojson, series, names, shapes.total_bounds


def _crosswalk_pairs(locs20, locs41):
    """uid pairs of the 738 units comparable across versions, restricted
    to units present in both panels, for the linked hover."""
    import pandas as pd

    cross = pd.read_parquet(compute.DATA / "adm2_crosswalk.parquet")
    uid20 = cross[["ISO", "ID_1", "ID_2"]].astype(str).agg("|".join, axis=1)
    uid41 = cross[["GID_0", "GID_1", "GID_2"]].astype(str).agg("|".join, axis=1)
    s20, s41 = set(locs20), set(locs41)
    pairs = {a: b for a, b in zip(uid20, uid41) if a in s20 and b in s41}
    print(f"linked hover pairs: {len(pairs)}")
    return pairs


def _map_figure(locs, gj, series, names, title, lon, lat, zmax, default):
    import plotly.graph_objects as go

    fig = go.Figure(go.Choropleth(
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
        showscale=False,
    ))
    fig.update_layout(
        geo=dict(
            visible=False,
            projection_type="mercator",
            lonaxis=dict(range=list(lon)),
            lataxis=dict(range=list(lat)),
            domain=dict(x=[0, 1], y=[0, 1]),
        ),
        title=dict(text=title, x=0.5, y=0.98, font=dict(size=14)),
        height=470,
        margin=dict(l=5, r=5, t=30, b=5),
    )
    return fig


def _colorbar_figure(zmax):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale="RdBu_r", cmin=-zmax, cmax=zmax, showscale=True,
            colorbar=dict(
                title=dict(text="deaths per 100,000", side="top"),
                orientation="h",
                x=0.5, xanchor="center",
                y=1.0, yanchor="top",
                len=0.5, thickness=12,
            ),
        ),
        hoverinfo="none",
    ))
    fig.update_layout(
        height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main() -> int:
    locs20, gj20, series20, names20, b20 = _panel("2.0", "mex_adm2_gadm20_plot.parquet")
    locs41, gj41, series41, names41, b41 = _panel("4.1", "mex_adm2_gadm41_plot.parquet")
    pairs = _crosswalk_pairs(locs20, locs41)
    zmax = max(
        abs(v) for s in (series20, series41) for arr in s.values() for v in arr
        if v == v
    )
    pad = 0.5
    lon = (min(b20[0], b41[0]) - pad, max(b20[2], b41[2]) + pad)
    lat = (min(b20[1], b41[1]) - pad, max(b20[3], b41[3]) + pad)

    default = "2080-2099|q50"
    fig20 = _map_figure(locs20, gj20, series20, names20, "GADM 2.0",
                        lon, lat, zmax, default)
    fig41 = _map_figure(locs41, gj41, series41, names41, "GADM 4.1",
                        lon, lat, zmax, default)
    figbar = _colorbar_figure(zmax)

    div20 = fig20.to_html(include_plotlyjs=True, full_html=False, div_id="map20")
    div41 = fig41.to_html(include_plotlyjs=False, full_html=False, div_id="map41")
    divbar = figbar.to_html(include_plotlyjs=False, full_html=False, div_id="cbar")

    script = f"""
<script>
var series20 = {json.dumps(series20)};
var series41 = {json.dumps(series41)};
var pair20 = {json.dumps(pairs)};
var pair41 = {{}};
Object.keys(pair20).forEach(function(k) {{ pair41[pair20[k]] = k; }});
function refresh() {{
  var key = document.getElementById("win").value + "|" +
            document.getElementById("pct").value;
  Plotly.restyle("map20", {{z: [series20[key]]}}, [0]);
  Plotly.restyle("map41", {{z: [series41[key]]}}, [0]);
}}
document.getElementById("pct").addEventListener("change", refresh);
document.getElementById("win").addEventListener("change", refresh);

// hovering a unit outlines its counterpart in the other panel through
// a highlight trace added at runtime, and the readout line under the
// controls shows the unit's value in both versions
function wire() {{
  var g20 = document.getElementById("map20");
  var g41 = document.getElementById("map41");
  var readout = document.getElementById("readout");
  var idle = readout.textContent;
  if (!g20.data || !g41.data || !g20.on) {{ setTimeout(wire, 100); return; }}
  var lookup20 = {{}}, lookup41 = {{}};
  g20.data[0].locations.forEach(function(u, i) {{ lookup20[u] = i; }});
  g41.data[0].locations.forEach(function(u, i) {{ lookup41[u] = i; }});
  var clear = ["rgba(0,0,0,0)"];
  [g20, g41].forEach(function(g) {{
    Plotly.addTraces(g, {{
      type: "choropleth",
      geojson: g.data[0].geojson,
      locations: [],
      z: [],
      colorscale: [[0, clear[0]], [1, clear[0]]],
      showscale: false,
      hoverinfo: "skip",
      marker: {{line: {{width: 2, color: "#222"}}}},
      geo: "geo",
    }});
  }});
  function key() {{
    return document.getElementById("win").value + "|" +
           document.getElementById("pct").value;
  }}
  function fmt(v) {{
    return (v === null || v !== v) ? "no value" : v.toFixed(1);
  }}
  function link(src, dst, map, srcSeries, dstSeries, srcLookup, dstLookup,
                srcLabel, dstLabel) {{
    src.on("plotly_hover", function(ev) {{
      var p = ev.points[0];
      var t = map[p.location];
      var k = key();
      if (t === undefined) {{
        Plotly.restyle(dst, {{locations: [[]], z: [[]]}}, [1]);
        readout.textContent = p.text + ": " + fmt(p.z) + " per 100,000 (" +
          srcLabel + "); no counterpart in " + dstLabel;
        return;
      }}
      Plotly.restyle(dst, {{locations: [[t]], z: [[0]]}}, [1]);
      var v = dstSeries[k][dstLookup[t]];
      readout.textContent = p.text + ": " + fmt(p.z) + " per 100,000 in " +
        srcLabel + ", " + fmt(v) + " in " + dstLabel;
    }});
    src.on("plotly_unhover", function(ev) {{
      Plotly.restyle(dst, {{locations: [[]], z: [[]]}}, [1]);
      readout.textContent = idle;
    }});
  }}
  link(g20, g41, pair20, series20, series41, lookup20, lookup41,
       "GADM 2.0", "GADM 4.1");
  link(g41, g20, pair41, series41, series20, lookup41, lookup20,
       "GADM 4.1", "GADM 2.0");
}}
wire();
</script>
"""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Effect of climate change on mortality rates in Mexico</title>
</head>
<body style="margin: 0; font-family: sans-serif;">
<div style="text-align: center; margin: 10px 0 0 0;">
  <div style="font-size: 17px;">Effect of climate change on mortality rates
    by Mexican municipality, GADM 2.0 and GADM 4.1</div>
  <div style="font-size: 12px; color: #444; margin-top: 2px;">
    deaths per 100,000 per year</div>
  <div style="margin-top: 8px;">
    Percentile across the 99 draws:
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
  <div id="readout" style="height: 18px; font-size: 13px; color: #444;
       margin-top: 6px;">hover a municipality for its value in both
       versions; comparable units outline in the other panel</div>
</div>
<div style="display: flex; flex-wrap: wrap; justify-content: center;">
  <div style="flex: 1 1 460px; min-width: 380px; max-width: 700px;">{div20}</div>
  <div style="flex: 1 1 460px; min-width: 380px; max-width: 700px;">{div41}</div>
</div>
<div style="max-width: 700px; margin: 0 auto;">{divbar}</div>
{script}
</body>
</html>
"""
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "mexico_rates.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
