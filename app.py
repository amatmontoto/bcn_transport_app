"""
app.py  —  Barcelona Transport Equity Dashboard
================================================
CBI4AI · Team 2.2 · Challenge 2.2 with TMB

Run:   streamlit run app.py

Four tabs:
  1. Network Map        — every stop, metro lines coloured, bus filterable
  2. Flagged Stops      — GNN Output 1: stops under supply pressure
  3. Topology Proposals — GNN Output 2: feasible relocations / new stops
  4. Predictive Layer   — elderly-population forecast 2025 → 2035
"""

import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

from gnn_model import load_model
from proposals import load_processed, flag_stops, generate_proposals
from forecast import load_forecast

st.set_page_config(page_title="Barcelona Transport Equity",
                   page_icon="🚇", layout="wide")

BASE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE, "processed")

BCN_CENTER = [41.395, 2.16]


# ----------------------------------------------------------------------
# cached loaders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading transport network …")
def get_data():
    nodes, edges, shapes, barris, demo = load_processed()
    return nodes, edges, shapes, barris, demo


@st.cache_resource(show_spinner="Loading GNN model …")
def get_model():
    return load_model()


@st.cache_data(show_spinner="Scoring stops with the GNN …")
def get_flagged(top_k):
    nodes, edges, shapes, barris, demo = get_data()
    flagged, _ = flag_stops(nodes, edges, model=get_model(), top_k=top_k)
    return flagged


@st.cache_data(show_spinner="Generating topology proposals …")
def get_proposals(top_k):
    nodes, edges, shapes, barris, demo = get_data()
    flagged = get_flagged(top_k)
    props = generate_proposals(flagged, edges, shapes, barris, demo,
                               model=get_model(), verbose=False)
    return props


@st.cache_data(show_spinner="Loading forecast …")
def get_forecast():
    return load_forecast()


# ----------------------------------------------------------------------
# header
# ----------------------------------------------------------------------
st.title("🚇 Barcelona Transport Equity Dashboard")
st.caption("CBI4AI · Team 2.2 · Challenge 2.2 with TMB — "
           "identifying where elderly residents are underserved by "
           "public transport, and what to do about it.")

nodes, edges, shapes, barris, demo = get_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Network Map",
    "⚠️ Flagged Stops",
    "🛠️ Topology Proposals",
    "📈 Predictive Layer",
])


# ======================================================================
# TAB 1 — NETWORK MAP
# ======================================================================
with tab1:
    st.subheader("Public transport network")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        show_metro = st.checkbox("Show metro + FGC", value=True)
        show_bus   = st.checkbox("Show bus", value=False,
                                 help="2 400+ stops — may be slow")
    with c2:
        metro_lines = sorted(
            {l for s in nodes[nodes["mode"] == "metro"]["lines"].dropna()
             for l in s.split(",") if l})
        sel_metro = st.multiselect("Filter metro lines", metro_lines,
                                   default=[])
        colour_by = st.radio("Colour stops by",
                             ["Line / mode", "Elderly coverage (GNN)"],
                             index=0)

    fmap = folium.Map(location=BCN_CENTER, zoom_start=12,
                      tiles="CartoDB positron")

    # neighbourhood outlines
    folium.GeoJson(
        barris.to_json(),
        style_function=lambda _: dict(color="#999999", weight=0.6,
                                      fill=False),
        name="Neighbourhoods",
    ).add_to(fmap)

    flagged_all = get_flagged(40)
    cov_cmap = LinearColormap(["#d73027", "#fee08b", "#1a9850"],
                              vmin=0, vmax=1,
                              caption="GNN elderly-coverage score")

    def stop_color(row):
        if colour_by == "Elderly coverage (GNN)":
            sc = flagged_all.loc[flagged_all["stop_id"] == row["stop_id"],
                                 "gnn_score"]
            return cov_cmap(float(sc.iloc[0])) if len(sc) else "#888888"
        return row["color"]

    # metro
    if show_metro:
        md = nodes[nodes["mode"] == "metro"]
        if sel_metro:
            md = md[md["lines"].apply(
                lambda s: any(l in (s or "").split(",") for l in sel_metro))]
        for _, r in md.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=4,
                color=stop_color(r), fill=True, fill_opacity=0.9,
                weight=1,
                tooltip=f"🚇 {r['name']}  ·  {r['lines']}",
            ).add_to(fmap)

    # bus
    if show_bus:
        bd = nodes[nodes["mode"] == "bus"]
        for _, r in bd.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=2,
                color=stop_color(r), fill=True, fill_opacity=0.6,
                weight=0.5,
                tooltip=f"🚍 {r['name']}  ·  {r['lines']}",
            ).add_to(fmap)

    if colour_by == "Elderly coverage (GNN)":
        cov_cmap.add_to(fmap)

    st_folium(fmap, height=560, width=None, returned_objects=[])

    m1, m2, m3 = st.columns(3)
    m1.metric("Metro + FGC stops", f"{(nodes['mode']=='metro').sum():,}")
    m2.metric("Bus stops", f"{(nodes['mode']=='bus').sum():,}")
    m3.metric("Network edges", f"{len(edges):,}")


# ======================================================================
# TAB 2 — FLAGGED STOPS  (GNN Output 1)
# ======================================================================
with tab2:
    st.subheader("Stops under supply pressure")
    st.markdown(
        "The GNN scores every stop on how well it covers the elderly "
        "residents who can realistically walk to it (250 m threshold). "
        "**Pressure** combines elderly demand with a low coverage score — "
        "high-pressure stops are the priority revision candidates.")

    top_k = st.slider("How many stops to flag", 10, 80, 40, step=5)
    flagged = get_flagged(top_k)
    fl = flagged[flagged["flagged"]].sort_values("pressure",
                                                 ascending=False)

    cL, cR = st.columns([3, 2])

    with cL:
        fmap2 = folium.Map(location=BCN_CENTER, zoom_start=12,
                           tiles="CartoDB positron")
        folium.GeoJson(
            barris.to_json(),
            style_function=lambda _: dict(color="#bbbbbb", weight=0.5,
                                          fill=False),
        ).add_to(fmap2)
        # all stops faded
        for _, r in nodes.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=1.5,
                color="#cccccc", fill=True, fill_opacity=0.3,
                weight=0).add_to(fmap2)
        # flagged stops highlighted
        for _, r in fl.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=7,
                color="#d73027", fill=True, fill_opacity=0.85, weight=1.5,
                tooltip=(f"⚠️ #{int(r['pressure_rank'])}  {r['name']}<br>"
                         f"Mode: {r['mode']}<br>"
                         f"Elderly within 400 m: {r['elderly_400']:.0f}<br>"
                         f"GNN coverage score: {r['gnn_score']:.2f}<br>"
                         f"Pressure: {r['pressure']:.2f}"),
            ).add_to(fmap2)
        st_folium(fmap2, height=520, width=None, returned_objects=[])

    with cR:
        st.markdown("**Top flagged stops**")
        st.dataframe(
            fl[["pressure_rank", "name", "mode", "elderly_400",
                "gnn_score", "pressure"]]
            .rename(columns={"pressure_rank": "#", "elderly_400":
                             "elderly (400 m)", "gnn_score": "coverage"})
            .set_index("#")
            .style.format({"elderly (400 m)": "{:.0f}",
                           "coverage": "{:.2f}", "pressure": "{:.2f}"}),
            height=480)

    st.info(f"**{len(fl)} stops** flagged. Mode split: "
            f"{(fl['mode']=='bus').sum()} bus · "
            f"{(fl['mode']=='metro').sum()} metro.")


# ======================================================================
# TAB 3 — TOPOLOGY PROPOSALS  (GNN Output 2)
# ======================================================================
with tab3:
    st.subheader("Feasible topology proposals")
    st.markdown(
        "For the highest-pressure stops the GNN evaluates **feasible** "
        "interventions. Every proposed coordinate lies on an existing "
        "GTFS route corridor — a path the current fleet already drives — "
        "so no new infrastructure is implied.\n\n"
        "- **RELOCATE** — shift a stop along its own line's corridor\n"
        "- **ADD** — insert a new stop on an existing corridor where a "
        "coverage gap meets real elderly demand")

    top_k = st.slider("Flagged-stop pool size", 10, 80, 40, step=5,
                      key="prop_k")
    props = get_proposals(top_k)

    if len(props) == 0:
        st.warning("No positive-delta proposals found for this pool size.")
    else:
        kinds = st.multiselect("Show proposal types", ["RELOCATE", "ADD"],
                               default=["RELOCATE", "ADD"])
        pv = props[props["kind"].isin(kinds)]

        cL, cR = st.columns([3, 2])
        with cL:
            fmap3 = folium.Map(location=BCN_CENTER, zoom_start=12,
                               tiles="CartoDB positron")
            folium.GeoJson(
                barris.to_json(),
                style_function=lambda _: dict(color="#bbbbbb", weight=0.5,
                                              fill=False),
            ).add_to(fmap3)
            for _, p in pv.iterrows():
                if p["kind"] == "RELOCATE":
                    # original (red) -> proposed (orange), dashed link
                    folium.CircleMarker(
                        [p["from_lat"], p["from_lon"]], radius=5,
                        color="#d73027", fill=True, fill_opacity=0.8,
                        weight=1, tooltip=f"Current: {p['name']}").add_to(fmap3)
                    folium.CircleMarker(
                        [p["to_lat"], p["to_lon"]], radius=6,
                        color="#fc8d59", fill=True, fill_opacity=0.9,
                        weight=1.5,
                        tooltip=(f"➡️ RELOCATE  {p['name']}<br>"
                                 f"{p['reason']}<br>"
                                 f"Δscore {p['delta_score']:+.3f}")
                    ).add_to(fmap3)
                    folium.PolyLine(
                        [[p["from_lat"], p["from_lon"]],
                         [p["to_lat"], p["to_lon"]]],
                        color="#888888", weight=2, dash_array="5,5",
                    ).add_to(fmap3)
                else:  # ADD
                    folium.CircleMarker(
                        [p["to_lat"], p["to_lon"]], radius=7,
                        color="#1a9850", fill=True, fill_opacity=0.9,
                        weight=1.5,
                        tooltip=(f"➕ NEW {p['mode']} stop<br>"
                                 f"{p['reason']}<br>"
                                 f"Δscore {p['delta_score']:+.3f}")
                    ).add_to(fmap3)
            st_folium(fmap3, height=520, width=None, returned_objects=[])

        with cR:
            st.markdown("**Ranked proposals**")
            st.dataframe(
                pv[["rank", "kind", "mode", "name",
                    "delta_score", "elderly_gained"]]
                .rename(columns={"delta_score": "Δ score",
                                 "elderly_gained": "elderly gained"})
                .set_index("rank")
                .style.format({"Δ score": "{:+.3f}",
                               "elderly gained": "{:+.0f}"}),
                height=480)

        st.markdown("**Proposal detail**")
        pick = st.selectbox("Inspect a proposal",
                            pv["rank"].tolist(),
                            format_func=lambda r:
                            f"#{r} — "
                            f"{pv[pv['rank']==r]['kind'].iloc[0]} — "
                            f"{pv[pv['rank']==r]['name'].iloc[0]}")
        prow = pv[pv["rank"] == pick].iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("Type", prow["kind"])
        d2.metric("GNN Δ coverage", f"{prow['delta_score']:+.3f}")
        d3.metric("Elderly within 250 m", f"{prow['elderly_gained']:+.0f}")
        st.success(prow["reason"])

        legend = ("🔴 current stop · 🟠 proposed relocation · "
                  "🟢 proposed new stop")
        st.caption(legend)


# ======================================================================
# TAB 4 — PREDICTIVE LAYER
# ======================================================================
with tab4:
    st.subheader("Elderly-population forecast 2025 → 2035")
    st.markdown(
        "A linear-trend model fitted to each neighbourhood's elderly-% "
        "history (1997-2025) projects the share of residents aged 65+. "
        "Fast-ageing neighbourhoods are tomorrow's equity deserts — "
        "they should be prioritised even if their coverage looks "
        "acceptable today.")

    fc = get_forecast()

    cL, cR = st.columns([2, 3])
    with cL:
        st.markdown("**Fastest-ageing neighbourhoods (10-yr Δ)**")
        st.dataframe(
            fc[["barri_name", "elderly_2025", "elderly_2035",
                "delta_10yr"]]
            .rename(columns={"barri_name": "neighbourhood",
                             "elderly_2025": "2025 %",
                             "elderly_2035": "2035 %",
                             "delta_10yr": "Δ 10-yr"})
            .head(15)
            .set_index("neighbourhood")
            .style.format({"2025 %": "{:.1f}", "2035 %": "{:.1f}",
                           "Δ 10-yr": "{:+.1f}"}),
            height=460)

    with cR:
        pick_b = st.selectbox("Inspect a neighbourhood",
                              fc["barri_name"].tolist())
        brow = fc[fc["barri_name"] == pick_b].iloc[0]
        series = json.loads(brow["series_json"])
        hist = pd.DataFrame({
            "year": [int(y) for y in series],
            "elderly_pct": [series[y] for y in series],
        }).dropna().sort_values("year")

        proj = pd.DataFrame({
            "year": [2025, 2030, 2035],
            "elderly_pct": [brow["elderly_2025"], brow["elderly_2030"],
                            brow["elderly_2035"]],
        })

        chart_df = (pd.concat([
            hist.assign(kind="historical"),
            proj.assign(kind="forecast"),
        ]))
        st.markdown(f"**{pick_b}** — elderly share of population")
        pivot = chart_df.pivot_table(index="year", columns="kind",
                                     values="elderly_pct")
        st.line_chart(pivot)

        f1, f2, f3 = st.columns(3)
        f1.metric("Elderly today (2025)", f"{brow['elderly_2025']:.1f}%")
        f2.metric("Forecast 2030", f"{brow['elderly_2030']:.1f}%",
                  f"{brow['elderly_2030']-brow['elderly_2025']:+.1f}")
        f3.metric("Forecast 2035", f"{brow['elderly_2035']:.1f}%",
                  f"{brow['elderly_2035']-brow['elderly_2025']:+.1f}")
        st.caption(f"Trend: {brow['trend_per_year']:+.2f} %-points per "
                   f"year · plausible band ±{brow['band']:.1f}")


st.divider()
st.caption("Data: TMB GTFS · Ajuntament de Barcelona Open Data · "
           "Model: GraphSAGE GNN (coverage regression) · "
           "CBI4AI Team 2.2")
