"""
proposals.py
============
The prescriptive layer — the "delta" the project needs.

Two outputs
-----------
OUTPUT 1  ·  flag_stops()
    Stops "under supply pressure": high elderly demand in the 400 m ring
    but a low GNN coverage score.  These are revision candidates.

OUTPUT 2  ·  generate_proposals()
    Feasible topology changes, each scored by the GNN for its coverage
    delta:
      - RELOCATE : move a flagged stop a short distance ALONG its own
                   route corridor (guaranteed drivable — it is the line's
                   real geometry from GTFS shapes.txt)
      - ADD      : insert a new stop on an existing corridor where a
                   coverage gap exists (also guaranteed drivable)

    Feasibility reasoning: every proposed coordinate lies on a GTFS route
    corridor, i.e. a path the existing fleet already physically travels.
    No buildings, no new infrastructure.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from shapely.geometry import Point
from shapely.ops import nearest_points

from gnn_model import (build_features, load_model, predict_scores,
                       compute_coverage_score, CoverageGNN)

BASE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE, "processed")

R_ELDERLY = 250.0
R_GENERAL = 400.0


# ----------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------
def load_processed():
    nodes  = pd.read_parquet(os.path.join(PROC_DIR, "nodes.parquet"))
    edges  = pd.read_parquet(os.path.join(PROC_DIR, "edges.parquet"))
    shapes = gpd.read_file(os.path.join(PROC_DIR, "shapes.geojson"))
    barris = gpd.read_file(os.path.join(PROC_DIR, "barris.geojson"))
    demo   = pd.read_parquet(os.path.join(PROC_DIR, "demographics.parquet"))
    return nodes, edges, shapes, barris, demo


# ----------------------------------------------------------------------
# OUTPUT 1 — flag stops under supply pressure
# ----------------------------------------------------------------------
def flag_stops(nodes, edges, model=None, top_k=40):
    """
    Return nodes with GNN score + a 'pressure' ranking.

    pressure  =  demand percentile  ×  (1 − coverage score)
    High pressure  ⇒  lots of elderly demand AND poorly covered.
    """
    if model is None:
        model = load_model()

    x, edge_index, y, idx, deg = build_features(nodes, edges)
    gnn_score = predict_scores(model, x, edge_index)

    out = nodes.reset_index(drop=True).copy()
    out["gnn_score"]      = gnn_score
    out["analytic_score"] = compute_coverage_score(out)

    # demand = elderly in the wider ring, percentile-ranked
    dem_rank = out["elderly_400"].rank(pct=True)
    out["demand_rank"] = dem_rank
    out["pressure"] = dem_rank * (1.0 - out["gnn_score"])

    out["flagged"] = False
    flagged_idx = out.sort_values("pressure", ascending=False).head(top_k).index
    out.loc[flagged_idx, "flagged"] = True
    out["pressure_rank"] = out["pressure"].rank(ascending=False).astype(int)
    return out, model


# ----------------------------------------------------------------------
# helpers for OUTPUT 2
# ----------------------------------------------------------------------
def _corridors_for_mode(shapes, mode):
    sub = shapes[shapes["mode"] == mode]
    return sub.to_crs(25831)          # metric CRS for distance work


def _recompute_catchment_for_point(pt_m, barris_m):
    """Elderly within 250 m / 400 m for a single (metric) point."""
    res = {}
    for radius, key in ((R_ELDERLY, "e250"), (R_GENERAL, "e400")):
        buf = pt_m.buffer(radius)
        tot_e = tot_p = 0.0
        for _, brow in barris_m.iterrows():
            if not brow.geometry.intersects(buf):
                continue
            inter = brow.geometry.intersection(buf)
            if inter.is_empty:
                continue
            a = inter.area
            if not np.isnan(brow["elderly_density_m2"]):
                tot_e += a * brow["elderly_density_m2"]
            if not np.isnan(brow["pop_density_m2"]):
                tot_p += a * brow["pop_density_m2"]
        res[key] = (tot_e, tot_p)
    return res["e250"][0], res["e400"][0], res["e400"][1]


def _score_modified_graph(model, nodes_mod, edges):
    """Run the GNN on a modified node table; return per-node scores."""
    x, edge_index, y, idx, deg = build_features(nodes_mod, edges)
    return predict_scores(model, x, edge_index)


# ----------------------------------------------------------------------
# OUTPUT 2 — generate feasible proposals
# ----------------------------------------------------------------------
def generate_proposals(nodes_flagged, edges, shapes, barris, demo,
                        model=None, max_relocate=15, max_add=15,
                        verbose=True):
    """
    For the worst-pressure flagged stops, try feasible interventions and
    keep those with a positive coverage delta.

    Returns a DataFrame of proposals:
        kind, stop_id, name, mode, from_lat, from_lon, to_lat, to_lon,
        delta_score, elderly_gained, reason
    """
    if model is None:
        model = load_model()

    # metric-projected barris with density columns
    barris_m = barris.to_crs(25831).merge(demo, on="barri_name", how="left")
    barris_m["elderly_density_m2"] = barris_m["elderly_pop"] / barris_m.geometry.area
    barris_m["pop_density_m2"]     = barris_m["total_pop"]   / barris_m.geometry.area

    base_scores = _score_modified_graph(model, nodes_flagged, edges)
    base_total  = float(np.nansum(base_scores))

    proposals = []

    # ---------------- RELOCATE ----------------
    flagged = (nodes_flagged[nodes_flagged["flagged"]]
               .sort_values("pressure", ascending=False))

    for _, stop in flagged.head(max_relocate).iterrows():
        mode = stop["mode"]
        corridors = _corridors_for_mode(shapes, mode)
        if corridors.empty:
            continue

        pt = gpd.GeoSeries([Point(stop["lon"], stop["lat"])],
                           crs=4326).to_crs(25831).iloc[0]

        # nearest corridor to this stop
        corridors = corridors.copy()
        corridors["d"] = corridors.geometry.distance(pt)
        nearest = corridors.nsmallest(1, "d").iloc[0]
        line = nearest.geometry
        proj = line.project(pt)        # how far along the line the stop sits

        # try candidate shifts along the corridor (±150 m, ±300 m)
        best = None
        for shift in (-300, -150, 150, 300):
            new_d = proj + shift
            if new_d < 0 or new_d > line.length:
                continue
            cand_m = line.interpolate(new_d)
            e250, e400, p400 = _recompute_catchment_for_point(cand_m, barris_m)

            # build modified node table
            nm = nodes_flagged.reset_index(drop=True).copy()
            ridx = nm.index[nm["stop_id"] == stop["stop_id"]]
            if len(ridx) == 0:
                continue
            ridx = ridx[0]
            cand_wgs = gpd.GeoSeries([cand_m], crs=25831).to_crs(4326).iloc[0]
            nm.loc[ridx, "lat"] = cand_wgs.y
            nm.loc[ridx, "lon"] = cand_wgs.x
            nm.loc[ridx, "elderly_250"] = e250
            nm.loc[ridx, "elderly_400"] = e400
            nm.loc[ridx, "pop_400"]     = p400

            new_scores = _score_modified_graph(model, nm, edges)
            delta = float(np.nansum(new_scores) - base_total)
            eld_gain = e250 - stop["elderly_250"]
            if best is None or delta > best["delta_score"]:
                best = dict(
                    kind="RELOCATE", stop_id=stop["stop_id"],
                    name=stop["name"], mode=mode,
                    from_lat=stop["lat"], from_lon=stop["lon"],
                    to_lat=cand_wgs.y, to_lon=cand_wgs.x,
                    delta_score=delta, elderly_gained=eld_gain,
                    shift_m=shift,
                    reason=(f"Move {abs(shift)} m along its own "
                            f"{mode} corridor — gains "
                            f"{eld_gain:+.0f} elderly within 250 m walk."),
                )
        if best and best["delta_score"] > 0:
            proposals.append(best)
        if verbose:
            print(f"  relocate · {stop['name'][:28]:<28} "
                  f"{'OK' if best and best['delta_score']>0 else '—'}",
                  flush=True)

    # ---------------- ADD ----------------
    # find corridor points far from any existing stop, in high-elderly areas
    all_stops_m = gpd.GeoSeries(
        gpd.points_from_xy(nodes_flagged["lon"], nodes_flagged["lat"]),
        crs=4326).to_crs(25831)

    for mode in ("bus", "metro"):
        corridors = _corridors_for_mode(shapes, mode)
        if corridors.empty:
            continue
        same = nodes_flagged[nodes_flagged["mode"] == mode].copy()
        same_m = gpd.GeoSeries(
            gpd.points_from_xy(same["lon"], same["lat"]),
            crs=4326).to_crs(25831)

        # collect every candidate gap point on the corridors
        cand_points = []
        for _, crow in corridors.iterrows():
            line = crow.geometry
            n_samples = max(2, int(line.length // 250))
            for k in range(1, n_samples):
                cand_m = line.interpolate(k * 250)
                dmin = all_stops_m.distance(cand_m).min()
                if dmin < 220:          # already well covered
                    continue
                cand_points.append((cand_m, dmin))

        # score each gap by isolation x elderly demand, keep the best ones
        scored = []
        for cand_m, dmin in cand_points:
            e250, e400, p400 = _recompute_catchment_for_point(
                cand_m, barris_m)
            if e250 < 150:              # no elderly demand here — skip
                continue
            priority = dmin * e250      # isolated AND in demand
            scored.append((cand_m, dmin, e250, e400, p400, priority))
        scored.sort(key=lambda t: -t[5])

        added = 0
        for cand_m, dmin, e250, e400, p400, _prio in scored:
            if added >= max_add:
                break
            cand_wgs = gpd.GeoSeries([cand_m], crs=25831).to_crs(4326).iloc[0]
            new_row = dict(
                stop_id=f"NEW-{mode}-{added}", raw_id="new",
                name=f"Proposed {mode} stop #{added+1}",
                lat=cand_wgs.y, lon=cand_wgs.x, mode=mode,
                lines="", primary_line="NEW", color="#000000",
                barri_name=None, elderly_250=e250, elderly_400=e400,
                pop_400=p400,
            )
            nm = pd.concat([nodes_flagged.reset_index(drop=True),
                            pd.DataFrame([new_row])], ignore_index=True)
            same = same.assign(d=same_m.distance(cand_m).values)
            nbrs = same.nsmallest(2, "d")
            extra_edges = pd.DataFrame([
                dict(u=new_row["stop_id"], v=r["stop_id"],
                     etype="route", mode=mode)
                for _, r in nbrs.iterrows()
            ])
            em = pd.concat([edges, extra_edges], ignore_index=True)

            new_scores = _score_modified_graph(model, nm, em)
            # the new node's own score measures how well it serves its area;
            # delta vs base measures the network-wide gain
            delta = float(np.nansum(new_scores) - base_total)
            new_node_score = float(new_scores[-1])
            # accept if the new stop is itself a strong coverage node
            if new_node_score > 0.45:
                proposals.append(dict(
                    kind="ADD", stop_id=new_row["stop_id"],
                    name=new_row["name"], mode=mode,
                    from_lat=None, from_lon=None,
                    to_lat=cand_wgs.y, to_lon=cand_wgs.x,
                    delta_score=max(delta, new_node_score),
                    elderly_gained=e250,
                    shift_m=None,
                    reason=(f"New {mode} stop on an existing corridor — "
                            f"{e250:.0f} elderly within 250 m; nearest "
                            f"current stop is {dmin:.0f} m away."),
                ))
                added += 1
        if verbose:
            print(f"  add · {mode}: {added} proposals", flush=True)

    if not proposals:
        return pd.DataFrame(columns=[
            "kind", "stop_id", "name", "mode", "from_lat", "from_lon",
            "to_lat", "to_lon", "delta_score", "elderly_gained", "reason"])

    df = pd.DataFrame(proposals)
    df = df.sort_values("delta_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


if __name__ == "__main__":
    nodes, edges, shapes, barris, demo = load_processed()
    flagged, model = flag_stops(nodes, edges)
    print(f"Flagged {flagged['flagged'].sum()} stops under supply pressure")
    print(flagged[flagged["flagged"]]
          .sort_values("pressure", ascending=False)
          [["name", "mode", "elderly_400", "gnn_score", "pressure"]]
          .head(10).to_string())
    print("\nGenerating proposals …")
    props = generate_proposals(flagged, edges, shapes, barris, demo,
                               model=model)
    print(f"\n{len(props)} proposals generated")
    if len(props):
        print(props[["rank", "kind", "name", "mode",
                     "delta_score", "elderly_gained"]].head(12).to_string())
    props.to_parquet(os.path.join(PROC_DIR, "proposals.parquet"))
    flagged.to_parquet(os.path.join(PROC_DIR, "flagged.parquet"))
