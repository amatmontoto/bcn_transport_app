"""
data_pipeline.py
================
Loads every data source for the Barcelona transport-equity project and builds
the graph the GNN consumes.

Outputs (all cached to /processed as parquet / json):
  - nodes.parquet      one row per transport stop (metro+FGC+bus)
  - edges.parquet      route / transfer edges between stops
  - barris.geojson     neighbourhood polygons (WGS84)
  - shapes.parquet     route corridor geometries (for feasible proposals)
  - elderly.parquet    elderly population & density per neighbourhood + forecast base

Run directly:  python data_pipeline.py
"""

import os, json, zipfile, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths — edit DATA_DIR if your raw files live elsewhere
# ----------------------------------------------------------------------
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "data")           # put raw files here
PROC_DIR   = os.path.join(BASE, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

GTFS_ZIP        = os.path.join(DATA_DIR, "data.zip")
TRANSPORTS_CSV  = os.path.join(DATA_DIR, "TRANSPORTS.csv")
BUS_CSV         = os.path.join(DATA_DIR, "ESTACIONS_BUS.csv")
AGE_CSV         = os.path.join(DATA_DIR, "Population_per_age.csv")
DENSITY_CSV     = os.path.join(DATA_DIR, "Population_per_districte_i_barri.csv")
POLYGONS_JSON   = os.path.join(DATA_DIR, "0301100100_UNITATS_ADM_POLIGONS.json")

# Barcelona bounding box (city proper)
BBOX = dict(lat_min=41.32, lat_max=41.47, lon_min=2.06, lon_max=2.23)

# Walking thresholds (metres)
R_ELDERLY = 250.0     # elderly comfortable walking radius
R_GENERAL = 400.0     # general-population walking radius

# Official TMB metro line colours (fallback if GTFS route_color missing)
METRO_COLORS = {
    "L1": "#E1261C", "L2": "#9B4E97", "L3": "#1E9145", "L4": "#FFCC00",
    "L5": "#0078C8", "L9N": "#F58220", "L9S": "#F58220",
    "L10N": "#00ADEF", "L10S": "#00ADEF", "L11": "#9ACD32",
    "FM": "#018D3A",  # Funicular de Montjuïc
}


# ======================================================================
# 1. GEOGRAPHY
# ======================================================================
def load_boundaries():
    """Return (barris, districts, city_polygon) as GeoDataFrames in WGS84."""
    with open(POLYGONS_JSON, encoding="utf-8") as f:
        poly = json.load(f)

    def grab(tipus):
        feats = [f for f in poly["features"]
                 if f["properties"]["TIPUS_UA"] == tipus]
        return gpd.GeoDataFrame.from_features(feats, crs="EPSG:25831").to_crs(4326)

    barris    = grab("BARRI")
    districts = grab("DISTRICTE")
    terme     = grab("TERME")
    barris["barri_name"] = barris["NOM"]
    districts["dist_name"] = districts["NOM"]
    return barris, districts, terme


# ======================================================================
# 2. STOPS  (metro + FGC + bus)
# ======================================================================
def _in_bbox(df, latcol, loncol):
    return df[(df[latcol] > BBOX["lat_min"]) & (df[latcol] < BBOX["lat_max"]) &
              (df[loncol] > BBOX["lon_min"]) & (df[loncol] < BBOX["lon_max"])]


def load_metro_stops():
    """
    Metro + FGC stops from GTFS (gives line identity + colour).
    One node per (parent) station — deduplicated by name+proximity.
    """
    with zipfile.ZipFile(GTFS_ZIP) as z:
        with z.open("data/stops.txt") as fh:      stops  = pd.read_csv(fh)
        with z.open("data/routes.txt") as fh:     routes = pd.read_csv(fh)
        with z.open("data/trips.txt") as fh:      trips  = pd.read_csv(fh)
        with z.open("data/stop_times.txt") as fh:
            stop_times = pd.read_csv(fh, usecols=["trip_id", "stop_id"],
                                     low_memory=False)

    # route_type 1 = metro/subway (includes the FGC urban lines in this feed)
    metro_routes = routes[routes["route_type"] == 1].copy()
    metro_rids   = set(metro_routes["route_id"])
    metro_trips  = trips[trips["route_id"].isin(metro_rids)]
    metro_tids   = set(metro_trips["trip_id"])
    metro_sids   = set(stop_times[stop_times["trip_id"].isin(metro_tids)]["stop_id"])

    mstops = stops[stops["stop_id"].isin(metro_sids) &
                   (stops["location_type"] == 0)].copy()
    mstops = _in_bbox(mstops, "stop_lat", "stop_lon")

    # Map each stop -> its line(s) and colour
    st_line = (stop_times[stop_times["stop_id"].isin(metro_sids)]
               .merge(metro_trips[["trip_id", "route_id"]], on="trip_id")
               .merge(metro_routes[["route_id", "route_short_name",
                                    "route_color"]], on="route_id")
               [["stop_id", "route_short_name", "route_color"]]
               .drop_duplicates())
    line_by_stop = (st_line.groupby("stop_id")["route_short_name"]
                    .apply(lambda s: sorted(set(s))).to_dict())
    color_by_stop = (st_line.groupby("stop_id")["route_color"]
                     .first().to_dict())

    rows = []
    for _, r in mstops.iterrows():
        lines = line_by_stop.get(r["stop_id"], [])
        prim  = lines[0] if lines else "?"
        col   = color_by_stop.get(r["stop_id"])
        if isinstance(col, str) and len(col) == 6:
            color = "#" + col
        else:
            color = METRO_COLORS.get(prim, "#888888")
        rows.append(dict(
            stop_id   = f"M-{r['stop_id']}",
            raw_id    = r["stop_id"],
            name      = r["stop_name"],
            lat       = r["stop_lat"],
            lon       = r["stop_lon"],
            mode      = "metro",
            lines     = ",".join(lines),
            primary_line = prim,
            color     = color,
        ))
    df = pd.DataFrame(rows)

    # Deduplicate platforms of the same station: round coords to ~40 m grid
    df["k"] = (df["lat"].round(3).astype(str) + "_" +
               df["lon"].round(3).astype(str) + "_" + df["primary_line"])
    df = df.drop_duplicates("k").drop(columns="k").reset_index(drop=True)
    return df


def load_bus_stops():
    """Daytime bus stops from GTFS (route_type 3)."""
    with zipfile.ZipFile(GTFS_ZIP) as z:
        with z.open("data/stops.txt") as fh:      stops  = pd.read_csv(fh)
        with z.open("data/routes.txt") as fh:     routes = pd.read_csv(fh)
        with z.open("data/trips.txt") as fh:      trips  = pd.read_csv(fh)
        with z.open("data/stop_times.txt") as fh:
            stop_times = pd.read_csv(fh, usecols=["trip_id", "stop_id"],
                                     low_memory=False)

    bus_routes = routes[routes["route_type"] == 3]
    bus_rids   = set(bus_routes["route_id"])
    bus_trips  = trips[trips["route_id"].isin(bus_rids)]
    bus_tids   = set(bus_trips["trip_id"])
    bus_sids   = set(stop_times[stop_times["trip_id"].isin(bus_tids)]["stop_id"])

    bstops = stops[stops["stop_id"].isin(bus_sids) &
                   (stops["location_type"] == 0)].copy()
    bstops = _in_bbox(bstops, "stop_lat", "stop_lon")

    # Lines serving each stop
    st_line = (stop_times[stop_times["stop_id"].isin(bus_sids)]
               .merge(bus_trips[["trip_id", "route_id"]], on="trip_id")
               .merge(bus_routes[["route_id", "route_short_name"]],
                      on="route_id")
               [["stop_id", "route_short_name"]].drop_duplicates())
    line_by_stop = (st_line.groupby("stop_id")["route_short_name"]
                    .apply(lambda s: sorted(set(s))).to_dict())

    rows = []
    for _, r in bstops.iterrows():
        lines = line_by_stop.get(r["stop_id"], [])
        rows.append(dict(
            stop_id   = f"B-{r['stop_id']}",
            raw_id    = r["stop_id"],
            name      = r["stop_name"],
            lat       = r["stop_lat"],
            lon       = r["stop_lon"],
            mode      = "bus",
            lines     = ",".join(lines),
            primary_line = lines[0] if lines else "?",
            color     = "#5B6770",
        ))
    return pd.DataFrame(rows).drop_duplicates("stop_id").reset_index(drop=True)


# ======================================================================
# 3. EDGES  (route adjacency + transfers)
# ======================================================================
def build_edges(nodes):
    """
    Build edges from GTFS:
      - route edges: consecutive stops on the same trip
      - transfer edges: from transfers.txt
    Edge weight = travel/transfer time in minutes (approx).
    """
    with zipfile.ZipFile(GTFS_ZIP) as z:
        with z.open("data/stop_times.txt") as fh:
            stop_times = pd.read_csv(fh, low_memory=False)
        with z.open("data/trips.txt") as fh:      trips  = pd.read_csv(fh)
        with z.open("data/routes.txt") as fh:     routes = pd.read_csv(fh)
        with z.open("data/transfers.txt") as fh:  transfers = pd.read_csv(fh)

    # raw_id -> our stop_id  (a raw id may map to both M- and B-, keep mode)
    id_lookup = {}
    for _, n in nodes.iterrows():
        id_lookup.setdefault(n["raw_id"], []).append(n["stop_id"])

    rt_by_trip = (trips.merge(routes[["route_id", "route_type"]],
                              on="route_id")[["trip_id", "route_type"]]
                  .set_index("trip_id")["route_type"].to_dict())

    edge_set = set()
    edges = []

    # ---- route adjacency ----
    st = stop_times.sort_values(["trip_id", "stop_sequence"])
    for tid, grp in st.groupby("trip_id"):
        rtype = rt_by_trip.get(tid)
        if rtype not in (1, 3):
            continue
        prefix = "M-" if rtype == 1 else "B-"
        seq = grp["stop_id"].tolist()
        for a, b in zip(seq[:-1], seq[1:]):
            ia, ib = f"{prefix}{a}", f"{prefix}{b}"
            key = tuple(sorted((ia, ib)))
            if key in edge_set or ia == ib:
                continue
            edge_set.add(key)
            edges.append(dict(u=ia, v=ib, etype="route", mode=
                              "metro" if rtype == 1 else "bus"))

    # ---- transfers ----
    for _, t in transfers.iterrows():
        for ia in id_lookup.get(t["from_stop_id"], []):
            for ib in id_lookup.get(t["to_stop_id"], []):
                key = tuple(sorted((ia, ib)))
                if key in edge_set or ia == ib:
                    continue
                edge_set.add(key)
                edges.append(dict(u=ia, v=ib, etype="transfer", mode="walk"))

    edf = pd.DataFrame(edges)
    # keep only edges whose endpoints survived bbox filtering
    valid = set(nodes["stop_id"])
    edf = edf[edf["u"].isin(valid) & edf["v"].isin(valid)].reset_index(drop=True)
    return edf


# ======================================================================
# 4. ROUTE CORRIDORS  (shapes — for feasible stop proposals)
# ======================================================================
def load_shapes():
    """Route corridor polylines from GTFS shapes.txt (WGS84)."""
    with zipfile.ZipFile(GTFS_ZIP) as z:
        with z.open("data/shapes.txt") as fh:  shapes = pd.read_csv(fh)
        with z.open("data/trips.txt") as fh:   trips  = pd.read_csv(fh)
        with z.open("data/routes.txt") as fh:  routes = pd.read_csv(fh)

    shapes = shapes[(shapes["shape_pt_lat"] > BBOX["lat_min"]) &
                    (shapes["shape_pt_lat"] < BBOX["lat_max"]) &
                    (shapes["shape_pt_lon"] > BBOX["lon_min"]) &
                    (shapes["shape_pt_lon"] < BBOX["lon_max"])]

    # shape_id -> mode
    shp_route = (trips[["shape_id", "route_id"]].drop_duplicates()
                 .merge(routes[["route_id", "route_type"]], on="route_id"))
    mode_by_shape = {}
    for _, r in shp_route.iterrows():
        if r["route_type"] in (1, 3):
            mode_by_shape[r["shape_id"]] = "metro" if r["route_type"] == 1 else "bus"

    rows = []
    for sid, grp in shapes.sort_values("shape_pt_sequence").groupby("shape_id"):
        if sid not in mode_by_shape or len(grp) < 2:
            continue
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        rows.append(dict(shape_id=sid, mode=mode_by_shape[sid],
                         geometry=LineString(coords)))
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ======================================================================
# 5. DEMOGRAPHICS  (elderly population per neighbourhood)
# ======================================================================
def _num(s):
    """Spanish-format number string -> float."""
    if pd.isna(s) or s == "-":
        return np.nan
    return float(str(s).replace(".", "").replace(",", "."))


def load_demographics(barris):
    """
    Combine:
      - Population_per_age.csv  -> % elderly (>=65) per neighbourhood, 1997-2025
      - density CSV             -> inhabitants / hectare per neighbourhood
    Compute absolute elderly head-count per neighbourhood (latest year),
    plus the full elderly-% time series for the forecasting layer.
    """
    age = pd.read_csv(AGE_CSV)
    dens = pd.read_csv(DENSITY_CSV)

    year_cols_age = [c for c in age.columns if c.isdigit()]
    for c in year_cols_age:
        age[c] = age[c].map(_num)

    year_cols_dens = [c for c in dens.columns if c.isdigit()]
    for c in year_cols_dens:
        dens[c] = dens[c].map(_num)

    eld = age[(age["Edat en grans grups"].astype(str).str.contains("65")) &
              (age["Tipus de territori"] == "Barri")].copy()
    eld = eld.rename(columns={"Territori": "barri_name"})

    dens_b = dens[dens["Tipus de territori"] == "Barri"].copy()
    dens_b = dens_b.rename(columns={"Territori": "barri_name"})

    # neighbourhood area (hectares) from polygons (EPSG:25831 -> m^2)
    barris_m = barris.to_crs(25831)
    area_ha = (barris_m.geometry.area / 1e4)
    area_by_name = dict(zip(barris["barri_name"], area_ha))

    # pick the latest year that actually has neighbourhood-level data
    # (some trailing columns are empty '-' for barris)
    def latest_with_data(df, cols):
        for c in reversed(cols):
            if df[c].notna().sum() > 0:
                return c
        return cols[-1]

    latest_dens = latest_with_data(dens_b, year_cols_dens)
    latest_eld  = latest_with_data(eld, year_cols_age)

    recs = []
    for _, r in eld.iterrows():
        name = r["barri_name"]
        eld_pct = r[latest_eld]
        drow = dens_b[dens_b["barri_name"] == name]
        density = drow[latest_dens].values[0] if len(drow) else np.nan
        area = area_by_name.get(name, np.nan)
        if np.isnan(density) or np.isnan(area):
            total_pop = np.nan
        else:
            total_pop = density * area
        elderly_cnt = (total_pop * eld_pct / 100.0
                       if not np.isnan(total_pop) and not np.isnan(eld_pct)
                       else np.nan)
        series = {y: r[y] for y in year_cols_age}
        recs.append(dict(
            barri_name   = name,
            elderly_pct  = eld_pct,
            density      = density,
            area_ha      = area,
            total_pop    = total_pop,
            elderly_pop  = elderly_cnt,
            pct_series   = json.dumps(series),
        ))
    return pd.DataFrame(recs)


# ======================================================================
# 6. SPATIAL JOIN + CATCHMENT  (elderly reachable per stop)
# ======================================================================
def enrich_nodes(nodes, barris, demo):
    """
    For every stop:
      - assign its neighbourhood (point-in-polygon)
      - estimate elderly residents within R_ELDERLY and R_GENERAL
    Elderly residents are spread uniformly across each neighbourhood polygon;
    a stop's catchment elderly count = sum over neighbourhoods of
    (overlap_area_within_radius / barri_area) * barri_elderly_pop.
    """
    gdf = gpd.GeoDataFrame(
        nodes.copy(),
        geometry=gpd.points_from_xy(nodes["lon"], nodes["lat"]),
        crs="EPSG:4326",
    )

    # assign neighbourhood
    joined = gpd.sjoin(gdf, barris[["barri_name", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    gdf["barri_name"] = joined["barri_name"].values

    # project to metric CRS for buffering
    gdf_m   = gdf.to_crs(25831)
    barris_m = barris.to_crs(25831).merge(demo, on="barri_name", how="left")
    barris_m["elderly_density_m2"] = (
        barris_m["elderly_pop"] / barris_m.geometry.area)
    barris_m["pop_density_m2"] = (
        barris_m["total_pop"] / barris_m.geometry.area)

    sindex = barris_m.sindex

    eld_250, eld_400, pop_400 = [], [], []
    for geom in gdf_m.geometry:
        out = {}
        for radius, key in ((R_ELDERLY, "e250"), (R_GENERAL, "e400")):
            buf = geom.buffer(radius)
            cand = list(sindex.intersection(buf.bounds))
            tot_e = tot_p = 0.0
            for ci in cand:
                brow = barris_m.iloc[ci]
                inter = brow.geometry.intersection(buf)
                if inter.is_empty:
                    continue
                a = inter.area
                ed = brow["elderly_density_m2"]
                pd_ = brow["pop_density_m2"]
                if not np.isnan(ed):
                    tot_e += a * ed
                if not np.isnan(pd_):
                    tot_p += a * pd_
            out[key] = (tot_e, tot_p)
        eld_250.append(out["e250"][0])
        eld_400.append(out["e400"][0])
        pop_400.append(out["e400"][1])

    nodes = nodes.copy()
    nodes["barri_name"]  = gdf["barri_name"].values
    nodes["elderly_250"] = eld_250    # elderly residents within 250 m
    nodes["elderly_400"] = eld_400    # elderly residents within 400 m
    nodes["pop_400"]     = pop_400    # general population within 400 m
    return nodes


# ======================================================================
# 7. MAIN
# ======================================================================
def build_all(verbose=True):
    def log(*a):
        if verbose: print(*a, flush=True)

    log("· loading boundaries …")
    barris, districts, terme = load_boundaries()

    log("· loading metro + FGC stops …")
    metro = load_metro_stops()
    log(f"    {len(metro)} metro/FGC nodes")

    log("· loading bus stops …")
    bus = load_bus_stops()
    log(f"    {len(bus)} bus nodes")

    nodes = pd.concat([metro, bus], ignore_index=True)

    log("· building edges …")
    edges = build_edges(nodes)
    log(f"    {len(edges)} edges")

    log("· loading route corridors (shapes) …")
    shapes = load_shapes()
    log(f"    {len(shapes)} corridor polylines")

    log("· loading demographics …")
    demo = load_demographics(barris)
    log(f"    {len(demo)} neighbourhoods")

    log("· spatial join + catchment (this is the slow part) …")
    nodes = enrich_nodes(nodes, barris, demo)

    # save
    nodes.to_parquet(os.path.join(PROC_DIR, "nodes.parquet"))
    edges.to_parquet(os.path.join(PROC_DIR, "edges.parquet"))
    shapes.to_file(os.path.join(PROC_DIR, "shapes.geojson"), driver="GeoJSON")
    barris.to_file(os.path.join(PROC_DIR, "barris.geojson"), driver="GeoJSON")
    districts.to_file(os.path.join(PROC_DIR, "districts.geojson"),
                      driver="GeoJSON")
    demo.to_parquet(os.path.join(PROC_DIR, "demographics.parquet"))
    log("✓ pipeline complete — processed/ written")
    return nodes, edges, shapes, barris, demo


if __name__ == "__main__":
    build_all()
