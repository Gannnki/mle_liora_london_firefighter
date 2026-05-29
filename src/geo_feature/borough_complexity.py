from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOROUGH_PATH = PROJECT_ROOT / "utils" / "London_Boroughs.gpkg"
DEFAULT_OSM_PATH = PROJECT_ROOT / "utils" / "greater-london.gpkg"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "utils" / "borough_intersection_features.csv"
PROJECTED_CRS = "EPSG:27700"
DEFAULT_TOLERANCE_METERS = 15

BOROUGH_NAME_CANDIDATES = ["NAME", "name", "BOROUGH", "borough", "borough_name"]
DRIVING_ROAD_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
}


def detect_borough_name_col(boroughs: gpd.GeoDataFrame) -> str:
    for column in BOROUGH_NAME_CANDIDATES:
        if column in boroughs.columns:
            return column

    raise ValueError(
        "Cannot find borough name column. Available columns: "
        f"{boroughs.columns.tolist()}"
    )


def read_boroughs(borough_path: Path | str = DEFAULT_BOROUGH_PATH) -> tuple[gpd.GeoDataFrame, str]:
    boroughs = gpd.read_file(borough_path).to_crs(PROJECTED_CRS)
    borough_name_col = detect_borough_name_col(boroughs)
    return boroughs, borough_name_col


def load_intersections_from_osm_pbf(
    osm_path: Path | str,
    tolerance_meters: float = DEFAULT_TOLERANCE_METERS,
) -> gpd.GeoDataFrame:
    try:
        from pyrosm import OSM
    except ImportError as exc:
        raise ImportError(
            "pyrosm is required for .osm.pbf inputs. Install pyrosm or use the "
            "provided greater-london.gpkg road layer."
        ) from exc

    osm = OSM(str(osm_path))
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    nodes = nodes.to_crs(PROJECTED_CRS)
    edges = edges.to_crs(PROJECTED_CRS)

    graph = ox.graph_from_gdfs(nodes, edges)
    intersection_points = ox.consolidate_intersections(
        graph,
        tolerance=tolerance_meters,
        rebuild_graph=False,
        dead_ends=False,
    )

    return gpd.GeoDataFrame(
        geometry=list(intersection_points),
        crs=PROJECTED_CRS,
    )


def load_intersections_from_roads_gpkg(
    osm_path: Path | str = DEFAULT_OSM_PATH,
    layer: str = "gis_osm_roads_free",
    tolerance_meters: float = DEFAULT_TOLERANCE_METERS,
) -> gpd.GeoDataFrame:
    roads = gpd.read_file(osm_path, layer=layer)

    if "fclass" in roads.columns:
        roads = roads[roads["fclass"].isin(DRIVING_ROAD_CLASSES)].copy()

    roads = roads.to_crs(PROJECTED_CRS)
    roads = roads[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    roads = roads.explode(index_parts=False, ignore_index=True)

    endpoint_rows = []
    for geometry in roads.geometry:
        if geometry.geom_type != "LineString" or len(geometry.coords) < 2:
            continue

        start = geometry.coords[0]
        end = geometry.coords[-1]
        endpoint_rows.append((round(start[0], 3), round(start[1], 3)))
        endpoint_rows.append((round(end[0], 3), round(end[1], 3)))

    if not endpoint_rows:
        return gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)

    endpoint_counts = (
        pd.DataFrame(endpoint_rows, columns=["x", "y"])
        .value_counts(["x", "y"])
        .reset_index(name="road_endpoint_count")
    )
    junctions = endpoint_counts[endpoint_counts["road_endpoint_count"] > 1].copy()

    if junctions.empty:
        return gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)

    junction_points = gpd.GeoDataFrame(
        junctions,
        geometry=gpd.points_from_xy(junctions["x"], junctions["y"]),
        crs=PROJECTED_CRS,
    )

    consolidated = junction_points.buffer(tolerance_meters).union_all()
    polygons = list(consolidated.geoms) if hasattr(consolidated, "geoms") else [consolidated]

    return gpd.GeoDataFrame(
        geometry=[polygon.centroid for polygon in polygons if not polygon.is_empty],
        crs=PROJECTED_CRS,
    )


def load_intersections(
    osm_path: Path | str = DEFAULT_OSM_PATH,
    tolerance_meters: float = DEFAULT_TOLERANCE_METERS,
) -> gpd.GeoDataFrame:
    osm_path = Path(osm_path)

    if osm_path.suffix.lower() == ".pbf":
        return load_intersections_from_osm_pbf(osm_path, tolerance_meters)

    return load_intersections_from_roads_gpkg(osm_path, tolerance_meters=tolerance_meters)


def build_borough_intersection_features(
    borough_path: Path | str = DEFAULT_BOROUGH_PATH,
    osm_path: Path | str = DEFAULT_OSM_PATH,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
    tolerance_meters: float = DEFAULT_TOLERANCE_METERS,
) -> pd.DataFrame:
    boroughs, borough_name_col = read_boroughs(borough_path)
    print("Borough columns:", boroughs.columns.tolist())
    print("Using borough name column:", borough_name_col)

    intersections = load_intersections(osm_path, tolerance_meters=tolerance_meters)

    joined = gpd.sjoin(
        intersections,
        boroughs[[borough_name_col, "geometry"]],
        how="left",
        predicate="within",
    )

    intersection_count = (
        joined
        .groupby(borough_name_col)
        .size()
        .reset_index(name="borough_intersection_count")
    )

    borough_features = boroughs[[borough_name_col, "geometry"]].copy()
    borough_features["borough_area_km2"] = borough_features.geometry.area / 1_000_000
    borough_features = borough_features.merge(
        intersection_count,
        on=borough_name_col,
        how="left",
    )
    borough_features["borough_intersection_count"] = (
        borough_features["borough_intersection_count"].fillna(0).astype(int)
    )
    borough_features["borough_intersection_density"] = (
        borough_features["borough_intersection_count"]
        / borough_features["borough_area_km2"]
    )

    feature_df = borough_features[
        [
            borough_name_col,
            "borough_area_km2",
            "borough_intersection_count",
            "borough_intersection_density",
        ]
    ].copy()
    feature_df = feature_df.rename(columns={borough_name_col: "IncGeo_BoroughName"})
    feature_df["IncGeo_BoroughName"] = (
        feature_df["IncGeo_BoroughName"].astype(str).str.strip().str.upper()
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path, index=False)
        print(f"Saved borough intersection features to: {output_path}")

    return feature_df


if __name__ == "__main__":
    build_borough_intersection_features()
