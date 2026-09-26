#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic SRTM / ETOPO topography acquisition for MTpy-v2 ModEM models.

The module determines the geographic footprint of the already-built UTM mesh,
downloads only the DEM tiles needed to cover that footprint, mosaics them, and
resamples the DEM directly onto the MTpy-v2 model cell centres.

Data sources
------------
SRTM 1 arc-second (~30 m):
    AWS public elevation tile archive used by the open-source ``elevation``
    downloader: elevation-tiles-prod/skadi.

ETOPO 2022 15 arc-second (~450 m at the equator):
    NOAA/NCEI bedrock elevation GeoTIFF tiles.
"""

from __future__ import annotations

import gzip
import math
import shutil
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from pyproj import Transformer

SRTM_BASE = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
ETOPO_BASE = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/"
    "15s_bed_elev_gtif"
)

SRTM_SOUTH = -56.0
SRTM_NORTH = 60.0


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def _tile_name_srtm(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def _tile_name_etopo(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def _download(url: str, destination: Path, log=None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    tmp = destination.with_suffix(destination.suffix + ".part")
    _log(log, f"  Downloading: {url}")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 180),
            headers={"User-Agent": "MTpy-v2-EDI-ModEM-topography/1.0"},
        ) as response:
            response.raise_for_status()
            with tmp.open("wb") as fout:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fout.write(chunk)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _srtm_tile_ranges(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    west, south, east, north = bounds
    lon0 = math.floor(west)
    lon1 = math.floor(np.nextafter(east, -np.inf))
    lat0 = math.floor(south)
    lat1 = math.floor(np.nextafter(north, -np.inf))
    return [(lat, lon) for lat in range(lat0, lat1 + 1) for lon in range(lon0, lon1 + 1)]


def _etopo_tile_ranges(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    west, south, east, north = bounds
    lon0 = math.floor(west / 15.0) * 15
    lon1 = math.floor(np.nextafter(east, -np.inf) / 15.0) * 15
    lat0 = math.floor(south / 15.0) * 15
    lat1 = math.floor(np.nextafter(north, -np.inf) / 15.0) * 15
    return [
        (lat, lon)
        for lat in range(lat0, lat1 + 15, 15)
        for lon in range(lon0, lon1 + 15, 15)
    ]


def mesh_absolute_utm_bounds(mesh, margin_m: float = 0.0) -> tuple[float, float, float, float]:
    """Return model footprint in absolute UTM metres."""
    cp = mesh.center_point
    if hasattr(cp, "east"):
        ce = float(cp.east)
        cn = float(cp.north)
    else:
        ce = float(cp["east"])
        cn = float(cp["north"])

    east0 = ce + float(mesh.grid_east[0])
    east1 = ce + float(mesh.grid_east[-1])
    north0 = cn + float(mesh.grid_north[0])
    north1 = cn + float(mesh.grid_north[-1])
    return (
        min(east0, east1) - margin_m,
        min(north0, north1) - margin_m,
        max(east0, east1) + margin_m,
        max(north0, north1) + margin_m,
    )


def utm_bounds_to_lonlat(
    utm_bounds: tuple[float, float, float, float],
    utm_epsg: int,
) -> tuple[float, float, float, float]:
    west_e, south_n, east_e, north_n = utm_bounds
    transformer = Transformer.from_crs(
        f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
    )
    corners = [
        transformer.transform(west_e, south_n),
        transformer.transform(west_e, north_n),
        transformer.transform(east_e, south_n),
        transformer.transform(east_e, north_n),
    ]
    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]
    return min(lons), min(lats), max(lons), max(lats)


def choose_source(
    bounds_lonlat: tuple[float, float, float, float],
    preferred: str = "auto",
    max_srtm_tiles: int = 16,
) -> tuple[str, str]:
    """Choose SRTM or ETOPO.

    Automatic mode prefers SRTM 1 arc-second whenever the model is inside SRTM
    coverage and the footprint does not require an excessive number of tiles.
    Otherwise ETOPO 2022 15 arc-second is selected as a global fallback.
    """
    west, south, east, north = bounds_lonlat
    inside_srtm = south >= SRTM_SOUTH and north <= SRTM_NORTH
    tile_count = len(_srtm_tile_ranges(bounds_lonlat)) if inside_srtm else 10**9

    p = preferred.strip().lower()
    if p in {"srtm", "srtm 30m", "srtm1", "srtm 1 arc-sec"}:
        if not inside_srtm:
            raise ValueError(
                f"SRTM 1 arc-second coverage is limited to about {SRTM_SOUTH:g}° to "
                f"{SRTM_NORTH:g}°. Model bounds are outside that range."
            )
        return "srtm1", f"SRTM 1 arc-second (~30 m), {tile_count} tile(s)"
    if p in {"etopo", "etopo 15s", "etopo 2022"}:
        return "etopo15", "ETOPO 2022 15 arc-second"

    if inside_srtm and tile_count <= max_srtm_tiles:
        return "srtm1", f"SRTM 1 arc-second (~30 m), {tile_count} tile(s)"
    return "etopo15", "ETOPO 2022 15 arc-second"


def _download_srtm_tiles(
    bounds_lonlat: tuple[float, float, float, float],
    cache_dir: Path,
    log=None,
) -> list[Path]:
    paths: list[Path] = []
    for lat0, lon0 in _srtm_tile_ranges(bounds_lonlat):
        tile = _tile_name_srtm(lat0, lon0)
        folder = cache_dir / tile[:3]
        hgt_path = folder / f"{tile}.hgt"
        if not hgt_path.exists():
            gz_path = folder / f"{tile}.hgt.gz"
            url = f"{SRTM_BASE}/{tile[:3]}/{tile}.hgt.gz"
            _download(url, gz_path, log)
            _log(log, f"  Decompressing: {gz_path.name}")
            with gzip.open(gz_path, "rb") as fin, hgt_path.open("wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
        paths.append(hgt_path)
    return paths


def _download_etopo_tiles(
    bounds_lonlat: tuple[float, float, float, float],
    cache_dir: Path,
    log=None,
) -> list[Path]:
    paths: list[Path] = []
    for lat0, lon0 in _etopo_tile_ranges(bounds_lonlat):
        tile = _tile_name_etopo(lat0, lon0)
        path = cache_dir / f"ETOPO_2022_v1_15s_{tile}_bed.tif"
        if not path.exists():
            url = f"{ETOPO_BASE}/{path.name}"
            _download(url, path, log)
        paths.append(path)
    return paths


def _mosaic_to_wgs84(
    tile_paths: Iterable[Path],
    bounds_lonlat: tuple[float, float, float, float],
    output: Path,
    log=None,
) -> Path:
    sources = [rasterio.open(p) for p in tile_paths]
    try:
        _log(log, f"  Mosaicking {len(sources)} DEM tile(s) ...")
        west, south, east, north = bounds_lonlat
        mosaic, transform = merge(
            sources,
            bounds=(west, south, east, north),
            nodata=-32768.0,
            dtype="float32",
        )
        data = mosaic[0].astype(np.float32, copy=False)
        data[data <= -32767] = np.nan
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 2,
        }
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data, 1)
    finally:
        for src in sources:
            src.close()
    return output


def _default_margin(mesh) -> float:
    dx = abs(float(np.median(np.diff(mesh.grid_east))))
    dy = abs(float(np.median(np.diff(mesh.grid_north))))
    return max(2000.0, 5.0 * math.hypot(dx, dy))


def _save_model_grid_topography(
    topo_model: np.ndarray,
    mesh,
    utm_epsg: int,
    output: Path,
) -> None:
    centers_e = 0.5 * (mesh.grid_east[:-1] + mesh.grid_east[1:])
    centers_n = 0.5 * (mesh.grid_north[:-1] + mesh.grid_north[1:])
    dx = float(np.median(np.diff(mesh.grid_east)))
    dy = float(np.median(np.diff(mesh.grid_north)))
    cp = mesh.center_point
    ce = float(cp.east) if hasattr(cp, "east") else float(cp["east"])
    cn = float(cp.north) if hasattr(cp, "north") else float(cp["north"])
    x0 = ce + float(centers_e[0]) - dx / 2
    y1 = cn + float(centers_n[-1]) + dy / 2
    transform = from_origin(x0, y1, dx, dy)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": len(centers_n),
        "width": len(centers_e),
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{utm_epsg}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 2,
    }
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(np.asarray(topo_model, dtype=np.float32), 1)


def resample_dem_to_model_grid(
    wgs84_dem: Path,
    mesh,
    utm_epsg: int,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Reproject the WGS84 DEM directly to the model-cell-centre grid."""
    centers_e = 0.5 * (mesh.grid_east[:-1] + mesh.grid_east[1:])
    centers_n = 0.5 * (mesh.grid_north[:-1] + mesh.grid_north[1:])
    dx = float(np.median(np.diff(mesh.grid_east)))
    dy = float(np.median(np.diff(mesh.grid_north)))

    cp = mesh.center_point
    ce = float(cp.east) if hasattr(cp, "east") else float(cp["east"])
    cn = float(cp.north) if hasattr(cp, "north") else float(cp["north"])
    dst_x0 = ce + float(centers_e[0]) - dx / 2
    dst_y1 = cn + float(centers_n[-1]) + dy / 2
    dst_transform = from_origin(dst_x0, dst_y1, dx, dy)

    resampling = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "linear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }.get(interpolation.lower(), Resampling.bilinear)

    destination = np.full(
        (len(centers_n), len(centers_e)), np.nan, dtype=np.float32
    )
    with rasterio.open(wgs84_dem) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=f"EPSG:{utm_epsg}",
            dst_nodata=np.nan,
            resampling=resampling,
        )

    finite = np.isfinite(destination)
    if not np.all(finite):
        missing = int(np.size(destination) - finite.sum())
        raise RuntimeError(
            f"Automatic DEM contains {missing} model cells without valid elevation "
            "after reprojection. Increase DEM margin or use the other DEM source."
        )
    return destination



def prepare_local_topography(
    mesh,
    utm_epsg: int,
    out_dir: Path,
    source_geotiff: Path,
    interpolation: str = "bilinear",
    log: Callable[[str], None] | None = None,
) -> dict:
    """Prepare a local GeoTIFF on the exact MTpy-v2 model-cell grid.

    Unlike ``add_topography_to_model(topography_file=...)``, this route accepts
    ordinary GeoTIFF/Cloud-Optimized GeoTIFF inputs with their own CRS and
    reprojects them directly onto the model grid using rasterio. The resulting
    ``surface_tuple`` is then passed to MTpy-v2, avoiding assumptions about the
    ArcGIS ASCII-grid file format documented for ``topography_file``.
    """
    source_geotiff = Path(source_geotiff).expanduser().resolve()
    if not source_geotiff.is_file():
        raise FileNotFoundError(f"Local topography GeoTIFF not found: {source_geotiff}")

    with rasterio.open(source_geotiff) as src:
        source_crs = src.crs
        source_count = src.count
    if source_crs is None:
        raise ValueError(
            f"Local GeoTIFF has no CRS: {source_geotiff}. "
            "Define its CRS before using it."
        )
    if source_count < 1:
        raise ValueError(f"Local GeoTIFF has no raster bands: {source_geotiff}")

    _log(log, f"  Local DEM: {source_geotiff}")
    _log(log, f"  Source CRS: {source_crs}")

    # Reuse the exact same model-grid reprojection machinery used by SRTM/ETOPO.
    out_dir = Path(out_dir)
    dem_dir = out_dir / "topography"
    dem_dir.mkdir(parents=True, exist_ok=True)

    topo_model = resample_dem_to_model_grid(
        source_geotiff, mesh, utm_epsg, interpolation=interpolation
    )

    model_tif = dem_dir / "local_topography_on_model_grid.tif"
    _save_model_grid_topography(topo_model, mesh, utm_epsg, model_tif)
    np.save(dem_dir / "local_topography_on_model_grid.npy", topo_model)

    centers_e = 0.5 * (mesh.grid_east[:-1] + mesh.grid_east[1:])
    centers_n = 0.5 * (mesh.grid_north[:-1] + mesh.grid_north[1:])
    cp = mesh.center_point
    ce = float(cp.east) if hasattr(cp, "east") else float(cp["east"])
    cn = float(cp.north) if hasattr(cp, "north") else float(cp["north"])
    abs_e, abs_n = np.meshgrid(ce + centers_e, cn + centers_n)
    to_wgs84 = Transformer.from_crs(
        f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
    )
    lon_grid, lat_grid = to_wgs84.transform(abs_e, abs_n)

    surface_tuple = (
        np.asarray(lon_grid, dtype=float),
        np.asarray(lat_grid, dtype=float),
        np.asarray(topo_model, dtype=float),
    )

    finite = np.isfinite(topo_model)
    if not np.all(finite):
        missing = int(np.size(topo_model) - finite.sum())
        raise RuntimeError(
            f"Local GeoTIFF does not cover {missing} model cells after reprojection. "
            "Make sure the DEM covers the entire model extent (including padding)."
        )

    metadata = {
        "source": "local_geotiff",
        "description": f"Local GeoTIFF: {source_geotiff.name}",
        "utm_epsg": int(utm_epsg),
        "source_geotiff": str(source_geotiff),
        "model_grid_geotiff": str(model_tif),
        "model_grid_array_north_east": [int(v) for v in topo_model.shape],
        "elevation_min_m": float(np.nanmin(topo_model)),
        "elevation_max_m": float(np.nanmax(topo_model)),
        "elevation_mean_m": float(np.nanmean(topo_model)),
    }
    with (dem_dir / "local_topography_metadata.json").open("w", encoding="utf-8") as fid:
        import json
        json.dump(metadata, fid, indent=2, ensure_ascii=False)

    _log(
        log,
        f"  Local DEM elevation range on model grid: "
        f"{metadata['elevation_min_m']:.1f} to {metadata['elevation_max_m']:.1f} m",
    )
    _log(log, f"  Model-grid DEM saved: {model_tif}")

    return {
        **metadata,
        "topography_array": topo_model,
        "surface_tuple": surface_tuple,
    }

def prepare_topography(
    mesh,
    utm_epsg: int,
    out_dir: Path,
    source: str = "auto",
    interpolation: str = "bilinear",
    log: Callable[[str], None] | None = None,
) -> dict:
    """Acquire and prepare a DEM for an MTpy-v2 mesh.

    Returns a dictionary containing the chosen source, geographic bounds,
    source GeoTIFF, model-grid GeoTIFF and model-grid elevation array.
    """
    margin = _default_margin(mesh)
    utm_bounds = mesh_absolute_utm_bounds(mesh, margin_m=margin)
    lonlat_bounds = utm_bounds_to_lonlat(utm_bounds, utm_epsg)
    chosen, description = choose_source(lonlat_bounds, source)

    west, south, east, north = lonlat_bounds
    _log(log, "Automatic topography:")
    _log(log, f"  Model+DEM margin in UTM : {margin:.1f} m")
    _log(log, f"  Geographic bounds       : W={west:.6f}, S={south:.6f}, E={east:.6f}, N={north:.6f}")
    _log(log, f"  Selected DEM            : {description}")

    cache_root = Path.home() / ".cache" / "mtpy_modem_topography"
    dem_dir = Path(out_dir) / "topography"
    cache_dir = cache_root / ("srtm1" if chosen == "srtm1" else "etopo15")

    if chosen == "srtm1":
        try:
            tiles = _download_srtm_tiles(lonlat_bounds, cache_dir, log)
        except Exception as exc:
            if source.strip().lower() == "auto":
                _log(log, f"  SRTM download failed ({exc}); falling back to ETOPO 2022.")
                chosen = "etopo15"
                description = "ETOPO 2022 15 arc-second (automatic fallback)"
                tiles = _download_etopo_tiles(lonlat_bounds, cache_root / "etopo15", log)
            else:
                raise
    else:
        tiles = _download_etopo_tiles(lonlat_bounds, cache_dir, log)

    source_tif = dem_dir / (
        "auto_SRTM1_clipped.tif" if chosen == "srtm1" else "auto_ETOPO2022_clipped.tif"
    )
    _mosaic_to_wgs84(tiles, lonlat_bounds, source_tif, log)

    _log(log, "  Reprojecting DEM to the exact MTpy-v2 model-cell grid ...")
    topo_model = resample_dem_to_model_grid(
        source_tif,
        mesh,
        utm_epsg,
        interpolation=interpolation,
    )
    model_tif = dem_dir / "auto_topography_on_model_grid.tif"
    _save_model_grid_topography(topo_model, mesh, utm_epsg, model_tif)

    np.save(dem_dir / "auto_topography_on_model_grid.npy", topo_model)

    # Build a surface tuple on the exact model-cell-centre grid. MTpy-v2
    # accepts this tuple directly and will interpolate it to the same model
    # grid; using the exact grid here avoids any geographic resampling gap.
    centers_e = 0.5 * (mesh.grid_east[:-1] + mesh.grid_east[1:])
    centers_n = 0.5 * (mesh.grid_north[:-1] + mesh.grid_north[1:])
    cp = mesh.center_point
    ce = float(cp.east) if hasattr(cp, "east") else float(cp["east"])
    cn = float(cp.north) if hasattr(cp, "north") else float(cp["north"])
    abs_e, abs_n = np.meshgrid(ce + centers_e, cn + centers_n)
    to_wgs84 = Transformer.from_crs(
        f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
    )
    lon_grid, lat_grid = to_wgs84.transform(abs_e, abs_n)

    surface_tuple = (
        np.asarray(lon_grid, dtype=float),
        np.asarray(lat_grid, dtype=float),
        np.asarray(topo_model, dtype=float),
    )

    metadata = {
        "source": chosen,
        "description": description,
        "utm_epsg": int(utm_epsg),
        "margin_m": float(margin),
        "bounds_lonlat": [float(v) for v in lonlat_bounds],
        "source_geotiff": str(source_tif),
        "model_grid_geotiff": str(model_tif),
        "model_grid_array_north_east": [int(v) for v in topo_model.shape],
        "elevation_min_m": float(np.nanmin(topo_model)),
        "elevation_max_m": float(np.nanmax(topo_model)),
        "elevation_mean_m": float(np.nanmean(topo_model)),
    }
    with (dem_dir / "topography_metadata.json").open("w", encoding="utf-8") as fid:
        import json
        json.dump(metadata, fid, indent=2, ensure_ascii=False)

    _log(log, f"  Elevation range on model grid: {metadata['elevation_min_m']:.1f} to {metadata['elevation_max_m']:.1f} m")
    _log(log, f"  Source DEM saved             : {source_tif}")
    _log(log, f"  Model-grid DEM saved         : {model_tif}")

    return {
        **metadata,
        "topography_array": topo_model,
        "surface_tuple": surface_tuple,
    }
