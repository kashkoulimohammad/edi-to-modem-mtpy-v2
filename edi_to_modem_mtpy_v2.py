#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDI -> ModEM preparation workflow based on MTpy-v2 (2.1.x / 2.1.4 API)

Outputs
-------
1) ModEM_Model_File.rho        : initial 3-D resistivity model
2) ModEM_Data.dat              : ModEM data file
3) covariance.cov              : ModEM covariance file
4) stations.csv                : final model coordinates of stations
5) selected_frequencies_Hz.txt : target frequency grid
6) selected_frequencies_Hz.csv : target frequency grid + station counts
7) figures/vertical_section_mesh.png : mesh-exact vertical section along the
                                         best-fit MT-station alignment

Important coordinate convention
--------------------------------
MTpy-v2 / ModEM model coordinates are:
    x = North
    y = East
    z = +Down

The script uses a WGS84 UTM EPSG supplied by the user.  Topography is read
from a GeoTIFF through MTpy-v2.  When topography is enabled, stations are
centered on model cells and projected onto the model topography following the
MTpy-v2 ModEM workflow.

Author: Mohammad F. Kashkouli / generated workflow
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Matplotlib is intentionally configured headlessly so this script can be used
# on a workstation/terminal without opening figures.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtpy import MT
from mtpy.core import MTData
from mtpy.modeling.structured_mesh_3d import StructuredGrid3D
from mtpy.modeling.modem import Covariance
from auto_topography import prepare_topography, prepare_local_topography


# -----------------------------------------------------------------------------
# Generic input helpers
# -----------------------------------------------------------------------------

def ask_str(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def ask_int(prompt: str, default: int | None = None, minimum: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("  Please enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"  Value must be >= {minimum}.")
            continue
        return value


def ask_float(prompt: str, default: float | None = None, minimum: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
        if minimum is not None and value < minimum:
            print(f"  Value must be >= {minimum}.")
            continue
        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1", "true"}:
            return True
        if raw in {"n", "no", "0", "false"}:
            return False
        print("  Please answer y/yes or n/no.")


def ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    choice_text = ", ".join(f"{k}={v}" for k, v in choices.items())
    while True:
        raw = input(f"{prompt} ({choice_text}) [{default}]: ").strip()
        if not raw:
            return default
        if raw in choices:
            return raw
        print("  Invalid choice.")


def load_config_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fid:
        cfg = json.load(fid)
    # Backward compatibility with earlier GUI/config files that used
    # n_horizontal and n_vertical. Those plot controls are intentionally
    # ignored; the current workflow creates one mesh-exact vertical section.
    cfg.setdefault("section_depth_m", 10000.0)
    return cfg


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def collect_inputs() -> dict:
    print("\n" + "=" * 78)
    print("EDI -> ModEM preparation using MTpy-v2")
    print("=" * 78)

    edi_dir = Path(ask_str("EDI folder"))
    out_dir = Path(ask_str("Output folder", str(edi_dir / "modem_input")))

    if not edi_dir.exists() or not edi_dir.is_dir():
        raise FileNotFoundError(f"EDI folder not found: {edi_dir}")

    utm_epsg = ask_int("WGS84 UTM EPSG", minimum=1000)

    print("\n--- Frequency grid ---")
    n_freq = ask_int("Number of target frequencies", 24, 2)
    f_min = ask_float("Minimum frequency [Hz]", 0.001, 1e-12)
    f_max = ask_float("Maximum frequency [Hz]", 1000.0, 1e-12)
    if f_max <= f_min:
        raise ValueError("Maximum frequency must be larger than minimum frequency.")

    print("\n--- Horizontal core model ---")
    nx = ask_int("Number of core cells in East (Y)", 40, 1)
    ny = ask_int("Number of core cells in North (X)", 40, 1)
    dx = ask_float("Core cell size East [m]", 500.0, 1e-6)
    dy = ask_float("Core cell size North [m]", 500.0, 1e-6)

    print("\n--- Horizontal padding ---")
    pad_x = ask_int("Number of North/South padding cells per side", 8, 0)
    pad_y = ask_int("Number of East/West padding cells per side", 8, 0)
    pad_factor = ask_float("Horizontal padding growth factor (>1)", 1.3, 1.0)

    print("\n--- Vertical model ---")
    nz = ask_int("Number of Earth cells in depth", 35, 2)
    dz1 = ask_float("Thickness of first Earth layer [m]", 30.0, 1e-6)
    dz_factor = ask_float("Vertical growth factor (>1)", 1.15, 1.0)

    topo = ask_yes_no("Apply topography from GeoTIFF", True)
    topo_file = None
    n_air = 0
    topo_interp = "nearest"
    if topo:
        topo_file = Path(ask_str("Topography GeoTIFF path"))
        if not topo_file.exists():
            raise FileNotFoundError(f"Topography file not found: {topo_file}")
        n_air = ask_int("Number of air layers", 20, 1)
        air_thickness = ask_float("Thickness of each air layer [m]", 100.0, 1e-6)
        topo_interp = ask_choice(
            "GeoTIFF interpolation",
            {"nearest": "nearest", "linear": "linear", "cubic": "cubic"},
            "linear",
        )

    print("\n--- ModEM data/error options ---")
    z_error_pct = ask_float("Impedance model-error floor [%]", 5.0, 0.01)
    tipper_error = ask_float("Tipper absolute model-error floor", 0.02, 0.0)
    inv_mode = ask_choice(
        "ModEM inversion mode",
        {
            "1": "Full impedance + vertical components",
            "2": "Full impedance only",
        },
        "1",
    )

    print("\n--- Covariance ---")
    smooth_e = ask_float("Covariance east smoothing", 0.3, 0.0)
    smooth_n = ask_float("Covariance north smoothing", 0.3, 0.0)
    smooth_z = ask_float("Covariance vertical smoothing", 0.3, 0.0)
    smooth_num = ask_int("Number of covariance smoothing iterations", 1, 0)

    print("\n--- Plot ---")
    section_depth_m = ask_float(
        "Maximum depth of vertical section below sea level [m]",
        10000.0,
        1.0,
    )
    initial_res = ask_float("Initial half-space resistivity [Ohm-m]", 100.0, 1e-6)

    return {
        "edi_dir": str(edi_dir),
        "out_dir": str(out_dir),
        "utm_epsg": utm_epsg,
        "n_freq": n_freq,
        "f_min": f_min,
        "f_max": f_max,
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "pad_factor": pad_factor,
        "nz": nz,
        "dz1": dz1,
        "dz_factor": dz_factor,
        "topography": topo,
        "topography_source": "local",
        "topography_file": str(topo_file) if topo_file else None,
        "n_air_layers": n_air,
        "air_layer_thickness": air_thickness,
        "topography_interp": topo_interp,
        "z_error_pct": z_error_pct,
        "tipper_error": tipper_error,
        "inv_mode": inv_mode,
        "smooth_e": smooth_e,
        "smooth_n": smooth_n,
        "smooth_z": smooth_z,
        "smooth_num": smooth_num,
        "section_depth_m": section_depth_m,
        "initial_res": initial_res,
    }


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------

def geometric_cell_sizes(first: float, factor: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if math.isclose(factor, 1.0):
        return np.full(n, first, dtype=float)
    return first * factor ** np.arange(n, dtype=float)


def edges_from_cell_sizes(cell_sizes: np.ndarray, center: float = 0.0) -> np.ndarray:
    total = float(np.sum(cell_sizes))
    start = center - total / 2.0
    return np.r_[start, start + np.cumsum(cell_sizes)]


def build_custom_mesh(mtd: MTData, cfg: dict) -> StructuredGrid3D:
    """Create a StructuredGrid3D using exact user-specified core counts/sizes.

    The current MTpy-v2 StructuredGrid3D API documents cell_number_ew/ns but the
    make_mesh() implementation still determines the inner horizontal grid from
    station extents.  For an exact user-controlled Nx/Ny we therefore build the
    node arrays explicitly and then use the normal MTpy-v2 object for topography
    and ModEM writing.
    """
    station_df = mtd.station_locations.copy()
    if station_df.empty:
        raise ValueError("No station locations found in the MTData object.")

    x = station_df["model_north"].to_numpy(dtype=float)
    y = station_df["model_east"].to_numpy(dtype=float)

    core_x_width = cfg["ny"] * cfg["dy"]
    core_y_width = cfg["nx"] * cfg["dx"]

    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())
    # Stations need to be inside the user-defined core area.  The small margin
    # avoids placing a station exactly on the outermost node.
    if x_span >= core_x_width:
        needed = int(math.ceil(x_span / cfg["dy"])) + 2
        raise ValueError(
            f"North/South core is too small: station span={x_span:.1f} m, "
            f"core width={core_x_width:.1f} m. Increase ny to at least about {needed}."
        )
    if y_span >= core_y_width:
        needed = int(math.ceil(y_span / cfg["dx"])) + 2
        raise ValueError(
            f"East/West core is too small: station span={y_span:.1f} m, "
            f"core width={core_y_width:.1f} m. Increase nx to at least about {needed}."
        )

    # MTpy-v2 computes model-relative station coordinates from the geographic
    # center of the station area. Therefore the station-area center is (0, 0)
    # in model coordinates. Build the user-defined core around (0, 0) rather
    # than around an arbitrary/non-zero coordinate.
    cx = 0.0
    cy = 0.0

    core_dx = np.full(cfg["ny"], cfg["dy"], dtype=float)
    core_dy = np.full(cfg["nx"], cfg["dx"], dtype=float)
    pad_x = geometric_cell_sizes(cfg["dy"], cfg["pad_factor"], cfg["pad_x"])
    pad_y = geometric_cell_sizes(cfg["dx"], cfg["pad_factor"], cfg["pad_y"])

    # Outermost -> core -> outermost sequence.
    cells_x = np.r_[pad_x[::-1], core_dx, pad_x]
    cells_y = np.r_[pad_y[::-1], core_dy, pad_y]

    x_edges = edges_from_cell_sizes(cells_x, cx)
    y_edges = edges_from_cell_sizes(cells_y, cy)

    z_cells = geometric_cell_sizes(cfg["dz1"], cfg["dz_factor"], cfg["nz"])
    z_edges = np.r_[0.0, np.cumsum(z_cells)]

    mesh = StructuredGrid3D(
        station_locations=station_df,
        center_point=mtd.center_point,
        cell_size_east=cfg["dx"],
        cell_size_north=cfg["dy"],
        res_initial_value=cfg["initial_res"],
        z1_layer=cfg["dz1"],
        n_layers=cfg["nz"],
        n_air_layers=cfg["n_air_layers"],
        pad_east=cfg["pad_y"],
        pad_north=cfg["pad_x"],
        pad_stretch_h=cfg["pad_factor"],
        pad_z=0,
        z_mesh_method="custom",
        grid_z=z_edges,
    )

    # Store node coordinates in the same model-relative frame used by ModEM
    # data. The model grid is explicitly centred at (0, 0). In a ModEM model
    # file, grid_center is NOT the geographic centre; it is the lower-left
    # corner of the model relative to the data/model centre. This is the same
    # convention used internally by MTpy-v2 StructuredGrid3D.make_mesh().
    mesh.grid_east = y_edges
    mesh.grid_north = x_edges
    mesh.grid_z = z_edges

    # Force exact numerical centring in case an odd number of cells or padding
    # arithmetic introduces a tiny floating-point offset.
    east_shift = 0.5 * (mesh.grid_east[0] + mesh.grid_east[-1])
    north_shift = 0.5 * (mesh.grid_north[0] + mesh.grid_north[-1])
    mesh.grid_east = mesh.grid_east - east_shift
    mesh.grid_north = mesh.grid_north - north_shift

    # ModEM stores cell widths plus the model origin/centre line. Since the
    # in-memory grid is centred on (0,0), the correct grid_center is the
    # lower-left model coordinate.
    mesh.grid_center = np.array(
        [mesh.grid_north[0], mesh.grid_east[0], 0.0],
        dtype=float,
    )

    # Keep the exact core-cell index ranges on the mesh so that model plots can
    # deliberately exclude all padding cells. The core itself is the original
    # user-defined nx (East) × ny (North) block around (0, 0).
    mesh.core_north_start = int(cfg["pad_x"])
    mesh.core_north_end = int(cfg["pad_x"] + cfg["ny"])
    mesh.core_east_start = int(cfg["pad_y"])
    mesh.core_east_end = int(cfg["pad_y"] + cfg["nx"])
    mesh.core_north_cell_count = int(cfg["ny"])
    mesh.core_east_cell_count = int(cfg["nx"])
    mesh.core_cell_size_north = float(cfg["dy"])
    mesh.core_cell_size_east = float(cfg["dx"])

    mesh.model_epsg = cfg["utm_epsg"]
    mesh.res_model = np.full(
        (mesh.nodes_north.size, mesh.nodes_east.size, mesh.nodes_z.size),
        cfg["initial_res"],
        dtype=float,
    )
    mesh.save_path = Path(cfg["out_dir"])

    return mesh


# -----------------------------------------------------------------------------
# Topography with user-controlled, constant-thickness air layers
# -----------------------------------------------------------------------------

def apply_fixed_air_topography(
    mesh: StructuredGrid3D,
    topo_array: np.ndarray,
    n_air_layers: int,
    air_layer_thickness: float,
    air_resistivity: float = 1e12,
    topography_buffer: float | None = None,
) -> dict:
    """Apply topography while preserving an exact, user-defined air mesh.

    MTpy-v2's built-in ``airlayer_type='constant'`` uses ``z1_layer`` as the
    thickness but *recomputes* the number of air layers from topographic relief.
    For ModEM preparation we instead want the user to control both values
    explicitly, e.g. 32 layers x 100 m = 3200 m total air thickness.

    The vertical placement follows the MTpy-v2 ``log_up``/``constant`` logic:
    the topography is positive-up relative to sea level, z is positive down,
    and the flat-earth part of the existing model is shifted downward beneath
    the fixed air stack.
    """
    n_air_layers = int(n_air_layers)
    air_layer_thickness = float(air_layer_thickness)

    if n_air_layers < 1:
        raise ValueError("Number of air layers must be at least 1 when topography is enabled.")
    if air_layer_thickness <= 0:
        raise ValueError("Air layer thickness must be > 0 m.")

    topo = np.asarray(topo_array, dtype=float)
    expected_shape = (mesh.nodes_north.size, mesh.nodes_east.size)
    if topo.shape != expected_shape:
        raise ValueError(
            "Topography array shape does not match the model horizontal grid: "
            f"got {topo.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(topo)):
        raise ValueError("Topography contains NaN or infinite values on the model grid.")

    mesh.surface_dict["topography"] = topo.copy()

    # Same station-buffer logic used by MTpy-v2's add_topography_to_model().
    gcx, gcy = [
        np.mean([arr[:-1], arr[1:]], axis=0)
        for arr in (mesh.grid_east, mesh.grid_north)
    ]
    if topography_buffer is None:
        topography_buffer = 5 * (mesh.cell_size_east**2 + mesh.cell_size_north**2) ** 0.5

    from mtpy.modeling import mesh_tools as mtmesh

    core_cells = mtmesh.get_station_buffer(
        gcx,
        gcy,
        mesh.station_locations["model_east"],
        mesh.station_locations["model_north"],
        buf=topography_buffer,
    )
    topo_core = topo[core_cells]
    if topo_core.size == 0:
        topo_core = topo

    topo_core_min = max(float(np.nanmin(topo_core)), 0.0)
    topo_core_max = float(np.nanmax(topo_core))
    relief = max(0.0, topo_core_max - topo_core_min)
    total_air = n_air_layers * air_layer_thickness

    if total_air + 1e-9 < relief:
        required = int(np.ceil(relief / air_layer_thickness))
        raise ValueError(
            "The requested air mesh does not cover the topographic relief. "
            f"Topographic relief over the model core is {relief:.1f} m, but "
            f"{n_air_layers} × {air_layer_thickness:.1f} m = {total_air:.1f} m. "
            f"Use at least {required} air layers at {air_layer_thickness:.1f} m, "
            "or increase the air-layer thickness."
        )

    # Exact constant-thickness air stack. Do NOT round these values: the user
    # explicitly requested the same thickness for every air cell.
    air_edges_from_top = np.arange(
        0.0,
        total_air + air_layer_thickness,
        air_layer_thickness,
        dtype=float,
    )
    if air_edges_from_top.size != n_air_layers + 1:
        air_edges_from_top = np.r_[
            np.arange(n_air_layers, dtype=float) * air_layer_thickness,
            total_air,
        ]

    # Match MTpy-v2's vertical reference convention: sea level is tied to
    # topo_core_min and the complete air column sits above that reference.
    topo_max_grid = topo_core_min + total_air
    new_airlayers = air_edges_from_top - topo_max_grid

    # Existing earth cells start at z=0. Shift the complete earth mesh down so
    # its top coincides with the bottom of the requested air stack.
    shifted_earth_grid = mesh.grid_z + new_airlayers[-1]
    mesh.grid_z = np.concatenate([new_airlayers[:-1], shifted_earth_grid])

    old_res_model = np.array(mesh.res_model, copy=True)
    mesh.n_air_layers = n_air_layers
    mesh.n_layers = mesh.grid_z.size - 1
    mesh.grid_center[2] = mesh.grid_z[0]

    new_res_model = np.ones(
        (
            mesh.nodes_north.size,
            mesh.nodes_east.size,
            mesh.nodes_z.size,
        ),
        dtype=float,
    ) * mesh.res_initial_value
    new_res_model[:, :, n_air_layers:] = old_res_model
    mesh.res_model = new_res_model

    # Assign air above the DEM surface and seawater/bathymetry below sea level,
    # following the MTpy-v2 implementation.
    top = np.zeros_like(topo) + mesh.grid_z[0]
    bottom = -topo
    mesh.assign_resistivity_from_surface_data(top, bottom, air_resistivity)
    mesh.assign_resistivity_from_surface_data(np.zeros_like(top), bottom, 0.3)

    actual_air_thicknesses = np.diff(mesh.grid_z[: n_air_layers + 1])
    max_deviation = float(np.max(np.abs(actual_air_thicknesses - air_layer_thickness)))

    return {
        "n_air_layers": n_air_layers,
        "air_layer_thickness_m": air_layer_thickness,
        "total_air_thickness_m": total_air,
        "topographic_relief_m": relief,
        "topographic_core_min_m": topo_core_min,
        "topographic_core_max_m": topo_core_max,
        "max_air_cell_thickness_deviation_m": max_deviation,
        "air_cell_thicknesses_m": actual_air_thicknesses.tolist(),
    }


# -----------------------------------------------------------------------------
# Data loading and interpolation
# -----------------------------------------------------------------------------

def find_edi_files(edi_dir: Path) -> list[Path]:
    files = sorted(edi_dir.glob("*.edi"))
    if not files:
        files = sorted(edi_dir.glob("*.EDI"))
    if not files:
        raise FileNotFoundError(f"No EDI files found in {edi_dir}")
    return files


def read_edi_into_mtdata(edi_files: Iterable[Path], utm_epsg: int) -> MTData:
    mt_objects = []
    failures = []
    for fn in edi_files:
        try:
            # IMPORTANT for MTpy-v2 2.1.x/2.1.4:
            # MT(filename) sets the filename but does not itself perform the
            # EDI read.  The transfer-function and station metadata therefore
            # have to be populated explicitly with read().
            mt_obj = MT()
            mt_obj.read(fn)

            lat = float(getattr(mt_obj, "latitude", 0.0) or 0.0)
            lon = float(getattr(mt_obj, "longitude", 0.0) or 0.0)
            if not (np.isfinite(lat) and np.isfinite(lon)):
                raise ValueError(
                    f"Invalid station coordinates: latitude={lat}, longitude={lon}"
                )
            if abs(lat) > 90 or abs(lon) > 180:
                raise ValueError(
                    f"Station coordinates out of range: latitude={lat}, longitude={lon}"
                )
            if abs(lat) < 1e-12 and abs(lon) < 1e-12:
                raise ValueError(
                    "Station latitude/longitude are both zero after reading the EDI. "
                    "Check the EDI coordinate metadata."
                )

            mt_objects.append(mt_obj)
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.name, repr(exc)))

    if not mt_objects:
        details = "\n".join(f"  {name}: {exc}" for name, exc in failures[:10])
        raise RuntimeError(
            "None of the EDI files could be read by MTpy-v2 with valid station "
            f"coordinates.\n{details}"
        )

    if failures:
        print(f"\nWarning: {len(failures)} EDI file(s) could not be read.")
        for name, exc in failures[:10]:
            print(f"  {name}: {exc}")

    mtd = MTData()
    # Use one survey so station keys are simple and ModEM output is clean.
    mtd.add_stations(mt_objects, survey_id="edi", dataset_copy_mode="shallow")
    mtd.utm_epsg = utm_epsg

    # Fail early with a useful diagnostic instead of MTpy's generic
    # "Station locations are all 0" error.
    station_df = mtd.station_locations
    if station_df is None or station_df.empty:
        raise RuntimeError("MTpy-v2 created no station-location records from the EDI files.")

    lat_ok = np.isfinite(station_df["latitude"].to_numpy(float))
    lon_ok = np.isfinite(station_df["longitude"].to_numpy(float))
    xy_nonzero = (
        np.abs(station_df["latitude"].to_numpy(float)) > 1e-12
    ) | (
        np.abs(station_df["longitude"].to_numpy(float)) > 1e-12
    )
    if not np.all(lat_ok & lon_ok & xy_nonzero):
        bad = station_df.loc[~(lat_ok & lon_ok & xy_nonzero), ["station", "latitude", "longitude"]]
        raise RuntimeError(
            "Some station coordinates are missing/zero after EDI ingestion:\n"
            + bad.to_string(index=False)
        )

    mtd.compute_relative_locations()
    return mtd


def build_frequency_grid(cfg: dict) -> np.ndarray:
    return np.logspace(
        np.log10(cfg["f_min"]),
        np.log10(cfg["f_max"]),
        cfg["n_freq"],
    )


def interpolate_data(mtd: MTData, frequencies: np.ndarray, cfg: dict) -> None:
    """Interpolate within each station's native range; do not extrapolate."""
    mtd.interpolate(
        frequencies,
        f_type="frequency",
        inplace=True,
        bounds_error=True,
    )
    mtd.compute_model_errors(
        z_error_value=cfg["z_error_pct"],
        z_error_type="geometric_mean",
        z_floor=True,
        t_error_value=cfg["tipper_error"],
        t_error_type="absolute",
        t_floor=True,
    )


# -----------------------------------------------------------------------------
# ModEM files
# -----------------------------------------------------------------------------

def write_modem_files(mtd: MTData, mesh: StructuredGrid3D, cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = out_dir / "ModEM_Model_File.rho"
    data_file = out_dir / "ModEM_Data.dat"
    cov_file = out_dir / "covariance.cov"

    mesh.to_modem(model_fn=model_file)

    # MTpy-v2 uses this dictionary when MTData.to_modem constructs the ModEM
    # Data object.
    mtd.impedance_units = "ohm"
    mtd.model_parameters = {
        "inv_mode": cfg["inv_mode"],
        "formatting": "1",
        "topography": bool(cfg["topography"]),
    }
    modem_data = mtd.to_modem(data_filename=data_file)

    # Current MTpy-v2 covariance API expects grid_dimensions and res_model.
    cov = Covariance()
    cov.grid_dimensions = mesh.res_model.shape
    cov.smoothing_east = cfg["smooth_e"]
    cov.smoothing_north = cfg["smooth_n"]
    cov.smoothing_z = cfg["smooth_z"]
    cov.smoothing_num = cfg["smooth_num"]
    cov.write_covariance_file(
        cov_fn=cov_file,
        res_model=mesh.res_model,
    )

    return {
        "model": model_file,
        "data": data_file,
        "covariance": cov_file,
    }


# -----------------------------------------------------------------------------
# Reporting / plots
# -----------------------------------------------------------------------------

def cell_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def save_station_table(mtd: MTData, out_dir: Path) -> pd.DataFrame:
    sdf = mtd.station_locations.copy()
    cols = [
        c for c in [
            "survey", "station", "latitude", "longitude", "elevation",
            "east", "north", "utm_epsg", "model_east", "model_north",
            "model_elevation", "profile_offset",
        ] if c in sdf.columns
    ]
    sdf[cols].to_csv(out_dir / "stations.csv", index=False)
    return sdf


def frequency_availability_table(mtd: MTData, frequencies: np.ndarray) -> pd.DataFrame:
    df = mtd.to_dataframe(impedance_units="ohm")
    if df.empty:
        return pd.DataFrame()

    rows = []
    for station, g in df.groupby("station"):
        periods = np.asarray(sorted(g["period"].dropna().unique()), dtype=float)
        freq = 1.0 / periods
        present = []
        for f in frequencies:
            # Frequencies should be essentially exact after interpolation;
            # use a relative tolerance to avoid floating-point formatting issues.
            present.append(np.any(np.isclose(freq, f, rtol=2e-7, atol=0.0)))
        row = {"station": station}
        row["n_available"] = int(np.sum(present))
        row["f_min_Hz"] = float(np.min(freq)) if freq.size else np.nan
        row["f_max_Hz"] = float(np.max(freq)) if freq.size else np.nan
        for f, ok in zip(frequencies, present):
            row[f"{f:.8g}_Hz"] = int(ok)
        rows.append(row)
    return pd.DataFrame(rows)


def save_frequency_reports(mtd: MTData, frequencies: np.ndarray, out_dir: Path) -> pd.DataFrame:
    np.savetxt(
        out_dir / "selected_frequencies_Hz.txt",
        frequencies,
        fmt="%.10e",
        header="Selected target frequencies in Hz (log-spaced)",
    )
    coverage = frequency_availability_table(mtd, frequencies)
    coverage.to_csv(out_dir / "selected_frequencies_Hz.csv", index=False)
    return coverage


def plot_station_map(mesh: StructuredGrid3D, stations: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(stations["model_east"], stations["model_north"], "^", ms=5, label="MT stations")
    ax.plot(
        [mesh.grid_east[0], mesh.grid_east[-1], mesh.grid_east[-1], mesh.grid_east[0], mesh.grid_east[0]],
        [mesh.grid_north[0], mesh.grid_north[0], mesh.grid_north[-1], mesh.grid_north[-1], mesh.grid_north[0]],
        "k-", lw=0.8, label="Model boundary",
    )
    ax.axhline(0.0, color="0.45", lw=0.7, ls="--", label="Model N=0")
    ax.axvline(0.0, color="0.45", lw=0.7, ls=":", label="Model E=0")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Model East (m)")
    ax.set_ylabel("Model North (m)")
    ax.set_title("MT stations and centred ModEM model extent")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "01_stations_and_model_extent.png", dpi=220)
    plt.close(fig)


def save_topography_station_diagnostics(
    mesh: StructuredGrid3D,
    topo_array: np.ndarray,
    stations_after: pd.DataFrame,
    stations_before: pd.DataFrame,
    out_dir: Path,
    utm_epsg: int,
) -> None:
    """Save diagnostics proving which DEM elevation each centred station uses."""
    centers_e = 0.5 * (mesh.grid_east[:-1] + mesh.grid_east[1:])
    centers_n = 0.5 * (mesh.grid_north[:-1] + mesh.grid_north[1:])

    rows = []
    for _, sta in stations_after.iterrows():
        e = float(sta["model_east"])
        n = float(sta["model_north"])
        ie = int(np.clip(np.searchsorted(mesh.grid_east, e, side="right") - 1, 0, len(centers_e) - 1))
        inn = int(np.clip(np.searchsorted(mesh.grid_north, n, side="right") - 1, 0, len(centers_n) - 1))
        topo_here = float(topo_array[inn, ie])

        # Find original station row for comparison before centering.
        b = stations_before.loc[stations_before["station"].astype(str) == str(sta["station"])]
        if len(b):
            b = b.iloc[0]
            e0 = float(b["model_east_before"])
            n0 = float(b["model_north_before"])
        else:
            e0 = np.nan
            n0 = np.nan

        rows.append({
            "station": str(sta["station"]),
            "original_model_east_m": e0,
            "original_model_north_m": n0,
            "centered_model_east_m": e,
            "centered_model_north_m": n,
            "horizontal_shift_m": float(np.hypot(e - e0, n - n0)) if np.isfinite(e0) else np.nan,
            "assigned_cell_east_center_m": float(centers_e[ie]),
            "assigned_cell_north_center_m": float(centers_n[inn]),
            "dem_elevation_at_assigned_cell_m": topo_here,
            "station_elevation_after_projection_m": float(sta.get("elevation", np.nan)),
        })

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "topography_station_diagnostics.csv", index=False)


def valid_earth_indices(mesh: StructuredGrid3D) -> np.ndarray:
    zc = cell_centers(mesh.grid_z)
    # Air is represented by very high resistivity in MTpy-v2 topographic models.
    if mesh.res_model is None:
        return np.arange(len(zc))
    air_like = np.nanmedian(mesh.res_model, axis=(0, 1)) > 1e10
    idx = np.where(~air_like)[0]
    return idx if idx.size else np.arange(len(zc))


def choose_horizontal_indices(mesh: StructuredGrid3D, count: int) -> np.ndarray:
    earth_idx = valid_earth_indices(mesh)
    if count >= earth_idx.size:
        return earth_idx
    # Select approximately logarithmically with depth, while keeping ordering.
    positions = np.linspace(0, earth_idx.size - 1, count)
    sel = earth_idx[np.unique(np.round(positions).astype(int))]
    return sel


def plot_horizontal_slices(mesh: StructuredGrid3D, out_dir: Path, count: int) -> list[float]:
    y = mesh.grid_east
    x = mesh.grid_north
    zc = cell_centers(mesh.grid_z)
    idxs = choose_horizontal_indices(mesh, count)

    log_rho = np.log10(np.clip(mesh.res_model, 1e-6, 1e15))
    depths = []
    for number, iz in enumerate(idxs, start=1):
        fig, ax = plt.subplots(figsize=(9, 7))
        pcm = ax.pcolormesh(
            y / 1000.0,
            x / 1000.0,
            log_rho[:, :, iz],
            shading="auto",
            cmap="turbo",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        depth = float(zc[iz])
        depths.append(depth)
        ax.set_title(f"Initial model: horizontal slice, z = {depth:.1f} m")
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("log10 Resistivity (Ohm-m)")
        fig.tight_layout()
        fig.savefig(out_dir / f"horizontal_slice_{number:02d}_{depth:.0f}m.png", dpi=220)
        plt.close(fig)
    return depths


def pca_station_line(stations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return best-fit line center and unit direction in (east, north)."""
    pts = stations[["model_east", "model_north"]].to_numpy(dtype=float)
    center = pts.mean(axis=0)
    if len(pts) < 2:
        return center, np.array([1.0, 0.0])
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0:
        direction = -direction
    return center, direction / np.linalg.norm(direction)


def _line_box_intersection(
    point_e: float,
    point_n: float,
    direction_e: float,
    direction_n: float,
    e0: float,
    e1: float,
    n0: float,
    n1: float,
    tol: float = 1e-12,
) -> tuple[float, float] | None:
    """Return line parameter interval where P+t*d lies inside a cell."""
    t_min = -np.inf
    t_max = np.inf

    if abs(direction_e) < tol:
        if point_e < e0 or point_e > e1:
            return None
    else:
        ta = (e0 - point_e) / direction_e
        tb = (e1 - point_e) / direction_e
        if ta > tb:
            ta, tb = tb, ta
        t_min = max(t_min, ta)
        t_max = min(t_max, tb)

    if abs(direction_n) < tol:
        if point_n < n0 or point_n > n1:
            return None
    else:
        ta = (n0 - point_n) / direction_n
        tb = (n1 - point_n) / direction_n
        if ta > tb:
            ta, tb = tb, ta
        t_min = max(t_min, ta)
        t_max = min(t_max, tb)

    if t_max <= t_min + 1e-9:
        return None
    return float(t_min), float(t_max)


def plot_vertical_mesh_section(
    mesh: StructuredGrid3D,
    stations: pd.DataFrame,
    out_dir: Path,
    max_depth_m: float,
) -> dict:
    """Plot one vertical section through the *core mesh only*.

    The section follows the PCA best-fit line through the already-centered MT
    stations, but it is clipped strictly to the user-defined core rectangle.
    Padding cells are never plotted.  Real station markers are also restricted
    to the core and their vertical position is taken from ``model_elevation``
    after MTpy-v2's ``project_stations_on_topography`` call, so each marker sits
    on the model topography rather than at sea level or at an arbitrary mesh
    depth.

    The horizontal station centering itself is still controlled by the model
    core cell size (e.g. 500 m), because ``MTData.center_stations(mesh)`` is run
    before this function.  We do *not* invent synthetic stations at 500 m
    spacing; only the actual EDI stations are plotted.
    """
    if mesh.res_model is None:
        raise ValueError("Cannot plot a vertical section because res_model is None.")
    if max_depth_m <= 0:
        raise ValueError("Maximum section depth must be > 0 m.")

    # Exact core cell ranges stored during build_custom_mesh().  Fall back to
    # the full mesh only for backward compatibility with an older mesh object.
    cn0 = int(getattr(mesh, "core_north_start", 0))
    cn1 = int(getattr(mesh, "core_north_end", len(mesh.grid_north) - 1))
    ce0 = int(getattr(mesh, "core_east_start", 0))
    ce1 = int(getattr(mesh, "core_east_end", len(mesh.grid_east) - 1))

    if not (0 <= cn0 < cn1 <= len(mesh.grid_north) - 1):
        raise ValueError("Invalid core North index range stored on the mesh.")
    if not (0 <= ce0 < ce1 <= len(mesh.grid_east) - 1):
        raise ValueError("Invalid core East index range stored on the mesh.")

    core_e0 = float(mesh.grid_east[ce0])
    core_e1 = float(mesh.grid_east[ce1])
    core_n0 = float(mesh.grid_north[cn0])
    core_n1 = float(mesh.grid_north[cn1])

    # Restrict the stations used for the profile fit to the core.  The workflow
    # should already place them there; this explicit filter prevents padding
    # stations from extending the plotted section if a future mesh configuration
    # changes.
    station_pts_all = stations[["model_east", "model_north"]].to_numpy(dtype=float)
    tol = 1e-6
    station_core_mask = (
        (station_pts_all[:, 0] >= core_e0 - tol)
        & (station_pts_all[:, 0] <= core_e1 + tol)
        & (station_pts_all[:, 1] >= core_n0 - tol)
        & (station_pts_all[:, 1] <= core_n1 + tol)
    )
    if int(np.count_nonzero(station_core_mask)) < 2:
        raise RuntimeError(
            "Fewer than two centered MT stations lie inside the core mesh. "
            "Increase the core dimensions or check station centering."
        )
    stations_core = stations.loc[station_core_mask].copy()
    station_pts = stations_core[["model_east", "model_north"]].to_numpy(dtype=float)

    center, direction = pca_station_line(stations_core)

    # Intersect the profile line with the exact core rectangle first. This is the
    # key change from the previous version: no padding-cell segment can enter the
    # figure even when the best-fit line extends beyond the stations.
    core_hit = _line_box_intersection(
        center[0], center[1], direction[0], direction[1],
        core_e0, core_e1, core_n0, core_n1,
    )
    if core_hit is None:
        raise RuntimeError("The station-alignment line does not intersect the core mesh.")
    core_t0, core_t1 = core_hit

    # Determine only the core cells intersected by the line.
    e_edges = np.asarray(mesh.grid_east, dtype=float)
    n_edges = np.asarray(mesh.grid_north, dtype=float)
    segments: list[tuple[float, float, int, int]] = []
    for j in range(cn0, cn1):
        n0, n1 = n_edges[j], n_edges[j + 1]
        for i in range(ce0, ce1):
            e0, e1 = e_edges[i], e_edges[i + 1]
            hit = _line_box_intersection(
                center[0], center[1], direction[0], direction[1],
                e0, e1, n0, n1,
            )
            if hit is None:
                continue
            t0 = max(hit[0], core_t0)
            t1 = min(hit[1], core_t1)
            if t1 > t0 + 1e-9:
                segments.append((t0, t1, j, i))

    if not segments:
        raise RuntimeError("The station-alignment line does not intersect any core mesh cells.")
    segments.sort(key=lambda x: x[0])

    # Requested maximum depth is in ModEM z coordinates (+down). Plot only
    # cells from the real mesh and clip the last cell at the requested depth.
    z_edges = np.asarray(mesh.grid_z, dtype=float)
    if max_depth_m >= z_edges[-1]:
        plot_depth = float(z_edges[-1])
        print(
            f"Requested section depth {max_depth_m:.1f} m exceeds the model bottom; "
            f"using model bottom {plot_depth:.1f} m."
        )
    else:
        plot_depth = float(max_depth_m)

    z_segments = []
    for k in range(len(z_edges) - 1):
        z0, z1 = float(z_edges[k]), float(z_edges[k + 1])
        if z0 >= plot_depth:
            break
        z1_clip = min(z1, plot_depth)
        if z1_clip > z0:
            z_segments.append((k, z0, z1_clip))
    if not z_segments:
        raise RuntimeError("No mesh depth cells are available for the requested section depth.")

    from matplotlib.collections import PolyCollection

    polygons = []
    values = []
    for t0, t1, j, i in segments:
        for k, z0, z1 in z_segments:
            polygons.append(
                [
                    (t0 / 1000.0, z0),
                    (t1 / 1000.0, z0),
                    (t1 / 1000.0, z1),
                    (t0 / 1000.0, z1),
                ]
            )
            values.append(float(np.log10(np.clip(mesh.res_model[j, i, k], 1e-6, 1e15))))

    fig, ax = plt.subplots(figsize=(13, 7.5))
    collection = PolyCollection(
        polygons,
        array=np.asarray(values, dtype=float),
        cmap="turbo",
        edgecolors="0.25",
        linewidths=0.22,
    )
    ax.add_collection(collection)

    xmin = min(seg[0] for seg in segments) / 1000.0
    xmax = max(seg[1] for seg in segments) / 1000.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(float(plot_depth), float(z_edges[0]))

    # The topographic surface is stored on the horizontal model grid.  Draw
    # the surface only across the core cells that the section actually crosses.
    topo = mesh.surface_dict.get("topography") if hasattr(mesh, "surface_dict") else None
    if topo is not None:
        topo_x = []
        topo_z = []
        for t0, t1, j, i in segments:
            zsurf = -float(topo[j, i])
            topo_x.extend([t0 / 1000.0, t1 / 1000.0, np.nan])
            topo_z.extend([zsurf, zsurf, np.nan])
        ax.plot(
            topo_x,
            topo_z,
            color="black",
            lw=1.7,
            solid_capstyle="round",
            label="Topographic surface",
            zorder=4,
        )

    # Plot only real EDI stations that fall inside the core.  Their elevation is
    # the final MTpy-v2 model elevation after projection onto topography.
    station_s = (station_pts - center) @ direction / 1000.0
    if "model_elevation" in stations_core.columns:
        station_z = pd.to_numeric(stations_core["model_elevation"], errors="coerce").to_numpy(dtype=float)
    elif "elevation" in stations_core.columns:
        station_z = -pd.to_numeric(stations_core["elevation"], errors="coerce").to_numpy(dtype=float)
    else:
        station_z = np.full(len(stations_core), np.nan, dtype=float)

    visible = (
        np.isfinite(station_s)
        & np.isfinite(station_z)
        & (station_s >= xmin - 1e-9)
        & (station_s <= xmax + 1e-9)
        & (station_z <= plot_depth + 1e-9)
        & (station_z >= float(z_edges[0]) - 1e-9)
    )
    if np.any(visible):
        ax.scatter(
            station_s[visible],
            station_z[visible],
            marker="^",
            s=46,
            facecolor="white",
            edgecolor="black",
            linewidth=0.85,
            zorder=6,
            label="MT stations (projected on topography)",
        )

        # Station labels make it easy to check that only core stations are shown.
        for sx, sz, name in zip(
            station_s[visible],
            station_z[visible],
            stations_core.loc[visible, "station"].astype(str).to_numpy(),
        ):
            ax.annotate(
                name,
                (sx, sz),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                zorder=7,
            )

    ax.axhline(0.0, color="0.35", lw=0.8, ls="--", label="Sea level (z=0)", zorder=3)
    ax.set_xlabel("Distance along core MT-station alignment (km)")
    ax.set_ylabel("Model z / depth (+down, m)")
    ax.set_title(
        "Initial resistivity model — core-only vertical section "
        f"| depth = {plot_depth/1000:.2f} km"
    )
    ax.grid(False)
    cbar = fig.colorbar(collection, ax=ax, pad=0.02)
    cbar.set_label("log10 Resistivity (Ohm-m)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    out_file = out_dir / "vertical_section_mesh.png"
    fig.savefig(out_file, dpi=240)
    plt.close(fig)

    return {
        "file": str(out_file),
        "max_depth_m": plot_depth,
        "n_horizontal_mesh_cells_intersected": len(segments),
        "n_vertical_layers_plotted": len(z_segments),
        "n_core_stations_plotted": int(np.count_nonzero(visible)),
        "n_core_stations_total": int(len(stations_core)),
        "core_east_extent_m": float(core_e1 - core_e0),
        "core_north_extent_m": float(core_n1 - core_n0),
        "core_cell_size_east_m": float(getattr(mesh, "core_cell_size_east", np.nan)),
        "core_cell_size_north_m": float(getattr(mesh, "core_cell_size_north", np.nan)),
        "profile_azimuth_deg": float(
            (np.degrees(np.arctan2(direction[0], direction[1])) + 360.0) % 180.0
        ),
    }

def plot_frequency_coverage(
    coverage: pd.DataFrame,
    frequencies: np.ndarray,
    out_dir: Path,
) -> None:
    if coverage.empty:
        return

    station_names = coverage["station"].astype(str).tolist()
    binary_cols = [f"{f:.8g}_Hz" for f in frequencies]
    arr = coverage[binary_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.24 * len(station_names))))
    # Show only the target grid.  White = unavailable, filled = available.
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", origin="upper", cmap="Greens")
    ax.set_xlabel("Selected frequency (Hz)")
    ax.set_ylabel("Station")
    ax.set_yticks(np.arange(len(station_names)))
    ax.set_yticklabels(station_names, fontsize=7)
    ax.set_xticks(np.arange(len(frequencies)))
    ax.set_xticklabels([f"{f:.3g}" for f in frequencies], rotation=70, ha="right", fontsize=7)
    ax.set_title("Availability of selected frequency grid at each MT station")
    fig.colorbar(im, ax=ax, label="Data available (1=yes, 0=no)")
    fig.tight_layout()
    fig.savefig(out_dir / "frequency_availability_by_station.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(coverage)), coverage["n_available"])
    ax.set_xlabel("Station index")
    ax.set_ylabel("Number of available selected frequencies")
    ax.set_title("Number of selected frequencies available at each station")
    ax.set_ylim(0, max(len(frequencies), coverage["n_available"].max()) * 1.05)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "frequency_count_per_station.png", dpi=220)
    plt.close(fig)

    # Global count plot: how many stations contain each selected frequency.
    station_count_per_freq = arr.sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies, station_count_per_freq, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Number of stations with data")
    ax.set_title("Station coverage of the selected frequency grid")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "station_count_per_frequency.png", dpi=220)
    plt.close(fig)


def write_run_summary(cfg: dict, mtd: MTData, mesh: StructuredGrid3D, frequencies: np.ndarray, out_dir: Path) -> None:
    summary = {
        "mtpy_workflow": "MTpy-v2 EDI -> ModEM",
        "n_stations": int(mtd.n_stations),
        "target_frequency_count": int(len(frequencies)),
        "target_frequency_min_Hz": float(frequencies.min()),
        "target_frequency_max_Hz": float(frequencies.max()),
        "model_shape_north_east_depth": [int(v) for v in mesh.res_model.shape],
        "model_depth_m": float(mesh.grid_z[-1]),
        "horizontal_extent_east_m": float(mesh.grid_east[-1] - mesh.grid_east[0]),
        "horizontal_extent_north_m": float(mesh.grid_north[-1] - mesh.grid_north[0]),
        "grid_center_north_m": float(mesh.grid_center[0]),
        "grid_center_east_m": float(mesh.grid_center[1]),
        "model_coordinate_center_east_m": float(0.5 * (mesh.grid_east[0] + mesh.grid_east[-1])),
        "model_coordinate_center_north_m": float(0.5 * (mesh.grid_north[0] + mesh.grid_north[-1])),
        "topography": bool(cfg["topography"]),
        "n_air_layers": int(cfg.get("n_air_layers", 0)),
        "air_layer_thickness_m": float(cfg.get("air_layer_thickness", 0.0)),
        "topography_source_requested": str(cfg.get("topography_source", "local")),
        "topography_source_resolved": str(cfg.get("resolved_topography_source", "")),
        "topography_source_description": str(cfg.get("resolved_topography_description", "")),
        "auto_topography_source_file": str(cfg.get("auto_topography_source_file", "")),
        "auto_topography_model_grid_file": str(cfg.get("auto_topography_model_grid_file", "")),
        "utm_epsg": int(cfg["utm_epsg"]),
        "initial_resistivity_ohm_m": float(cfg["initial_res"]),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as fid:
        json.dump(summary, fid, indent=2)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run(cfg: dict) -> None:
    edi_dir = Path(cfg["edi_dir"])
    out_dir = Path(cfg["out_dir"])
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    edi_files = find_edi_files(edi_dir)
    print(f"\nFound {len(edi_files)} EDI files.")

    # 1) Load EDI -> MTData, project station positions to UTM and compute
    #    model-relative coordinates.
    mtd = read_edi_into_mtdata(edi_files, cfg["utm_epsg"])
    print(f"Loaded {mtd.n_stations} stations into MTData.")

    # 2) Create exact user-controlled custom structured mesh.
    mesh = build_custom_mesh(mtd, cfg)
    print("\nInitial custom mesh created:")
    print(f"  cells (North, East, Depth): {mesh.res_model.shape}")
    print(f"  horizontal extent North: {mesh.grid_north[-1] - mesh.grid_north[0]:.1f} m")
    print(f"  horizontal extent East : {mesh.grid_east[-1] - mesh.grid_east[0]:.1f} m")
    print(f"  earth depth: {mesh.grid_z[-1]:.1f} m")

    # 3) Apply topography. In automatic mode the workflow derives the model
    #    geographic footprint from the UTM mesh, downloads only the necessary
    #    SRTM/ETOPO tiles, reprojects the DEM directly onto model-cell centres,
    #    and passes the resulting array to MTpy-v2. Local GeoTIFF remains an
    #    option for reproducibility or when a higher-resolution site DEM exists.
    if cfg["topography"]:
        topo_source_raw = str(cfg.get("topography_source", "local"))
        topo_source = topo_source_raw.strip().lower()

        # Normalize GUI labels to the internal source names used by
        # auto_topography.prepare_topography().
        if topo_source in {
            "auto",
            "automatic",
            "automatic (srtm → etopo)",
            "automatic (srtm -> etopo)",
        }:
            dem_source = "auto"
        elif topo_source in {
            "srtm 30 m",
            "srtm 1 arc-sec",
            "srtm1",
            "srtm",
        }:
            dem_source = "srtm1"
        elif topo_source in {
            "etopo 2022 15 arc-sec",
            "etopo15",
            "etopo",
            "etopo 2022",
        }:
            dem_source = "etopo15"
        elif topo_source in {"local geotiff", "local", "geotiff"}:
            dem_source = "local"
        else:
            raise ValueError(
                f"Unknown topography source: {topo_source_raw}. "
                "Use Automatic (SRTM → ETOPO), SRTM 30 m, ETOPO 2022 15 arc-sec, or Local GeoTIFF."
            )

        if dem_source == "local":
            print("\nPreparing local GeoTIFF topography on the exact MTpy-v2 model grid ...")
            topo_info = prepare_local_topography(
                mesh,
                cfg["utm_epsg"],
                out_dir,
                Path(cfg["topography_file"]),
                interpolation=cfg["topography_interp"],
                log=print,
            )
            topo_air_info = apply_fixed_air_topography(
                mesh,
                topo_info["topography_array"],
                cfg.get("n_air_layers", 20),
                cfg.get("air_layer_thickness", 100.0),
                air_resistivity=1e12,
            )
            print(
                f"  Air mesh: {topo_air_info['n_air_layers']} layers × "
                f"{topo_air_info['air_layer_thickness_m']:.3f} m "
                f"(total {topo_air_info['total_air_thickness_m']:.3f} m)"
            )
            print(
                f"  Topographic relief used for air coverage: "
                f"{topo_air_info['topographic_relief_m']:.1f} m"
            )
            print(
                f"  Maximum air-cell thickness deviation: "
                f"{topo_air_info['max_air_cell_thickness_deviation_m']:.3e} m"
            )
            cfg["resolved_topography_source"] = "local_geotiff"
            cfg["resolved_topography_description"] = topo_info["description"]
            cfg["auto_topography_source_file"] = topo_info["source_geotiff"]
            cfg["auto_topography_model_grid_file"] = topo_info["model_grid_geotiff"]
        else:
            print(f"\nPreparing {dem_source} topography ...")
            topo_info = prepare_topography(
                mesh,
                cfg["utm_epsg"],
                out_dir,
                source=dem_source,
                interpolation=cfg["topography_interp"],
                log=print,
            )
            topo_air_info = apply_fixed_air_topography(
                mesh,
                topo_info["topography_array"],
                cfg.get("n_air_layers", 20),
                cfg.get("air_layer_thickness", 100.0),
                air_resistivity=1e12,
            )
            print(
                f"  Air mesh: {topo_air_info['n_air_layers']} layers × "
                f"{topo_air_info['air_layer_thickness_m']:.3f} m "
                f"(total {topo_air_info['total_air_thickness_m']:.3f} m)"
            )
            print(
                f"  Topographic relief used for air coverage: "
                f"{topo_air_info['topographic_relief_m']:.1f} m"
            )
            print(
                f"  Maximum air-cell thickness deviation: "
                f"{topo_air_info['max_air_cell_thickness_deviation_m']:.3e} m"
            )
            cfg["resolved_topography_source"] = topo_info["source"]
            cfg["resolved_topography_description"] = topo_info["description"]
            cfg["auto_topography_source_file"] = topo_info["source_geotiff"]
            cfg["auto_topography_model_grid_file"] = topo_info["model_grid_geotiff"]

    # 4) Put stations exactly at cell centres, following the MTpy-v2 / SAGE
    #    ModEM workflow. This MUST happen after the final horizontal mesh is
    #    available and before exporting the data file. For topography the
    #    stations are then projected vertically onto the corresponding surface.
    before_center = mtd.station_locations[["station", "model_east", "model_north"]].copy()
    before_center = before_center.rename(columns={"model_east": "model_east_before", "model_north": "model_north_before"})
    mtd.center_stations(mesh)

    if cfg["topography"]:
        mtd.project_stations_on_topography(mesh)

    mesh.station_locations = mtd.station_locations

    # Verify that every station is exactly at the centre of its assigned cell.
    s_after = mtd.station_locations.copy()
    e_idx = np.searchsorted(
        mesh.grid_east, s_after["model_east"].to_numpy(float), side="right"
    ) - 1
    n_idx = np.searchsorted(
        mesh.grid_north, s_after["model_north"].to_numpy(float), side="right"
    ) - 1
    e_idx = np.clip(e_idx, 0, mesh.grid_east.size - 2)
    n_idx = np.clip(n_idx, 0, mesh.grid_north.size - 2)
    expected_e = 0.5 * (mesh.grid_east[e_idx] + mesh.grid_east[e_idx + 1])
    expected_n = 0.5 * (mesh.grid_north[n_idx] + mesh.grid_north[n_idx + 1])
    center_error = np.sqrt(
        (s_after["model_east"].to_numpy(float) - expected_e) ** 2
        + (s_after["model_north"].to_numpy(float) - expected_n) ** 2
    )

    print("\nStation centering (MTpy-v2 center_stations):")
    print(f"  Model centre (E, N): (0.000, 0.000) m")
    print(f"  Station E range after centering: {s_after.model_east.min():.3f} to {s_after.model_east.max():.3f} m")
    print(f"  Station N range after centering: {s_after.model_north.min():.3f} to {s_after.model_north.max():.3f} m")
    print(f"  Maximum horizontal centre error: {center_error.max():.6g} m")
    if center_error.max() > 1e-6:
        raise RuntimeError(
            "Station-centering verification failed: at least one station is not "
            "at the centre of its assigned model cell."
        )

    if cfg["topography"]:
        save_topography_station_diagnostics(
            mesh,
            topo_info["topography_array"],
            s_after,
            before_center,
            out_dir / "figures",
            cfg["utm_epsg"],
        )

    # Save the pre/post coordinates to make centring auditable.
    after_center = s_after[["station", "model_east", "model_north", "model_elevation"]].copy()
    after_center = after_center.rename(
        columns={"model_east": "model_east_after", "model_north": "model_north_after"}
    )
    centering_report = before_center.merge(after_center, on="station", how="outer")
    centering_report["horizontal_shift_m"] = np.sqrt(
        (centering_report["model_east_after"] - centering_report["model_east_before"]) ** 2
        + (centering_report["model_north_after"] - centering_report["model_north_before"]) ** 2
    )
    centering_report.to_csv(out_dir / "station_centering_report.csv", index=False)

    # 5) Frequency selection and interpolation, no extrapolation.
    frequencies = build_frequency_grid(cfg)
    print("\nTarget frequencies [Hz]:")
    print("  " + ", ".join(f"{f:.6g}" for f in frequencies))
    interpolate_data(mtd, frequencies, cfg)

    # If the user requested tipper but essentially no tipper exists, fall back
    # to full impedance mode so the produced file remains useful.
    df_check = mtd.to_dataframe(impedance_units="ohm")
    has_tipper = False
    if not df_check.empty and {"t_zx", "t_zy"}.issubset(df_check.columns):
        tvals = pd.to_numeric(df_check[["t_zx", "t_zy"]].stack(), errors="coerce")
        has_tipper = bool(np.isfinite(tvals.to_numpy()).any() and np.any(np.abs(tvals.fillna(0).to_numpy()) > 0))
    if cfg["inv_mode"] == "1" and not has_tipper:
        print("No usable tipper values detected; switching ModEM inversion mode from 1 to 2 (full impedance).")
        cfg["inv_mode"] = "2"

    # 6) Write ModEM files.
    files = write_modem_files(mtd, mesh, cfg)
    print("\nCreated ModEM files:")
    for kind, fn in files.items():
        print(f"  {kind:10s}: {fn}")

    # 7) Reporting and the single requested model plot.
    stations = save_station_table(mtd, out_dir)
    coverage = save_frequency_reports(mtd, frequencies, out_dir)
    section_info = plot_vertical_mesh_section(
        mesh,
        stations,
        fig_dir,
        cfg["section_depth_m"],
    )
    cfg["vertical_section_info"] = section_info
    print("\nCreated model figure:")
    print(f"  Vertical section: {section_info['file']}")
    print(f"  Intersected horizontal mesh cells: {section_info['n_horizontal_mesh_cells_intersected']}")
    print(f"  Vertical layers plotted: {section_info['n_vertical_layers_plotted']}")
    write_run_summary(cfg, mtd, mesh, frequencies, out_dir)

    # Save config exactly as used, including the possible automatic inversion
    # mode fallback.
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as fid:
        json.dump(cfg, fid, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("Completed.")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Figures directory: {fig_dir.resolve()}")
    print("=" * 78)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EDI MT data into ModEM input files using MTpy-v2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON configuration file. If omitted, an interactive wizard is used.",
    )
    args = parser.parse_args()

    cfg = load_config_file(args.config) if args.config else collect_inputs()
    run(cfg)


if __name__ == "__main__":
    main()
