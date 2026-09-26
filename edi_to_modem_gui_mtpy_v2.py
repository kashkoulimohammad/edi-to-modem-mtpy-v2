#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI for EDI -> ModEM preparation using the existing MTpy-v2 workflow.

This GUI is a front-end for edi_to_modem_mtpy_v2.py.  Keep both files in the
same directory.  The scientific workflow itself is imported from the existing
script so that the GUI does not duplicate or silently alter the conversion
logic.

Run:
    python edi_to_modem_gui_mtpy_v2.py

Requirements:
    tkinter (usually included with Python)
    MTpy-v2 dependencies required by edi_to_modem_mtpy_v2.py
"""

from __future__ import annotations

import contextlib
import io
import json
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "EDI → ModEM | MTpy-v2 GUI"

# The workflow file is expected to sit beside this GUI file.
WORKFLOW_FILE = Path(__file__).with_name("edi_to_modem_mtpy_v2.py")


class QueueWriter(io.TextIOBase):
    """Redirect stdout/stderr into the Tkinter log queue."""

    def __init__(self, q: queue.Queue[str]):
        super().__init__()
        self.q = q

    def write(self, text: str) -> int:
        if text:
            self.q.put(text)
        return len(text)

    def flush(self) -> None:
        return None


class Application(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1220x900")
        self.minsize(1050, 760)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.running = False

        self.vars: dict[str, tk.Variable] = {}
        self._build_variables()
        self._build_style()
        self._build_ui()
        self._refresh_topography_state()
        self._poll_log_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Variables / defaults
    # ------------------------------------------------------------------

    def _build_variables(self) -> None:
        self.vars.update({
            "edi_dir": tk.StringVar(),
            "out_dir": tk.StringVar(),
            "utm_epsg": tk.StringVar(value="32639"),

            "n_freq": tk.StringVar(value="24"),
            "f_min": tk.StringVar(value="0.001"),
            "f_max": tk.StringVar(value="1000.0"),

            # IMPORTANT: keep the naming aligned with the workflow script.
            "nx": tk.StringVar(value="40"),
            "ny": tk.StringVar(value="40"),
            "dx": tk.StringVar(value="500.0"),
            "dy": tk.StringVar(value="500.0"),

            "pad_x": tk.StringVar(value="8"),
            "pad_y": tk.StringVar(value="8"),
            "pad_factor": tk.StringVar(value="1.3"),

            "nz": tk.StringVar(value="35"),
            "dz1": tk.StringVar(value="30.0"),
            "dz_factor": tk.StringVar(value="1.15"),

            "topography": tk.BooleanVar(value=True),
            "topography_source": tk.StringVar(value="Automatic (SRTM → ETOPO)"),
            "topography_file": tk.StringVar(),
            "n_air_layers": tk.StringVar(value="20"),
            "air_layer_thickness": tk.StringVar(value="100.0"),
            "topography_interp": tk.StringVar(value="linear"),

            "z_error_pct": tk.StringVar(value="5.0"),
            "tipper_error": tk.StringVar(value="0.02"),
            "inv_mode": tk.StringVar(value="1"),

            "smooth_e": tk.StringVar(value="0.3"),
            "smooth_n": tk.StringVar(value="0.3"),
            "smooth_z": tk.StringVar(value="0.3"),
            "smooth_num": tk.StringVar(value="1"),

            "section_depth_m": tk.StringVar(value="10000.0"),
            "initial_res": tk.StringVar(value="100.0"),

            "status": tk.StringVar(value="Ready"),
        })

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="EDI → ModEM", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="MTpy-v2 structured 3-D model preparation",
        ).pack(side="left", padx=(15, 0), pady=(5, 0))
        ttk.Label(header, textvariable=self.vars["status"]).pack(side="right", pady=(5, 0))

        paned = ttk.Panedwindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)

        top = ttk.Frame(paned)
        bottom = ttk.Frame(paned)
        paned.add(top, weight=4)
        paned.add(bottom, weight=2)

        # Scrollable main parameter area.
        canvas = tk.Canvas(top, highlightthickness=0)
        scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = ttk.Frame(canvas, padding=(4, 4, 12, 8))
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        content.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))

        self._build_general(content)
        self._build_frequency(content)
        self._build_mesh(content)
        self._build_topography(content)
        self._build_modem(content)
        self._build_covariance(content)
        self._build_plots(content)
        self._build_actions(content)

        # Log panel.
        log_frame = ttk.LabelFrame(bottom, text="Processing log", padding=7, style="Section.TLabelframe")
        log_frame.pack(fill="both", expand=True)

        log = tk.Text(log_frame, wrap="none", height=12, font=("TkFixedFont", 9))
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
        xscroll = ttk.Scrollbar(log_frame, orient="horizontal", command=log.xview)
        log.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        log.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = log

    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=10, style="Section.TLabelframe")
        frame.pack(fill="x", pady=(0, 8))
        return frame

    def _entry_row(self, frame: ttk.LabelFrame, row: int, label: str, varname: str,
                   browse: str | None = None, width: int = 22) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(frame, textvariable=self.vars[varname], width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if browse == "dir":
            ttk.Button(frame, text="Browse…", command=lambda v=varname: self._browse_dir(v)).grid(
                row=row, column=2, padx=(7, 0), pady=4
            )
        elif browse == "file":
            ttk.Button(frame, text="Browse…", command=lambda v=varname: self._browse_file(v)).grid(
                row=row, column=2, padx=(7, 0), pady=4
            )

    def _build_general(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "1. Input / Output")
        f.columnconfigure(1, weight=1)
        self._entry_row(f, 0, "EDI folder", "edi_dir", "dir", 70)
        self._entry_row(f, 1, "Output folder", "out_dir", "dir", 70)
        self._entry_row(f, 2, "WGS84 UTM EPSG", "utm_epsg")

        ttk.Button(f, text="Load JSON", command=self._load_json).grid(row=0, column=3, padx=(25, 4), pady=4)
        ttk.Button(f, text="Save JSON", command=self._save_json).grid(row=1, column=3, padx=(25, 4), pady=4)
        ttk.Label(
            f,
            text="Both workflow scripts should be in the same directory.",
            foreground="#555555",
        ).grid(row=2, column=3, sticky="w", padx=(25, 4), pady=4)

    def _build_frequency(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "2. Frequency grid")
        f.columnconfigure(1, weight=1)
        self._entry_row(f, 0, "Number of target frequencies", "n_freq")
        self._entry_row(f, 1, "Minimum frequency [Hz]", "f_min")
        self._entry_row(f, 2, "Maximum frequency [Hz]", "f_max")
        ttk.Label(f, text="The target grid is logarithmically spaced; MTpy-v2 interpolation does not extrapolate.",
                  foreground="#555555").grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

    def _build_mesh(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "3. 3-D model mesh")
        f.columnconfigure(1, weight=1)
        f.columnconfigure(4, weight=1)

        ttk.Label(f, text="Core cells East (Y)").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.vars["nx"], width=14).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(f, text="Core cell size East [m]").grid(row=0, column=3, sticky="w", padx=(30, 8), pady=4)
        ttk.Entry(f, textvariable=self.vars["dx"], width=14).grid(row=0, column=4, sticky="w", pady=4)

        ttk.Label(f, text="Core cells North (X)").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.vars["ny"], width=14).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(f, text="Core cell size North [m]").grid(row=1, column=3, sticky="w", padx=(30, 8), pady=4)
        ttk.Entry(f, textvariable=self.vars["dy"], width=14).grid(row=1, column=4, sticky="w", pady=4)

        ttk.Separator(f).grid(row=2, column=0, columnspan=5, sticky="ew", pady=7)

        ttk.Label(f, text="N/S padding cells per side").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.vars["pad_x"], width=14).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(f, text="E/W padding cells per side").grid(row=3, column=3, sticky="w", padx=(30, 8), pady=4)
        ttk.Entry(f, textvariable=self.vars["pad_y"], width=14).grid(row=3, column=4, sticky="w", pady=4)

        ttk.Label(f, text="Horizontal padding growth factor").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.vars["pad_factor"], width=14).grid(row=4, column=1, sticky="w", pady=4)

        ttk.Separator(f).grid(row=5, column=0, columnspan=5, sticky="ew", pady=7)

        ttk.Label(f, text="Earth cells in depth").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.vars["nz"], width=14).grid(row=6, column=1, sticky="w", pady=4)
        ttk.Label(f, text="First Earth layer thickness [m]").grid(row=6, column=3, sticky="w", padx=(30, 8), pady=4)
        ttk.Entry(f, textvariable=self.vars["dz1"], width=14).grid(row=6, column=4, sticky="w", pady=4)

        ttk.Label(f, text="Vertical growth factor").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.vars["dz_factor"], width=14).grid(row=7, column=1, sticky="w", pady=4)

    def _build_topography(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "4. Topography")
        f.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            f,
            text="Apply topography",
            variable=self.vars["topography"],
            command=self._refresh_topography_state,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 7))

        ttk.Label(f, text="Topography source").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self._topo_source_combo = ttk.Combobox(
            f,
            textvariable=self.vars["topography_source"],
            values=[
                "Automatic (SRTM → ETOPO)",
                "SRTM 30 m",
                "ETOPO 2022 15 arc-sec",
                "Local GeoTIFF",
            ],
            state="readonly",
            width=30,
        )
        self._topo_source_combo.grid(row=1, column=1, sticky="w", pady=4)
        self._topo_source_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_topography_state())
        ttk.Label(
            f,
            text="Automatic = use the model extent/coordinates to select SRTM; ETOPO is the fallback.",
            foreground="#555555",
        ).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=4)

        ttk.Label(f, text="Local GeoTIFF DEM").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self._topo_file_entry = ttk.Entry(f, textvariable=self.vars["topography_file"], width=70)
        self._topo_file_entry.grid(row=2, column=1, sticky="ew", pady=4)
        self._topo_file_button = ttk.Button(f, text="Browse…", command=lambda: self._browse_file("topography_file"))
        self._topo_file_button.grid(row=2, column=2, padx=(7, 0), pady=4)

        ttk.Label(f, text="Number of air layers").grid(row=3, column=0, sticky="w", pady=4)
        self._air_entry = ttk.Entry(f, textvariable=self.vars["n_air_layers"], width=18)
        self._air_entry.grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(f, text="Air layer thickness [m]").grid(row=3, column=3, sticky="w", padx=(30, 8), pady=4)
        self._air_thickness_entry = ttk.Entry(f, textvariable=self.vars["air_layer_thickness"], width=18)
        self._air_thickness_entry.grid(row=3, column=4, sticky="w", pady=4)

        ttk.Label(
            f,
            text="Used above the Earth mesh when topography is enabled; all air cells have this same thickness.",
            foreground="#555555",
        ).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=4)

        ttk.Label(f, text="DEM/model-grid interpolation").grid(row=4, column=0, sticky="w", pady=4)
        self._topo_combo = ttk.Combobox(
            f,
            textvariable=self.vars["topography_interp"],
            values=["nearest", "linear", "cubic"],
            state="readonly",
            width=17,
        )
        self._topo_combo.grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(
            f,
            text="For automatic DEMs the program downloads only the tiles covering the model + a small automatic margin, then resamples directly to model-cell centres.",
            foreground="#555555",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_modem(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "5. ModEM data / error settings")
        f.columnconfigure(1, weight=1)
        self._entry_row(f, 0, "Impedance model-error floor [%]", "z_error_pct")
        self._entry_row(f, 1, "Tipper absolute error floor", "tipper_error")

        ttk.Label(f, text="ModEM inversion mode").grid(row=2, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(
            f,
            textvariable=self.vars["inv_mode"],
            values=["1", "2"],
            state="readonly",
            width=20,
        )
        combo.grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(f, text="1 = impedance + vertical components; 2 = impedance only",
                  foreground="#555555").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=4)

    def _build_covariance(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "6. Covariance")
        f.columnconfigure(1, weight=1)
        self._entry_row(f, 0, "East smoothing", "smooth_e")
        self._entry_row(f, 1, "North smoothing", "smooth_n")
        self._entry_row(f, 2, "Vertical smoothing", "smooth_z")
        self._entry_row(f, 3, "Smoothing iterations", "smooth_num")

    def _build_plots(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "7. Model plot")
        f.columnconfigure(1, weight=1)
        self._entry_row(f, 0, "Maximum vertical-section depth below sea level [m]", "section_depth_m")
        self._entry_row(f, 1, "Initial half-space resistivity [Ohm-m]", "initial_res")
        ttk.Label(
            f,
            text="One vertical section is generated along the best-fit MT-station alignment. The section uses the actual mesh cells and cell boundaries; no regular-grid interpolation is used.",
            foreground="#555555",
            wraplength=900,
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(
            f,
            text="Depth is ModEM z-coordinate (+down), referenced to sea level (z = 0).",
            foreground="#555555",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(3, 0))

    def _build_actions(self, parent: ttk.Frame) -> None:
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(2, 10))

        self.run_button = ttk.Button(f, text="▶  Generate ModEM Files", style="Run.TButton", command=self._start_run)
        self.run_button.pack(side="left", padx=(0, 8))

        ttk.Button(f, text="Clear log", command=self._clear_log).pack(side="left", padx=4)
        ttk.Button(f, text="Open output folder", command=self._open_output).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(f, mode="indeterminate", length=230)
        self.progress.pack(side="right", padx=(10, 0))

    # ------------------------------------------------------------------
    # Browse / state
    # ------------------------------------------------------------------

    def _browse_dir(self, varname: str) -> None:
        initial = self.vars[varname].get() or str(Path.home())
        path = filedialog.askdirectory(initialdir=initial, title="Select folder")
        if path:
            self.vars[varname].set(path)
            if varname == "edi_dir" and not self.vars["out_dir"].get().strip():
                self.vars["out_dir"].set(str(Path(path) / "modem_input"))

    def _browse_file(self, varname: str) -> None:
        initial = str(Path(self.vars[varname].get()).parent) if self.vars[varname].get().strip() else str(Path.home())
        if varname == "topography_file":
            path = filedialog.askopenfilename(
                initialdir=initial,
                title="Select GeoTIFF topography",
                filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(initialdir=initial, title="Select file")
        if path:
            self.vars[varname].set(path)

    def _refresh_topography_state(self) -> None:
        enabled = bool(self.vars["topography"].get())
        source = self.vars["topography_source"].get().strip().lower()
        local = enabled and source == "local geotiff"
        self._topo_source_combo.configure(state="readonly" if enabled else "disabled")
        self._topo_file_entry.configure(state="normal" if local else "disabled")
        self._topo_file_button.configure(state="normal" if local else "disabled")
        self._air_entry.configure(state="normal" if enabled else "disabled")
        self._air_thickness_entry.configure(state="normal" if enabled else "disabled")
        self._topo_combo.configure(state="readonly" if enabled else "disabled")

    # ------------------------------------------------------------------
    # Config conversion
    # ------------------------------------------------------------------

    def _config_from_ui(self) -> dict:
        """Convert GUI fields to exactly the dictionary expected by run()."""
        def integer(name: str, label: str, minimum: int | None = None) -> int:
            try:
                value = int(self.vars[name].get().strip())
            except Exception as exc:
                raise ValueError(f"{label}: enter an integer.") from exc
            if minimum is not None and value < minimum:
                raise ValueError(f"{label}: value must be >= {minimum}.")
            return value

        def number(name: str, label: str, minimum: float | None = None) -> float:
            try:
                value = float(self.vars[name].get().strip())
            except Exception as exc:
                raise ValueError(f"{label}: enter a number.") from exc
            if minimum is not None and value < minimum:
                raise ValueError(f"{label}: value must be >= {minimum}.")
            return value

        edi_dir = Path(self.vars["edi_dir"].get().strip()).expanduser()
        out_dir = Path(self.vars["out_dir"].get().strip()).expanduser()
        if not edi_dir.is_dir():
            raise ValueError(f"EDI folder does not exist: {edi_dir}")
        if not out_dir:
            raise ValueError("Output folder is required.")

        utm_epsg = integer("utm_epsg", "WGS84 UTM EPSG", 1000)
        n_freq = integer("n_freq", "Number of target frequencies", 2)
        f_min = number("f_min", "Minimum frequency", 1e-12)
        f_max = number("f_max", "Maximum frequency", 1e-12)
        if f_max <= f_min:
            raise ValueError("Maximum frequency must be larger than minimum frequency.")

        nx = integer("nx", "Core cells East", 1)
        ny = integer("ny", "Core cells North", 1)
        dx = number("dx", "Core cell size East", 1e-6)
        dy = number("dy", "Core cell size North", 1e-6)
        pad_x = integer("pad_x", "N/S padding cells", 0)
        pad_y = integer("pad_y", "E/W padding cells", 0)
        pad_factor = number("pad_factor", "Horizontal padding growth factor", 1.0)

        nz = integer("nz", "Earth cells in depth", 2)
        dz1 = number("dz1", "First Earth layer thickness", 1e-6)
        dz_factor = number("dz_factor", "Vertical growth factor", 1.0)

        topography = bool(self.vars["topography"].get())
        topo_source = self.vars["topography_source"].get().strip()
        topo_file = self.vars["topography_file"].get().strip()
        n_air_layers = integer("n_air_layers", "Number of air layers", 1)
        air_layer_thickness = number("air_layer_thickness", "Air layer thickness", 1e-6)
        if topography and topo_source.lower() == "local geotiff":
            if not topo_file:
                raise ValueError("Local GeoTIFF mode is selected, but no GeoTIFF was selected.")
            if not Path(topo_file).expanduser().is_file():
                raise ValueError(f"Topography GeoTIFF not found: {topo_file}")

        z_error_pct = number("z_error_pct", "Impedance error floor", 0.01)
        tipper_error = number("tipper_error", "Tipper error floor", 0.0)
        inv_mode = self.vars["inv_mode"].get().strip()
        if inv_mode not in {"1", "2"}:
            raise ValueError("ModEM inversion mode must be 1 or 2.")

        smooth_e = number("smooth_e", "Covariance east smoothing", 0.0)
        smooth_n = number("smooth_n", "Covariance north smoothing", 0.0)
        smooth_z = number("smooth_z", "Covariance vertical smoothing", 0.0)
        smooth_num = integer("smooth_num", "Covariance smoothing iterations", 0)

        section_depth_m = number(
            "section_depth_m",
            "Maximum vertical-section depth below sea level",
            1.0,
        )
        initial_res = number("initial_res", "Initial half-space resistivity", 1e-6)

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
            "topography": topography,
            "topography_source": topo_source,
            "topography_file": str(Path(topo_file).expanduser()) if topo_file else None,
            "n_air_layers": n_air_layers,
            "air_layer_thickness": air_layer_thickness,
            "topography_interp": self.vars["topography_interp"].get(),
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

    def _apply_config(self, cfg: dict) -> None:
        for key, value in cfg.items():
            if key not in self.vars:
                continue
            self.vars[key].set(value)
        self._refresh_topography_state()

    def _load_json(self) -> None:
        filename = filedialog.askopenfilename(
            title="Load JSON configuration",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            with Path(filename).open("r", encoding="utf-8") as fid:
                cfg = json.load(fid)
            self._apply_config(cfg)
            self._append_log(f"Loaded configuration: {filename}\n")
            self.vars["status"].set("Configuration loaded")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _save_json(self) -> None:
        try:
            cfg = self._config_from_ui()
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))
            return

        filename = filedialog.asksaveasfilename(
            title="Save JSON configuration",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="modem_config.json",
        )
        if not filename:
            return
        try:
            with Path(filename).open("w", encoding="utf-8") as fid:
                json.dump(cfg, fid, indent=2, ensure_ascii=False)
            self._append_log(f"Saved configuration: {filename}\n")
            self.vars["status"].set("Configuration saved")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        if self.running:
            return

        if not WORKFLOW_FILE.is_file():
            messagebox.showerror(
                "Workflow file not found",
                f"The following file is required beside the GUI:\n\n{WORKFLOW_FILE}",
            )
            return

        try:
            cfg = self._config_from_ui()
        except Exception as exc:
            messagebox.showerror("Input validation", str(exc))
            return

        self._clear_log()
        self._append_log("Starting EDI → ModEM workflow...\n")
        self._append_log(f"Workflow: {WORKFLOW_FILE}\n\n")

        self.running = True
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.vars["status"].set("Running…")

        self.worker = threading.Thread(target=self._worker_run, args=(cfg,), daemon=True)
        self.worker.start()

    def _worker_run(self, cfg: dict) -> None:
        try:
            # Import lazily so the GUI itself can open even when MTpy is not
            # installed. The error will then be shown inside the log.
            import importlib.util

            spec = importlib.util.spec_from_file_location("edi_to_modem_workflow", WORKFLOW_FILE)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load workflow module: {WORKFLOW_FILE}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            writer = QueueWriter(self.log_queue)
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                module.run(cfg)

            self.log_queue.put("\nGUI_STATUS_SUCCESS\n")
        except Exception:
            self.log_queue.put("\nGUI_STATUS_ERROR\n")
            self.log_queue.put(traceback.format_exc())

    def _poll_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "\nGUI_STATUS_SUCCESS\n":
                    self._finish_run(success=True)
                elif item == "\nGUI_STATUS_ERROR\n":
                    self._finish_run(success=False)
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _finish_run(self, success: bool) -> None:
        self.running = False
        self.run_button.configure(state="normal")
        self.progress.stop()
        self.vars["status"].set("Completed" if success else "Failed")

        if success:
            out = self.vars["out_dir"].get().strip()
            self._append_log("\nProcess completed successfully.\n")
            messagebox.showinfo(
                "Completed",
                f"ModEM files and figures were generated successfully.\n\nOutput:\n{out}",
            )
        else:
            self._append_log("\nProcess failed. See the traceback above.\n")
            messagebox.showerror(
                "Processing error",
                "The workflow stopped with an error. See the processing log for the traceback.",
            )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.update_idletasks()

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _open_output(self) -> None:
        path = Path(self.vars["out_dir"].get().strip()).expanduser()
        if not path.exists():
            messagebox.showwarning("Output folder", "The output folder does not exist yet.")
            return

        import os
        import platform
        import subprocess

        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open folder", str(exc))

    def _on_close(self) -> None:
        if self.running:
            answer = messagebox.askyesno(
                "Processing in progress",
                "A conversion is currently running. Close the GUI anyway?",
            )
            if not answer:
                return
        self.destroy()


def main() -> None:
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
