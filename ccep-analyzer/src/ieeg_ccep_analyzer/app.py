# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import textwrap
import time
import base64
import json
import math
import webbrowser
from collections import Counter
from functools import partial
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import stats
from scipy.signal import hilbert
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import mne

# Reduce MNE console chatter (fsaverage fetch/status, filter logs, etc.)
mne.set_log_level('WARNING')
from mne.surface import decimate_surface

# Optional interactive exports (Plotly). If missing, interactive export is skipped gracefully.
try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:  # pragma: no cover
    go = None
    pio = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("ieeg_analysis.log"), logging.StreamHandler()],
)
LOG = logging.getLogger("ieeg")


# -----------------------------------------------------------------------------
# Publication-oriented Matplotlib configuration
# -----------------------------------------------------------------------------
def _base_publication_rc(scale: float = 1.0) -> Dict[str, Any]:
    s = 10 * scale
    return {
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": s,
        "axes.titlesize": 1.2 * s,
        "axes.labelsize": s,
        "xtick.labelsize": 0.9 * s,
        "ytick.labelsize": 0.9 * s,
        "legend.fontsize": 0.9 * s,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.8,
    }


STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Publication": _base_publication_rc(1.0),
    "Classic": {**_base_publication_rc(1.0), "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"},
    "Dark": {
        **_base_publication_rc(1.0),
        "axes.facecolor": "#212121",
        "figure.facecolor": "#121212",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "grid.color": "#424242",
        "axes.edgecolor": "#e0e0e0",
    },
}


# -----------------------------------------------------------------------------
# UI/Export data classes
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class ExportOptions:
    formats: Tuple[str, ...] = ("png", "pdf", "svg")
    dpi: int = 600
    transparent: bool = False
    interactive_html: bool = True
    html_inline_js: bool = True  # True => fully offline (plotly.js embedded)

    def ensure_valid(self) -> None:
        valid = {"png", "pdf", "svg"}
        fmts = tuple(f.lower() for f in self.formats if f and f.lower() in valid)
        self.formats = fmts or ("png",)


@dataclass(slots=True)
class ThemeSettings:
    preset: str = "Publication"
    colormap: str = "plasma"
    erp_line_width: float = 1.6
    ci_alpha: float = 0.2
    font_scale: float = 1.0
    show_zero_time: bool = True

    def apply(self) -> None:
        rc = STYLE_PRESETS.get(self.preset, STYLE_PRESETS["Publication"]).copy()
        scale = max(0.7, min(1.8, float(self.font_scale)))
        rc.update(_base_publication_rc(scale))
        mpl.rcParams.update(rc)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def evoked_to_dataframe(evoked: Any) -> pd.DataFrame:
    times_ms = evoked.times * 1000.0
    df = pd.DataFrame((evoked.data * 1e6).T, columns=evoked.ch_names)
    df.insert(0, "time_ms", times_ms)
    return df


def _ignore_event(func: Callable[[], Any], _event: Any = None) -> Any:
    """Adapter: call a no-arg callback from an event-based binding."""
    return func()


def write_gallery_html(output_html: Path, title: str, images: Dict[str, List[Path]]) -> None:
    """Write a lightweight local HTML gallery for exported images and interactive HTML."""

    def rel_path(p: Path) -> str:
        return os.path.relpath(p, output_html.parent)

    css = """
    :root {
      color-scheme: light dark;
      --bg: #ffffff;
      --fg: #111111;
      --muted: #666666;
      --card: #fafafa;
      --border: #e5e5e5;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f1115;
        --fg: #e6e6e6;
        --muted: #a0a0a0;
        --card: #171a21;
        --border: #2a2e39;
      }
    }
    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--fg);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji",
            "Segoe UI Symbol";
    }
    main { max-width: 1200px; margin: 18px auto; padding: 0 12px; }
    h1 { margin: 4px 0 2px; }
    p.meta { color: var(--muted); margin-top: 0; }
    .section { margin-top: 20px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
    }
    figure {
      margin: 0;
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 8px;
      overflow: hidden;
    }
    figure img, figure iframe {
      width: 100%;
      height: auto;
      display: block;
      border: 0;
    }
    figure iframe { height: 420px; }
    figcaption { padding: 8px 10px; font-size: 12px; color: var(--muted); }
    a.inline { color: inherit; text-decoration: underline; }
    """

    lines: List[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        f"<style>{css}</style>",
        "</head><body><main>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>Generated on {datetime.now().isoformat(timespec='seconds')}</p>",
    ]

    for section, paths in images.items():
        if not paths:
            continue
        lines.append(f"<div class='section'><h2>{section}</h2><div class='grid'>")
        for p in paths:
            src = rel_path(p)
            if p.suffix.lower() in (".html", ".pdf"):
                lines.append(
                    "<figure>"
                    f"<iframe src='{src}' loading='lazy'></iframe>"
                    f"<figcaption>{p.name} — "
                    f"<a class='inline' href='{src}' target='_blank' rel='noopener'>open</a>"
                    f"</figcaption></figure>"
                )
            else:
                lines.append(
                    "<figure>"
                    f"<img src='{src}' alt='{p.name}'>"
                    f"<figcaption>{p.name}</figcaption>"
                    "</figure>"
                )
        lines.append("</div></div>")
    lines.append("</main></body></html>")
    output_html.write_text("\n".join(lines), encoding="utf-8")


class CreateToolTip:
    """Tiny tooltip helper."""

    def __init__(self, widget: tk.Widget, text: str = "widget info") -> None:
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _event: Any = None) -> None:
        if self.tip_window or not self.text:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", 8, "normal"),
        ).pack(ipadx=1)

    def hide_tip(self, _event: Any = None) -> None:
        tw = self.tip_window
        self.tip_window = None
        if tw is not None:
            tw.destroy()





class ScrollableFrame(ttk.Frame):
    """Scrollable container for long control panels (Canvas + interior Frame)."""

    def __init__(self, parent: tk.Widget, *, width: int = 290, **kwargs: Any):
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width)
        self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self.interior = ttk.Frame(self._canvas)
        self._win_id = self._canvas.create_window((0, 0), window=self.interior, anchor="nw")

        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._vsb.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.interior.bind("<Configure>", self._on_interior_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._bind_mousewheel(self._canvas)
        self._bind_mousewheel(self.interior)

    def _on_interior_configure(self, _event: Any) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: Any) -> None:
        self._canvas.itemconfigure(self._win_id, width=event.width)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        def _wheel(e: Any) -> str:
            if getattr(e, "delta", 0):
                self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            return "break"

        def _wheel_up(_e: Any) -> str:
            self._canvas.yview_scroll(-1, "units")
            return "break"

        def _wheel_down(_e: Any) -> str:
            self._canvas.yview_scroll(1, "units")
            return "break"

        widget.bind("<MouseWheel>", _wheel)     # Windows / macOS
        widget.bind("<Button-4>", _wheel_up)    # Linux scroll up
        widget.bind("<Button-5>", _wheel_down)  # Linux scroll down


@dataclass
class SurfCache:
    pial: Optional[Tuple[np.ndarray, np.ndarray]] = None
    white: Optional[Tuple[np.ndarray, np.ndarray]] = None
    inflated: Optional[Tuple[np.ndarray, np.ndarray]] = None


SURFACE_KEY_TO_LABEL = {"pial": "Pial", "white": "White Matter", "inflated": "Inflated"}
SURFACE_LABEL_TO_KEY = {v: k for k, v in SURFACE_KEY_TO_LABEL.items()}
ORIENTATIONS = {
    "top": (90, 0),
    "bottom": (-90, 0),
    "front": (0, 0),
    "back": (0, 180),
    "left": (0, 90),
    "right": (0, -90),
}

PLOTLY_ELECTRODE_OFFSET_M = 0.003
MPL_ELECTRODE_OFFSET_M = 0.0025
MPL_ELECTRODE_JITTER_M = 0.0008
TOP_ROI_OPTIONS = ("Top 5 responding ROIs", "Top 10 responding ROIs")

LIGHT_DIR = np.array([0.35, -0.55, 0.75], float)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
LIGHT_AMBIENT, LIGHT_DIFFUSE = 0.35, 0.75


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
class IEEGAnalyzer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("iEEG ERP Analyzer — Publication-Quality + Interactive 3D + Early/Late")
        if os.name == "nt":
            root.state("zoomed")
        else:
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

        # Data
        self.raw: Optional[mne.io.BaseRaw] = None
        self.events: Optional[np.ndarray] = None
        self.event_times: List[Tuple[float, str]] = []
        self.stim_pairs: List[str] = []
        self.file_path: Optional[Path] = None
        self.results_dir: Optional[Path] = None

        self.montage: Optional[mne.channels.DigMontage] = None
        self.electrode_coords: Optional[pd.DataFrame] = None
        self._pending_coords: Optional[pd.DataFrame] = None
        self._elecnum_to_channel: Dict[int, str] = {}
        self._channel_to_region: Dict[str, str] = {}
        self.workflow_timings: List[Dict[str, Any]] = []

        self._name_index: Dict[str, str] = {}
        self._name_index_list: List[Tuple[str, str]] = []

        self.selected_channel: Optional[str] = None
        self.selected_electrodes: set[str] = set()
        # Channel listbox mapping (display name -> full channel name)
        self._display_to_channel: Dict[str, str] = {}
        self._channel_to_listbox_index: Dict[str, int] = {}
        self._suspend_listbox_callback: bool = False

        self._pick_cid: Optional[int] = None
        self._loading_edf = False
        self._loaded_edf_path: Optional[Path] = None

        self.scatter_3d = None
        self.scatter_underlay = None
        self.scatter_sel = None
        self.ax3d = None
        self._matched_channels_3d: List[str] = []
        self._matched_pos_plot: Optional[np.ndarray] = None
        self._matched_sizes: List[float] = []

        self.last_epochs: Optional[mne.Epochs] = None
        self.last_evoked: Optional[mne.Evoked] = None

        # HTML export cache (pair → epochs). Built on-demand for HTML export.
        self.epochs_by_pair: Dict[str, mne.Epochs] = {}
        self._epochs_cache_sig: Optional[Tuple[float, float, float, float, float]] = None

        self.roi_var = tk.StringVar(value="All ROIs")
        self._current_roi = "All ROIs"
        self._surf_cache = SurfCache()
        self._fsaverage_dir: Optional[str] = None

        # Figures
        self.fig3d = plt.Figure(figsize=(5.2, 4.0), dpi=100, constrained_layout=False)
        # Combined 3D scene: template mesh (ROI projection) + electrodes
        self.figdetail = plt.Figure(figsize=(5.2, 4.0), dpi=100, constrained_layout=True)

        self.canvas3d: Optional[FigureCanvasTkAgg] = None
        self.canvasdetail: Optional[FigureCanvasTkAgg] = None

        # 3D display controls (GUI)
        self.mesh_opacity_var = tk.DoubleVar(value=0.35)
        self.project_only_selected_var = tk.BooleanVar(value=False)
        self.show_left_hemi_var = tk.BooleanVar(value=True)
        self.show_right_hemi_var = tk.BooleanVar(value=True)
        self._roi_mask_3d: list[bool] = []

        self.selection_listbox: Optional[tk.Listbox] = None
        self.status_bar: Optional[tk.Label] = None

        # Fonts
        self.title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
        self.subtitle_font = tkfont.Font(family="Helvetica", size=13, slant="italic")
        self.button_font = tkfont.Font(family="Helvetica", size=10)

        # Theme/export
        self.theme = ThemeSettings()
        self.export_opts = ExportOptions()
        self.theme.apply()

        self._setup_style()
        self._create_menubar()
        self._create_title_banner()
        self._create_widgets()
        self._create_status_bar()

        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self._expanded_panel: Optional[str] = None

        LOG.info("Application initialized")

    # ---------- Small UI helpers ----------
    def _ui(self, fn, *a, **k) -> None:
        self.root.after(0, lambda: fn(*a, **k))

    def _update_status(self, msg: str) -> None:
        if self.status_bar:
            self.status_bar.config(text=msg)
            self.status_bar.update_idletasks()

    @staticmethod
    def _run_thread(fn, *a, **k) -> None:
        threading.Thread(target=fn, args=a, kwargs=k, daemon=True).start()

    # ----- UI -----
    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TButton", padding=6, relief="flat", background="#004080", foreground="white", font=self.button_font
        )
        style.map("TButton", background=[("active", "#0059b3")])
        style.configure("Card.TFrame", background="white", relief="raised", borderwidth=1)

    def _create_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open EDF...", accelerator="Ctrl+O", command=self.load_file)
        file_menu.add_command(
            label="Open Coordinates CSV...", accelerator="Ctrl+Shift+O", command=self.load_electrode_coords
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        opt_menu = tk.Menu(menubar, tearoff=0)
        opt_menu.add_command(label="Reset", accelerator="F5", command=self.reset)
        opt_menu.add_command(label="Export HTML App", accelerator="Ctrl+S", command=self.export_html_app)
        opt_menu.add_command(label="Export PDF Report", accelerator="Ctrl+Shift+S", command=self.export_results)
        opt_menu.add_command(label="Toggle Fullscreen", accelerator="F11", command=self.toggle_fullscreen)
        menubar.add_cascade(label="Options", menu=opt_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        for seq, cmd in (
            ("<Control-o>", self.load_file),
            ("<Control-O>", self.load_electrode_coords),
            ("<Control-q>", self.root.quit),
            ("<F5>", self.reset),
            ("<Control-s>", self.export_html_app),
            ("<Control-S>", self.export_results),
            ("<F11>", self.toggle_fullscreen),
        ):
            self.root.bind(seq, partial(_ignore_event, cmd))

    def toggle_fullscreen(self, _e: Optional[Any] = None) -> None:
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def _create_title_banner(self) -> None:
        banner = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        banner.grid(row=0, column=0, sticky="ew")
        tk.Label(banner, text="iEEG ERP Analyzer", font=self.title_font, fg="white", bg="#004080").pack(
            side=tk.TOP, fill=tk.X
        )
        tk.Label(
            banner,
            text="ERP • RMS • Gamma — Publication‑quality GUI, static & interactive 3D exports",
            font=self.subtitle_font,
            fg="white",
            bg="#004080",
        ).pack(side=tk.TOP, fill=tk.X)

    def _create_widgets(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.grid(row=1, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)


        left_outer = ttk.Frame(container, style="Card.TFrame")
        left_outer.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        left_outer.grid_rowconfigure(0, weight=1)
        left_outer.grid_columnconfigure(0, weight=1)

        left_sf = ScrollableFrame(left_outer, width=290)
        left_sf.grid(row=0, column=0, sticky="nsew")
        left = left_sf.interior

        def labelframe(parent, text):
            f = ttk.LabelFrame(parent, text=text, padding=5)
            f.pack(fill="x", padx=5, pady=5)
            return f

        # File controls
        lf = labelframe(left, "File Controls")
        self.file_button = ttk.Button(lf, text="Select EDF File", command=self.load_file)
        self.file_button.pack(fill="x", pady=2)
        CreateToolTip(self.file_button, "Open an EDF file for analysis.")
        self.file_label = ttk.Label(lf, text="No file selected", wraplength=220)
        self.file_label.pack(fill="x", pady=2)

        # Coordinates
        cf = labelframe(left, "Electrode Coordinates")
        self.coord_button = ttk.Button(cf, text="Load Coordinates CSV", command=self.load_electrode_coords)
        self.coord_button.pack(fill="x", pady=2)
        CreateToolTip(self.coord_button, "Load a CSV with electrode coordinates (mm).")
        self.coord_label = ttk.Label(cf, text="No coordinates loaded", wraplength=220)
        self.coord_label.pack(fill="x", pady=2)

        # Analysis parameters
        pf = labelframe(left, "Analysis Parameters")
        ttk.Label(pf, text="Stim Pair:").pack(anchor="w")
        self.stim_pair_var = tk.StringVar()
        self.stim_pair_combo = ttk.Combobox(pf, textvariable=self.stim_pair_var, state="readonly")
        self.stim_pair_combo.pack(fill="x", pady=2)
        self.stim_pair_combo.bind("<<ComboboxSelected>>", self._on_pair_change)

        def row(parent: tk.Widget, pady: int = 2) -> ttk.Frame:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=pady)
            return frame

        self.min_time_var = tk.StringVar(value="0.5")
        fr = row(pf)
        ttk.Label(fr, text="Min Time Between Events (s):").pack(side="left")
        ttk.Entry(fr, textvariable=self.min_time_var, width=8).pack(side="right")

        self.tmin_var, self.tmax_var = tk.StringVar(value="-0.2"), tk.StringVar(value="1.0")
        fr = row(pf)
        ttk.Label(fr, text="Epoch Window (s):").pack(side="left")
        ttk.Entry(fr, textvariable=self.tmin_var, width=8).pack(side="left", padx=2)
        ttk.Label(fr, text="to").pack(side="left", padx=2)
        ttk.Entry(fr, textvariable=self.tmax_var, width=8).pack(side="left", padx=2)

        self.baseline_min_var, self.baseline_max_var = tk.StringVar(value="-0.2"), tk.StringVar(value="0")
        fr = row(pf)
        ttk.Label(fr, text="Baseline (s):").pack(side="left")
        ttk.Entry(fr, textvariable=self.baseline_min_var, width=8).pack(side="left", padx=2)
        ttk.Label(fr, text="to").pack(side="left", padx=2)
        ttk.Entry(fr, textvariable=self.baseline_max_var, width=8).pack(side="left", padx=2)

        # Early/Late windows
        wf = labelframe(left, "Windows (ms): Early vs Late")
        self.early_start_ms_var, self.early_end_ms_var = tk.StringVar(value="10"), tk.StringVar(value="80")
        self.late_start_ms_var, self.late_end_ms_var = tk.StringVar(value="80"), tk.StringVar(value="300")

        def window_row(parent: tk.Widget, label: str, v0: tk.StringVar, v1: tk.StringVar) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=1)
            ttk.Label(frame, text=f"{label}:").pack(side="left")
            ttk.Entry(frame, textvariable=v0, width=6).pack(side="left", padx=2)
            ttk.Label(frame, text="to").pack(side="left")
            ttk.Entry(frame, textvariable=v1, width=6).pack(side="left", padx=2)

        window_row(wf, "Early", self.early_start_ms_var, self.early_end_ms_var)
        window_row(wf, "Late", self.late_start_ms_var, self.late_end_ms_var)
        CreateToolTip(
            wf,
            "Defaults follow SPES/CCEP literature:\n"
            "Early 10–80 ms (short-latency), Late 80–300 ms (longer-latency).",
        )


        # View selector: controls whether 3D peaks (GUI + export) use the full epoch or Early/Late windows.
        view_row = ttk.Frame(wf)
        view_row.pack(fill="x", pady=4)
        ttk.Label(view_row, text="View:").pack(side="left")
        self.view_window_var = tk.StringVar(value="Average")
        for txt, val in (("Average", "Average"), ("Early", "Early"), ("Late", "Late")):
            ttk.Radiobutton(
                view_row,
                text=txt,
                value=val,
                variable=self.view_window_var,
                command=self._on_view_window_change,
            ).pack(side="left", padx=2)
        CreateToolTip(
            view_row,
            "Select which time window defines the peak amplitude used for 3D maps and export:\n"
            "• Average: peak over the full epoch window\n"
            "• Early/Late: peak within the selected ms window",
        )

        # Advanced options
        af = labelframe(left, "Advanced Options")
        ln_row = row(af)
        ttk.Label(ln_row, text="Line-noise notch:").pack(side="left")
        self.notch_50_var, self.notch_60_var = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(ln_row, text="50 Hz", variable=self.notch_50_var).pack(side="left", padx=4)
        ttk.Checkbutton(ln_row, text="60 Hz", variable=self.notch_60_var).pack(side="left", padx=4)
        CreateToolTip(ln_row, "Apply notch filters for mains hum at 50 and/or 60 Hz.")

        fr = row(af)
        self.bandpass_var = tk.BooleanVar(value=True)
        self.hp_freq_var, self.lp_freq_var = tk.StringVar(value="1"), tk.StringVar(value="100")
        ttk.Checkbutton(fr, text="Bandpass Filter (Hz):", variable=self.bandpass_var).pack(side="left")
        ttk.Entry(fr, textvariable=self.hp_freq_var, width=5).pack(side="left", padx=2)
        ttk.Label(fr, text="-").pack(side="left")
        ttk.Entry(fr, textvariable=self.lp_freq_var, width=5).pack(side="left", padx=2)

        # Mode + ROI + surface
        am = labelframe(left, "Analysis Mode")
        self.analysis_mode_var = tk.StringVar(value="ERP")
        ttk.Label(am, text="Compute:").pack(anchor="w")
        self.analysis_mode_combo = ttk.Combobox(
            am, textvariable=self.analysis_mode_var, values=["ERP", "RMS", "Gamma"], state="readonly"
        )
        self.analysis_mode_combo.pack(fill="x", pady=2)
        CreateToolTip(self.analysis_mode_combo, "Choose ERP, RMS, or Gamma band power.")
        self.analysis_mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        surf_row = row(am)
        ttk.Label(surf_row, text="Template surface:").pack(side="left")
        self.surface_var = tk.StringVar(value=SURFACE_KEY_TO_LABEL["pial"])
        self.surface_combo = ttk.Combobox(
            surf_row, textvariable=self.surface_var, values=list(SURFACE_KEY_TO_LABEL.values()), state="readonly", width=14
        )
        self.surface_combo.pack(side="left", padx=4)
        self.surface_combo.bind("<<ComboboxSelected>>", self._on_surface_change)
        CreateToolTip(self.surface_combo, "Template mesh used for ROI projection and 3D exports.")

        rf = labelframe(left, "ROI Selection")
        ttk.Label(rf, text="Select ROI:").pack(anchor="w")
        self.roi_combo = ttk.Combobox(rf, textvariable=self.roi_var, values=["All ROIs"], state="readonly")
        self.roi_combo.pack(fill="x", pady=2)
        CreateToolTip(self.roi_combo, "Only electrodes in this ROI are used for projection.")
        # 3D display controls
        # The left sidebar uses `pack`, so this control group must also be packed.
        # (Mixing `grid` and `pack` in the same parent raises a Tk error.)
        display = ttk.LabelFrame(left, text="3D display")
        display.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(display, text="Brain opacity").grid(row=0, column=0, sticky="w")
        self.mesh_opacity_scale = ttk.Scale(
            display,
            from_=0.05,
            to=1.0,
            orient="horizontal",
            variable=self.mesh_opacity_var,
            command=lambda _v=None: self._on_mesh_opacity_change(),
        )
        self.mesh_opacity_scale.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        display.grid_columnconfigure(0, weight=1)
        CreateToolTip(display, "Adjust template mesh transparency so implanted electrodes are visible through the surface.")
        # Hemisphere visibility toggles (removes one side of the surface to see electrodes inside).
        hemi = ttk.Frame(display)
        hemi.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(hemi, text="Hemispheres").pack(anchor="w")
        hemi_row = ttk.Frame(hemi)
        hemi_row.pack(fill="x", pady=(2, 0))
        ttk.Checkbutton(
            hemi_row,
            text="Left",
            variable=self.show_left_hemi_var,
            command=self.plot_3d_visualization,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            hemi_row,
            text="Right",
            variable=self.show_right_hemi_var,
            command=self.plot_3d_visualization,
        ).pack(side="left")
        CreateToolTip(hemi, "Toggle hemisphere visibility. Uncheck to remove that half of the surface.")
        proj_cb = ttk.Checkbutton(
            display,
            text="Project only selected sensor montage",
            variable=self.project_only_selected_var,
            command=self.plot_3d_visualization,
        )
        proj_cb.grid(row=3, column=0, sticky="w", pady=(6, 0))
        CreateToolTip(
            proj_cb,
            "When enabled, the ROI projection on the cortical surface is computed using only the currently selected sensor montage (multi-select list).",
        )
        # (No row counter here; sidebar uses `pack`.)

        self.roi_combo.bind("<<ComboboxSelected>>", self._on_roi_selected)

        # Appearance & Export
        axf = labelframe(left, "Appearance & Export")
        fr = row(axf)
        ttk.Label(fr, text="Theme:").pack(side="left")
        self.preset_var = tk.StringVar(value=self.theme.preset)
        ttk.Combobox(fr, textvariable=self.preset_var, values=list(STYLE_PRESETS.keys()), state="readonly", width=14).pack(
            side="left", padx=4
        )
        ttk.Label(fr, text="Colormap:").pack(side="left")
        self.cmap_var = tk.StringVar(value=self.theme.colormap)
        ttk.Combobox(
            fr,
            textvariable=self.cmap_var,
            values=["plasma", "viridis", "magma", "inferno", "cividis", "coolwarm"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=4)

        fr = row(axf)
        ttk.Label(fr, text="Line width:").pack(side="left")
        self.lw_var = tk.StringVar(value=str(self.theme.erp_line_width))
        ttk.Entry(fr, textvariable=self.lw_var, width=5).pack(side="left", padx=4)
        ttk.Label(fr, text="CI α:").pack(side="left")
        self.ci_var = tk.StringVar(value=str(self.theme.ci_alpha))
        ttk.Entry(fr, textvariable=self.ci_var, width=5).pack(side="left", padx=4)
        self.zero_line_var = tk.BooleanVar(value=self.theme.show_zero_time)
        ttk.Checkbutton(fr, text="Zero-time line", variable=self.zero_line_var).pack(side="left", padx=4)

        fr = row(axf)
        ttk.Label(fr, text="Font scale:").pack(side="left")
        self.font_scale_var = tk.StringVar(value=str(self.theme.font_scale))
        ttk.Entry(fr, textvariable=self.font_scale_var, width=5).pack(side="left", padx=4)
        ttk.Label(fr, text="Export DPI:").pack(side="left")
        self.dpi_var = tk.StringVar(value=str(self.export_opts.dpi))
        ttk.Entry(fr, textvariable=self.dpi_var, width=5).pack(side="left", padx=4)

        self.png_var, self.pdf_var, self.svg_var = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(fr, text="PNG", variable=self.png_var).pack(side="left", padx=2)
        ttk.Checkbutton(fr, text="PDF", variable=self.pdf_var).pack(side="left", padx=2)
        ttk.Checkbutton(fr, text="SVG", variable=self.svg_var).pack(side="left", padx=2)
        self.transparent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fr, text="Transparent", variable=self.transparent_var).pack(side="left", padx=4)

        fr = row(axf)
        self.html_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr, text="HTML (interactive 3D)", variable=self.html_var).pack(side="left", padx=2)
        CreateToolTip(fr, "Export Plotly-based 3D HTML (standalone, offline if enabled).")

        fr = row(axf)
        ttk.Button(fr, text="Apply Style", command=self._apply_style_now).pack(side="left", padx=2)
        CreateToolTip(fr, "Apply selected theme/colormap and refresh canvases.")

        # Action buttons
        bf = ttk.Frame(left)
        bf.pack(fill="x", padx=5, pady=5)
        self.analyze_button = ttk.Button(bf, text="Run ERP", command=lambda: self._run_thread(self.generate_erps), state="disabled")
        self.analyze_button.pack(side="left", padx=2)
        CreateToolTip(self.analyze_button, "Compute ERP/RMS/Gamma from the loaded EDF.")
        self.export_button = ttk.Button(bf, text="Export HTML App", command=self.export_html_app, state="disabled")
        self.export_button.pack(side="left", padx=2)
        CreateToolTip(self.export_button, "Export a single self-contained HTML viewer (no import buttons) with the current data + interactive plots.")
        ttk.Button(bf, text="Reset", command=self.reset).pack(side="left", padx=2)

        # Selection list
        sf = ttk.LabelFrame(left, text="Selected Electrodes", padding=5)
        sf.pack(fill="both", expand=True, padx=5, pady=5)
        self.selection_listbox = tk.Listbox(sf, selectmode=tk.EXTENDED, height=10)
        self.selection_listbox.pack(fill="both", expand=True, pady=2)
        self.selection_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        
        # Right figure grid
        right = ttk.Frame(container, style="Card.TFrame")
        right.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        vf = ttk.Frame(right)
        vf.pack(fill="both", expand=True)

        # Combined 3D scene (mesh + electrodes)
        self.frame3d = ttk.Frame(vf)
        self.frame3d.grid(row=0, column=0, padx=6, pady=(6, 3), sticky="nsew")
        ttk.Label(
            self.frame3d,
            text="3D brain + electrodes (ROI projection)",
            style="PlotTitle.TLabel",
        ).pack(anchor="w", pady=(0, 4))

        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self.frame3d)
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)

        # Detailed waveform panel
        self.framedetail = ttk.Frame(vf)
        self.framedetail.grid(row=1, column=0, padx=6, pady=(3, 6), sticky="nsew")
        ttk.Label(self.framedetail, text="Detailed waveform", style="PlotTitle.TLabel").pack(
            anchor="w", pady=(0, 4)
        )

        self.canvasdetail = FigureCanvasTkAgg(self.figdetail, master=self.framedetail)
        self.canvasdetail.get_tk_widget().pack(fill="both", expand=True)

        vf.grid_rowconfigure(0, weight=3)
        vf.grid_rowconfigure(1, weight=2)
        vf.grid_columnconfigure(0, weight=1)

        # Interactions
        self.canvas3d.mpl_connect("pick_event", self._on_3d_pick)
        self.canvas3d.mpl_connect(
            "button_press_event",
            lambda e: self._on_canvas_double_click(e, which="3d")
            if getattr(e, "dblclick", False)
            else None,
        )
        self.canvasdetail.mpl_connect(
            "button_press_event",
            lambda e: self._on_canvas_double_click(e, which="detail")
            if getattr(e, "dblclick", False)
            else None,
        )

        if hasattr(self, "project_only_selected_var"):
            try:
                self.project_only_selected_var.set(False)
            except Exception:
                pass
        self._update_status("Ready")

    def _create_status_bar(self) -> None:
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, background="#f0f0f0")
        self.status_bar.grid(row=2, column=0, sticky="ew")

    # ---------- Channel-name normalization & lookup ----------
    @staticmethod
    def _normalize_label(s: str) -> str:
        s = re.sub(r"(?i)^pol\s*", "", s.strip()).lower()
        parts = re.findall(r"[a-z]+|\d+", s)
        out = []
        for p in parts:
            out.append(str(int(p)) if p.isdigit() else p)
        return "".join(out)

    def _build_channel_name_index(self) -> None:
        self._name_index.clear()
        self._name_index_list.clear()
        if self.raw is None:
            return
        for ch in self.raw.ch_names:
            nmain = self._normalize_label(ch)
            self._name_index[nmain] = ch
            self._name_index_list.append((nmain, ch))
            for tok in re.split(r"[^A-Za-z0-9]+", ch):
                if not tok:
                    continue
                nt = self._normalize_label(tok)
                self._name_index.setdefault(nt, ch)
                self._name_index_list.append((nt, ch))

    def _resolve_stim_token(self, token: str) -> Optional[str]:
        if not token:
            return None
        key = self._normalize_label(token)
        if key in self._name_index:
            return self._name_index[key]
        # fallback: contains match
        for nk, ch in self._name_index_list:
            if key and key in nk:
                return ch
        return None


    def _lookup_channel_pos(self, ch_name: str) -> Optional[np.ndarray]:
        """Return a channel 3D position in **meters** if available.

        This is used by the desktop GUI 3D renderer. We try, in order:
          1) exact channel match in ``self.raw`` and ``raw.info['chs'][idx]['loc'][:3]``
          2) montage positions (``raw.get_montage()`` or ``self.montage``)

        The lookup tolerates POL prefixes and stim tokens (e.g., "LC3") by using
        :meth:`_resolve_stim_token`.
        """
        if not ch_name:
            return None

        raw = getattr(self, "raw", None)
        idx: Optional[int] = None

        if raw is not None and hasattr(raw, "ch_names"):
            try:
                idx = raw.ch_names.index(ch_name)
            except ValueError:
                resolved = self._resolve_stim_token(ch_name)
                if resolved and resolved in raw.ch_names:
                    idx = raw.ch_names.index(resolved)
                else:
                    stripped = re.sub(r"(?i)^pol\s*", "", ch_name).strip()
                    if stripped in raw.ch_names:
                        idx = raw.ch_names.index(stripped)

        # Try raw.info first (fast + authoritative once montage applied).
        if idx is not None and raw is not None:
            try:
                loc = np.asarray(raw.info["chs"][idx].get("loc", None), dtype=float)
                if loc.size >= 3 and np.all(np.isfinite(loc[:3])):
                    return loc[:3].copy()
            except Exception:
                pass

        # Montage fallback.
        mont = None
        try:
            if raw is not None and hasattr(raw, "get_montage"):
                mont = raw.get_montage()
        except Exception:
            mont = None

        if mont is None:
            mont = getattr(self, "montage", None)

        try:
            if mont is not None:
                ch_pos = mont.get_positions().get("ch_pos", {}) or {}
                # direct
                if ch_name in ch_pos:
                    pos = np.asarray(ch_pos[ch_name], dtype=float)
                    if pos.size == 3 and np.all(np.isfinite(pos)):
                        return pos.copy()

                # resolved token
                resolved = self._resolve_stim_token(ch_name)
                if resolved and resolved in ch_pos:
                    pos = np.asarray(ch_pos[resolved], dtype=float)
                    if pos.size == 3 and np.all(np.isfinite(pos)):
                        return pos.copy()

                # stripped POL prefix
                stripped = re.sub(r"(?i)^pol\s*", "", ch_name).strip()
                if stripped in ch_pos:
                    pos = np.asarray(ch_pos[stripped], dtype=float)
                    if pos.size == 3 and np.all(np.isfinite(pos)):
                        return pos.copy()
        except Exception:
            pass

        return None

    @staticmethod
    def _strip_pol(name: str) -> str:
        """Display-only cleanup: remove leading 'POL' and any '$' markers."""
        s = re.sub(r"(?i)^pol\s*", "", name).strip()
        return s.replace("$", "")
    @staticmethod
    def _strip_pol_prefix(name: str) -> str:
        """Alias kept for backwards compatibility with earlier iterations."""
        return IEEGAnalyzer._strip_pol(name)

    def _split_pair(self, pair: str) -> Tuple[str, str]:
        """Split a stim-pair label into (a, b) electrode names.

        Accepts a variety of separators (e.g., '-', '–', '—', ':', ',').
        """
        s = (pair or "").strip()
        for dash in ("–", "—", "−", "‑", "‒", "―"):
            s = s.replace(dash, "-")
        if "-" in s:
            parts = [p.strip() for p in s.split("-") if p.strip()]
        elif ":" in s:
            parts = [p.strip() for p in s.split(":") if p.strip()]
        elif "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in s.split() if p.strip()]

        if len(parts) >= 2:
            a, b = parts[0], parts[1]
        elif len(parts) == 1:
            a = b = parts[0]
        else:
            a = b = ""

        return self._strip_pol(a), self._strip_pol(b)


    def _apply_style(self) -> None:
        """Apply current Matplotlib theme/appearance settings."""
        try:
            self.theme.apply()
        except Exception as exc:
            LOG.warning("Failed to apply style: %s", exc)

    def _apply_style_now(self) -> None:
        self._apply_style()
        self.plot_3d_visualization()
        self.update_detailed_view()

    def _on_mode_change(self, _event=None) -> None:
        mode = self.analysis_mode_var.get()
        self.analyze_button.config(text=f"Run {mode}")
        if self.raw is not None and self.events is not None and len(self.events):
            self._run_thread(self.generate_erps)

    def _on_pair_change(self, _event=None) -> None:
        # Update the stim-site marker immediately in the GUI 3D view, then recompute.
        if self.raw is not None and self.events is not None and len(self.events):
            try:
                self.plot_3d_visualization()
            except Exception:
                pass
            self._run_thread(self.generate_erps)


    def _on_view_window_change(self, _event=None) -> None:
        self.plot_3d_visualization()
        self.update_detailed_view()

    def _on_canvas_double_click(self, event: Any, which: str) -> None:
        # Toggle between normal and "expanded" mode for the clicked panel.
        if which not in {"3d", "detail"}:
            return

        if self._expanded_panel == which:
            self._show_all()
            self._expanded_panel = None
        else:
            self._hide_others(which)
            self._expanded_panel = which

    def _toggle_panel(self, which: str) -> None:
        if self._expanded_panel is None:
            self._expanded_panel = which
            self._hide_others(which)
        else:
            self._expanded_panel = None
            self._show_all()

    def _hide_others(self, which: str) -> None:
        if which == "3d":
            if getattr(self, "framedetail", None) is not None:
                self.framedetail.grid_remove()
        elif which == "detail":
            if getattr(self, "frame3d", None) is not None:
                self.frame3d.grid_remove()

    def _show_all(self) -> None:
        if getattr(self, "frame3d", None) is not None:
            self.frame3d.grid()
        if getattr(self, "framedetail", None) is not None:
            self.framedetail.grid()

    def _record_workflow_timing(self, step: str, elapsed_s: float, **metadata: Any) -> None:
        """Record objective workflow timing metrics for later audit/reporting."""
        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": str(step),
            "elapsed_s": round(float(elapsed_s), 4),
        }
        row.update({k: v for k, v in metadata.items() if v is not None})
        self.workflow_timings.append(row)
        LOG.info("Workflow timing | %s | %.4f s | %s", step, elapsed_s, metadata)
        self._write_workflow_timing_csv()

    def _write_workflow_timing_csv(self) -> None:
        """Persist workflow timings next to the current results when a results folder exists."""
        if not self.workflow_timings:
            return
        out_dir = getattr(self, "results_dir", None)
        if out_dir is None:
            return
        try:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            pd.DataFrame(self.workflow_timings).to_csv(Path(out_dir) / "workflow_timing_metrics.csv", index=False)
        except Exception as exc:
            LOG.warning("Could not write workflow timing metrics: %s", exc)

    def load_electrode_coords(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self._update_status("Loading electrode coordinates...")
            df = pd.read_csv(path)
            self._apply_electrode_coords(df)
            self.coord_label.config(text=f"Loaded: {Path(path).name}" + (" (awaiting EDF)" if self.raw is None else ""))
            messagebox.showinfo(
                "Success", "Electrode coordinates loaded" + (" and applied." if self.raw else " (will apply after EDF load).")
            )
            self._update_status("Ready")
        except Exception as exc:
            LOG.exception("Coordinates load error: %s", exc)
            messagebox.showerror("Error", f"Failed to load coordinates: {exc}")
            self._update_status("Error loading electrode coordinates")

    def load_file(self) -> None:
        if getattr(self, "_loading_edf", False):
            return
        path = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf"), ("All files", "*.*")])
        if not path:
            return
        p = Path(path).resolve()
        if self._loaded_edf_path is not None and p == self._loaded_edf_path and self.raw is not None:
            self._update_status("EDF already loaded.")
            LOG.info("EDF already loaded: %s", p)
            return
        self._loading_edf = True
        try:
            self._update_status("Loading EDF file...")

            t0 = time.perf_counter()
            self.raw = mne.io.read_raw_edf(str(p), preload=True)
            data_import_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            self._ensure_montage()
            self._apply_filters()
            events, evt_times = self._extract_stim_events()
            preprocessing_s = time.perf_counter() - t1

            if len(events) == 0:
                raise ValueError("No stimulation events found.")
            self.events, self.event_times = events, evt_times
            self.file_path = p
            self._loaded_edf_path = p
            self.results_dir = self.file_path.parent / f"results_{self.file_path.stem}"
            self.results_dir.mkdir(exist_ok=True)

            self._record_workflow_timing("data_import", data_import_s, file=p.name)
            self._record_workflow_timing(
                "preprocessing_and_event_detection",
                preprocessing_s,
                file=p.name,
                n_events=int(len(events)),
                n_stim_pairs=int(len(self.stim_pairs)),
            )

            self.file_label.config(
                text=f"Loaded: {self.file_path.name}\nEvents: {len(events)}\nStim Pairs: {len(self.stim_pairs)}"
            )
            self.analyze_button.config(state="normal")
            # Show electrode placements immediately (even before analysis) with the current stim-site marker.
            try:
                self.plot_3d_visualization()
            except Exception:
                pass
            self._update_status("Ready")
        except Exception as exc:
            LOG.exception("EDF load error: %s", exc)
            messagebox.showerror("Error", f"Failed to load file: {exc}")
            self._update_status("Error loading file")
        finally:
            self._loading_edf = False

    def _apply_electrode_coords(self, df: pd.DataFrame) -> None:
        required = {"ElecNumber", "X", "Y", "Z"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required}")

        # Regions are optional; used only for ROI filtering / summaries.
        if "Regions" not in df.columns:
            LOG.warning("Missing 'Regions'; setting to 'Unknown'.")
            df = df.copy()
            df["Regions"] = "Unknown"
        elif df["Regions"].isnull().any():
            LOG.warning("Missing 'Regions' entries; filling with 'Unknown'.")
            df = df.copy()
            df["Regions"] = df["Regions"].fillna("Unknown")

        df = df.sort_values("ElecNumber").reset_index(drop=True)

        rois = ["All ROIs"] + list(TOP_ROI_OPTIONS) + sorted(df["Regions"].astype(str).unique().tolist())
        self.roi_combo["values"] = rois
        self.roi_var.set("All ROIs")
        self.electrode_coords = df

        # If EDF is not loaded yet, stage coords for later.
        if self.raw is None:
            self._pending_coords = df
            return

        ch_names = list(self.raw.ch_names)
        n = min(len(ch_names), len(df))

        pos_m = df.loc[: n - 1, ["X", "Y", "Z"]].to_numpy(dtype=float) / 1000.0
        ch_pos: Dict[str, Tuple[float, float, float]] = {
            ch_names[i]: (float(pos_m[i, 0]), float(pos_m[i, 1]), float(pos_m[i, 2])) for i in range(n)
        }

        # Fill remaining channels with 10–20 positions (or origin when unknown).
        std = mne.channels.make_standard_montage("standard_1020")
        std_pos = std.get_positions()["ch_pos"]

        missing: List[str] = []
        for ch in ch_names[n:]:
            xyz = std_pos.get(ch)
            if xyz is None:
                missing.append(ch)
                xyz = (0.0, 0.0, 0.0)
            ch_pos[ch] = xyz

        if missing:
            preview = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
            LOG.warning("%d channels not in 10–20; set to (0,0,0): %s", len(missing), preview)

        self.montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        self.raw.set_montage(self.montage, on_missing="ignore")

        # Build ElecNumber -> channel mapping (assumes EDF channel order aligns with sorted ElecNumber)
        self._elecnum_to_channel.clear()
        self._channel_to_region.clear()
        for i, row in enumerate(df.itertuples(index=False)):
            if i >= len(ch_names):
                break
            try:
                elec_num = int(getattr(row, "ElecNumber"))
            except (TypeError, ValueError):
                continue
            ch = ch_names[i]
            self._elecnum_to_channel[elec_num] = ch
            self._channel_to_region[ch] = str(getattr(row, "Regions", "Unknown"))

        self._build_channel_name_index()



    def _ensure_montage(self) -> None:
        if self.raw is None:
            return
        if self.montage is not None:
            self.raw.set_montage(self.montage, on_missing="ignore")
            self._build_channel_name_index()
            return
        if self._pending_coords is not None:
            self._apply_electrode_coords(self._pending_coords)
            self._pending_coords = None
            return
        LOG.warning("No electrode coordinates; applying standard montage.")
        std = mne.channels.make_standard_montage("standard_1020")
        self.raw.set_montage(std, on_missing="ignore")
        self.montage = std
        self._build_channel_name_index()
        messagebox.showinfo("Info", "Standard montage applied.")

    def _apply_filters(self) -> None:
        if self.raw is None:
            return

        freqs: List[float] = []
        if self.notch_50_var.get():
            freqs.append(50.0)
        if self.notch_60_var.get():
            freqs.append(60.0)
        if freqs:
            self.raw.notch_filter(np.asarray(freqs, dtype=float))

        if self.bandpass_var.get():
            self.raw.filter(
                l_freq=float(self.hp_freq_var.get()),
                h_freq=float(self.lp_freq_var.get()),
            )



    def _extract_stim_events(self) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
        if self.raw is None:
            return np.empty((0, 3), int), []
        pat = re.compile(r"([A-Za-z]+\d+)-([A-Za-z]+\d+)")
        events, event_times, pairs = [], [], set()
        for desc, onset in zip(self.raw.annotations.description, self.raw.annotations.onset):
            m = pat.search(desc)
            if not m:
                continue
            pair = f"{m.group(1)}-{m.group(2)}"
            pairs.add(pair)
            events.append([int(onset * self.raw.info["sfreq"]), 0, 1])
            event_times.append((onset, pair))
        self.stim_pairs = sorted(pairs)
        self.stim_pair_combo["values"] = self.stim_pairs
        if self.stim_pairs:
            self.stim_pair_combo.set(self.stim_pairs[0])
        return np.asarray(events, int), event_times

    def _clean_events(self, stim_pair: str, min_time: float) -> np.ndarray:
        if self.events is None or self.raw is None or not len(self.events):
            return np.empty((0, 3), int)
        min_samp = int(min_time * float(self.raw.info["sfreq"]))
        keep: List[np.ndarray] = []
        for ev, (_, p) in zip(self.events, self.event_times):
            if p != stim_pair:
                continue
            if not keep or (ev[0] - keep[-1][0]) >= min_samp:
                keep.append(ev)
        return np.asarray(keep, int)

    # ----- Core computation -----
    def generate_erps(self) -> None:
        mode = self.analysis_mode_var.get()
        self._ui(self._update_status, f"Running {mode} analysis...")
        try:
            if self.raw is None or self.events is None:
                raise RuntimeError("Load an EDF first.")

            stim_pair = self.stim_pair_var.get()
            analysis_t0 = time.perf_counter()
            clean = self._clean_events(stim_pair, float(self.min_time_var.get()))
            if len(clean) == 0:
                raise ValueError(f"No events for pair {stim_pair}")

            epochs = mne.Epochs(
                self.raw,
                clean,
                event_id={f"stim_{stim_pair}": 1},
                tmin=float(self.tmin_var.get()),
                tmax=float(self.tmax_var.get()),
                baseline=(float(self.baseline_min_var.get()), float(self.baseline_max_var.get())),
                preload=True,
            )
            evoked = self._compute_evoked_for_mode(epochs, mode)
            self.last_epochs, self.last_evoked = epochs, evoked
            self._record_workflow_timing(
                "feature_extraction",
                time.perf_counter() - analysis_t0,
                stim_pair=stim_pair,
                mode=mode,
                n_trials=int(len(clean)),
                n_channels=int(len(evoked.ch_names)),
            )
            self._ui(self._on_analysis_complete)
        except Exception as exc:
            LOG.exception("Analysis error: %s", exc)
            self._ui(messagebox.showerror, "Error", f"Failed to run analysis: {exc}")
        finally:
            self._ui(self._update_status, "Ready")

    def _on_analysis_complete(self) -> None:
        self.export_button.config(state="normal")

        # Populate channel list (for montage selection) once we have an evoked object
        if self.last_evoked is not None:
            self._populate_channel_listbox()

            # Choose a reasonable default selection so the detailed waveform isn't blank
            if not self.selected_electrodes and self.last_evoked.ch_names:
                self.selected_channel = self.last_evoked.ch_names[0]
                self.selected_electrodes = {self.selected_channel}

            self._refresh_selection_list()

        self.plot_3d_visualization()
        self.update_detailed_view(self.selected_channel)

    @staticmethod
    def _compute_gamma_envelope(data: np.ndarray, fs: float, fmin: float = 30.0, fmax: float = 80.0) -> np.ndarray:
        ne, nc, nt = data.shape
        flat = data.reshape(ne * nc, nt)
        filt = mne.filter.filter_data(flat, sfreq=fs, l_freq=fmin, h_freq=fmax, method="iir", verbose=False)
        env = np.abs(hilbert(filt, axis=-1))
        return env.reshape(ne, nc, nt)

    def _compute_evoked_for_mode(self, epochs: mne.Epochs, mode: str) -> mne.Evoked:
        if mode == "ERP":
            return epochs.average()
        data = epochs.get_data()
        evk = epochs.average()
        if mode == "RMS":
            evk.data = np.sqrt(np.mean(data**2, axis=0))
            return evk
        if mode == "Gamma":
            evk.data = self._compute_gamma_envelope(data, fs=float(epochs.info["sfreq"])).mean(axis=0)
            return evk
        raise ValueError(f"Unknown analysis mode: {mode}")

    # ----- Detailed channel view -----
    @staticmethod
    def _ci95(trials_uv: np.ndarray, mode: str) -> np.ndarray:
        if trials_uv.shape[0] <= 1:
            return np.zeros(trials_uv.shape[1], float)
        if mode in ("ERP", "Gamma"):
            return 1.96 * stats.sem(trials_uv, axis=0)
        # RMS delta-method CI
        x = trials_uv
        n = x.shape[0]
        mu2 = np.mean(x**2, axis=0)
        rms = np.sqrt(mu2) + 1e-12
        std2 = np.std(x**2, axis=0, ddof=1)
        sem_mu2 = std2 / np.sqrt(n)
        return 1.96 * (sem_mu2 / (2.0 * rms))

    
    def update_detailed_view(self, channel_name: Optional[str] = None) -> None:
        """Update the detailed waveform panel.

        - If multiple electrodes are selected (via listbox multi-select or 3D picking), render a montage.
        - If none are selected, fall back to `channel_name`, otherwise the first channel.
        """
        if self.last_epochs is None or self.last_evoked is None:
            return

        all_ch = list(self.last_evoked.ch_names)

        # Determine which channels to show
        selected: List[str] = []
        if self.selected_electrodes:
            selected = [ch for ch in all_ch if ch in self.selected_electrodes]

        if not selected:
            if channel_name and channel_name in all_ch:
                selected = [channel_name]
            elif all_ch:
                selected = [all_ch[0]]

        if not selected:
            return

        # Pull data for selected channels
        try:
            data_v = self.last_epochs.get_data(picks=selected)  # (n_epochs, n_sel, n_times)
        except Exception:
            # Fallback: if picks fail for any reason, show the first channel
            selected = [all_ch[0]]
            data_v = self.last_epochs.get_data(picks=selected)

        n_epochs, n_sel, n_times = data_v.shape
        t_ms = self.last_epochs.times * 1000.0
        data_uv = data_v * 1e6

        mean_uv = data_uv.mean(axis=0)
        if n_epochs > 1:
            sem = data_uv.std(axis=0, ddof=1) / max(1.0, math.sqrt(float(n_epochs)))
            ci_uv = 1.96 * sem
        else:
            ci_uv = np.zeros_like(mean_uv)

        # Compute vertical spacing (robust)
        abs_vals = np.abs(mean_uv).ravel()
        if abs_vals.size:
            span = float(np.percentile(abs_vals, 95))
        else:
            span = 1.0
        spacing = max(10.0, span * 3.0)

        # Clear and redraw
        self.figdetail.clf()
        ax = self.figdetail.add_subplot(1, 1, 1)

        # Apply theme (rcParams affects new axes)
        try:
            self.theme.apply()
        except Exception:
            pass

        # Plot montage (top-to-bottom)
        order = list(range(n_sel))[::-1]
        yticks: List[float] = []
        ylabels: List[str] = []

        for rank, i_ch in enumerate(order):
            offset = rank * spacing
            y = mean_uv[i_ch] + offset
            ylo = (mean_uv[i_ch] - ci_uv[i_ch]) + offset
            yhi = (mean_uv[i_ch] + ci_uv[i_ch]) + offset

            ax.plot(t_ms, y, linewidth=1.0)
            ax.fill_between(t_ms, ylo, yhi, alpha=0.18, linewidth=0)

            yticks.append(offset)
            ylabels.append(self._strip_pol(selected[i_ch]))

        # Stim time marker
        ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)

        # Highlight the current early/late window (if applicable)
        try:
            window_ms = self._current_view_window_ms()
            if window_ms is not None:
                ax.axvspan(window_ms[0], window_ms[1], alpha=0.08)
        except Exception:
            pass

        title = "Detailed waveform montage" if len(selected) > 1 else f"Detailed waveform — {self._strip_pol(selected[0])}"
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Channels")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)

        # Improve layout
        self.figdetail.tight_layout()
        self.canvasdetail.draw_idle()

    def on_span_select(self, xmin: float, xmax: float) -> None:
        if not (self.last_epochs and self.last_evoked and self.selected_channel):
            return
        times = self.last_evoked.times * 1000.0
        imin, imax = int(np.argmin(np.abs(times - xmin))), int(np.argmin(np.abs(times - xmax)))
        if imax <= imin:
            imax = imin + 1
        ch_idx = self.last_evoked.ch_names.index(self.selected_channel)
        mode = self.analysis_mode_var.get()

        if mode == "Gamma":
            env = self._compute_gamma_envelope(self.last_epochs.get_data(), fs=float(self.last_epochs.info["sfreq"])) * 1e6
            data = env[:, ch_idx, imin:imax]
        else:
            data = self.last_epochs.get_data()[:, ch_idx, imin:imax] * 1e6

        mean_amp = float(np.mean(data))
        peak_amp = float(np.max(np.abs(data)))
        std_amp = float(np.std(data))
        median_amp = float(np.median(data))
        q25, q75 = (float(v) for v in np.percentile(data, [25, 75]))
        slope = float(np.mean(np.diff(data, axis=1)))
        msg = (
            "Selected Window Statistics\n------------------------\n"
            f"Time Range: {xmin:.1f} - {xmax:.1f} ms\n"
            f"Mean Amplitude: {mean_amp:.2f} µV\nMedian Amplitude: {median_amp:.2f} µV\n"
            f"Peak Amplitude: {peak_amp:.2f} µV\nStd Deviation: {std_amp:.2f} µV\n"
            f"Q25 - Q75: {q25:.2f} - {q75:.2f} µV\nAverage Slope: {slope:.2f} µV/sample"
        )
        messagebox.showinfo("Window Statistics", msg)

    # ----- 3D visual helpers -----

    @staticmethod
    def _stable_pair_from_name(name: str) -> float:
        """Deterministic scalar in [0, 1) derived from a channel name."""
        h = hashlib.sha256(name.encode("utf-8")).digest()
        # Use first 8 bytes as a uint64 for stable pseudo-randomness.
        u = int.from_bytes(h[:8], "little", signed=False)
        return (u % 10_000_000) / 10_000_000.0

    def _offset_and_jitter_positions(
        self,
        pos: np.ndarray,
        names: Sequence[str],
        center: np.ndarray,
        offset_m: float = 0.004,
        jitter_m: float = 0.002,
    ) -> np.ndarray:
        """Separate coincident electrodes deterministically for visualization."""
        v = pos - center[None, :]
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        nrm = v / (norms + 1e-12)

        # Tangent basis: cross with a fixed axis; fallback if near-parallel.
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        t1 = np.cross(nrm, z_axis)
        bad = np.linalg.norm(t1, axis=1) < 1e-6
        if np.any(bad):
            t1[bad] = np.cross(nrm[bad], y_axis)

        t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
        t2 = np.cross(nrm, t1)
        t2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)

        angles = np.array([2 * np.pi * self._stable_pair_from_name(n) for n in names], dtype=float)
        jitter = jitter_m * (np.cos(angles)[:, None] * t1 + np.sin(angles)[:, None] * t2)
        return pos + nrm * offset_m + jitter

    def _draw_axes_triad(self, ax: Axes3D) -> None:
        """Draw an orientation triad in the lower-left corner."""
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        origin = np.array([xlim[0], ylim[0], zlim[0]], dtype=float)
        length = 0.1 * float(max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

        ax.plot([origin[0], origin[0] + length], [origin[1], origin[1]], [origin[2], origin[2]], color="r")
        ax.plot([origin[0], origin[0]], [origin[1], origin[1] + length], [origin[2], origin[2]], color="g")
        ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + length], color="b")

        ax.text(origin[0] + length, origin[1], origin[2], "X", color="r")
        ax.text(origin[0], origin[1] + length, origin[2], "Y", color="g")
        ax.text(origin[0], origin[1], origin[2] + length, "Z", color="b")

    @staticmethod
    def _mark_stim_center(
        ax: Axes3D,
        center: Optional[np.ndarray],
        *,
        brain_center: Optional[np.ndarray] = None,
        offset_m: float = 0.003,
    ) -> None:
        """Draw a visible stim-site marker (yellow circle) on a 3D axis."""
        if center is None:
            return
        c = np.asarray(center, dtype=float)
        if brain_center is not None:
            bc = np.asarray(brain_center, dtype=float)
            v = c - bc
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                c = c + (v / n) * float(offset_m)        # Single high-contrast stim marker (hollow ring) to avoid confusion with colored electrodes
        ax.scatter(
            [c[0]], [c[1]], [c[2]],
            s=260, facecolors="none", edgecolors="#ffd700",
            linewidths=2.6, marker="o", zorder=8
        )


    @staticmethod
    def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n

    def _face_shading(self, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        ndotl = np.clip(self._compute_face_normals(verts, faces) @ LIGHT_DIR, 0.0, 1.0)
        return np.clip(LIGHT_AMBIENT + LIGHT_DIFFUSE * ndotl, 0.0, 1.0)

    
    def _apply_shaded_mesh(
        self,
        poly: Poly3DCollection,
        verts: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray] = None,
        *,
        face_values: Optional[np.ndarray] = None,
        cmap: Union[str, mcolors.Colormap] = "magma",
        cmap_name: Optional[Union[str, mcolors.Colormap]] = None,
        norm: Optional[mcolors.Normalize] = None,
        alpha: Optional[float] = None,
        base_color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float]] = "lightgrey",
        shade: bool = True,
        edgecolor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        linewidth: float = 0.0,
    ) -> mpl.cm.ScalarMappable:
        """Apply a per-face colormap (optionally shaded) to a Poly3DCollection.

        This helper is used by both the GUI renderer and report exports.

        It supports legacy keyword aliases:
          - face_values instead of values
          - cmap_name instead of cmap
        """
        # Backwards-compatible aliases
        if face_values is not None:
            fvals = np.asarray(face_values, dtype=float)
        elif values is not None:
            fvals = np.asarray(values, dtype=float)
        else:
            fvals = None

        if cmap_name is not None:
            cmap = cmap_name

        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

        if norm is None:
            if fvals is not None and np.isfinite(fvals).any():
                vmin = float(np.nanmin(fvals))
                vmax = float(np.nanmax(fvals))
            else:
                vmin, vmax = 0.0, 1.0
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        if alpha is None:
            a = poly.get_alpha()
            alpha = 1.0 if a is None else float(a)
        alpha = float(max(0.0, min(1.0, alpha)))

        # Start from a base color, then (optionally) override with mapped values.
        base_rgba = np.array(mcolors.to_rgba(base_color, alpha=alpha), dtype=float)
        n_faces = int(faces.shape[0]) if faces is not None else 0
        colors = np.tile(base_rgba, (n_faces, 1))

        if fvals is not None and fvals.shape[0] == n_faces:
            mapped = np.asarray(cmap_obj(norm(np.nan_to_num(fvals, nan=norm.vmin))), dtype=float)
            if mapped.ndim == 2 and mapped.shape[1] >= 3:
                colors[:, :3] = mapped[:, :3]
            colors[:, 3] = alpha

        # Simple Lambert shading to give depth cues without changing the colormap.
        if shade and verts.size and faces.size:
            try:
                face_normals = self._compute_face_normals(verts, faces)
                light_dir = np.array([0.2, 0.3, 0.9], dtype=float)
                light_dir /= np.linalg.norm(light_dir)
                intensity = np.clip(face_normals @ light_dir, 0.15, 1.0)
                colors[:, :3] = (colors[:, :3].T * intensity).T
            except Exception:
                # Shading is aesthetic; never fail the plot because of it.
                pass

        poly.set_facecolor(colors)
        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(float(linewidth))
        poly.set_alpha(alpha)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        if fvals is not None:
            sm.set_array(fvals)
        else:
            sm.set_array(np.array([]))
        return sm

    def _format_3d_axes_mm(self, ax: Axes3D) -> None:
        """GUI: show 3D axes in millimeters with readable ticks."""
        mm = FuncFormatter(lambda v, _pos: f"{v * 1000:.0f}")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(5))
            axis.set_major_formatter(mm)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        try:
            ax.tick_params(labelsize=8 * self.theme.font_scale)
        except Exception:
            pass
        ax.grid(False)

    def _stim_center_from_pair(self, pair_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Midpoint between stimulating electrodes (meters). Returns None if unresolved."""
        if self.montage is None:
            return None
        stim = (pair_name or (self.stim_pair_var.get() if hasattr(self, "stim_pair_var") else "")).strip()
        ch_pair = self._stim_channels_from_pair(stim)
        if ch_pair is None:
            return None
        lc, rc = ch_pair
        chp = self.montage.get_positions().get("ch_pos", {})
        p1 = np.asarray(chp.get(lc, (0.0, 0.0, 0.0)), dtype=float)
        p2 = np.asarray(chp.get(rc, (0.0, 0.0, 0.0)), dtype=float)
        if np.allclose(p1, 0.0) or np.allclose(p2, 0.0):
            return None
        return 0.5 * (p1 + p2)

    def _stim_channels_from_pair(self, pair_name: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Resolve a bipolar stimulation pair to the channel names used in the montage/raw data."""
        stim = (pair_name or (self.stim_pair_var.get() if hasattr(self, "stim_pair_var") else "")).strip()
        if not stim:
            return None
        a_tok, b_tok = self._split_pair(stim)
        a = self._resolve_stim_token(a_tok) or a_tok
        b = self._resolve_stim_token(b_tok) or b_tok
        if not a or not b:
            return None
        return a, b

    def _stim_geometry_from_display_positions(
        self,
        pair_name: Optional[str],
        pos_by_channel: Dict[str, np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, str]]]:
        """Return displayed stim-contact endpoints and midpoint.

        The stim marker is derived from the same displayed coordinates as the electrodes,
        so it remains centered on the selected bipole even when radial offsets or jitter are
        applied for visualization.
        """
        ch_pair = self._stim_channels_from_pair(pair_name)
        if ch_pair is None:
            return None
        a, b = ch_pair
        if a not in pos_by_channel or b not in pos_by_channel:
            return None
        p1 = np.asarray(pos_by_channel[a], dtype=float)
        p2 = np.asarray(pos_by_channel[b], dtype=float)
        if p1.shape != (3,) or p2.shape != (3,) or not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            return None
        center = 0.5 * (p1 + p2)
        return p1, p2, center, (a, b)

    @staticmethod
    def _top_roi_labels_from_values(
        channels: Sequence[str],
        values: np.ndarray,
        channel_to_region: Dict[str, str],
        n_top: int,
    ) -> List[str]:
        """Return ROI labels ranked by maximum response amplitude."""
        best: Dict[str, float] = {}
        for ch, val in zip(channels, values):
            region = str(channel_to_region.get(ch, "Unknown"))
            if not region or region == "Unknown":
                continue
            v = float(val) if np.isfinite(val) else 0.0
            if region not in best or v > best[region]:
                best[region] = v
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [r for r, _v in ranked[:max(1, int(n_top))]]

    @staticmethod
    def _style_3d_axes_for_export(ax: Axes3D) -> None:
        """Hide 3D axes/panes/ticks for clean presentation (GUI + export)."""
        ax.grid(False)
        try:
            ax.set_axis_off()
        except Exception:
            pass
        # Make 3D panes and gridlines transparent (mpl_toolkits.mplot3d specifics)
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
            if axis is None:
                continue
            try:
                pane = axis.pane
                pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
            try:
                axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _style_axes_for_export(self, ax: Axes3D) -> None:
        """Backward-compatible wrapper used by older call sites."""
        self._style_3d_axes_for_export(ax)




    def plot_3d_visualization(self) -> None:
        """
        GUI: Combined 3D template surface + electrode peak scatter (with stim-site marker).

        Notes
        -----
        - Electrode coordinates may be missing; channels without known coordinates are placed at (0,0,0) and then
          deterministically jittered so the user can still browse stim pairs in a consistent layout.
        - Hemisphere visibility toggles (Left/Right) remove that half of the template mesh (electrodes remain).
        - Axes/grid/labels are intentionally hidden (GUI matches export styling).
        """
        render_t0 = time.perf_counter()
        self.fig3d.clf()
        ax = self.fig3d.add_subplot(111, projection="3d")
        self.ax3d = ax

        ax.set_facecolor("#0b1020")

        if self.raw is None:
            ax.text(0.5, 0.5, 0.5, "Load data to view 3D.", color="#d0d8ff", ha="center", va="center")
            self._style_axes_for_export(ax)
            self.canvas3d.draw_idle()
            return

        montage = self.raw.get_montage()
        if montage is None:
            ax.text(0.5, 0.5, 0.5, "No montage / electrode positions.", color="#d0d8ff", ha="center", va="center")
            self._style_axes_for_export(ax)
            self.canvas3d.draw_idle()
            return

        pair = (self.stim_pair_var.get() or "").strip()
        mode = (self.analysis_mode_var.get() or "ERP").strip()
        roi_label = (self.roi_var.get() or "All ROIs").strip()
        view_mode = (self.view_window_var.get() or "Average (full window)").strip()

        # --- Electrode positions (meters) ---
        # IMPORTANT: do NOT drop (0,0,0) positions here. Many clinical channel labels are not in 10–20; we keep them
        # and later apply a deterministic offset/jitter so the GUI still renders a stable electrode layout.
        pos_by_ch: dict[str, np.ndarray] = {}
        for ch in self.raw.ch_names:
            loc = self._lookup_channel_pos(ch)
            if loc is None:
                continue
            loc = np.asarray(loc, dtype=float).reshape(-1)
            if loc.size != 3 or not np.isfinite(loc).all():
                continue
            pos_by_ch[ch] = loc

        matched = list(pos_by_ch.keys())
        if not matched:
            ax.text(0.5, 0.5, 0.5, "No valid electrode positions.", color="#d0d8ff", ha="center", va="center")
            self._style_axes_for_export(ax)
            self.canvas3d.draw_idle()
            return

        pos_arr = np.asarray([pos_by_ch[ch] for ch in matched], dtype=float)

        # --- Template mesh ---
        surface = (self.surface_var.get() or "pial").strip().lower()
        verts, faces_all = self._get_fsaverage_mesh(surface)
        brain_center = np.nanmean(verts, axis=0) if verts.size else np.zeros(3, dtype=float)

        # Pick a center for the deterministic jitter: prefer non-zero coordinates if available.
        norms = np.linalg.norm(pos_arr, axis=1)
        if np.any(norms > 1e-6):
            jitter_center = np.nanmean(pos_arr[norms > 1e-6], axis=0)
        else:
            jitter_center = brain_center

        pos_plot = self._offset_and_jitter_positions(pos_arr, matched, jitter_center)
        pos_plot_by_ch = {ch: pos_plot[i] for i, ch in enumerate(matched)}

        # --- Values per channel (peak |uV| in selected window) ---
        vals = np.zeros(len(matched), dtype=float)
        if pair and hasattr(self, "epochs_by_pair") and pair in self.epochs_by_pair:
            try:
                epochs = self.epochs_by_pair[pair]
                evk = self._compute_evoked_for_mode(epochs, mode)
                data_uv = np.asarray(evk.data, dtype=float) * 1e6  # V -> µV
                times_ms = np.asarray(evk.times, dtype=float) * 1000.0

                if view_mode.startswith("Early"):
                    t0, t1 = self._get_windows_ms()[0]
                    tmask = (times_ms >= t0) & (times_ms <= t1)
                elif view_mode.startswith("Late"):
                    t0, t1 = self._get_windows_ms()[1]
                    tmask = (times_ms >= t0) & (times_ms <= t1)
                else:
                    tmask = np.ones_like(times_ms, dtype=bool)

                name_to_idx = {n: i for i, n in enumerate(evk.ch_names)}
                for i, ch in enumerate(matched):
                    j = name_to_idx.get(ch)
                    if j is None:
                        continue
                    seg = data_uv[j, tmask]
                    if seg.size:
                        vals[i] = float(np.nanmax(np.abs(seg)))
            except Exception as exc:
                logging.getLogger(__name__).warning("3D values failed (pair=%s, mode=%s): %s", pair, mode, exc)

        # Robust vmax (avoid a single outlier flattening the map)
        vmax = float(np.nanpercentile(vals[vals > 0], 98)) if np.any(vals > 0) else 1.0
        vmax = max(vmax, 1.0)

        # ROI mask (electrodes outside ROI are faded). Special "Top responding" options
        # rank ROIs by the largest response in the currently selected mode/window.
        roi_mask: np.ndarray | None = None
        if roi_label and roi_label != "All ROIs":
            if roi_label == "Top 5 responding ROIs":
                top_labels = set(self._top_roi_labels_from_values(matched, vals, self._channel_to_region, 5))
                roi_mask = np.asarray([self._channel_to_region.get(ch, "Unknown") in top_labels for ch in matched], dtype=bool)
            elif roi_label == "Top 10 responding ROIs":
                top_labels = set(self._top_roi_labels_from_values(matched, vals, self._channel_to_region, 10))
                roi_mask = np.asarray([self._channel_to_region.get(ch, "Unknown") in top_labels for ch in matched], dtype=bool)
            else:
                roi_mask = np.asarray(
                    [self._channel_to_region.get(ch, "Unknown") == roi_label for ch in matched],
                    dtype=bool,
                )

        # save for click-selection bookkeeping
        self._roi_mask_3d = (roi_mask.tolist() if roi_mask is not None else [True] * len(matched))
        proj_mask = None
        if (
            hasattr(self, "project_only_selected_var")
            and bool(self.project_only_selected_var.get())
            and getattr(self, "selected_electrodes", None)
        ):
            sel = set(self.selected_electrodes)
            pm = np.asarray([ch in sel for ch in matched], dtype=bool)
            if pm.any():
                proj_mask = pm

        # --- Hemisphere visibility (mesh only) ---
        show_left = True if not hasattr(self, "show_left_hemi_var") else bool(self.show_left_hemi_var.get())
        show_right = True if not hasattr(self, "show_right_hemi_var") else bool(self.show_right_hemi_var.get())

        faces = faces_all
        if verts.size and faces_all.size and not (show_left and show_right):
            # Use the mesh's X extent to define the mid-sagittal plane.
            mid_x = float((np.nanmin(verts[:, 0]) + np.nanmax(verts[:, 0])) / 2.0)
            is_left_v = verts[:, 0] <= mid_x
            left_counts = is_left_v[faces_all].sum(axis=1)
            face_left = left_counts >= 2
            face_right = left_counts <= 1

            if show_left and not show_right:
                faces = faces_all[face_left]
            elif show_right and not show_left:
                faces = faces_all[face_right]
            else:
                faces = faces_all[:0]

        # --- Shared color mapping ---
        cmap = plt.get_cmap("plasma")
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

        # --- Draw mesh (if any faces visible) ---
        if verts.size and faces.size:
            try:
                pos_proj = pos_plot
                vals_proj = vals
                if proj_mask is not None:
                    pos_proj = pos_plot[proj_mask]
                    vals_proj = vals[proj_mask]
                vvals = self._project_to_vertices(verts, pos_proj, vals_proj)
                fvals = np.nanmean(vvals[faces], axis=1)

                poly = Poly3DCollection(
                    verts[faces],
                    linewidths=0.0,
                    antialiased=False,
                )
                mesh_opacity = float(self.mesh_opacity_var.get()) if hasattr(self, "mesh_opacity_var") else 0.35
                poly.set_alpha(max(0.02, min(1.0, mesh_opacity)))
                self._apply_shaded_mesh(poly, verts, faces, face_values=fvals, cmap_name=cmap, norm=norm, alpha=mesh_opacity)
                ax.add_collection3d(poly)
            except Exception as exc:
                logging.getLogger(__name__).warning("3D mesh render failed: %s", exc)

        # --- Draw electrodes ---
        cvals = vals.copy()
        sizes = np.full(len(matched), 18.0, dtype=float)
        alphas = np.full(len(matched), 1.0, dtype=float)
        if roi_mask is not None:
            alphas[~roi_mask] = 0.22
            sizes[~roi_mask] = 14.0
            cvals[~roi_mask] = 0.0
        if proj_mask is not None:
            alphas[~proj_mask] = np.minimum(alphas[~proj_mask], 0.12)
            sizes[~proj_mask] = np.minimum(sizes[~proj_mask], 10.0)

        sc = ax.scatter(
            pos_plot[:, 0],
            pos_plot[:, 1],
            pos_plot[:, 2],
            c=cvals,
            s=sizes,
            cmap=cmap,
            norm=norm,
            depthshade=False,
            edgecolors="none",
            picker=True,
        )

        self.scatter_3d = sc
        self._matched_channels_3d = matched
        self._matched_pos_plot = pos_plot
        self._matched_sizes = sizes.tolist()

        # per-point alpha (matplotlib workaround)
        try:
            fc = sc.get_facecolors()
            if fc is not None and len(fc) == len(alphas):
                fc[:, 3] = alphas
                sc.set_facecolors(fc)
        except Exception:
            pass

        # --- Stim marker: exact midpoint of the selected displayed bipole ---
        stim_geometry = self._stim_geometry_from_display_positions(pair, pos_plot_by_ch) if pair else None
        if stim_geometry is not None:
            p1, p2, stim_center, _stim_chans = stim_geometry
            ax.plot(
                [float(p1[0]), float(p2[0])],
                [float(p1[1]), float(p2[1])],
                [float(p1[2]), float(p2[2])],
                color="#ffdf4d",
                linewidth=2.0,
                alpha=0.95,
                zorder=9,
            )
            ax.scatter(
                [float(stim_center[0])],
                [float(stim_center[1])],
                [float(stim_center[2])],
                s=140,
                c="#ffdf4d",
                edgecolors="black",
                linewidths=1.0,
                depthshade=False,
                zorder=10,
            )

        # --- Colorbar (single, unified) ---
        try:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = self.fig3d.colorbar(sm, ax=ax, shrink=0.62, pad=0.02, fraction=0.05)
            cbar.set_label("Peak |µV| (current window)", color="#cbd3ff")
            cbar.ax.yaxis.set_tick_params(color="#cbd3ff")
            plt.setp(cbar.ax.get_yticklabels(), color="#cbd3ff")
        except Exception:
            pass

        # --- Camera / extents ---
        try:
            # Use either mesh or electrode extents for equal scaling.
            bbox_pts = verts if (verts.size and faces.size) else pos_plot
            self._set_axes_equal_3d(ax, bbox_pts)
        except Exception:
            pass

        ax.set_title(f"{pair or 'All pairs'} | {mode} | {roi_label} | {view_mode}", color="#d0d8ff", fontsize=10)

        # Hide axes/grid/labels to match export styling.
        self._style_axes_for_export(ax)

        self._record_workflow_timing(
            "visualization_generation",
            time.perf_counter() - render_t0,
            pair=pair or "",
            mode=mode,
            roi=roi_label,
            view=view_mode,
        )
        self.canvas3d.draw_idle()


    def _update_scatter_colors(self) -> None:
        if self.scatter_3d is None or not self.canvas3d:
            return

        cmap, norm = self.scatter_3d.get_cmap(), self.scatter_3d.norm
        arr = self.scatter_3d.get_array()
        if arr is None:
            return

        base = np.asarray(cmap(norm(arr)))

        # Grey out non-ROI electrodes (context)
        if self._roi_mask_3d and len(self._roi_mask_3d) == len(base):
            roi_mask = np.asarray(self._roi_mask_3d, dtype=bool)
            base[~roi_mask, :3] = np.array([0.72, 0.72, 0.72])
            base[~roi_mask, 3] = 0.70

        # Selected electrodes in bright red
        for i, ch in enumerate(self._matched_channels_3d):
            if ch in self.selected_electrodes:
                base[i] = [1.0, 0.0, 0.0, 1.0]

        self.scatter_3d.set_facecolors(base)
        self._update_selection_overlay()
        self.canvas3d.draw_idle()

    def _update_selection_overlay(self) -> None:
        if self.ax3d is None or self._matched_pos_plot is None:
            return
        if self.scatter_sel is not None:
            try:
                self.scatter_sel.remove()
            except Exception:
                pass
            self.scatter_sel = None
        if not self.selected_electrodes:
            return
        idxs = [i for i, ch in enumerate(self._matched_channels_3d) if ch in self.selected_electrodes]
        if not idxs:
            return
        pos_sel = self._matched_pos_plot[idxs]
        size_sel = (np.asarray([self._matched_sizes[i] for i in idxs]) * 1.35).tolist()
        self.scatter_sel = self.ax3d.scatter(
            pos_sel[:, 0], pos_sel[:, 1], pos_sel[:, 2], s=size_sel, facecolor="none", edgecolor="#ff4136", linewidth=1.3, zorder=4
        )

    def _on_3d_pick(self, event: Any) -> Any:
        if event.artist != self.scatter_3d:
            return
        key = (getattr(event.mouseevent, "key", "") or "").lower()
        multi = ("shift" in key) or ("control" in key)
        for ind in event.ind:  # type: ignore[attr-defined]
            ch = self._matched_channels_3d[ind]
            if multi:
                (self.selected_electrodes.remove(ch) if ch in self.selected_electrodes else self.selected_electrodes.add(ch))
            else:
                self.selected_electrodes = set() if (ch in self.selected_electrodes and len(self.selected_electrodes) == 1) else {ch}
            self._update_status(f"Selected: {self._strip_pol(ch)}  (use Shift/Ctrl for multi-select)")

        self._refresh_selection_list()
        self._update_scatter_colors()
        self.selected_channel = next(iter(self.selected_electrodes)) if len(self.selected_electrodes) == 1 else None
        if self.selected_channel:
            self.update_detailed_view(self.selected_channel)

    def on_listbox_select(self, _event: Any) -> None:
        if not self.selection_listbox or self._suspend_listbox_callback:
            return

        idxs = list(self.selection_listbox.curselection())
        sel_channels: List[str] = []
        for i in idxs:
            disp = str(self.selection_listbox.get(i))
            ch = self._display_to_channel.get(disp)
            if ch and self.last_evoked and ch in self.last_evoked.ch_names:
                sel_channels.append(ch)

        self.selected_electrodes = set(sel_channels)

        # Sync 3D highlight (if available)
        try:
            self._update_scatter_colors()
        except Exception:
            pass

        # Update detailed waveform. If multiple electrodes are selected, show a montage.
        if len(sel_channels) == 1:
            self.selected_channel = sel_channels[0]
            self.update_detailed_view(self.selected_channel)
        else:
            self.selected_channel = sel_channels[-1] if sel_channels else None
            self.update_detailed_view(None)

    def _populate_channel_listbox(self) -> None:
        """Populate the channel listbox with *all* channels (POL-stripped for display)."""
        if not self.selection_listbox:
            return
        self.selection_listbox.delete(0, tk.END)
        self._display_to_channel.clear()
        self._channel_to_listbox_index.clear()
        if self.last_evoked is None:
            return

        for ch in self.last_evoked.ch_names:
            base = self._strip_pol(ch)
            disp = base
            n = 2
            while disp in self._display_to_channel:
                disp = f"{base} ({n})"
                n += 1
            idx = self.selection_listbox.size()
            self.selection_listbox.insert(tk.END, disp)
            self._display_to_channel[disp] = ch
            self._channel_to_listbox_index[ch] = idx

    def _refresh_selection_list(self) -> None:
        """Sync listbox selection to `self.selected_electrodes` without rebuilding the list."""
        if not self.selection_listbox:
            return
        if self.selection_listbox.size() == 0 and self.last_evoked is not None:
            self._populate_channel_listbox()

        self._suspend_listbox_callback = True
        try:
            self.selection_listbox.selection_clear(0, tk.END)
            for ch in sorted(self.selected_electrodes, key=lambda c: (self.last_evoked.ch_names.index(c) if self.last_evoked and c in self.last_evoked.ch_names else 10**9)):
                idx = self._channel_to_listbox_index.get(ch)
                if idx is not None:
                    self.selection_listbox.selection_set(idx)
        finally:
            self._suspend_listbox_callback = False

    # ----- Mesh helpers -----
    def _get_fsaverage_mesh(self, surface: str) -> Tuple[np.ndarray, np.ndarray]:
        surf_key = surface if surface in SURFACE_KEY_TO_LABEL else "pial"
        cached = getattr(self._surf_cache, surf_key, None)
        if cached is not None:
            return cached

        fs_dir = self._fsaverage_dir
        if fs_dir is None:
            try:
                fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            except TypeError:
                fs_dir = mne.datasets.fetch_fsaverage()
            self._fsaverage_dir = fs_dir
        lh_path = Path(fs_dir) / "surf" / f"lh.{surf_key}"
        rh_path = Path(fs_dir) / "surf" / f"rh.{surf_key}"
        v_lh, f_lh = mne.read_surface(str(lh_path))
        v_rh, f_rh = mne.read_surface(str(rh_path))
        v_lh, v_rh = v_lh / 1000.0, v_rh / 1000.0

        ntri_lh = max(int(f_lh.shape[0] * 0.01), 2000)
        ntri_rh = max(int(f_rh.shape[0] * 0.01), 2000)
        v_lh, f_lh = decimate_surface(v_lh, f_lh, n_triangles=ntri_lh)
        v_rh, f_rh = decimate_surface(v_rh, f_rh, n_triangles=ntri_rh)
        f_rh = f_rh + v_lh.shape[0]

        verts = np.vstack((v_lh, v_rh))
        faces = np.vstack((f_lh, f_rh))
        setattr(self._surf_cache, surf_key, (verts, faces))
        return verts, faces

    def _current_surface_key(self) -> str:
        return SURFACE_LABEL_TO_KEY.get(self.surface_var.get().strip(), "pial")

    @staticmethod
    def _surface_display_name(surface: str) -> str:
        return SURFACE_KEY_TO_LABEL.get(surface, SURFACE_KEY_TO_LABEL["pial"])

    @staticmethod
    def _surface_short_label(surface: str) -> str:
        return {"pial": "Pial", "white": "White", "inflated": "Inflated"}.get(surface, surface.capitalize())

    def _on_surface_change(self, _event=None) -> None:
        self.plot_3d_visualization()

    def _on_mesh_opacity_change(self) -> None:
        """Update only the mesh transparency (GUI)."""
        alpha = float(self.mesh_opacity_var.get()) if self.mesh_opacity_var is not None else 0.35
        poly = getattr(self, "mesh_poly", None)
        if poly is None:
            return

        try:
            fc = poly.get_facecolor()
            if fc is None or len(fc) == 0:
                poly.set_alpha(alpha)
            else:
                arr = np.array(fc, copy=True)
                if arr.ndim == 1:
                    if arr.shape[0] >= 4:
                        arr[3] = alpha
                    poly.set_facecolor(arr)
                else:
                    if arr.shape[1] >= 4:
                        arr[:, 3] = alpha
                    poly.set_facecolor(arr)

            if self.canvas3d:
                self.canvas3d.draw_idle()
        except Exception:
            # Fallback: full redraw
            self.plot_3d_visualization()

    def _on_roi_selected(self, _event=None) -> None:
        self.plot_3d_visualization()

    @staticmethod
    def _window_mask(times_ms: np.ndarray, t0: float, t1: float) -> np.ndarray:
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        return (times_ms >= lo) & (times_ms <= hi)

    def _get_windows_ms(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        try:
            e0, e1 = float(self.early_start_ms_var.get()), float(self.early_end_ms_var.get())
            l0, l1 = float(self.late_start_ms_var.get()), float(self.late_end_ms_var.get())
        except Exception:
            e0, e1, l0, l1 = 10.0, 80.0, 80.0, 300.0
        if e1 < e0:
            e0, e1 = e1, e0
        if l1 < l0:
            l0, l1 = l1, l0
        return (e0, e1), (l0, l1)


    def _current_view_window_ms(self) -> Optional[Tuple[float, float]]:
        """Return the currently selected peak window in ms (or None for full-epoch Average)."""
        sel = getattr(self, "view_window_var", None)
        choice = sel.get() if sel is not None else "Average"
        if choice == "Early":
            early, _late = self._get_windows_ms()
            return early
        if choice == "Late":
            _early, late = self._get_windows_ms()
            return late
        return None

    def _current_view_label(self) -> str:
        """Human-readable label for the current view window."""
        sel = getattr(self, "view_window_var", None)
        choice = sel.get() if sel is not None else "Average"
        win = self._current_view_window_ms()
        if win is None:
            return "Average (full epoch)"
        return f"{choice} ({win[0]:.0f}–{win[1]:.0f} ms)"


    def _roi_positions_and_values(
        self, evoked: mne.Evoked, window_ms: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.electrode_coords is None or self.raw is None:
            return np.empty((0, 3)), np.empty((0,))

        data = evoked.data
        if window_ms is not None:
            mask = self._window_mask(evoked.times * 1000.0, *window_ms)
            data = data[:, mask] if mask.any() else np.zeros((data.shape[0], 0))

        peaks_uv = (np.max(np.abs(data), axis=1) * 1e6) if data.shape[1] else np.zeros(len(evoked.ch_names), float)
        names = evoked.ch_names
        idx_map = {ch: i for i, ch in enumerate(names)}

        roi_choice = self.roi_var.get()
        top_labels: Optional[set[str]] = None
        if roi_choice == "Top 5 responding ROIs":
            top_labels = set(self._top_roi_labels_from_values(names, peaks_uv, self._channel_to_region, 5))
        elif roi_choice == "Top 10 responding ROIs":
            top_labels = set(self._top_roi_labels_from_values(names, peaks_uv, self._channel_to_region, 10))

        df = self.electrode_coords
        if roi_choice not in ("All ROIs", "Top 5 responding ROIs", "Top 10 responding ROIs"):
            df = df[df["Regions"] == roi_choice]
        if df.empty:
            return np.empty((0, 3)), np.empty((0,))

        pos, vals = [], []
        for _, row in df.iterrows():
            idx1 = int(row["ElecNumber"])
            ch = self._elecnum_to_channel.get(idx1)
            if not ch or ch not in idx_map:
                continue
            if top_labels is not None and self._channel_to_region.get(ch, "Unknown") not in top_labels:
                continue
            vals.append(float(peaks_uv[idx_map[ch]]))
            pos.append((row["X"] / 1000.0, row["Y"] / 1000.0, row["Z"] / 1000.0))
        return (np.asarray(pos) if pos else np.empty((0, 3))), (np.asarray(vals) if vals else np.empty((0,)))

    @staticmethod
    def _project_to_vertices(verts: np.ndarray, elec_pos: np.ndarray, elec_vals: np.ndarray) -> np.ndarray:
        _, idx = cKDTree(elec_pos).query(verts)
        return elec_vals[idx]

    @staticmethod
    def _compress_raw_roi_values(values: np.ndarray, ratio_trigger: float = 3.5) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        if values.size < 4:
            return values, None
        abs_vals = np.abs(values)
        max_abs = float(abs_vals.max())
        if max_abs <= 0:
            return values, None
        sorted_abs = np.sort(abs_vals)
        idx_90 = int(np.floor(0.9 * (sorted_abs.size - 1)))
        baseline = float(sorted_abs[idx_90]) if sorted_abs.size > 1 else max_abs
        if baseline <= 0 and sorted_abs.size >= 2:
            baseline = float(sorted_abs[-2])
        if baseline <= 0:
            return values, None
        ratio = max_abs / (baseline + 1e-9)
        if ratio <= 1.5:
            return values, None

        clip_limit = baseline
        if ratio >= ratio_trigger and sorted_abs.size > 2:
            idx_75 = int(np.floor(0.75 * (sorted_abs.size - 1)))
            cand = float(sorted_abs[idx_75])
            if cand > 0:
                clip_limit = max(cand, baseline * 0.8)

        clip_limit = max(clip_limit, 1e-6)
        if max_abs <= clip_limit * 1.05:
            return values, None

        lower = float(values.min())
        if np.any(values < 0):
            lower = max(lower, -clip_limit)

        clipped = np.clip(values, lower, clip_limit)
        return (values, None) if np.allclose(clipped, values) else (clipped, (lower, clip_limit))

    # ----- Plotting: Template mesh ROI projection -----
    def plot_template_mesh(self) -> None:
        # Backwards-compatible alias: the GUI now renders a single combined 3D scene.
        self.plot_3d_visualization()

    def _compute_early_late_metrics(
        self,
        epochs: mne.Epochs,
        evoked: mne.Evoked,
        mode: str,
        stim_pair: str,
        early_ms: Tuple[float, float],
        late_ms: Tuple[float, float],
    ) -> pd.DataFrame:
        times_ms = evoked.times * 1000.0
        me, ml = self._window_mask(times_ms, *early_ms), self._window_mask(times_ms, *late_ms)
        trials_v = epochs.get_data()
        if mode == "Gamma":
            trials_v = self._compute_gamma_envelope(trials_v, fs=float(epochs.info["sfreq"]))
        trials_uv = trials_v * 1e6
        ev_uv = evoked.data * 1e6

        rows = []
        for i, ch in enumerate(evoked.ch_names):
            region = self._channel_to_region.get(ch, "Unknown")
            if not me.any() or not ml.any():
                rows.append(
                    dict(
                        pair=stim_pair,
                        mode=mode,
                        channel=ch,
                        region=region,
                        early_start_ms=early_ms[0],
                        early_end_ms=early_ms[1],
                        late_start_ms=late_ms[0],
                        late_end_ms=late_ms[1],
                        early_mean_abs_uv=np.nan,
                        late_mean_abs_uv=np.nan,
                        early_peak_abs_uv=np.nan,
                        early_peak_lat_ms=np.nan,
                        early_auc_abs_uv_ms=np.nan,
                        late_peak_abs_uv=np.nan,
                        late_peak_lat_ms=np.nan,
                        late_auc_abs_uv_ms=np.nan,
                        t_stat=np.nan,
                        p_value=np.nan,
                        cohens_d_paired=np.nan,
                    )
                )
                continue

            s = ev_uv[i]
            s_e, t_e = s[me], times_ms[me]
            s_l, t_l = s[ml], times_ms[ml]

            early_peak = float(np.max(np.abs(s_e))) if s_e.size else np.nan
            early_lat = float(t_e[int(np.argmax(np.abs(s_e)))]) if s_e.size else np.nan
            early_auc = float(np.trapz(np.abs(s_e), t_e)) if s_e.size else np.nan

            late_peak = float(np.max(np.abs(s_l))) if s_l.size else np.nan
            late_lat = float(t_l[int(np.argmax(np.abs(s_l)))]) if s_l.size else np.nan
            late_auc = float(np.trapz(np.abs(s_l), t_l)) if s_l.size else np.nan

            x = trials_uv[:, i, :]
            e_vals = np.mean(np.abs(x[:, me]), axis=1)
            l_vals = np.mean(np.abs(x[:, ml]), axis=1)
            early_mean, late_mean = float(np.mean(e_vals)), float(np.mean(l_vals))

            if len(e_vals) > 1 and np.all(np.isfinite(e_vals)) and np.all(np.isfinite(l_vals)):
                tstat, pval = stats.ttest_rel(e_vals, l_vals, nan_policy="omit")
                diff = e_vals - l_vals
                dcohen = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
                tval, pval = float(tstat), float(pval)
            else:
                tval, pval, dcohen = np.nan, np.nan, np.nan

            rows.append(
                dict(
                    pair=stim_pair,
                    mode=mode,
                    channel=ch,
                    region=region,
                    early_start_ms=early_ms[0],
                    early_end_ms=early_ms[1],
                    late_start_ms=late_ms[0],
                    late_end_ms=late_ms[1],
                    early_mean_abs_uv=early_mean,
                    late_mean_abs_uv=late_mean,
                    early_peak_abs_uv=early_peak,
                    early_peak_lat_ms=early_lat,
                    early_auc_abs_uv_ms=early_auc,
                    late_peak_abs_uv=late_peak,
                    late_peak_lat_ms=late_lat,
                    late_auc_abs_uv_ms=late_auc,
                    t_stat=tval,
                    p_value=pval,
                    cohens_d_paired=dcohen,
                )
            )
        return pd.DataFrame(rows)

    def _evoked_masked_to_window(self, epochs: mne.Epochs, mode: str, window_ms: Tuple[float, float]) -> mne.Evoked:
        evk = self._compute_evoked_for_mode(epochs, mode)
        mask = self._window_mask(evk.times * 1000.0, *window_ms)
        evk2 = evk.copy()
        dat = evk2.data.copy()
        dat[:, ~mask] = 0.0 if mask.any() else 0.0
        evk2.data = dat
        return evk2

    # ----- Export -----
    @staticmethod
    def _sanitize_pair_name(sp: str) -> str:
        return re.sub(r"[^A-Za-z0-9\-]+", "_", sp)

    @staticmethod
    def _pick_preview(paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        pref = {".png": 0, ".svg": 1, ".pdf": 2}
        return sorted(paths, key=lambda p: pref.get(p.suffix.lower(), 9))[0]

    def _save_figure_set(self, fig: plt.Figure, base_path: Path) -> List[Path]:
        self.export_opts.ensure_valid()
        out_paths: List[Path] = []
        for fmt in self.export_opts.formats:
            out_path = base_path.with_suffix(f".{fmt}")
            kwargs: Dict[str, Any] = dict(dpi=self.export_opts.dpi, transparent=self.export_opts.transparent)
            if fmt == "pdf":
                kwargs["metadata"] = dict(
                    Title=base_path.name,
                    Author="iEEG ERP Analyzer",
                    Creator="iEEG ERP Analyzer",
                    Subject="Publication-quality export",
                    CreationDate=datetime.now(),
                )
            elif fmt == "svg":
                kwargs["metadata"] = dict(
                    Title=base_path.name,
                    Description="Publication-quality export",
                    Creator="iEEG ERP Analyzer",
                    Date=datetime.now().isoformat(timespec="seconds"),
                )
            fig.savefig(out_path, **kwargs)
            out_paths.append(out_path)
        return out_paths

    def _save_plotly_html(self, fig: "go.Figure", path: Path) -> None:
        if pio is None:
            raise RuntimeError("Plotly I/O not available.")
        path.parent.mkdir(parents=True, exist_ok=True)
        include_js: Any = True if self.export_opts.html_inline_js else "cdn"
        pio.write_html(fig, file=str(path), include_plotlyjs=include_js, full_html=True)

    def _build_grid_figure(self, evoked: mne.Evoked, epochs: mne.Epochs, stim_pair: str, mode: str) -> plt.Figure:
        self.theme.apply()
        fig = plt.Figure(figsize=(10.0, 8.0), dpi=100, constrained_layout=True)
        n_ch = len(evoked.ch_names)
        n_cols = int(np.ceil(np.sqrt(n_ch)))
        n_rows = int(np.ceil(n_ch / n_cols))
        times = evoked.times * 1000.0

        win = self._current_view_window_ms()
        view_lbl = self._current_view_label()

        if mode == "Gamma":
            env = self._compute_gamma_envelope(epochs.get_data(), fs=float(epochs.info["sfreq"])) * 1e6
            ep_uv, ev_uv = env, env.mean(axis=0)
        else:
            ep_uv, ev_uv = epochs.get_data() * 1e6, evoked.data * 1e6

        for idx in range(n_ch):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            y = ev_uv[idx]
            ax.plot(times, y, "-", linewidth=self.theme.erp_line_width)
            if len(epochs) > 1:
                ci = self._ci95(ep_uv[:, idx, :], mode)
                ax.fill_between(times, y - ci, y + ci, alpha=self.theme.ci_alpha, linewidth=0)
            if self.theme.show_zero_time:
                ax.axvline(0.0, lw=0.8, ls="--", color="#9c9c9c", zorder=0)

            if win is not None:
                ax.axvspan(win[0], win[1], color="#ffddaa", alpha=0.12, zorder=0)

            ax.tick_params(labelsize=6 * self.theme.font_scale)
            if idx % n_cols == 0:
                ax.set_ylabel("µV")
            if idx >= n_ch - n_cols:
                ax.set_xlabel("ms")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

        fig.suptitle(f"{mode} · Pair {stim_pair}  (n={len(epochs)} trials) — {view_lbl}", fontsize=10 * self.theme.font_scale)
        return fig

    def _build_scatter_for_evoked(self, evoked: mne.Evoked, mode: str) -> Tuple[plt.Figure, Any]:
        self.theme.apply()
        if self.montage is None:
            raise ValueError("No montage available.")
        fig = plt.Figure(figsize=(6.4, 5.2), dpi=100, constrained_layout=False)
        ax = fig.add_subplot(111, projection="3d")
        verts, faces = self._get_fsaverage_mesh("pial")
        mesh = Poly3DCollection(verts[faces])
        ax.add_collection3d(mesh)
        self._apply_shaded_mesh(mesh, verts, faces, alpha=0.22, base_color="lightgrey")
        self._set_axes_equal_3d(ax, verts)

        ch_pos = self.montage.get_positions()["ch_pos"]
        names = evoked.ch_names
        matched, pos = [], []
        for ch in names:
            if ch in ch_pos and not np.allclose(ch_pos[ch], [0, 0, 0]):
                matched.append(ch)
                pos.append(ch_pos[ch])
        if not pos:
            raise ValueError("No valid electrode positions for scatter export.")
        pos = np.asarray(pos)

        peaks = np.max(np.abs(evoked.data), axis=1) * 1e6
        amps = np.asarray([peaks[names.index(ch)] for ch in matched])
        vmin, vmax = float(amps.min()), float(amps.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        pos_plot = self._offset_and_jitter_positions(pos, matched, verts.mean(axis=0))
        s = 20.0 + 160.0 * (amps - vmin) / (vmax - vmin + 1e-12)

        ax.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2], s=s * 1.45, c="white", alpha=0.85, edgecolor="none", zorder=2)
        sc = ax.scatter(
            pos_plot[:, 0],
            pos_plot[:, 1],
            pos_plot[:, 2],
            c=amps,
            cmap=plt.get_cmap(self.theme.colormap),
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            s=s,
            edgecolor="k",
            linewidth=0.35,
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.55, aspect=12)
        cbar.set_label("Peak |µV|")
        cbar.ax.set_in_layout(False)
        ax.set_title(f"3D Electrode Visualization — {mode} (Export)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=90, azim=0)
        self._draw_axes_triad(ax)
        return fig, ax

    def _determine_raw_mesh_limits(
        self, surface: str, evoked_windows: Iterable[Tuple[Optional[mne.Evoked], Optional[Tuple[float, float]]]]
    ) -> Optional[Tuple[float, float]]:
        if self.montage is None:
            return None
        try:
            verts, faces = self._get_fsaverage_mesh(surface)
        except Exception:
            LOG.exception("Failed to fetch fsaverage mesh for raw limit computation.")
            return None

        mins, maxs = [], []
        for evk, window in evoked_windows:
            if evk is None:
                continue
            pos_roi, vals_roi = self._roi_positions_and_values(evk, window_ms=window)
            if pos_roi.size == 0 or vals_roi.size == 0:
                continue
            adjusted, _ = self._compress_raw_roi_values(vals_roi.copy())
            vvals = self._project_to_vertices(verts, pos_roi, adjusted)
            fvals = np.mean(vvals[faces], axis=1)
            mins.append(float(np.min(fvals)))
            maxs.append(float(np.max(fvals)))

        if not mins:
            return None
        vmin, vmax = float(min(mins)), float(max(maxs))
        if np.isclose(vmin, vmax):
            vmax = vmin + max(1.0, abs(vmin) * 0.05 + 1e-6)
        return vmin, vmax

    def _build_mesh_for_evoked(
        self,
        evoked: mne.Evoked,
        surface: str,
        window_ms: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        color_limits: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, Any]:
        self.theme.apply()
        fig = plt.Figure(figsize=(6.4, 5.2), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf_key = surface if surface in SURFACE_KEY_TO_LABEL else "pial"
        verts, faces = self._get_fsaverage_mesh(surf_key)
        poly = Poly3DCollection(verts[faces])
        ax.add_collection3d(poly)
        self._set_axes_equal_3d(ax, verts)

        sm, center, clip_info = None, None, None
        orig_min = orig_max = None

        if self.montage is None:
            self._apply_shaded_mesh(poly, verts, faces, alpha=0.90, base_color="lightsteelblue")
        else:
            pos_roi, vals_roi = self._roi_positions_and_values(evoked, window_ms=window_ms)
            if pos_roi.size == 0:
                self._apply_shaded_mesh(poly, verts, faces, alpha=0.90, base_color="lightsteelblue")
            else:
                center = self._stim_center_from_pair()

                if normalize:
                    d2 = np.sum((pos_roi - (center if center is not None else 0)) ** 2, axis=1)
                    eps = 1e-7
                    amp_z = (vals_roi - vals_roi.mean()) / (vals_roi.std() + eps)
                    dist_z = (d2 - d2.mean()) / (d2.std() + eps)
                    final = amp_z / (dist_z + eps)
                else:
                    if vals_roi.size:
                        orig_min, orig_max = float(vals_roi.min()), float(vals_roi.max())
                    final, clip_info = self._compress_raw_roi_values(vals_roi.copy())

                vvals = self._project_to_vertices(verts, pos_roi, final)
                fvals = np.mean(vvals[faces], axis=1)

                if color_limits is None:
                    vmin, vmax = float(fvals.min()), float(fvals.max())
                else:
                    vmin, vmax = color_limits if color_limits[0] <= color_limits[1] else (color_limits[1], color_limits[0])

                if np.isclose(vmin, vmax):
                    vmax = vmin + (1e-6 if normalize else max(1.0, abs(vmin) * 0.05 + 1e-6))
                norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if (normalize and vmin < 0 < vmax) else plt.Normalize(vmin=vmin, vmax=vmax)
                sm = self._apply_shaded_mesh(poly, verts, faces, face_values=fvals, cmap_name=self.theme.colormap, norm=norm, alpha=0.95)
                self._mark_stim_center(ax, center, brain_center=verts.mean(axis=0))

        lohi = f" — {window_ms[0]:.0f}–{window_ms[1]:.0f} ms" if window_ms else ""
        surface_label = self._surface_display_name(surf_key)
        ax.set_title(f"{surface_label} ROI — {'Normalized' if normalize else 'Raw µV'}{lohi}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=30, azim=-60)

        if sm is not None:
            label = "Normalized amplitude" if normalize else "Peak amplitude (µV)"
            if window_ms:
                label += f"\n{window_ms[0]:.0f}–{window_ms[1]:.0f} ms"
            if (not normalize) and clip_info is not None:
                lo_clip, hi_clip = clip_info
                clip_lines = []
                if orig_max is not None and hi_clip < orig_max - 1e-6:
                    clip_lines.append(f"High clip ≤ {hi_clip:.2f} µV")
                if orig_min is not None and lo_clip > orig_min + 1e-6:
                    clip_lines.append(f"Low clip ≥ {lo_clip:.2f} µV")
                if clip_lines:
                    label += "\n" + " / ".join(clip_lines)
            fig.colorbar(sm, ax=ax, shrink=0.55, aspect=12).set_label(label)

        self._draw_axes_triad(ax)
        return fig, ax

    # ----- Plotly helpers -----
    def _mpl_to_plotly_colorscale(self, name: str, n: int = 256) -> List[List[Any]]:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(name, n)
        return [[i / (n - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"] for i, (r, g, b, _a) in enumerate(cmap(range(n)))]

    def _plotly_layout(self, title: str) -> Dict[str, Any]:
        return dict(
            title=title,
            margin=dict(l=0, r=0, t=50, b=0),
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
            template="plotly_white",
        )

    def _plotly_scatter_for_evoked(self, evoked: mne.Evoked, mode: str) -> "go.Figure":
        if go is None:
            raise RuntimeError("Plotly not available.")
        if self.montage is None:
            raise RuntimeError("No montage available.")
        verts, faces = self._get_fsaverage_mesh("pial")
        x, y, z = verts.T
        i, j, k = faces.T

        brain = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color="lightgrey",
            opacity=0.08,
            name="Template brain",
            lighting=dict(ambient=0.9, diffuse=0.8, specular=0.0, roughness=1.0),
            flatshading=False,
            showscale=False,
        )

        ch_pos = self.montage.get_positions()["ch_pos"]
        names = evoked.ch_names
        matched, pos = [], []
        for ch in names:
            if ch in ch_pos and not np.allclose(ch_pos[ch], [0, 0, 0]):
                matched.append(ch)
                pos.append(ch_pos[ch])
        if not pos:
            raise ValueError("No valid electrode positions for interactive scatter.")
        pos = np.asarray(pos)

        center = verts.mean(axis=0)
        vecs = pos - center
        pos_plot = pos + (vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)) * PLOTLY_ELECTRODE_OFFSET_M

        peaks_uv = np.max(np.abs(evoked.data), axis=1) * 1e6
        amps = np.asarray([peaks_uv[names.index(ch)] for ch in matched])
        vmin, vmax = float(amps.min()), float(amps.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        sizes = 6.0 + 18.0 * (amps - vmin) / (vmax - vmin + 1e-12)

        scatter = go.Scatter3d(
            x=pos_plot[:, 0],
            y=pos_plot[:, 1],
            z=pos_plot[:, 2],
            mode="markers",
            text=[self._strip_pol(ch) for ch in matched],
            marker=dict(
                size=sizes,
                color=amps,
                colorscale=self._mpl_to_plotly_colorscale(self.theme.colormap),
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title="Peak |µV|"),
                line=dict(width=0.9, color="black"),
                opacity=1.0,
            ),
            name="Electrodes",
            hovertemplate="<b>%{text}</b><br>Peak: %{marker.color:.2f} µV<extra></extra>",
        )
        fig = go.Figure(data=[brain, scatter])
        fig.update_layout(**self._plotly_layout(f"Interactive — 3D Electrode Scatter ({mode})"))
        fig.update_scenes(camera=dict(eye=dict(x=1.2, y=0.0, z=0.6)))
        return fig

    def _plotly_mesh_for_evoked(
        self, evoked: mne.Evoked, surface: str, window_ms: Optional[Tuple[float, float]] = None, normalize: bool = True
    ) -> "go.Figure":
        if go is None:
            raise RuntimeError("Plotly not available.")
        surf_key = surface if surface in SURFACE_KEY_TO_LABEL else "pial"
        surface_label = self._surface_display_name(surf_key)
        verts, faces = self._get_fsaverage_mesh(surf_key)
        x, y, z = verts.T
        i, j, k = faces.T

        lighting = dict(ambient=0.9, diffuse=0.9, specular=0.0, roughness=1.0)

        if self.montage is None:
            mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="lightsteelblue", opacity=0.90, lighting=lighting, showscale=False)
            fig = go.Figure(data=[mesh])
            title = "Interactive — ROI (Plain)"
        else:
            pos_roi, vals_roi = self._roi_positions_and_values(evoked, window_ms=window_ms)
            if pos_roi.size == 0:
                mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="lightsteelblue", opacity=0.90, lighting=lighting, showscale=False)
                fig = go.Figure(data=[mesh])
                title = "Interactive — ROI (Plain)"
            else:
                center = None
                center = self._stim_center_from_pair()

                d2 = np.sum((pos_roi - (center if center is not None else 0)) ** 2, axis=1)
                eps = 1e-7
                if normalize:
                    amp_z = (vals_roi - vals_roi.mean()) / (vals_roi.std() + eps)
                    dist_z = (d2 - d2.mean()) / (d2.std() + eps)
                    final = amp_z / (dist_z + eps)
                    label_base, title_suffix = "Normalized amplitude", "Normalized"
                else:
                    final = vals_roi
                    label_base, title_suffix = "Peak amplitude (µV)", "Raw µV"

                vvals = self._project_to_vertices(verts, pos_roi, final)
                vmin, vmax = float(vvals.min()), float(vvals.max())
                if np.isclose(vmin, vmax):
                    vmax = vmin + (1e-6 if normalize else max(1.0, abs(vmin) * 0.05 + 1e-6))

                mesh = go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    intensity=vvals,
                    colorscale=self._mpl_to_plotly_colorscale(self.theme.colormap),
                    cmin=vmin,
                    cmax=vmax,
                    opacity=0.95,
                    lighting=lighting,
                    flatshading=False,
                    showscale=True,
                    colorbar=dict(
                        title=label_base if window_ms is None else f"{label_base}<br>{window_ms[0]:.0f}–{window_ms[1]:.0f} ms"
                    ),
                    name="ROI projection",
                )
                fig = go.Figure(data=[mesh])
                title = f"Interactive — ROI Projection ({title_suffix})"

        window_txt = f" ({window_ms[0]:.0f}–{window_ms[1]:.0f} ms)" if window_ms else ""
        fig.update_layout(**self._plotly_layout(f"{title} — {surface_label}{window_txt}"))
        fig.update_scenes(camera=dict(eye=dict(x=1.1, y=-0.6, z=0.6)))
        return fig

    # ----- ROI bar figures -----
    def _fig_roi_bar(self, df: pd.DataFrame, pair: str, mode: str, which: str = "early") -> Optional[plt.Figure]:
        if "region" not in df.columns or df["region"].isna().all():
            return None
        col = "early_mean_abs_uv" if which.lower().startswith("e") else "late_mean_abs_uv"
        label = "Early (mean |µV|)" if col == "early_mean_abs_uv" else "Late (mean |µV|)"
        g = df.groupby("region", dropna=False).agg(val=(col, "mean"), n=("channel", "count")).reset_index()
        g = g.sort_values("val", ascending=False).head(20)
        fig = plt.Figure(figsize=(6.6, 5.0), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.barh(g["region"].astype(str), g["val"])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title(f"ROI bars — {which.capitalize()} — {mode} — {pair}")
        for i, v in enumerate(g["val"].values):
            ax.text(v if v >= 0 else 0, i, f" {v:.2f}", va="center", ha="left", fontsize=8)
        return fig

    def _fig_roi_diff_bar(self, df: pd.DataFrame, pair: str, mode: str) -> Optional[plt.Figure]:
        if "region" not in df.columns or df["region"].isna().all():
            return None
        roi = df.groupby("region", dropna=False).agg(early=("early_mean_abs_uv", "mean"), late=("late_mean_abs_uv", "mean")).reset_index()
        roi["late_minus_early"] = roi["late"] - roi["early"]
        gg = roi.sort_values("late_minus_early", ascending=False).head(20)
        fig = plt.Figure(figsize=(6.6, 5.0), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.barh(gg["region"].astype(str), gg["late_minus_early"])
        ax.invert_yaxis()
        ax.set_xlabel("Late − Early (mean |µV|)")
        ax.set_title(f"ROI difference (Late−Early) — {mode} — {pair}")
        for i, v in enumerate(gg["late_minus_early"].values):
            ax.text(v if v >= 0 else 0, i, f" {v:.2f}", va="center", ha="left", fontsize=8)
        return fig

    def _max_activation_summary(self, evoked: mne.Evoked) -> Tuple[str, float]:
        """Return (region_label, peak_uv) for the channel with max |peak|."""
        peaks_uv = np.max(np.abs(evoked.data), axis=1) * 1e6
        idx = int(np.argmax(peaks_uv)) if peaks_uv.size else 0
        ch = evoked.ch_names[idx] if evoked.ch_names else "Unknown"
        region = self._channel_to_region.get(ch, "Unknown")
        return f"{region}  ({self._strip_pol(ch)})", float(peaks_uv[idx] if peaks_uv.size else 0.0)

    def _draw_mesh_projection(
        self,
        ax: Axes3D,
        evoked: mne.Evoked,
        surface: str,
        *,
        window_ms: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        title: str = "",
    ) -> Optional[mpl.cm.ScalarMappable]:
        """Draw ROI mesh projection on an existing 3D axis; return ScalarMappable for colorbar."""
        verts, faces = self._get_fsaverage_mesh(surface)
        poly = Poly3DCollection(verts[faces])
        ax.add_collection3d(poly)
        self._set_axes_equal_3d(ax, verts)

        sm: Optional[mpl.cm.ScalarMappable] = None
        center = None

        if self.montage is None:
            self._apply_shaded_mesh(poly, verts, faces, face_values=None, alpha=0.90, base_color="lightsteelblue")
        else:
            pos_roi, vals_roi = self._roi_positions_and_values(evoked, window_ms=window_ms)
            if pos_roi.size == 0:
                self._apply_shaded_mesh(poly, verts, faces, face_values=None, alpha=0.90, base_color="lightsteelblue")
            else:
                stim = self.stim_pair_var.get() or ""
                if "-" in stim:
                    left_tok, right_tok = stim.split("-", 1)
                    left_ch = self._resolve_stim_token(left_tok)
                    right_ch = self._resolve_stim_token(right_tok)
                    chp = self.montage.get_positions()["ch_pos"]
                    if left_ch in chp and right_ch in chp:
                        center = 0.5 * (np.array(chp[left_ch]) + np.array(chp[right_ch]))

                if normalize:
                    d2 = np.sum((pos_roi - (center if center is not None else 0.0)) ** 2, axis=1)
                    eps = 1e-7
                    amp_z = (vals_roi - vals_roi.mean()) / (vals_roi.std() + eps)
                    dist_z = (d2 - d2.mean()) / (d2.std() + eps)
                    final = amp_z / (dist_z + eps)
                else:
                    final = vals_roi.copy()

                vvals = self._project_to_vertices(verts, pos_roi, final)
                fvals = np.mean(vvals[faces], axis=1)
                vmin, vmax = float(fvals.min()), float(fvals.max())
                if np.isclose(vmin, vmax):
                    vmax = vmin + (1e-6 if normalize else max(1.0, abs(vmin) * 0.05 + 1e-6))
                norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if (normalize and vmin < 0 < vmax) else plt.Normalize(vmin=vmin, vmax=vmax)
                sm = self._apply_shaded_mesh(poly, verts, faces, face_values=fvals, cmap_name=self.theme.colormap, norm=norm, alpha=0.95)
                self._mark_stim_center(ax, center, brain_center=verts.mean(axis=0))

        ax.set_title(title, fontsize=10 * self.theme.font_scale)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        self._draw_axes_triad(ax)
        return sm

    
    # ----- PDF Export helpers -----
    @staticmethod
    def _export_view_specs() -> List[Tuple[str, int, int]]:
        """
        View specs used in PDF export.

        NOTE: In this coordinate frame, azimuth values that visually corresponded to
        "left/right" were actually "front/back" on your data. We therefore map:
          - Left/Right  -> azim 180 / 0
          - Front/Back  -> azim  90 / -90
        """
        return [
            ("Top", 90, 0),
            ("Bottom", -90, 0),
            ("Left", 0, 180),
            ("Right", 0, 0),
            ("Front", 0, 90),
            ("Back", 0, -90),
        ]

    def _draw_electrode_placements(
        self,
        ax: Axes3D,
        surface: str,
        *,
        title: str = "",
        stim_center: Optional[np.ndarray] = None,
        for_export: bool = False,
    ) -> None:
        """Draw template mesh + electrode locations (optionally with stim-site marker)."""
        verts, faces = self._get_fsaverage_mesh(surface)
        poly = Poly3DCollection(verts[faces])
        ax.add_collection3d(poly)
        self._set_axes_equal_3d(ax, verts)
        self._apply_shaded_mesh(poly, verts, faces, face_values=None, alpha=0.22, base_color="lightgrey")

        brain_center = verts.mean(axis=0)

        if self.montage is not None and self.raw is not None:
            ch_pos = self.montage.get_positions()["ch_pos"]
            names = list(self.raw.ch_names)
            pos_list: List[Tuple[float, float, float]] = []
            matched: List[str] = []
            for ch in names:
                if ch in ch_pos and not np.allclose(ch_pos[ch], [0, 0, 0]):
                    pos_list.append(ch_pos[ch])
                    matched.append(ch)
            if pos_list:
                pos_arr = np.asarray(pos_list, dtype=float)
                pos_plot = self._offset_and_jitter_positions(
                    pos_arr,
                    matched,
                    brain_center,
                    offset_m=MPL_ELECTRODE_OFFSET_M,
                    jitter_m=MPL_ELECTRODE_JITTER_M,
                )
                # Halo + markers
                ax.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2],
                           s=28, c="white", alpha=0.85, edgecolor="none", zorder=2)
                ax.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2],
                           s=18, c="black", alpha=0.95, edgecolor="black", linewidth=0.2, zorder=3)

        # Stim-site marker (yellow circle) if provided
        if stim_center is not None:
            self._mark_stim_center(ax, stim_center, brain_center=brain_center)

        ax.set_title(title, fontsize=10 * self.theme.font_scale)
        if for_export:
            self._style_3d_axes_for_export(ax)
        else:
            self._format_3d_axes_mm(ax)
            self._draw_axes_triad(ax)

    def _build_clinical_summary_page(
        self,
        summary_rows: List[Dict[str, Any]],
        *,
        export_path: Path,
        surface_label: str,
        view_label: str,
        window_ms: Tuple[float, float],
        early_ms: Tuple[float, float],
        late_ms: Tuple[float, float],
    ) -> plt.Figure:
        """Create the final PDF page: an overall clinical decision-support summary.

        IMPORTANT: This is *not* a medical diagnosis. It is a structured summary of the
        computed stimulation-evoked response metrics to help a clinician review results.
        """

        def _trunc(s: str, n: int) -> str:
            s = str(s) if s is not None else ""
            return s if len(s) <= n else s[: max(0, n - 1)] + "…"

        fig = plt.figure(figsize=(11, 8.5), dpi=120)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src_name = self.file_path.name if getattr(self, "file_path", None) else "(no file)"

        # Extract key settings (best-effort; keep this page robust even if some vars are missing).
        try:
            tmin = float(self.tmin_var.get())
            tmax = float(self.tmax_var.get())
        except Exception:
            tmin, tmax = -0.1, 0.5
        try:
            baseline = (float(self.baseline_min_var.get()), float(self.baseline_max_var.get()))
        except Exception:
            baseline = (-0.1, 0.0)
        try:
            roi_sel = str(self.roi_var.get())
        except Exception:
            roi_sel = "All ROIs"

        disclaimer = (
            "Decision-support summary generated from stimulation-evoked iEEG metrics. "
            "Not a medical diagnosis. Requires clinical correlation and review of raw data, "
            "stimulation parameters, and standard presurgical evaluation."
        )

        # Header
        ax.text(
            0.05,
            0.97,
            "Overall Clinical Summary (Decision Support)",
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.05,
            0.94,
            f"Source: {src_name}    |    Export: {export_path.name}    |    Generated: {now_s}",
            fontsize=9,
            va="top",
        )
        ax.text(0.05, 0.915, disclaimer, fontsize=8.5, style="italic", va="top")

        # If nothing processed, still create a valid page.
        if not summary_rows:
            ax.text(
                0.05,
                0.83,
                "No stimulation pairs/modes were exported.\n"
                "Check that stimulation events were detected and that the selected ROI contains electrodes.",
                fontsize=11,
                va="top",
            )
            return fig

        pairs = sorted({r.get("pair", "") for r in summary_rows if r.get("pair")})
        modes = sorted({r.get("mode", "") for r in summary_rows if r.get("mode")})

        # QC flags
        low_trial = [
            f"{r.get('pair','?')} / {r.get('mode','?')} (n={int(r.get('n_trials',0))})"
            for r in summary_rows
            if int(r.get("n_trials", 0)) < 3
        ]
        missing_stim = sorted({r.get("pair", "") for r in summary_rows if not r.get("stim_center_ok", True)})

        # Top activated regions (counts of the max responder region across rows)
        top_regions_lines: List[str] = []
        for m in ("ERP", "RMS", "Gamma"):
            if m not in modes:
                continue
            c = Counter(
                r.get("max_region", "Unknown")
                for r in summary_rows
                if r.get("mode") == m and r.get("max_region") not in (None, "", "Unknown")
            )
            if not c:
                top_regions_lines.append(f"{m}: (no region labels available)")
            else:
                top = ", ".join(f"{reg} ({n})" for reg, n in c.most_common(5))
                top_regions_lines.append(f"{m}: {top}")

        # Early vs late max regions (region counts)
        early_counts = Counter(
            r.get("early_max_region", "Unknown")
            for r in summary_rows
            if r.get("early_max_region") not in (None, "", "Unknown")
        )
        late_counts = Counter(
            r.get("late_max_region", "Unknown")
            for r in summary_rows
            if r.get("late_max_region") not in (None, "", "Unknown")
        )

        early_top = ", ".join(f"{reg} ({n})" for reg, n in early_counts.most_common(5)) or "(no region labels available)"
        late_top = ", ".join(f"{reg} ({n})" for reg, n in late_counts.most_common(5)) or "(no region labels available)"

        # Sort combinations by peak magnitude in the plotted window.
        rows_sorted = sorted(
            summary_rows,
            key=lambda r: float(r.get("peak_display_uv", 0.0) or 0.0),
            reverse=True,
        )
        top_rows = rows_sorted[: min(10, len(rows_sorted))]

        # Compose overview block
        overview_lines = [
            f"Surface: {surface_label}",
            f"ROI filter: {roi_sel}",
            f"Epoch: {tmin:.3f}–{tmax:.3f} s (baseline {baseline[0]:.3f}–{baseline[1]:.3f} s)",
            f"Exported view window: {view_label} ({window_ms[0]:.0f}–{window_ms[1]:.0f} ms)",
            f"Early window: {early_ms[0]:.0f}–{early_ms[1]:.0f} ms    |    Late window: {late_ms[0]:.0f}–{late_ms[1]:.0f} ms",
            f"Stim pairs exported: {len(pairs)}    |    Modes: {', '.join(modes)}",
        ]
        ax.text(0.05, 0.86, "\n".join(overview_lines), fontsize=9.5, va="top")

        # Summary findings
        ax.text(0.05, 0.74, "Summary of max-activation regions (counts):", fontsize=11, fontweight="bold", va="top")
        ax.text(0.07, 0.715, "\n".join(top_regions_lines), fontsize=9, va="top")

        ax.text(0.05, 0.63, "Early vs late max regions (counts across all pairs × modes):", fontsize=11, fontweight="bold", va="top")
        ax.text(0.07, 0.605, f"Early: {early_top}", fontsize=9, va="top")
        ax.text(0.07, 0.582, f"Late:  {late_top}", fontsize=9, va="top")

        # Top responses table
        ax.text(0.05, 0.53, "Highest-magnitude responses (in exported view window):", fontsize=11, fontweight="bold", va="top")

        # Monospaced table
        header = (
            f"{'Pair':<14}  {'Mode':<5}  {'n':>3}  {'Stim ROI':<18}  {'Max ROI':<18}  {'Peak|µV|':>9}  {'Spread%':>7}"
        )
        lines = [header, "-" * len(header)]
        for r in top_rows:
            pair = _trunc(r.get("pair", ""), 14)
            mode = _trunc(r.get("mode", ""), 5)
            ntr = int(r.get("n_trials", 0))
            stim_roi = _trunc(r.get("stim_region", "Unknown"), 18)
            max_roi = _trunc(r.get("max_region", "Unknown"), 18)
            pk = float(r.get("peak_display_uv", 0.0) or 0.0)
            spr = float(r.get("spread_frac", 0.0) or 0.0) * 100.0
            lines.append(
                f"{pair:<14}  {mode:<5}  {ntr:>3d}  {stim_roi:<18}  {max_roi:<18}  {pk:>9.0f}  {spr:>6.0f}%"
            )
        ax.text(0.05, 0.505, "\n".join(lines), fontsize=8.2, family="monospace", va="top")

        # QC / interpretation notes
        ax.text(0.05, 0.30, "QC / interpretation notes:", fontsize=11, fontweight="bold", va="top")
        notes: List[str] = []
        if low_trial:
            notes.append(
                "• Low trial-count items (<3 trials): " + ", ".join(_trunc(s, 90) for s in low_trial[:8])
                + (" …" if len(low_trial) > 8 else "")
            )
        if missing_stim:
            notes.append(
                "• Missing or invalid stim-site coordinates for: "
                + ", ".join(_trunc(p, 20) for p in missing_stim[:12])
                + (" …" if len(missing_stim) > 12 else "")
            )
        notes.extend(
            [
                "• High-amplitude early responses can still be stimulation artifact; confirm by inspecting raw waveforms.",
                "• Early-window peaks are more consistent with direct/effective connectivity; late-window peaks with polysynaptic propagation.",
                "• Integrate with ictal onset patterns, interictal spikes, imaging, and semiology; do not use this page in isolation for clinical decisions.",
            ]
        )
        notes_wrapped = [textwrap.fill(n, width=120, subsequent_indent="  ") for n in notes]
        ax.text(0.07, 0.275, "\n".join(notes_wrapped), fontsize=9, va="top")

        return fig

    # ----- HTML Export helpers -----
    def _epochs_cache_signature(self) -> Tuple[float, float, float, float, float]:
        """Signature for epochs derived from current GUI epoch parameters."""
        def _f(var, default):
            try:
                return float(var.get())
            except Exception:
                return float(default)
        return (
            _f(self.min_time_var, 0.5),
            _f(self.tmin_var, -0.2),
            _f(self.tmax_var, 1.0),
            _f(self.baseline_min_var, -0.2),
            _f(self.baseline_max_var, 0.0),
        )

    def _rebuild_epochs_by_pair_cache(self, force: bool = False) -> None:
        """Build epochs for all stim pairs (used by HTML export)."""
        if self.raw is None or self.events is None or not len(self.events):
            self.epochs_by_pair = {}
            self._epochs_cache_sig = None
            return

        sig = self._epochs_cache_signature()
        if force or (self._epochs_cache_sig != sig):
            self.epochs_by_pair = {}
            self._epochs_cache_sig = sig

        if not self.stim_pairs:
            return

        min_time, tmin, tmax, b0, b1 = sig
        baseline = (b0, b1)

        for pair in self.stim_pairs:
            if pair in self.epochs_by_pair:
                continue
            clean_evs = self._clean_events(pair, min_time)
            if len(clean_evs) == 0:
                continue
            try:
                self.epochs_by_pair[pair] = mne.Epochs(
                    self.raw,
                    clean_evs,
                    event_id={f"stim_{pair}": 1},
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    preload=True,
                    verbose=False,
                )
            except Exception as exc:
                LOG.warning("Failed to build epochs for pair %s: %s", pair, exc)


    def export_html_app(self) -> None:
        """Export a single self-contained HTML viewer of the current dataset."""
        if self.raw is None or self.events is None or not len(self.events) or not self.stim_pairs:
            messagebox.showerror(
                "Export HTML App",
                "Load an EDF and detect stimulation events before exporting.",
            )
            return

        # Ensure a montage exists (needed for electrode positions / 3D views).
        self._ensure_montage()

        out_path = filedialog.asksaveasfilename(
            title="Export interactive HTML viewer",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
        )
        if not out_path:
            return

        # Disable during export (re-enabled on completion).
        try:
            self.export_button.config(state="disabled")
        except Exception:
            pass

        inline_js = True  # Prefer fully offline HTML if plotly.min.js is found.

        def worker() -> None:
            try:
                export_t0 = time.perf_counter()
                self._ui(self._update_status, "Building HTML export (computing epochs/CI)...")
                # Rebuild to guarantee export reflects the current GUI settings.
                self._rebuild_epochs_by_pair_cache(force=True)

                if not self.epochs_by_pair:
                    raise RuntimeError(
                        "No valid epochs found to export. "
                        "Confirm stim events exist and try running the analysis first."
                    )

                payload = self._build_html_app_payload()
                html = self._render_html_app(payload, inline_plotly_js=inline_js)
                Path(out_path).write_text(html, encoding="utf-8")
                self._record_workflow_timing(
                    "html_report_creation",
                    time.perf_counter() - export_t0,
                    file=Path(out_path).name,
                    n_pairs=int(len(self.epochs_by_pair)),
                )

                def _done() -> None:
                    try:
                        self.export_button.config(state="normal")
                    except Exception:
                        pass

                    if messagebox.askyesno("Open file?", "HTML export complete. Open in browser?"):
                        import webbrowser

                        webbrowser.open(out_path)

                    messagebox.showinfo("Export complete", f"Saved HTML viewer to:\n{out_path}")
                    self._update_status("Ready")

                self._ui(_done)

            except Exception as exc:
                LOG.exception("HTML export failed: %s", exc)

                err_msg = str(exc)

                def _err(msg: str = err_msg) -> None:
                    try:
                        self.export_button.config(state="normal")
                    except Exception:
                        pass
                    messagebox.showerror("Export failed", f"HTML export failed:\n{msg}")
                    self._update_status("Export failed")

                self._ui(_err)

        self._run_thread(worker)

    @staticmethod
    def _b64_nd(arr: np.ndarray, *, dtype: Optional[np.dtype] = None) -> Dict[str, Any]:
        """Encode a NumPy array as base64 for embedding into a single HTML file."""
        a = np.asarray(arr)
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        a = np.ascontiguousarray(a)
        return {
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "b64": base64.b64encode(a.tobytes()).decode("ascii"),
        }

    def _plotly_min_js_text(self) -> Optional[str]:
        """Return the contents of a local plotly.min.js if available, else None."""
        try:
            import plotly  # type: ignore

            pkg = Path(plotly.__file__).resolve().parent
            candidates = [
                pkg / "package_data" / "plotly.min.js",
                pkg / "package_data" / "plotly.min.js.gz",
                pkg / "plotly.min.js",
            ]
            for cand in candidates:
                if cand.exists() and cand.suffix == ".js":
                    return cand.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        return None

    def _compute_mean_ci_uv(self, epochs: mne.Epochs, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute evoked mean and 95% CI (both in µV) for all channels/timepoints."""
        evk = self._compute_evoked_for_mode(epochs, mode)
        mean_uv = (evk.data * 1e6).astype(np.float32)

        n = len(epochs)
        if n <= 1:
            return mean_uv, np.zeros_like(mean_uv)

        data_v = epochs.get_data()  # (n_trials, n_ch, n_t)
        if mode == "Gamma":
            env = self._compute_gamma_envelope(data_v, fs=float(epochs.info["sfreq"]))
            trials_uv = (env * 1e6).astype(np.float32)
        else:
            trials_uv = (data_v * 1e6).astype(np.float32)

        if mode in ("ERP", "Gamma"):
            ci_uv = (1.96 * stats.sem(trials_uv, axis=0)).astype(np.float32)
            return mean_uv, ci_uv

        # RMS delta-method (propagate variance of x^2 through sqrt(.))
        x = trials_uv.astype(np.float64, copy=False)
        mu2 = np.mean(x * x, axis=0)  # E[x^2]
        rms = np.sqrt(mu2) + 1e-12
        std2 = np.std(x * x, axis=0, ddof=1)  # std(x^2)
        sem_mu2 = std2 / np.sqrt(float(n))
        sem_rms = sem_mu2 / (2.0 * rms)
        ci_uv = (1.96 * sem_rms).astype(np.float32)
        return mean_uv, ci_uv

    def _build_html_app_payload(self) -> Dict[str, Any]:
        """Build a JSON-serializable payload for the interactive HTML viewer."""
        if self.raw is None:
            raise RuntimeError("No EDF loaded.")
        if not self.epochs_by_pair:
            raise RuntimeError("No analysis results found (epochs_by_pair is empty).")

        surface_key = "pial"
        if hasattr(self, "surface_var"):
            try:
                surface_key = self._current_surface_key()
            except Exception:
                surface_key = "pial"

        # Mesh (fsaverage) in meters, then convert to mm for display.
        verts_m, faces = self._get_fsaverage_mesh(surface_key)
        center_m = np.mean(verts_m, axis=0)

        # Electrode positions from montage (meters).
        montage = self.raw.get_montage()
        if montage is None:
            raise RuntimeError("No montage available.")
        pos_dict = montage.get_positions() or {}
        ch_pos = pos_dict.get("ch_pos", {}) or {}
        pos_by_name: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in ch_pos.items()}

        # Use all channels that have valid (non-zero) positions.
        raw_names = list(self.raw.ch_names)
        matched_all = []
        pos_arr_list = []
        for ch in raw_names:
            p = pos_by_name.get(ch)
            if p is None:
                continue
            if float(np.linalg.norm(p)) < 1e-12:
                continue
            matched_all.append(ch)
            pos_arr_list.append(p)
        if not matched_all:
            raise RuntimeError("No electrode positions available to export.")

        pos_arr_m = np.vstack(pos_arr_list).astype(np.float64)
        pos_plot_m = self._offset_and_jitter_positions(pos_arr_m, matched_all, center_m, offset_m=0.005, jitter_m=0.001)

        # Channel order for evoked arrays is from the epochs object.
        first_epochs = next(iter(self.epochs_by_pair.values()))
        ch_names = list(first_epochs.ch_names)
        times_ms = (first_epochs.times * 1000.0).astype(float)

        ch_index = {ch: i for i, ch in enumerate(ch_names)}
        keep = [i for i, ch in enumerate(matched_all) if ch in ch_index]
        matched = [matched_all[i] for i in keep]
        pos_arr_m = pos_arr_m[keep]
        pos_plot_m = pos_plot_m[keep]
        elec_ch_idx = [int(ch_index[ch]) for ch in matched]

        # Region labels (optional).
        chan_to_region: Dict[str, str] = {}
        if self.electrode_coords is not None and "Regions" in self.electrode_coords.columns:
            df = self.electrode_coords
            n = min(len(self.raw.ch_names), len(df))
            for i in range(n):
                chan_to_region[str(self.raw.ch_names[i])] = str(df.loc[i, "Regions"])
        regions = [chan_to_region.get(ch, "Unknown") for ch in matched]
        unique_rois = ["All ROIs"] + list(TOP_ROI_OPTIONS) + sorted({r for r in regions if r is not None and str(r).strip() != ""})

        # KDTree: for each mesh vertex, store k-nearest electrode indices (in the matched list).
        k_near = int(min(4, len(matched)))
        if k_near < 1:
            k_near = 1
        tree = cKDTree(pos_arr_m)
        idxs = tree.query(verts_m, k=k_near)[1]
        if idxs.ndim == 1:
            idxs = idxs[:, None]
        idxs = idxs.astype(np.int32, copy=False)

        # Stim marker positions per pair: use the same displayed electrode coordinates
        # as the Plotly markers so the stim site is centered exactly on the selected bipole.
        plot_pos_by_name = {ch: pos_plot_m[i] for i, ch in enumerate(matched)}
        stim_markers_mm: Dict[str, List[float]] = {}
        stim_bipoles_mm: Dict[str, Dict[str, Any]] = {}
        for pair in sorted(self.epochs_by_pair.keys()):
            geom = self._stim_geometry_from_display_positions(pair, plot_pos_by_name)
            if geom is None:
                continue
            p1, p2, c_plot, ch_pair = geom
            stim_markers_mm[pair] = (c_plot * 1000.0).astype(float).tolist()
            stim_bipoles_mm[pair] = {
                "contacts": [self._strip_pol_prefix(ch_pair[0]), self._strip_pol_prefix(ch_pair[1])],
                "p1_mm": (p1 * 1000.0).astype(float).tolist(),
                "p2_mm": (p2 * 1000.0).astype(float).tolist(),
                "midpoint_mm": (c_plot * 1000.0).astype(float).tolist(),
            }

        # Evoked mean + CI for each pair × mode.
        modes = ["ERP", "RMS", "Gamma"]
        evoked_pack: Dict[str, Any] = {}
        for pair, epochs in sorted(self.epochs_by_pair.items()):
            evoked_pack[pair] = {}
            for mode in modes:
                mean_uv, ci_uv = self._compute_mean_ci_uv(epochs, mode)
                evoked_pack[pair][mode] = {
                    "n_trials": int(len(epochs)),
                    "mean_uv": self._b64_nd(mean_uv, dtype=np.float32),
                    "ci_uv": self._b64_nd(ci_uv, dtype=np.float32),
                }

        payload: Dict[str, Any] = {
            "meta": {
                "created": datetime.now().isoformat(timespec="seconds"),
                "source_file": str(self.file_path.name if self.file_path else ""),
                "surface": surface_key,
            },
            "pairs": sorted(self.epochs_by_pair.keys()),
            "modes": modes,
            "times_ms": times_ms.astype(float).tolist(),
            "ch_names": ch_names,
            "defaults": {
                "pair": (
                    str(self.stim_pair_var.get())
                    if hasattr(self, "stim_pair_var") and str(self.stim_pair_var.get()).strip()
                    else sorted(self.epochs_by_pair.keys())[0]
                ),
                "mode": str(self.analysis_mode_var.get()) if hasattr(self, "analysis_mode_var") else "ERP",
                "view": str(self.view_window_var.get()) if hasattr(self, "view_window_var") else "Average",
                "early_start_ms": float(self.early_start_ms_var.get()) if hasattr(self, "early_start_ms_var") else 10.0,
                "early_end_ms": float(self.early_end_ms_var.get()) if hasattr(self, "early_end_ms_var") else 80.0,
                "late_start_ms": float(self.late_start_ms_var.get()) if hasattr(self, "late_start_ms_var") else 80.0,
                "late_end_ms": float(self.late_end_ms_var.get()) if hasattr(self, "late_end_ms_var") else 300.0,
                "roi": str(self.roi_var.get()) if hasattr(self, "roi_var") else "All ROIs",
            },
            "electrodes": {
                "names": [self._strip_pol_prefix(n) for n in matched],
                    "names_raw": matched,
                "ch_idx": elec_ch_idx,
                "regions": regions,
                "x_plot_mm": self._b64_nd(pos_plot_m[:, 0] * 1000.0, dtype=np.float32),
                "y_plot_mm": self._b64_nd(pos_plot_m[:, 1] * 1000.0, dtype=np.float32),
                "z_plot_mm": self._b64_nd(pos_plot_m[:, 2] * 1000.0, dtype=np.float32),
                "x_mm": self._b64_nd(pos_arr_m[:, 0] * 1000.0, dtype=np.float32),
                "y_mm": self._b64_nd(pos_arr_m[:, 1] * 1000.0, dtype=np.float32),
                "z_mm": self._b64_nd(pos_arr_m[:, 2] * 1000.0, dtype=np.float32),
            },
            "mesh": {
                "surface": surface_key,
                "center_mm": (center_m * 1000.0).astype(float).tolist(),
                "x_mm": self._b64_nd(verts_m[:, 0] * 1000.0, dtype=np.float32),
                "y_mm": self._b64_nd(verts_m[:, 1] * 1000.0, dtype=np.float32),
                "z_mm": self._b64_nd(verts_m[:, 2] * 1000.0, dtype=np.float32),
                "i": self._b64_nd(faces[:, 0].astype(np.int32, copy=False)),
                "j": self._b64_nd(faces[:, 1].astype(np.int32, copy=False)),
                "k": self._b64_nd(faces[:, 2].astype(np.int32, copy=False)),
                "k_nearest": int(k_near),
                "nearest_idx": self._b64_nd(idxs.astype(np.int32, copy=False)),
            },
            "stim_markers_mm": stim_markers_mm,
            "stim_bipoles_mm": stim_bipoles_mm,
            "rois": unique_rois,
            "evoked": evoked_pack,
        }
        return payload

    def _render_html_app(self, payload: Dict[str, Any], *, inline_plotly_js: bool) -> str:
        """Render a standalone HTML viewer (single file) with embedded data."""
        # Plotly JS: inline (offline) if requested and available; otherwise CDN.
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        if inline_plotly_js:
            js = self._plotly_min_js_text()
            if js:
                plotly_tag = f"<script>\n{js}\n</script>"

        # Persist current GUI mesh opacity into the exported HTML (if available)
        try:
            mesh_opacity_default = float(self.mesh_opacity_var.get()) if self.mesh_opacity_var is not None else 0.35
        except Exception:
            mesh_opacity_default = 0.35
        mesh_opacity_default = float(np.clip(mesh_opacity_default, 0.05, 1.0))

        # IMPORTANT: The HTML viewer embeds JSON in a <script type="application/json"> tag.

        # Two common failure modes when opening exported HTML locally are:

        #   1) NaN/Inf values serialized by Python as `NaN`/`Infinity` (invalid for JSON.parse in browsers)

        #   2) Raw '<' characters (e.g., '</script>') prematurely terminating the script tag

        # We sanitize floats and escape '<' to keep the payload robust.


        def _sanitize_for_json(obj):

            if obj is None:

                return None


            # Numpy scalar types -> builtin Python scalars

            if isinstance(obj, (np.floating,)):

                v = float(obj)

                return v if math.isfinite(v) else None

            if isinstance(obj, (np.integer,)):

                return int(obj)


            # Builtin floats: drop NaN/Inf

            if isinstance(obj, float):

                return obj if math.isfinite(obj) else None


            if isinstance(obj, (list, tuple)):

                return [_sanitize_for_json(x) for x in obj]

            if isinstance(obj, dict):

                return {str(k): _sanitize_for_json(v) for k, v in obj.items()}


            return obj


        safe_payload = _sanitize_for_json(payload)

        data_json = json.dumps(

            safe_payload,

            ensure_ascii=False,

            allow_nan=False,

            separators=(",", ":"),

        )

        # Escape '<' to prevent '</script>' from breaking the HTML.

        data_json = data_json.replace("<", "\u003c")
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>iEEG CCEP Viewer</title>
  <style>
    :root {{
      --panel-bg: #111827;
      --panel-fg: #e5e7eb;
      --muted: #9ca3af;
      --border: #374151;
      --accent: #22c55e;
      --warn: #f59e0b;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: #0b1020;
      color: var(--panel-fg);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    .app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      height: 100vh;
      gap: 0;
    }}
    .sidebar {{
      background: var(--panel-bg);
      border-right: 1px solid var(--border);
      padding: 14px 12px;
      overflow: auto;
    }}
    .content {{
      display: grid;
      grid-template-rows: 1fr 0.9fr;
      gap: 8px;
      padding: 8px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      min-height: 300px;
    }}
    .panel {{
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      position: relative;
    }}
    .panelHeader {{
      position: absolute;
      top: 8px;
      left: 10px;
      z-index: 10;
      font-size: 12px;
      color: var(--muted);
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(55, 65, 81, 0.6);
      padding: 4px 8px;
      border-radius: 999px;
      user-select: none;
      pointer-events: none;
    }}
    .plot {{
      width: 100%;
      height: 100%;
    }}
    h2 {{
      margin: 0 0 10px 0;
      font-size: 14px;
      font-weight: 600;
    }}
    .field {{
      margin-bottom: 10px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    select, input {{
      width: 100%;
      box-sizing: border-box;
      background: #0b1224;
      border: 1px solid var(--border);
      color: var(--panel-fg);
      padding: 8px 10px;
      border-radius: 8px;
      outline: none;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .hint {{
      font-size: 11px;
      color: var(--muted);
      line-height: 1.35;
    }}
    .btnRow {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-top: 10px;
    }}
    .hemiRow {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .check {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--panel-fg);
      user-select: none;
    }}
    .check input {{
      transform: translateY(1px);
    }}

    button {{
      background: #0b1224;
      border: 1px solid var(--border);
      color: var(--panel-fg);
      padding: 8px 10px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 12px;
    }}
    button:hover {{
      border-color: #6b7280;
    }}
    .status {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
      font-size: 12px;
      color: var(--muted);
      white-space: pre-wrap;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      margin-right: 6px;
      font-size: 11px;
      color: var(--muted);
    }}
  
    .panel.span2 {{
      grid-column: 1 / span 2;
    }}

    .plotWithSlider {{
      height: calc(100% - 34px);
    }}

    .opacityRow {{
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 10px;
      height: 22px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
      color: var(--muted);
      pointer-events: auto;
      user-select: none;
    }}
    .opacityRow input[type="range"] {{
      flex: 1;
    }}
    .opacityRow span {{
      white-space: nowrap;
    }}
</style>
  {plotly_tag}
</head>
<body>
  <div class="app">
    <div class="sidebar">
      <h2>iEEG CCEP Viewer</h2>

      <div class="field">
        <label>Stim pair</label>
        <select id="pairSelect"></select>
      </div>

      <div class="field">
        <label>Analysis mode</label>
        <select id="modeSelect"></select>
      </div>

      <div class="field">
        <label>Epoch view</label>
        <select id="viewSelect">
          <option value="Average">Average (full window)</option>
          <option value="Early">Early</option>
          <option value="Late">Late</option>
        </select>
      </div>

      <div class="grid2">
        <div class="field">
          <label>Early start (ms)</label>
          <input id="earlyStart" type="number" step="1" />
        </div>
        <div class="field">
          <label>Early end (ms)</label>
          <input id="earlyEnd" type="number" step="1" />
        </div>
      </div>
      <div class="grid2">
        <div class="field">
          <label>Late start (ms)</label>
          <input id="lateStart" type="number" step="1" />
        </div>
        <div class="field">
          <label>Late end (ms)</label>
          <input id="lateEnd" type="number" step="1" />
        </div>
      </div>

      <div class="field">
        <label>ROI filter</label>
        <select id="roiSelect"></select>
      </div>

      <div class="field">
        <label>Channels (detailed waveform)</label>
        <select id="chanSelect" multiple size="8"></select>
        <div class="hint">Shift/Ctrl to multi-select. Clicking electrodes selects channels.</div>
        <div class="btnRow" style="margin-top:8px;">
          <button id="projBtn" title="Toggle: project surface values using only the selected sensor montage">Project Only Selected Sensor Montage</button>
          <button id="resetBtn" title="Reset all controls to their initial defaults">Reset</button>
        </div>
      </div>

      <div class="field">
      <label>Hemispheres</label>
      <div class="hemiRow">
        <label class="check"><input id="showLeftHemi" type="checkbox" checked> Left</label>
        <label class="check"><input id="showRightHemi" type="checkbox" checked> Right</label>
      </div>
    </div>

      <div class="status" id="statusBox">Loading data…</div>
      <div class="hint" style="margin-top:10px;">
        <span class="badge">No imports</span>
        <span class="badge">Single file</span>
        <span class="badge">Interactive</span>
        <div style="margin-top:6px;">
          This HTML is a standalone viewer. It contains the precomputed results from the desktop app and lets you explore stim pairs, modes, ROIs, and early/late windows.
        </div>
      </div>
    </div>

    <div class="content">
      <div class="row">
        <div class="panel span2">
          <div class="panelHeader">3D brain + electrodes (ROI projection)</div>
          <div id="meshDiv" class="plot plotWithSlider"></div>
          <div class="opacityRow">
            <span>Brain opacity</span>
            <input id="opacitySlider" type="range" min="0.05" max="1" step="0.05" value="{mesh_opacity_default:.2f}" />
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panelHeader">Detailed waveform</div>
        <div id="waveDiv" class="plot"></div>
      </div>
    </div>
  </div>

  <script id="app-data" type="application/json">{data_json}</script>
  <script>
  (async function() {{
    const $ = (id) => document.getElementById(id);
            function slice1D(arr, start, len) {{
                const s = Math.max(0, start | 0);
                const e = Math.min(arr.length, s + (len | 0));
                return arr.slice(s, e);
            }}

    const statusBox = $('statusBox');
    function setStatus(msg) {{
      if (statusBox) {{
        statusBox.textContent = msg;
      }}
    }}

    function fail(err) {{
      const msg = (err && err.stack) ? err.stack : String(err);
      if (statusBox) {{
        statusBox.textContent = 'ERROR loading viewer.\\n' + msg;
        statusBox.style.borderColor = '#ef4444';
      }}
      console.error(err);
    }}

    window.addEventListener('error', (e) => fail(e.error || e.message || e));
    window.addEventListener('unhandledrejection', (e) => fail(e.reason || e));

    setStatus('Parsing embedded data…');
    let APP = null;
    try {{
      const node = document.getElementById('app-data');
      if (!node) {{
        throw new Error(
          'Missing embedded payload element #app-data. (The export may have been truncated or corrupted.)'
        );
      }}
      APP = JSON.parse(node.textContent);
    }} catch (err) {{
      fail(err);
      return;
    }}


function stripPOL(name) {{
  if (name === null || name === undefined) return '';
  return String(name).replace(/^POL\\s+/i, '');
}}

let projectOnlySelected = false;
let DEFAULTS = null;

    async function b64ToArrayBuffer(b64) {{
      try {{
        const res = await fetch('data:application/octet-stream;base64,' + b64);
        return await res.arrayBuffer();
      }} catch (e) {{
        // Fallback for browsers that restrict fetch() on file:// or data: URLs.
        const bin = atob(b64);
        const len = bin.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {{
          bytes[i] = bin.charCodeAt(i);
        }}
        return bytes.buffer;
      }}
    }}

    async function decodeND(obj) {{
      const buf = await b64ToArrayBuffer(obj.b64);
      let arr;
      switch (obj.dtype) {{
        case 'float32': arr = new Float32Array(buf); break;
        case 'float64': arr = new Float64Array(buf); break;
        case 'int32': arr = new Int32Array(buf); break;
        case 'uint32': arr = new Uint32Array(buf); break;
        default: arr = new Float32Array(buf);
      }}
      return {{ data: arr, shape: obj.shape }};
    }}

    // Populate selects
    const pairSelect = $('pairSelect');
    const modeSelect = $('modeSelect');
    const roiSelect = $('roiSelect');
    const chanSelect = $('chanSelect');
    const projBtn = $('projBtn');
    const resetBtn = $('resetBtn');

    for (const p of APP.pairs) {{
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      pairSelect.appendChild(opt);
    }}
    for (const m of APP.modes) {{
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      modeSelect.appendChild(opt);
    }}
    for (const r of APP.rois) {{
      const opt = document.createElement('option');
      opt.value = r;
      opt.textContent = r;
      roiSelect.appendChild(opt);
    }}
    for (const ch of APP.ch_names) {{
      const opt = document.createElement('option');
      opt.value = ch;
      opt.textContent = stripPOL(ch);
      chanSelect.appendChild(opt);
    }}

    // Defaults
    pairSelect.value = APP.defaults.pair;
    modeSelect.value = APP.defaults.mode;
    $('viewSelect').value = APP.defaults.view;
    $('earlyStart').value = APP.defaults.early_start_ms;
    $('earlyEnd').value = APP.defaults.early_end_ms;
    $('lateStart').value = APP.defaults.late_start_ms;
    $('lateEnd').value = APP.defaults.late_end_ms;
    roiSelect.value = APP.defaults.roi;
    if (chanSelect.options.length) {{ chanSelect.options[0].selected = true; }}

    // Decode static arrays
    setStatus("Decoding mesh + electrode geometry…");

    const meshX = (await decodeND(APP.mesh.x_mm)).data;
    const meshY = (await decodeND(APP.mesh.y_mm)).data;
    const meshZ = (await decodeND(APP.mesh.z_mm)).data;
    const meshI = (await decodeND(APP.mesh.i)).data;
    const meshJ = (await decodeND(APP.mesh.j)).data;
    const meshK = (await decodeND(APP.mesh.k)).data;
    // Precompute hemisphere face lists (majority vote by X coordinate).
    // This lets the user hide Left/Right hemispheres without recomputing meshes.
    let hemiLeftI = null, hemiLeftJ = null, hemiLeftK = null;
    let hemiRightI = null, hemiRightJ = null, hemiRightK = null;
    if (meshX && meshI && meshJ && meshK) {{
      let minX = Infinity, maxX = -Infinity;
      for (let i = 0; i < meshX.length; i++) {{
        const v = meshX[i];
        if (v < minX) minX = v;
        if (v > maxX) maxX = v;
      }}
      const midX = (minX + maxX) / 2.0;
      const isLeft = new Uint8Array(meshX.length);
      for (let i = 0; i < meshX.length; i++) {{
        isLeft[i] = (meshX[i] <= midX) ? 1 : 0;
      }}
      const li = [], lj = [], lk = [];
      const ri = [], rj = [], rk = [];
      for (let t = 0; t < meshI.length; t++) {{
        const a = meshI[t], b = meshJ[t], c = meshK[t];
        const cnt = isLeft[a] + isLeft[b] + isLeft[c];
        if (cnt >= 2) {{ li.push(a); lj.push(b); lk.push(c); }}
        else {{ ri.push(a); rj.push(b); rk.push(c); }}
      }}
      hemiLeftI = li; hemiLeftJ = lj; hemiLeftK = lk;
      hemiRightI = ri; hemiRightJ = rj; hemiRightK = rk;
    }}

    const nearestIdx = (await decodeND(APP.mesh.nearest_idx)).data;
    const kNear = APP.mesh.k_nearest;

    const elecX = (await decodeND(APP.electrodes.x_plot_mm)).data;
    const elecY = (await decodeND(APP.electrodes.y_plot_mm)).data;
    const elecZ = (await decodeND(APP.electrodes.z_plot_mm)).data;

    const elecNames = APP.electrodes.names;
    const elecChIdx = APP.electrodes.ch_idx;
    const elecRegions = APP.electrodes.regions;

    const times = APP.times_ms;
    const nTimes = times.length;

    // Caches for evoked arrays (decoded lazily).
    const evokedCache = {{}};

    async function getEvoked(pair, mode) {{
      const key = pair + "||" + mode;
      if (evokedCache[key]) {{
        return await evokedCache[key];
      }}
      const item = APP.evoked[pair][mode];
      evokedCache[key] = (async () => {{
        const meanObj = await decodeND(item.mean_uv);
        const ciObj = await decodeND(item.ci_uv);
        return {{
          mean: meanObj.data,
          ci: ciObj.data,
          shape: meanObj.shape,
          nTrials: item.n_trials
        }};
      }})();
      return await evokedCache[key];
    }}

    function getWindow() {{
      const view = $('viewSelect').value;
      let t0 = (times && times.length) ? times[0] : -200;
      let t1 = (times && times.length) ? times[times.length-1] : 1000;
      if (view === 'Early') {{
        t0 = parseFloat($('earlyStart').value || "0");
        t1 = parseFloat($('earlyEnd').value || "0");
      }} else if (view === 'Late') {{
        t0 = parseFloat($('lateStart').value || "0");
        t1 = parseFloat($('lateEnd').value || "0");
      }}
      if (!isFinite(t0)) t0 = -1e9;
      if (!isFinite(t1)) t1 =  1e9;
      if (t1 < t0) {{
        const tmp = t0; t0 = t1; t1 = tmp;
      }}

      // Convert to index window [start,end)
      let start = 0;
      while (start < times.length && times[start] < t0) start++;
      let end = times.length;
      while (end > start && times[end - 1] > t1) end--;
      if (end <= start) {{
        start = 0; end = times.length;
      }}
      return {{view, t0, t1, start, end}};
    }}

    function quantile(arr, q) {{
      const tmp = [];
      for (let i = 0; i < arr.length; i++) {{
        const v = arr[i];
        if (isFinite(v)) tmp.push(v);
      }}
      if (tmp.length === 0) return 1.0;
      tmp.sort((a,b) => a - b);
      const idx = Math.max(0, Math.min(tmp.length - 1, Math.floor(q * (tmp.length - 1))));
      return tmp[idx];
    }}

    function computeElectrodePeaks(meanFlat, win) {{
      const peaks = new Float32Array(elecChIdx.length);
      const start = win.start;
      const end = win.end;
      for (let e = 0; e < elecChIdx.length; e++) {{
        const ch = elecChIdx[e];
        let peak = 0.0;
        const base = ch * nTimes;
        for (let t = start; t < end; t++) {{
          let v = meanFlat[base + t];
          if (v < 0) v = -v;
          if (v > peak) peak = v;
        }}
        peaks[e] = peak;
      }}
      return peaks;
    }}

    function topRoisByPeak(peaks, nTop) {{
      const best = new Map();
      for (let i = 0; i < elecNames.length; i++) {{
        const roi = String(elecRegions[i] || 'Unknown');
        if (!roi || roi === 'Unknown') continue;
        const v = Number(peaks[i] || 0);
        if (!best.has(roi) || v > best.get(roi)) best.set(roi, v);
      }}
      return Array.from(best.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, nTop)
        .map(x => x[0]);
    }}

    function roiMasks(roiName, peaks) {{
      const inRoi = new Uint8Array(elecNames.length);
      if (!roiName || roiName === 'All ROIs') {{
        inRoi.fill(1);
        return inRoi;
      }}
      let selected = null;
      if (roiName === 'Top 5 responding ROIs') {{
        selected = new Set(topRoisByPeak(peaks, 5));
      }} else if (roiName === 'Top 10 responding ROIs') {{
        selected = new Set(topRoisByPeak(peaks, 10));
      }}
      for (let i = 0; i < elecNames.length; i++) {{
        const roi = String(elecRegions[i] || 'Unknown');
        if (selected) {{
          if (selected.has(roi)) inRoi[i] = 1;
        }} else if (roi === String(roiName)) {{
          inRoi[i] = 1;
        }}
      }}
      return inRoi;
    }}

    function computeVertexIntensity(peaks, inRoi) {{
      const nV = meshX.length;
      const out = new Float32Array(nV);
      for (let v = 0; v < nV; v++) {{
        const base = v * kNear;
        let val = 0.0;
        for (let k = 0; k < kNear; k++) {{
          const ei = nearestIdx[base + k];
          if (inRoi[ei]) {{
            val = peaks[ei];
            break;
          }}
        }}
        out[v] = val;
      }}
      return out;
    }}

    function channelIndex(name) {{
      // Linear search is OK for ~200 channels.
      for (let i = 0; i < APP.ch_names.length; i++) {{
        if (APP.ch_names[i] === name) return i;
      }}
      return 0;
    }}

    const config3d = {{
      displayModeBar: true,
      responsive: true
    }};

    const config2d = {{
      displayModeBar: true,
      responsive: true
    }};

    function sceneLayoutBase() {{
      return {{
        xaxis: {{visible: false, showgrid: false, zeroline: false}},
        yaxis: {{visible: false, showgrid: false, zeroline: false}},
        zaxis: {{visible: false, showgrid: false, zeroline: false}},
        aspectmode: 'data',
        bgcolor: 'rgba(0,0,0,0)'
      }};
    }}

    const cameraPresets = {{
      Top:    {{eye: {{x: 0,   y: 0,   z: 2.5}}}},
      Bottom: {{eye: {{x: 0,   y: 0,   z: -2.5}}}},
      Left:   {{eye: {{x: -2.5,y: 0,   z: 0}}}},
      Right:  {{eye: {{x: 2.5, y: 0,   z: 0}}}},
      Front:  {{eye: {{x: 0,   y: 2.5, z: 0}}}},
      Back:   {{eye: {{x: 0,   y: -2.5,z: 0}}}},
    }};

    async function updateAll() {{
      const pair = pairSelect.value;
      const mode = modeSelect.value;
      const roiName = roiSelect.value;
      const win = getWindow();
      const viewWin = [win.t0, win.t1];

      setStatus("Rendering…\\n"
        + "File: " + APP.meta.source_file + "\\n"
        + "Surface: " + APP.meta.surface + "\\n"
        + "Pair: " + pair + "\\n"
        + "Mode: " + mode + "\\n"
        + "View: " + win.view + " (" + Math.round(win.t0) + "–" + Math.round(win.t1) + " ms)\\n"
        + "ROI: " + roiName);

      const evk = await getEvoked(pair, mode);
      const meanFlat = evk.mean;
      const ciFlat = evk.ci;

      // Peaks and scaling
      const peaks = computeElectrodePeaks(meanFlat, win);
      const cmax = Math.max(1e-6, quantile(peaks, 0.95));

      const inRoi = roiMasks(roiName, peaks);

      // Optional: restrict surface projection to the selected sensor montage
      let selMask = null;
      if (projectOnlySelected) {{
        const selNames = Array.from(chanSelect.selectedOptions).map(o => stripPOL(o.value));
        const selSet = new Set(selNames);
        const tmp = new Array(elecNames.length);
        let cnt = 0;
        for (let i = 0; i < elecNames.length; i++) {{
          const ok = selSet.has(elecNames[i]);
          tmp[i] = ok;
          if (ok) cnt += 1;
        }}
        if (cnt > 0) selMask = tmp;
      }}

      let peaksForProj = peaks;
      if (selMask) {{
        peaksForProj = peaks.map((v,i) => selMask[i] ? v : 0);
      }}

      // Electrode traces: ROI selected, ROI (not selected), and non-ROI
      const xR = [], yR = [], zR = [], cR = [], cdR = [];
      const xF = [], yF = [], zF = [], cdF = [];
      const xO = [], yO = [], zO = [], cdO = [];
      for (let i = 0; i < elecNames.length; i++) {{
        const isSel = (!selMask) ? true : !!selMask[i];
        if (inRoi[i] && isSel) {{
          xR.push(elecX[i]); yR.push(elecY[i]); zR.push(elecZ[i]);
          cR.push(peaks[i]);
          cdR.push(i);
        }} else if (inRoi[i] && !isSel) {{
          xF.push(elecX[i]); yF.push(elecY[i]); zF.push(elecZ[i]);
          cdF.push(i);
        }} else {{
          xO.push(elecX[i]); yO.push(elecY[i]); zO.push(elecZ[i]);
          cdO.push(i);
        }}
      }}

      const stim = APP.stim_markers_mm[pair] || null;
      const stimBipole = (APP.stim_bipoles_mm && APP.stim_bipoles_mm[pair]) ? APP.stim_bipoles_mm[pair] : null;
      const stimLineTrace = stimBipole ? {{
        type: 'scatter3d',
        mode: 'lines',
        x: [stimBipole.p1_mm[0], stimBipole.p2_mm[0]],
        y: [stimBipole.p1_mm[1], stimBipole.p2_mm[1]],
        z: [stimBipole.p1_mm[2], stimBipole.p2_mm[2]],
        line: {{color: 'yellow', width: 7}},
        name: 'Stim bipole',
        hoverinfo: 'skip',
        showlegend: false
      }} : null;
      const stimTrace = stim ? {{
        type: 'scatter3d',
        mode: 'markers',
        x: [stim[0]],
        y: [stim[1]],
        z: [stim[2]],
        marker: {{
          size: 10,
          color: 'yellow',
          symbol: 'circle',
          line: {{color: 'black', width: 2}}
        }},
        name: 'Stim site',
        hovertemplate: '<b>Stim site</b><br>' + pair + '<br>' + (stimBipole ? stimBipole.contacts.join(' - ') : '') + '<extra></extra>'
      }} : null;

            // Electrode-only plot removed: electrodes are rendered on the surface plot below.

      // Mesh intensity
      const intensity = computeVertexIntensity(peaksForProj, inRoi);

      const opEl = $('opacitySlider');
      const meshOpacity = opEl ? parseFloat(opEl.value || '0.35') : 0.35;

      const showLeftHemi = $('showLeftHemi').checked;
      const showRightHemi = $('showRightHemi').checked;

      const leftVisible = !!(showLeftHemi && hemiLeftI && hemiLeftI.length);
      const rightVisible = !!(showRightHemi && hemiRightI && hemiRightI.length);

      const showScaleLeft = leftVisible && !rightVisible;
      const showScaleRight = rightVisible;

      const meshLeftTrace = {{
        type: 'mesh3d',
        x: meshX, y: meshY, z: meshZ,
        i: (hemiLeftI || meshI), j: (hemiLeftJ || meshJ), k: (hemiLeftK || meshK),
        intensity: intensity,
        colorscale: 'Plasma',
        cmin: 0,
        cmax: cmax,
        opacity: meshOpacity,
        visible: leftVisible,
        showscale: showScaleLeft,
        colorbar: {{ title: 'Peak |µV|<br>(current window)', len: 0.75 }},
        name: 'Left hemisphere'
      }};

      const meshRightTrace = {{
        type: 'mesh3d',
        x: meshX, y: meshY, z: meshZ,
        i: (hemiRightI || meshI), j: (hemiRightJ || meshJ), k: (hemiRightK || meshK),
        intensity: intensity,
        colorscale: 'Plasma',
        cmin: 0,
        cmax: cmax,
        opacity: meshOpacity,
        visible: rightVisible,
        showscale: showScaleRight,
        colorbar: {{ title: 'Peak |µV|<br>(current window)', len: 0.75 }},
        name: 'Right hemisphere'
      }};

      const meshElecTrace = {{
        type: 'scatter3d',
        mode: 'markers',
        x: xR, y: yR, z: zR,
        marker: {{
          size: 3,
          color: cR,
          colorscale: 'Plasma',
          cmin: 0,
          cmax: cmax,
          opacity: 0.9,
          showscale: false
        }},
        hovertemplate: '<b>%{{text}}</b><br>ROI: %{{customdata[1]}}<br>Peak |µV|: %{{customdata[2]:.3f}}<extra></extra>',
        text: cdR.map(i => elecNames[i]),
        customdata: cdR.map(i => [i, elecRegions[i] || 'Unknown', peaks[i] || 0]),
        name: 'Electrodes'
      }};


      const meshFadeTrace = {{
        type: 'scatter3d',
        mode: 'markers',
        x: xF, y: yF, z: zF,
        customdata: cdF.map(i => [i, elecRegions[i] || 'Unknown', peaks[i] || 0]),
        marker: {{ size: 2, color: '#6b7280', opacity: 0.20 }},
        hovertemplate: '<b>%{{text}}</b><br>ROI: %{{customdata[1]}}<br>Peak |µV|: %{{customdata[2]:.3f}}<extra></extra>',
        text: cdF.map(i => elecNames[i]),
        name: 'ROI (not selected)',
        showlegend: false
      }};


      const meshOtherTrace = {{
        type: 'scatter3d',
        mode: 'markers',
        x: xO,
        y: yO,
        z: zO,
        customdata: cdO.map(i => [i, elecRegions[i] || 'Unknown', peaks[i] || 0]),
        marker: {{
          size: 3,
          color: 'rgba(190,190,190,0.55)'
        }},
        hovertemplate: '<b>%{{text}}</b><br>ROI: %{{customdata[1]}}<br>Peak |µV|: %{{customdata[2]:.3f}}<extra></extra>',
        text: cdO.map(i => elecNames[i]),
        name: 'Other electrodes'
      }};

      const meshData = [meshLeftTrace, meshRightTrace, meshOtherTrace];
      if (xF && xF.length) {{ meshData.push(meshFadeTrace); }}
      meshData.push(meshElecTrace);
      if (stimLineTrace) {{ meshData.push(stimLineTrace); }}
      if (stimTrace) {{ meshData.push(stimTrace); }}
      const meshLayout = {{
        margin: {{l: 0, r: 0, t: 0, b: 0}},
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: sceneLayoutBase()
      }};

      await Plotly.react('meshDiv', meshData, meshLayout, config3d);
      bindMeshClick();

      
    // Detailed waveform
    // (meanFlat/ciFlat/times defined above from decoded evoked)
      // const meanFlat = ...

    function percentile(arr, p) {{
      if (!arr || arr.length === 0) return 0;
      const a = arr.slice().sort((x, y) => x - y);
      const idx = (p / 100) * (a.length - 1);
      const lo = Math.floor(idx), hi = Math.ceil(idx);
      if (lo === hi) return a[lo];
      const w = idx - lo;
      return a[lo] * (1 - w) + a[hi] * w;
    }}

    function getSelectedChannels() {{
      if (!chanSelect) return [];
      const sel = Array.from(chanSelect.selectedOptions).map(o => o.value).filter(Boolean);
      if (sel.length === 0 && chanSelect.options.length) {{
        chanSelect.options[0].selected = true;
        return [chanSelect.options[0].value];
      }}
      return sel;
    }}

    let selectedCh = getSelectedChannels().filter(ch => channelIndex(ch) >= 0);
    if (selectedCh.length === 0 && APP.ch_names && APP.ch_names.length) {{
      selectedCh = [APP.ch_names[0]];
      if (chanSelect && chanSelect.options.length) {{
        for (const opt of chanSelect.options) {{ opt.selected = (opt.value === selectedCh[0]); }}
      }}
    }}
    const showCI = selectedCh.length <= 6;

    // Robust montage spacing based on 95th percentile of |mean|.
    let absVals = [];
    for (const ch of selectedCh) {{
      const idx = channelIndex(ch);
      if (idx < 0) continue;
      const m = slice1D(meanFlat, idx * nTimes, nTimes);
      for (let i = 0; i < nTimes; i++) {{
        absVals.push(Math.abs(m[i]));
      }}
    }}
    let spacing = percentile(absVals, 95) * 3.0;
    if (!isFinite(spacing) || spacing <= 0) spacing = 1.0;

    // Build traces (stacked/offset montage).
    const waveData = [];
    const offsets = [];
    const tickText = [];
    for (let i = 0; i < selectedCh.length; i++) {{
      const ch = selectedCh[i];
      const chIdx = channelIndex(ch);
      const off = (selectedCh.length - 1 - i) * spacing;
      offsets.push(off);
      tickText.push(stripPOL(ch));

      const mean = slice1D(meanFlat, chIdx * nTimes, nTimes).map(v => v + off);

      if (showCI) {{
        const ci = slice1D(ciFlat, chIdx * nTimes, nTimes);
        const upper = mean.map((v, j) => v + ci[j]);
        const lower = mean.map((v, j) => v - ci[j]);

        // CI band (2 traces) + mean trace.
        waveData.push({{x: times, y: lower, type: 'scatter', mode: 'lines', line: {{width: 0}}, hoverinfo: 'skip', showlegend: false}});
        waveData.push({{x: times, y: upper, type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(120,170,255,0.18)', line: {{width: 0}}, hoverinfo: 'skip', showlegend: false}});
      }}

      waveData.push({{x: times, y: mean, type: 'scatter', mode: 'lines', name: stripPOL(ch), line: {{width: 2}}}});
    }}

    // Layout
    const titleText = (selectedCh.length <= 1)
      ? ('Detailed waveform — ' + stripPOL(selectedCh[0]) + ' | ' + pair + ' | ' + mode)
      : ('Montage (' + selectedCh.length + ' ch) — ' + pair + ' | ' + mode);

    const waveLayout = {{
      margin: {{l: 60, r: 20, t: 40, b: 45}},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      title: {{text: titleText, font: {{size: 14, color: 'rgba(229,231,235,0.95)'}}}},
      xaxis: {{title: 'Time (ms)', gridcolor: 'rgba(255,255,255,0.08)', zeroline: false}},
      yaxis: {{
        title: (selectedCh.length <= 1) ? 'µV' : ('µV (offset; ~' + Math.round(spacing) + ' µV spacing)'),
        gridcolor: 'rgba(255,255,255,0.06)',
        zeroline: false,
        tickvals: offsets,
        ticktext: tickText
      }},
      shapes: [
        {{type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: {{color: 'rgba(200,200,200,0.6)', width: 1, dash: 'dash'}}}},
        {{type: 'rect', x0: viewWin[0], x1: viewWin[1], y0: 0, y1: 1, yref: 'paper', fillcolor: 'rgba(255,215,160,0.18)', line: {{width: 0}}}}
      ],
      annotations: [
        {{
          xref: 'paper', yref: 'paper', x: 0.0, y: 1.02,
          text: (selectedCh.length <= 1)
            ? ('n=' + evk.nTrials + '  |  ' + pair + '  |  ' + mode + '  |  ' + stripPOL(selectedCh[0]))
            : ('n=' + evk.nTrials + '  |  ' + pair + '  |  ' + mode + '  |  ' + selectedCh.length + ' channels'),
          showarrow: false,
          font: {{size: 12, color: 'rgba(229,231,235,0.9)'}},
          align: 'left'
        }}
      ]
    }};

    await Plotly.react('waveDiv', waveData, waveLayout, config2d);
    }}

    // Click on electrodes -> update channel
    // NOTE: Plotly adds the `.on()` helper to the graph div *after* the first plot
    // is created. Binding before `Plotly.react()` runs will throw:
    //   TypeError: meshDiv.on is not a function
    //
    // So we define a binder and call it after the electrode plot is first rendered.
    function bindMeshClick() {{
      const meshDiv = $('meshDiv');
      if (!meshDiv) return;
      if (typeof meshDiv.on !== 'function') return;

      if (typeof meshDiv.removeAllListeners === 'function') {{
        meshDiv.removeAllListeners('plotly_click');
      }}

      meshDiv.on('plotly_click', (data) => {{
        const pt = data && data.points && data.points[0];
        if (!pt) return;
        const cd = pt.customdata;
        const idx = Array.isArray(cd) ? cd[0] : cd;
        if (idx === undefined || idx === null) return;
        const ch = (APP.electrodes.names_raw && APP.electrodes.names_raw[idx]) ? APP.electrodes.names_raw[idx] : elecNames[idx];
        if (!ch) return;
        const sel = $('chanSelect');
        const add = data && data.event && (data.event.shiftKey || data.event.ctrlKey || data.event.metaKey);
        if (sel && sel.options) {{
          if (!add) {{ for (const opt of sel.options) {{ opt.selected = false; }} }}
          for (const opt of sel.options) {{ if (opt.value === ch) {{ opt.selected = true; }} }}
        }}
        updateAll();
      }});
    }}

    // Mesh opacity slider (surface transparency)
    const opacitySlider = $('opacitySlider');
    if (opacitySlider) {{
      opacitySlider.addEventListener('input', () => {{
        const op = parseFloat(opacitySlider.value || '0.35');
        // Traces 0–1 are the left/right hemisphere mesh surfaces.
        try {{ Plotly.restyle('meshDiv', {{opacity: op}}, [0, 1]); }} catch (e) {{ /* ignore */ }}
      }});
    }}

    // Control changes
    for (const el of [pairSelect, modeSelect, roiSelect, chanSelect, $('viewSelect'), $('earlyStart'), $('earlyEnd'), $('lateStart'), $('lateEnd'), $('showLeftHemi'), $('showRightHemi')]) {{
      el.addEventListener('change', () => updateAll());
      el.addEventListener('input', () => updateAll());
    }}

    function updateProjBtnLabel() {{
      if (!projBtn) return;
      projBtn.textContent = projectOnlySelected ? 'Project All Sensors' : 'Project Only Selected Sensor Montage';
    }}

    function captureDefaults() {{
      const sel = Array.from(chanSelect.options).filter(o => o.selected).map(o => o.value);
      return {{
        pair: pairSelect.value,
        mode: modeSelect.value,
        view: document.getElementById('viewSelect').value,
        earlyStart: document.getElementById('earlyStart').value,
        earlyEnd: document.getElementById('earlyEnd').value,
        lateStart: document.getElementById('lateStart').value,
        lateEnd: document.getElementById('lateEnd').value,
        roi: roiSelect.value,
        opacity: document.getElementById('opacitySlider').value,
        showLeft: document.getElementById('showLeftHemi').checked,
        showRight: document.getElementById('showRightHemi').checked,
        selChannels: sel
      }};
    }}

    function applyDefaults(d) {{
      if (!d) return;
      pairSelect.value = d.pair;
      modeSelect.value = d.mode;
      document.getElementById('viewSelect').value = d.view;
      document.getElementById('earlyStart').value = d.earlyStart;
      document.getElementById('earlyEnd').value = d.earlyEnd;
      document.getElementById('lateStart').value = d.lateStart;
      document.getElementById('lateEnd').value = d.lateEnd;
      roiSelect.value = d.roi;
      document.getElementById('opacitySlider').value = d.opacity;
      document.getElementById('showLeftHemi').checked = d.showLeft;
      document.getElementById('showRightHemi').checked = d.showRight;
      for (const opt of chanSelect.options) {{
        opt.selected = d.selChannels.includes(opt.value);
      }}
      projectOnlySelected = false;
      updateProjBtnLabel();
    }}

    function resetDefaults() {{
      applyDefaults(DEFAULTS);
      updateAll();
    }}

    if (projBtn) {{
      projBtn.addEventListener('click', () => {{
        projectOnlySelected = !projectOnlySelected;
        updateProjBtnLabel();
        updateAll();
      }});
    }}
    if (resetBtn) {{
      resetBtn.addEventListener('click', () => {{
        resetDefaults();
      }});
    }}

    DEFAULTS = captureDefaults();
    updateProjBtnLabel();
    setStatus("Ready.");
    await updateAll();
  }})();
  </script>
</body>
</html>
"""

    def export_results(self) -> None:
        """Export one PDF report (simplified).

        Structure (selected surface only):
          • One *electrode placement + stim-site* page per stim pair (6 views), at the start of the PDF.
          • For each stim pair × mode (ERP/RMS/Gamma):
              - Waveforms page (full-page grid)
              - 3D activation page (6-view mesh; stim-site marker; max-activation ROI)
        """
        if self.raw is None or self.events is None or not len(self.events):
            messagebox.showerror("Export", "Load an EDF and detect stimulation events first.")
            return

        # Apply current appearance settings before exporting
        try:
            self._apply_style_now()
        except Exception:
            pass
        self.theme.apply()

        default_name = f"ieeg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF", "*.pdf")],
        )
        if not out_path:
            return
        pdf_path = Path(out_path)
        export_t0 = time.perf_counter()

        surface = self._current_surface_key()  # ONLY selected texture
        surf_label = self._surface_display_name(surface)

        view_specs = self._export_view_specs()  # [(label, elev, azim), ...]

        window_ms = self._current_view_window_ms()
        view_lbl = self._current_view_label()

        min_time = float(self.min_time_var.get())
        tmin, tmax = float(self.tmin_var.get()), float(self.tmax_var.get())
        summary_window_ms = window_ms if window_ms is not None else (tmin * 1000.0, tmax * 1000.0)
        baseline = (float(self.baseline_min_var.get()), float(self.baseline_max_var.get()))

        # Early/late windows are used only for the final decision-support summary page.
        # (The plotted pages still respect the currently selected view window.)
        try:
            early_ms = (float(self.early_start_ms_var.get()), float(self.early_end_ms_var.get()))
            late_ms = (float(self.late_start_ms_var.get()), float(self.late_end_ms_var.get()))
        except Exception:
            # Sensible fallbacks if GUI vars are missing/corrupted.
            early_ms = (10.0, 80.0)
            late_ms = (80.0, 300.0)

        prev_pair = self.stim_pair_var.get() if hasattr(self, "stim_pair_var") else ""

        # Collect per-(stim pair × analysis mode) summary metrics for the final page.
        summary_rows: List[Dict[str, Any]] = []

        def stim_center_from_pair(pair_name: str) -> Optional[np.ndarray]:
            """Return stim-site midpoint in *meters* (or None if unresolved)."""
            if self.montage is None or "-" not in pair_name:
                return None
            left_tok, right_tok = pair_name.split("-", 1)
            left_ch = self._resolve_stim_token(left_tok)
            right_ch = self._resolve_stim_token(right_tok)
            if not left_ch or not right_ch:
                return None
            chp = self.montage.get_positions()["ch_pos"]
            if left_ch not in chp or right_ch not in chp:
                return None
            p1 = np.asarray(chp[left_ch], dtype=float)
            p2 = np.asarray(chp[right_ch], dtype=float)
            # If either stim electrode is missing (0,0,0), don't draw a misleading marker.
            if np.allclose(p1, 0.0) or np.allclose(p2, 0.0):
                return None
            return 0.5 * (p1 + p2)

        def stim_label_from_pair(pair_name: str) -> str:
            """Return a compact stim-site label including regions when available."""
            if "-" not in pair_name:
                return pair_name
            left_tok, right_tok = pair_name.split("-", 1)
            left_ch = self._resolve_stim_token(left_tok) or left_tok
            right_ch = self._resolve_stim_token(right_tok) or right_tok
            left_reg = self._channel_to_region.get(left_ch, "Unknown") if hasattr(self, "_channel_to_region") else "Unknown"
            right_reg = self._channel_to_region.get(right_ch, "Unknown") if hasattr(self, "_channel_to_region") else "Unknown"
            left_short = self._strip_pol_prefix(left_ch)
            right_short = self._strip_pol_prefix(right_ch)
            if left_reg == right_reg:
                return f"{left_short}-{right_short} ({left_reg})"
            return f"{left_short}-{right_short} ({left_reg} / {right_reg})"

        def roi_positions_values_channels(
            evk: mne.Evoked,
            *,
            window_ms_override: Optional[Tuple[float, float]] = None,
        ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
            """Electrode positions + RAW peak |µV| + channels, respecting ROI filter."""
            if self.electrode_coords is None or self.raw is None:
                return np.empty((0, 3)), np.empty((0,)), []
            data = evk.data
            use_window_ms = window_ms_override if window_ms_override is not None else window_ms
            if use_window_ms is not None:
                mask = self._window_mask(evk.times * 1000.0, *use_window_ms)
                data = data[:, mask] if mask.any() else np.zeros((data.shape[0], 0))
            peaks_uv = (np.max(np.abs(data), axis=1) * 1e6) if data.shape[1] else np.zeros(len(evk.ch_names), float)
            idx_map = {ch: i for i, ch in enumerate(evk.ch_names)}

            roi_choice = self.roi_var.get()
            top_labels: Optional[set[str]] = None
            if roi_choice == "Top 5 responding ROIs":
                top_labels = set(self._top_roi_labels_from_values(evk.ch_names, peaks_uv, self._channel_to_region, 5))
                df = self.electrode_coords
            elif roi_choice == "Top 10 responding ROIs":
                top_labels = set(self._top_roi_labels_from_values(evk.ch_names, peaks_uv, self._channel_to_region, 10))
                df = self.electrode_coords
            else:
                df = (
                    self.electrode_coords
                    if roi_choice == "All ROIs"
                    else self.electrode_coords[self.electrode_coords["Regions"] == roi_choice]
                )
            if df.empty or not evk.ch_names:
                return np.empty((0, 3)), np.empty((0,)), []

            pos: List[Tuple[float, float, float]] = []
            vals: List[float] = []
            chans: List[str] = []
            for _, row in df.iterrows():
                idx1 = int(row["ElecNumber"])
                ch = self._elecnum_to_channel.get(idx1)
                if not ch or ch not in idx_map:
                    continue
                if top_labels is not None and self._channel_to_region.get(ch, "Unknown") not in top_labels:
                    continue
                vals.append(float(peaks_uv[idx_map[ch]]))
                pos.append((row["X"] / 1000.0, row["Y"] / 1000.0, row["Z"] / 1000.0))
                chans.append(ch)
            return (np.asarray(pos) if pos else np.empty((0, 3))), (np.asarray(vals) if vals else np.empty((0,))), chans

        def mesh_face_values_for_evoked(
            evk: mne.Evoked,
            pair_name: str,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float, float, str, float, float, Optional[Tuple[float, float]]]:
            """Compute face values once per page (shared across all 6 views).

            Uses RAW per-electrode peak |µV| for the mesh. If outliers dominate,
            values are clipped for visibility; the annotation will report both the
            displayed peak and the raw peak when clipping occurs.

            Returns:
              verts, faces, fvals, stim_center, vmin, vmax,
              max_label, peak_display_uv, peak_raw_uv, clip_info
            """
            verts, faces = self._get_fsaverage_mesh(surface)
            brain_center = verts.mean(axis=0)

            if self.montage is None:
                return verts, faces, np.zeros((faces.shape[0],), float), None, 0.0, 1.0, "Unknown", 0.0, 0.0, None

            pos_roi, vals_roi, chans_roi = roi_positions_values_channels(evk)
            if pos_roi.size == 0 or vals_roi.size == 0 or not chans_roi:
                return verts, faces, np.zeros((faces.shape[0],), float), None, 0.0, 1.0, "Unknown", 0.0, 0.0, None

            # Clip extreme values for *display* so non-dominant responses remain visible
            vals_adj, clip_info = self._compress_raw_roi_values(vals_roi.copy())

            # Determine max activation (DISPLAY scale), and keep raw for reference
            idx_max = int(np.argmax(vals_adj)) if vals_adj.size else 0
            ch_max = chans_roi[idx_max] if chans_roi else "Unknown"
            region = self._channel_to_region.get(ch_max, "Unknown")
            max_label = f"{region} ({self._strip_pol(ch_max)})"
            peak_display_uv = float(vals_adj[idx_max]) if vals_adj.size else 0.0
            peak_raw_uv = float(vals_roi[idx_max]) if vals_roi.size else 0.0

            vvals = self._project_to_vertices(verts, pos_roi, vals_adj)
            fvals = np.mean(vvals[faces], axis=1)

            vmin = 0.0  # |µV| map: start at 0 for interpretability
            vmax = float(np.max(fvals)) if fvals.size else 1.0
            if np.isclose(vmax, vmin):
                vmax = vmin + max(1.0, abs(vmin) * 0.05 + 1e-6)

            stim_center = stim_center_from_pair(pair_name)
            return verts, faces, fvals, stim_center, vmin, vmax, max_label, peak_display_uv, peak_raw_uv, clip_info

        try:
            with PdfPages(str(pdf_path)) as pdf:
                # -----------------------------------------------------------------
                # START PAGES: electrode placements for each stim pair (6 views)
                # -----------------------------------------------------------------
                for pair in self.stim_pairs:
                    stim_center = stim_center_from_pair(pair)

                    fig0 = plt.Figure(figsize=(11, 8.5), dpi=120, constrained_layout=True)
                    gs0 = gridspec.GridSpec(2, 3, figure=fig0)
                    axes0: List[Axes3D] = [
                        fig0.add_subplot(gs0[r, c], projection="3d")  # type: ignore[arg-type]
                        for r in range(2) for c in range(3)
                    ]
                    for ax, (lab, el, az) in zip(axes0, view_specs):
                        self._draw_electrode_placements(
                            ax,
                            surface,
                            title=f"{lab} view",
                            stim_center=stim_center,
                            for_export=True,
                        )
                        ax.view_init(elev=el, azim=az)

                    fig0.suptitle(
                        f"Electrode Placements + Stim Site — Pair {pair} — {surf_label}",
                        fontsize=14 * self.theme.font_scale,
                    )

                    if stim_center is None:
                        fig0.text(
                            0.985,
                            0.02,
                            "Stim-site marker unavailable (missing coordinates for stim electrodes).",
                            ha="right",
                            va="bottom",
                            fontsize=9 * self.theme.font_scale,
                            color="#444",
                        )

                    pdf.savefig(fig0)
                    plt.close(fig0)

                # -----------------------------------------------------------------
                # MAIN CONTENT: per pair × mode pages
                # -----------------------------------------------------------------
                for pair in self.stim_pairs:
                    clean_evs = self._clean_events(pair, min_time)
                    if len(clean_evs) == 0:
                        continue

                    try:
                        self.stim_pair_var.set(pair)
                    except Exception:
                        pass

                    epochs = mne.Epochs(
                        self.raw,
                        clean_evs,
                        event_id={f"stim_{pair}": 1},
                        tmin=tmin,
                        tmax=tmax,
                        baseline=baseline,
                        preload=True,
                        verbose=False,
                    )

                    for mode in ("ERP", "RMS", "Gamma"):
                        evk = self._compute_evoked_for_mode(epochs, mode)

                        # Page A: waveforms
                        fig_wave = self._build_grid_figure(evk, epochs, pair, mode)
                        fig_wave.set_size_inches(11, 8.5, forward=True)
                        pdf.savefig(fig_wave)
                        plt.close(fig_wave)
                        # Page B: 3D activation (6 views; selected surface only)
                        fig = plt.Figure(figsize=(11, 8.5), dpi=120)
                        gs = gridspec.GridSpec(
                            2,
                            4,
                            figure=fig,
                            width_ratios=[1.0, 1.0, 1.0, 0.07],
                            wspace=0.02,
                            hspace=0.02,
                        )
                        axes: List[Axes3D] = [
                            fig.add_subplot(gs[r, c], projection="3d")  # type: ignore[arg-type]
                            for r in range(2) for c in range(3)
                        ]
                        cax = fig.add_subplot(gs[:, 3])

                        verts, faces, fvals, stim_center, vmin, vmax, max_label, pk_disp, pk_raw, clip_info = mesh_face_values_for_evoked(
                            evk, pair
                        )

                        # Collect per-(pair × mode) summary metrics for the final page.
                        try:
                            _pos_arr, vals_raw_uv, chans_roi = roi_positions_values_channels(evk)
                            vals_adj_uv, _clip_info2 = self._compress_raw_roi_values(vals_raw_uv)

                            if vals_adj_uv.size:
                                idx_max = int(np.argmax(vals_adj_uv))
                                ch_max = chans_roi[idx_max]
                                reg_max = self._channel_to_region.get(ch_max, "Unknown")
                            else:
                                ch_max = ""
                                reg_max = "Unknown"

                            # Early and late peak responders (used only for the summary page).
                            _pos_e, early_raw_uv, early_chans = roi_positions_values_channels(
                                evk, window_ms_override=early_ms
                            )
                            early_adj_uv, _ = self._compress_raw_roi_values(early_raw_uv)
                            if early_adj_uv.size:
                                i_e = int(np.argmax(early_adj_uv))
                                ch_e = early_chans[i_e]
                                reg_e = self._channel_to_region.get(ch_e, "Unknown")
                            else:
                                ch_e = ""
                                reg_e = "Unknown"
                            early_peak_disp_uv = float(np.max(early_adj_uv)) if early_adj_uv.size else 0.0

                            _pos_l, late_raw_uv, late_chans = roi_positions_values_channels(
                                evk, window_ms_override=late_ms
                            )
                            late_adj_uv, _ = self._compress_raw_roi_values(late_raw_uv)
                            if late_adj_uv.size:
                                i_l = int(np.argmax(late_adj_uv))
                                ch_l = late_chans[i_l]
                                reg_l = self._channel_to_region.get(ch_l, "Unknown")
                            else:
                                ch_l = ""
                                reg_l = "Unknown"
                            late_peak_disp_uv = float(np.max(late_adj_uv)) if late_adj_uv.size else 0.0

                            peak_disp_uv = float(pk_disp)
                            spread_frac = (
                                float(np.mean(vals_adj_uv >= (0.5 * peak_disp_uv)))
                                if peak_disp_uv > 0 and vals_adj_uv.size
                                else 0.0
                            )
                            summary_rows.append(
                                {
                                    "pair": pair,
                                    "mode": mode,
                                    "n_trials": int(len(clean_evs)),
                                    "roi_filter": str(self.roi_var.get()) if hasattr(self, "roi_var") else "All",
                                    "surface": str(surf_label),
                                    "view": str(view_lbl),
                                    "window_ms": tuple(float(x) for x in summary_window_ms),
                                    "stim_label": stim_label_from_pair(pair),
                                    "stim_center_ok": bool(stim_center is not None),
                                    "max_region": str(reg_max),
                                    "max_channel": str(self._strip_pol_prefix(ch_max)) if ch_max else "",
                                    "peak_display_uv": float(pk_disp),
                                    "peak_raw_uv": float(pk_raw),
                                    "early_max_region": str(reg_e),
                                    "early_max_channel": str(self._strip_pol_prefix(ch_e)) if ch_e else "",
                                    "early_peak_display_uv": float(early_peak_disp_uv),
                                    "late_max_region": str(reg_l),
                                    "late_max_channel": str(self._strip_pol_prefix(ch_l)) if ch_l else "",
                                    "late_peak_display_uv": float(late_peak_disp_uv),
                                    "clip": str(clip_info),
                                    "spread_frac": float(spread_frac),
                                    "n_channels": int(vals_adj_uv.size),
                                }
                            )
                        except Exception as exc:
                            LOG.exception(
                                "Failed to compute export summary metrics for pair=%s mode=%s: %s",
                                pair,
                                mode,
                                exc,
                            )
                        brain_center = verts.mean(axis=0)
                        norm = plt.Normalize(vmin=vmin, vmax=vmax)
                        sm0: Optional[mpl.cm.ScalarMappable] = None

                        for ax, (lab, el, az) in zip(axes, view_specs):
                            poly = Poly3DCollection(verts[faces])
                            ax.add_collection3d(poly)
                            self._set_axes_equal_3d(ax, verts)
                            sm = self._apply_shaded_mesh(
                                poly,
                                verts,
                                faces,
                                face_values=fvals,
                                cmap_name=self.theme.colormap,
                                norm=norm,
                                alpha=0.95,
                            )
                            if sm0 is None:
                                sm0 = sm
                            self._mark_stim_center(ax, stim_center, brain_center=brain_center)
                            ax.set_title(f"{lab}", fontsize=10 * self.theme.font_scale)
                            ax.view_init(elev=el, azim=az)
                            self._style_3d_axes_for_export(ax)

                        fig.suptitle(f"{mode} — Pair {pair} — {surf_label} — {view_lbl}", fontsize=14 * self.theme.font_scale)

                        if sm0 is not None:
                            base_label = "Peak |µV| (Gamma envelope)" if mode == "Gamma" else "Peak |µV|"
                            label = base_label + (" (clipped)" if clip_info is not None else "")
                            cb = fig.colorbar(sm0, cax=cax)
                            cb.set_label(label)
                        else:
                            cax.set_axis_off()

                        # Max activation annotation (display-consistent; show raw if clipped)
                        annot = f"Max activation:\n{max_label}\nPeak |µV|: {pk_disp:.2f}"
                        if clip_info is not None and pk_raw > pk_disp + 1e-6:
                            annot += f"\nRaw peak: {pk_raw:.2f}"
                        fig.text(
                            0.86,
                            0.97,
                            annot,
                            ha="left",
                            va="top",
                            fontsize=10 * self.theme.font_scale,
                            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#444", alpha=0.9),
                        )

                        pdf.savefig(fig)
                        plt.close(fig)

                # -----------------------------
                # FINAL PAGE: decision-support
                # -----------------------------
                try:
                    fig_summary = self._build_clinical_summary_page(
                        summary_rows,
                        export_path=pdf_path,
                        surface_label=surf_label,
                        view_label=view_lbl,
                        window_ms=summary_window_ms,
                        early_ms=early_ms,
                        late_ms=late_ms,
                    )
                    pdf.savefig(fig_summary)
                    plt.close(fig_summary)
                except Exception as exc:
                    LOG.exception("Failed to build clinical summary page: %s", exc)

            self._record_workflow_timing(
                "pdf_report_creation",
                time.perf_counter() - export_t0,
                file=pdf_path.name,
                n_stim_pairs=int(len(self.stim_pairs)),
            )
            messagebox.showinfo("Export", f"PDF report saved:\n{pdf_path}")
        except Exception as exc:
            LOG.exception("PDF export failed: %s", exc)
            messagebox.showerror("Export", f"Failed to export PDF report:\n{exc}")
        finally:
            try:
                if prev_pair:
                    self.stim_pair_var.set(prev_pair)
            except Exception:
                pass
    def reset(self) -> None:
        self.selected_channel = None
        self.selected_electrodes.clear()

        self.fig3d.clf()
        self.figdetail.clf()

        self.mesh_poly = None
        self.scatter_3d = None
        self.stim_marker_3d = None
        self._roi_mask_3d = []

        if self.canvas3d:
            self.canvas3d.draw()
        if self.canvasdetail:
            self.canvasdetail.draw()
        if self.selection_listbox:
            self.selection_listbox.delete(0, tk.END)

        self._update_status("Ready")

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "iEEG ERP Analyzer — Publication‑quality + Interactive 3D + Early/Late edition\n"
            "• ERP, RMS, time‑resolved Gamma; Pial/White ROI; multi‑view export & gallery; Plotly HTML exports.\n"
            "• Robust stim‑pair channel resolution, auto‑recompute on pair change, separate 50/60 Hz notch.\n"
            "• Upgraded 3D visuals (Lambert shading, halo markers, radial offsets+jitter, selection rings, axis triads).\n"
            "• Early (10–80 ms) vs Late (80–300 ms) windows, per‑channel metrics & ROI summaries for ALL stim pairs.\n"
            "• Early/Late ROI bar figures (Early, Late, and Late−Early difference).\n"
            "• Early vs Late 3D connectivity: static & interactive template meshes exported separately.",
        )

    def main(self) -> None:
        self.root.mainloop()


def main() -> None:
    root = tk.Tk()
    IEEGAnalyzer(root).main()


if __name__ == "__main__":
    main()
