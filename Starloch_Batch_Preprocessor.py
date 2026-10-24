#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Starloch Batch Preprocessor (SBPP) for Siril 1.4.4+.

A WBPP-style front end around the verified Starloch v3 native-scale workflow.
The script is intentionally self-contained so it can be placed directly in a
Siril script directory and launched from Siril's Python Scripts menu.

Core processing remains inside Siril. This program discovers and groups inputs,
matches calibration data, creates a reviewed plan, and calls Siril through
sirilpy; optional mosaic gradient correction uses the standalone GraXpert CLI.
"""

from __future__ import annotations

import ast
import faulthandler
import json
import math
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TextIO

import sirilpy as s

# Siril manages one shared Python environment for scripts. These are both on
# the official known-good list and ensure_installed is a no-op when present.
s.ensure_installed("astropy", "PyQt6")

from astropy.io import fits  # pylint: disable=wrong-import-position
from astropy.wcs import WCS  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from PyQt6.QtCore import (  # pylint: disable=wrong-import-position
    QPointF,
    QStandardPaths,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (  # pylint: disable=wrong-import-position
    QColor,
    QFont,
    QBrush,
    QGuiApplication,
    QPainter,
    QPalette,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (  # pylint: disable=wrong-import-position
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "Starloch Batch Preprocessor"
APP_SLUG = "starloch-batch-preprocessor"
VERSION = "1.12.0"
KOFI_URL = "https://ko-fi.com/rkremeier"
GITHUB_REPO = "YoruDev-Ryland/Starloch-Batch-Preprocessor"
SCRIPT_FILENAME = "Starloch_Batch_Preprocessor.py"
UPDATE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
UPDATE_CHECK_INTERVAL_SECONDS = 24 * 3600
SCHEMA_VERSION = 1
DEFAULTS_REVISION = 8
METADATA_CACHE_VERSION = 4
CLEANUP_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0)
TINY_STACK_MEDIAN_MAX = 7
MIN_LINEAR_STACK_COUNT = 40
TINY_STACK_HOT_SIGMA = "2.5"
STANDARD_HOT_SIGMA = "3"
SHARED_WCS_TOLERANCE_PIXELS = 0.05
ORIENTATION_WCS_TOLERANCE_ARCSEC = 0.05
TARGET_INFERENCE_HEADER_LIMIT = 32

FITS_SUFFIXES = (".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz")
XISF_SUFFIXES = (".xisf",)
RAW_SUFFIXES = (
    ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".dng",
)
OTHER_IMAGE_SUFFIXES = (".tif", ".tiff", ".ser")
SUPPORTED_SUFFIXES = FITS_SUFFIXES + XISF_SUFFIXES + RAW_SUFFIXES + OTHER_IMAGE_SUFFIXES

SKIP_DIR_NAMES = {
    ".bad", ".git", "cache", "calibrated", "registered", "rejected",
    "drizztmp", "output", "outputs", "process", "processing", "tmp",
}
SKIP_DIR_PREFIXES = (
    "process_", "starloch_process", "sbpp_", "wbpp", "pi_output",
)

FRAME_LIGHT = "LIGHT"
FRAME_BIAS = "BIAS"
FRAME_DARK = "DARK"
FRAME_FLAT = "FLAT"
FRAME_DARKFLAT = "DARKFLAT"
FRAME_UNKNOWN = "UNKNOWN"

FILTER_ORDER = {
    "H": 0, "HA": 0, "HALPHA": 0,
    "S": 1, "SII": 1,
    "O": 2, "OIII": 2,
    "L": 3, "LUM": 3, "LUMINANCE": 3,
    "R": 4, "RED": 4,
    "G": 5, "GREEN": 5,
    "B": 6, "BLUE": 6,
    "OSC": 7, "RGB": 7,
}

PANEL_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:panel|pane|tile|field|pointing)"
    r"[\s_.-]*([A-Za-z]*\d+[A-Za-z]*|\d+[A-Za-z]*)(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


def utc_now() -> str:
    """Return a stable UTC timestamp for manifests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: Any, fallback: str = "unknown") -> str:
    """Make a Siril/filesystem-safe identifier."""
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def normalized_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def comparable_instrument(value: Any) -> str:
    """Return a model-bearing camera ID, ignoring generic driver labels."""
    name = normalized_text(value)
    without_driver = name.replace("ASCOM", "")
    if re.fullmatch(r"(?:ZWO)?ASICAMERA\d*", without_driver):
        return ""
    if re.fullmatch(r"(?:ZWO)?CAMERA\d*", without_driver):
        return ""
    return name


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    return int(round(number)) if number is not None else None


def sexagesimal_degrees(value: Any, *, hours: bool = False) -> Optional[float]:
    """Parse decimal or sexagesimal coordinates without assuming folder names."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().strip("'\"")
    normalized = re.sub(r"[hHdD°mM'′sS\"″:]", " ", text)
    parts = [part for part in normalized.split() if part]
    try:
        if len(parts) == 1:
            number = float(parts[0])
            result = number * 15.0 if hours else number
        else:
            sign = -1.0 if parts[0].startswith("-") else 1.0
            major = abs(float(parts[0]))
            minutes = abs(float(parts[1])) if len(parts) > 1 else 0.0
            seconds = abs(float(parts[2])) if len(parts) > 2 else 0.0
            result = sign * (major + minutes / 60.0 + seconds / 3600.0)
            if hours:
                result *= 15.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result % 360.0 if hours else result


def position_angle_from_header(header: dict[str, Any]) -> Optional[float]:
    """Sky angle of the image +Y axis, degrees east of north, or None.

    WCS rotation wins over NINA's planned framing angle (OBJCTROT). The
    mechanical ROTATOR/ROTATANG angles are deliberately ignored: they are not
    sky angles.
    """
    def value(key: str) -> Optional[float]:
        return safe_float(first_header(header, key))

    cd12, cd22 = value("CD1_2"), value("CD2_2")
    if cd12 is not None and cd22 is not None and (cd12 or cd22):
        return math.degrees(math.atan2(cd12, cd22)) % 360.0
    pc12, pc22 = value("PC1_2"), value("PC2_2")
    cdelt1, cdelt2 = value("CDELT1"), value("CDELT2")
    if pc12 is not None and pc22 is not None and cdelt1 and cdelt2:
        # CD = diag(CDELT1, CDELT2) @ PC
        return math.degrees(math.atan2(cdelt1 * pc12, cdelt2 * pc22)) % 360.0
    crota2 = value("CROTA2")
    if crota2 is not None and cdelt2:
        return (-crota2) % 360.0
    planned = value("OBJCTROT")
    if planned is not None:
        return planned % 360.0
    return None


def pointing_from_header(
    header: dict[str, Any]
) -> tuple[Optional[float], Optional[float], str]:
    """Return the most stable available celestial center and its provenance."""
    object_ra = first_header(header, "OBJCTRA", "OBJRA")
    object_dec = first_header(header, "OBJCTDEC", "OBJDEC")
    if object_ra is not None and object_dec is not None:
        ra = sexagesimal_degrees(object_ra, hours=True)
        dec = sexagesimal_degrees(object_dec)
        if ra is not None and dec is not None and -90.0 <= dec <= 90.0:
            return ra, dec, "target"

    crval1 = safe_float(first_header(header, "CRVAL1"))
    crval2 = safe_float(first_header(header, "CRVAL2"))
    ctype1 = normalized_text(first_header(header, "CTYPE1"))
    ctype2 = normalized_text(first_header(header, "CTYPE2"))
    if (
        crval1 is not None and crval2 is not None
        and (not ctype1 or "RA" in ctype1)
        and (not ctype2 or "DEC" in ctype2)
        and -90.0 <= crval2 <= 90.0
    ):
        return crval1 % 360.0, crval2, "wcs"

    mount_ra = safe_float(first_header(header, "RA", "TELRA"))
    mount_dec = safe_float(first_header(header, "DEC", "TELDEC"))
    if (
        mount_ra is not None and mount_dec is not None
        and -90.0 <= mount_dec <= 90.0
    ):
        return mount_ra % 360.0, mount_dec, "mount"
    return None, None, ""


def almost_equal(left: Optional[float], right: Optional[float], tolerance: float) -> bool:
    return left is None or right is None or abs(left - right) <= tolerance


def display_number(value: Optional[float], digits: int = 1) -> str:
    return "?" if value is None else f"{value:.{digits}f}"


def file_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def is_supported(path: Path) -> bool:
    return path.is_file() and any(path.name.lower().endswith(s) for s in SUPPORTED_SUFFIXES)


def should_skip_dir(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in SKIP_DIR_NAMES
        or "_sbpp_" in name
        or any(name.startswith(prefix) for prefix in SKIP_DIR_PREFIXES)
    )


def clean_header_value(value: Any) -> Any:
    """Convert XISF FITSKeyword text into a normal Python scalar."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (str, int, float, bool)):
            return parsed
    except (SyntaxError, ValueError):
        pass
    upper = text.upper()
    if upper in {"T", "TRUE"}:
        return True
    if upper in {"F", "FALSE"}:
        return False
    number = safe_float(text)
    return number if number is not None else text.strip("'\"")


def first_header(header: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_filter(value: Any, bayer: str = "") -> str:
    text = str(value or "").strip()
    if not text or normalized_text(text) in {"NONE", "NOFILTER", "CLEAR"}:
        return "OSC" if bayer else "L"
    aliases = {
        "HALPHA": "H", "HA": "H", "HΑ": "H",
        "SII": "S", "SULFURII": "S", "SULPHURII": "S",
        "OIII": "O", "OXYGENIII": "O",
        "RED": "R", "GREEN": "G", "BLUE": "B",
        "LUM": "L", "LUMINANCE": "L",
    }
    return aliases.get(normalized_text(text), text)


def channel_key(dataset_id: str, filter_name: str) -> str:
    """Stable identifier used by the plan and UI for per-channel output mode."""
    return f"{dataset_id}:{filter_name}"


def natural_key(value: str) -> tuple[Any, ...]:
    """Sort panel identifiers as panel_2 before panel_10."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def detect_panel_id(meta: "FrameMeta", input_root: Path) -> str:
    """Find a target-independent mosaic panel label from folders, file, or OBJECT.

    Acquisition applications normally place mosaic lights below folders such as
    ``Panel 1`` or ``Tile_03``. The OBJECT value and filename are fallbacks for
    flatter layouts. A normal non-mosaic project is one implicit panel.
    """
    path = meta.path_obj
    try:
        relative = path.relative_to(input_root.resolve())
    except (OSError, ValueError):
        relative = path
    candidates = [*relative.parts[:-1], path.stem, meta.object_name]
    for candidate in candidates:
        match = PANEL_PATTERN.search(str(candidate))
        if match:
            token = slugify(match.group(1), "01").lower()
            return f"panel_{token}"
    return "panel_01"


def angular_separation_deg(
    left_ra: float, left_dec: float, right_ra: float, right_dec: float
) -> float:
    """Great-circle distance with stable behavior around RA=0."""
    ra1, dec1, ra2, dec2 = map(
        math.radians, (left_ra, left_dec, right_ra, right_dec)
    )
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _axis_fields_of_view(meta: "FrameMeta") -> list[Optional[float]]:
    """Angular size along image X and Y from acquisition headers (None if unknown)."""
    if not meta.focal_length_mm or meta.focal_length_mm <= 0:
        return [None, None]
    sizes: list[Optional[float]] = []
    for pixels, pixel_size in (
        (meta.width, meta.pixel_size_x_um),
        (meta.height, meta.pixel_size_y_um or meta.pixel_size_x_um),
    ):
        if pixels and pixel_size and pixels > 0 and pixel_size > 0:
            sensor_mm = pixels * pixel_size / 1000.0
            sizes.append(math.degrees(
                2.0 * math.atan(sensor_mm / (2.0 * meta.focal_length_mm))
            ))
        else:
            sizes.append(None)
    return sizes


def field_of_view_deg(meta: "FrameMeta") -> Optional[tuple[float, float]]:
    """Angular (width along image X, height along image Y), or None if unknown."""
    width, height = _axis_fields_of_view(meta)
    if width is None or height is None:
        return None
    return width, height


def short_field_of_view_deg(meta: "FrameMeta") -> Optional[float]:
    """Estimate the sensor's short angular dimension from acquisition headers."""
    dimensions = [size for size in _axis_fields_of_view(meta) if size is not None]
    return min(dimensions) if dimensions else None


def _cluster_center(cluster: dict[str, Any]) -> tuple[float, float]:
    ra = math.degrees(math.atan2(cluster["y"], cluster["x"])) % 360.0
    dec = math.degrees(math.atan2(
        cluster["z"], math.hypot(cluster["x"], cluster["y"])
    ))
    return ra, dec


def _add_cluster_member(cluster: dict[str, Any], meta: "FrameMeta") -> None:
    assert meta.pointing_ra_deg is not None and meta.pointing_dec_deg is not None
    ra = math.radians(meta.pointing_ra_deg)
    dec = math.radians(meta.pointing_dec_deg)
    cluster["x"] += math.cos(dec) * math.cos(ra)
    cluster["y"] += math.cos(dec) * math.sin(ra)
    cluster["z"] += math.sin(dec)
    cluster["members"].append(meta)


def assign_panel_ids(
    lights: list["FrameMeta"],
    input_root: Path,
    options: "PlanOptions",
) -> tuple[dict[str, str], str, list[str]]:
    """Cluster physical pointings, using names only when coordinates are absent."""
    if not options.mosaic_enabled:
        return ({meta.path: "panel_01" for meta in lights}, "Mosaic detection disabled", [])

    coordinate_lights = [
        meta for meta in lights
        if meta.pointing_ra_deg is not None and meta.pointing_dec_deg is not None
    ]
    stable_lights = [
        meta for meta in coordinate_lights
        if meta.pointing_source in {"target", "wcs"}
    ]
    cluster_lights = stable_lights or coordinate_lights
    fovs = sorted(
        value for value in (short_field_of_view_deg(meta) for meta in cluster_lights)
        if value is not None
    )
    short_fov = fovs[len(fovs) // 2] if fovs else None
    tolerance = (
        short_fov * options.mosaic_cluster_percent / 100.0
        if short_fov is not None else 0.08
    )
    tolerance = max(0.01, min(0.5, tolerance))

    clusters: list[dict[str, Any]] = []
    for meta in sorted(
        cluster_lights,
        key=lambda item: (
            item.pointing_ra_deg or 0.0,
            item.pointing_dec_deg or 0.0,
            item.path,
        ),
    ):
        distances = [
            angular_separation_deg(
                meta.pointing_ra_deg or 0.0,
                meta.pointing_dec_deg or 0.0,
                *_cluster_center(cluster),
            )
            for cluster in clusters
        ]
        nearest = min(range(len(distances)), key=distances.__getitem__) if distances else None
        if nearest is not None and distances[nearest] <= tolerance:
            _add_cluster_member(clusters[nearest], meta)
        else:
            cluster = {"x": 0.0, "y": 0.0, "z": 0.0, "members": []}
            _add_cluster_member(cluster, meta)
            clusters.append(cluster)

    ordered = sorted(
        clusters,
        key=lambda cluster: (*_cluster_center(cluster), -len(cluster["members"])),
    )
    assignments: dict[str, str] = {}
    centers: list[tuple[float, float]] = []
    for index, cluster in enumerate(ordered, start=1):
        panel_id = f"panel_{index:02d}"
        centers.append(_cluster_center(cluster))
        for meta in cluster["members"]:
            assignments[meta.path] = panel_id

    warnings: list[str] = []
    unassigned = [meta for meta in lights if meta.path not in assignments]
    # Lower-quality mount coordinates can be attached to a stable commanded/WCS
    # center, but cannot create extra panels that mimic gradual mount drift.
    for meta in list(unassigned):
        if meta.pointing_ra_deg is None or meta.pointing_dec_deg is None or not centers:
            continue
        distances = [
            angular_separation_deg(
                meta.pointing_ra_deg, meta.pointing_dec_deg, center[0], center[1]
            )
            for center in centers
        ]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        attachment_limit = max(tolerance * 3.0, (short_fov or 0.4) * 0.35)
        if distances[nearest] <= attachment_limit:
            assignments[meta.path] = f"panel_{nearest + 1:02d}"

    unassigned = [meta for meta in lights if meta.path not in assignments]
    if unassigned:
        hint_to_panel: dict[str, str] = {}
        for meta in lights:
            if meta.path in assignments:
                hint = detect_panel_id(meta, input_root)
                if hint != "panel_01":
                    hint_to_panel.setdefault(hint, assignments[meta.path])
        for meta in unassigned:
            hint = detect_panel_id(meta, input_root)
            if hint in hint_to_panel:
                assignments[meta.path] = hint_to_panel[hint]
            elif not centers:
                assignments[meta.path] = hint
            elif len(centers) == 1:
                assignments[meta.path] = "panel_01"

    unresolved = [meta for meta in lights if meta.path not in assignments]
    if unresolved:
        warnings.append(
            f"BLOCKING: {len(unresolved)} mosaic lights lack usable pointing coordinates "
            "or an explicit panel label; they cannot be assigned reliably."
        )
        for meta in unresolved:
            assignments[meta.path] = "panel_unresolved"

    if not centers:
        panel_count = len(set(assignments.values()))
        summary = f"Name fallback ({panel_count} panel{'s' if panel_count != 1 else ''})"
        warnings.append(
            "No usable target-center or WCS coordinates were found; mosaic detection "
            "fell back to explicit Panel/Tile/Field/Pointing labels."
        )
        return assignments, summary, warnings

    source_names = sorted({meta.pointing_source for meta in cluster_lights})
    layout = ""
    if len(centers) == 2:
        mean_dec = math.radians((centers[0][1] + centers[1][1]) / 2.0)
        horizontal = abs(centers[0][0] - centers[1][0]) * abs(math.cos(mean_dec))
        vertical = abs(centers[0][1] - centers[1][1])
        layout = "1x2" if horizontal >= vertical else "2x1"
    layout_text = f", inferred {layout}" if layout else ""
    fov_text = f", short FOV {short_fov:.3f}°" if short_fov is not None else ""
    summary = (
        f"Coordinate clustering ({'/'.join(source_names)}): {len(centers)} panel(s)"
        f"{layout_text}, tolerance {tolerance:.3f}°{fov_text}"
    )
    if not stable_lights and coordinate_lights:
        warnings.append(
            "Panel clustering used lower-confidence mount RA/DEC because commanded-center "
            "and solved WCS coordinates were unavailable; review every panel assignment."
        )
    return assignments, summary, warnings


# --- Mosaic panel layout (drawn by PanelMapWindow) -----------------------------


@dataclass
class PanelOutline:
    panel_id: str
    number: Optional[int]
    center_ra_deg: float
    center_dec_deg: float
    width_deg: float
    height_deg: float
    angle_deg: Optional[float]
    corners: list[tuple[float, float]]
    filter_counts: dict[str, int]
    missing_filters: list[str]
    overlaps: dict[str, float] = field(default_factory=dict)
    isolated: bool = False


@dataclass
class PanelLayout:
    dataset_id: str
    panels: list[PanelOutline]
    drawable: bool
    message: str
    rotation_known: bool
    unplaced: list[str]
    far: list[str] = field(default_factory=list)


def _unit_vector(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    return math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)


def _mean_direction(points: list[tuple[float, float]]) -> tuple[float, float]:
    x = y = z = 0.0
    for ra, dec in points:
        ux, uy, uz = _unit_vector(ra, dec)
        x, y, z = x + ux, y + uy, z + uz
    return _cluster_center({"x": x, "y": y, "z": z})


FAR_PANEL_DEG = 30.0


def _separation_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    ua, ub = _unit_vector(*a), _unit_vector(*b)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(ua, ub))))
    return math.degrees(math.acos(dot))


def _gnomonic(
    ra_deg: float, dec_deg: float, centre: tuple[float, float]
) -> tuple[float, float]:
    """Tangent-plane (xi east, eta north) in degrees about ``centre``."""
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    a0, d0 = math.radians(centre[0]), math.radians(centre[1])
    cosc = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(a - a0)
    xi = math.cos(d) * math.sin(a - a0) / cosc
    eta = (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(a - a0)) / cosc
    return math.degrees(xi), math.degrees(eta)


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
    )


def _convex_intersection_area(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Area shared by two convex polygons (Sutherland-Hodgman clipping)."""
    subject = a if _polygon_area(a) >= 0 else a[::-1]
    clip = b if _polygon_area(b) >= 0 else b[::-1]
    output = list(subject)
    for (cx1, cy1), (cx2, cy2) in zip(clip, clip[1:] + clip[:1]):
        if not output:
            break

        def inside(point: tuple[float, float]) -> bool:
            return (cx2 - cx1) * (point[1] - cy1) - (cy2 - cy1) * (point[0] - cx1) >= 0

        def crossing(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
            dx, dy = q[0] - p[0], q[1] - p[1]
            ex, ey = cx2 - cx1, cy2 - cy1
            denominator = dx * ey - dy * ex
            t = ((cx1 - p[0]) * ey - (cy1 - p[1]) * ex) / denominator
            return p[0] + t * dx, p[1] + t * dy

        points, output = output, []
        for current, following in zip(points, points[1:] + points[:1]):
            if inside(following):
                if not inside(current):
                    output.append(crossing(current, following))
                output.append(following)
            elif inside(current):
                output.append(crossing(current, following))
    return abs(_polygon_area(output)) if len(output) >= 3 else 0.0


def panel_layout(dataset: "DatasetPlan") -> PanelLayout:
    """Sky layout of a dataset's panels from scan metadata, for drawing."""
    filters = list(dataset.channels)
    frames_by_panel: dict[str, list[tuple[str, FrameMeta]]] = {}
    for filter_name, groups in dataset.channels.items():
        for group in groups:
            frames_by_panel.setdefault(group.panel_id, []).extend(
                (filter_name, frame) for frame in group.frames
            )
    all_sizes = [
        size for entries in frames_by_panel.values()
        for _filter, frame in entries
        if (size := field_of_view_deg(frame)) is not None
    ]

    placed: list[dict[str, Any]] = []
    unplaced: list[str] = []
    for panel_id in dataset.panels:
        entries = frames_by_panel.get(panel_id, [])
        located = [
            frame for _filter, frame in entries
            if frame.pointing_ra_deg is not None and frame.pointing_dec_deg is not None
        ]
        preferred = [
            frame for frame in located if frame.pointing_source in {"target", "wcs"}
        ] or located
        if not preferred:
            unplaced.append(panel_id)
            continue
        sizes = [size for _f, frame in entries if (size := field_of_view_deg(frame))]
        angles = [
            frame.position_angle_deg for _f, frame in entries
            if frame.position_angle_deg is not None
        ]
        counts = Counter(filter_name for filter_name, _frame in entries)
        placed.append({
            "panel_id": panel_id,
            "centre": _mean_direction([
                (frame.pointing_ra_deg, frame.pointing_dec_deg) for frame in preferred
            ]),
            "sizes": sizes,
            # Axial mean: an outline is the same rotated by 180 degrees, and
            # plate-solved frames from both sides of a meridian flip differ
            # by ~180 degrees, so a plain circular mean would cancel out.
            "angle": (
                0.5 * math.degrees(math.atan2(
                    sum(math.sin(math.radians(2 * a)) for a in angles),
                    sum(math.cos(math.radians(2 * a)) for a in angles),
                )) % 180.0
                if angles else None
            ),
            "counts": {name: counts[name] for name in filters if counts[name]},
            "missing": [name for name in filters if not counts[name]],
        })

    if not placed:
        return PanelLayout(
            dataset.dataset_id, [], False,
            "These frames have no sky coordinates, so the panel layout cannot be drawn.",
            False, unplaced,
        )
    if not all_sizes:
        return PanelLayout(
            dataset.dataset_id, [], False,
            "Panel size unknown: frames have no focal length or pixel size.",
            False, unplaced,
        )

    # Anchor on the panel closest to all others, so one wild coordinate (a
    # park position, another target) is reported instead of being projected
    # far outside the tangent plane's useful range.
    anchor = min(
        (entry["centre"] for entry in placed),
        key=lambda centre: sum(_separation_deg(centre, other["centre"]) for other in placed),
    )
    far = [
        entry["panel_id"] for entry in placed
        if _separation_deg(entry["centre"], anchor) > FAR_PANEL_DEG
    ]
    placed = [entry for entry in placed if entry["panel_id"] not in far]
    projection_centre = _mean_direction([entry["centre"] for entry in placed])
    panels: list[PanelOutline] = []
    for entry in placed:
        sizes = entry["sizes"] or all_sizes
        width = float(np.median([size[0] for size in sizes]))
        height = float(np.median([size[1] for size in sizes]))
        xi, eta = _gnomonic(*entry["centre"], projection_centre)
        angle = math.radians(entry["angle"] or 0.0)
        # Image +Y points ``angle`` east of north; image +X is perpendicular.
        y_axis = (math.sin(angle), math.cos(angle))
        x_axis = (-math.cos(angle), math.sin(angle))
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corner_xi = xi + sx * width / 2 * x_axis[0] + sy * height / 2 * y_axis[0]
            corner_eta = eta + sx * width / 2 * x_axis[1] + sy * height / 2 * y_axis[1]
            corners.append((-corner_xi, corner_eta))
        digits = re.findall(r"\d+", entry["panel_id"])
        panels.append(PanelOutline(
            panel_id=entry["panel_id"],
            number=int(digits[-1]) if digits else None,
            center_ra_deg=entry["centre"][0],
            center_dec_deg=entry["centre"][1],
            width_deg=width,
            height_deg=height,
            angle_deg=entry["angle"],
            corners=corners,
            filter_counts=entry["counts"],
            missing_filters=entry["missing"],
        ))

    for index, first in enumerate(panels):
        for second in panels[index + 1:]:
            shared = _convex_intersection_area(first.corners, second.corners)
            smaller = min(
                abs(_polygon_area(first.corners)), abs(_polygon_area(second.corners))
            )
            if smaller > 0 and shared / smaller > 1e-6:
                fraction = shared / smaller
                first.overlaps[second.panel_id] = fraction
                second.overlaps[first.panel_id] = fraction
    for panel in panels:
        panel.isolated = len(panels) > 1 and not panel.overlaps
    return PanelLayout(
        dataset_id=dataset.dataset_id,
        panels=panels,
        drawable=True,
        message="",
        rotation_known=all(panel.angle_deg is not None for panel in panels),
        unplaced=unplaced,
        far=far,
    )


def siril_theme_mode(siril: Any) -> str:
    """Return Siril's active theme as dark, light, or system fallback."""
    try:
        value = int(siril.get_siril_config("gui", "theme"))
    except Exception:
        return "system"
    # Siril 1.4 documents 0=dark and 1=light. Unknown/future values are left
    # to Qt's system appearance rather than guessing a color palette.
    return {0: "dark", 1: "light"}.get(value, "system")


def apply_siril_theme(application: QApplication, siril: Any) -> str:
    """Apply Siril's theme preference using Qt's native color-scheme support."""
    mode = siril_theme_mode(siril)
    application.setStyle("Fusion")
    scheme = {
        "dark": Qt.ColorScheme.Dark,
        "light": Qt.ColorScheme.Light,
        "system": Qt.ColorScheme.Unknown,
    }[mode]
    try:
        QGuiApplication.styleHints().setColorScheme(scheme)
    except AttributeError:
        # Older Qt versions simply retain their native system palette.
        pass
    if mode == "dark":
        palette = QPalette()
        colors = {
            QPalette.ColorRole.Window: QColor("#2b2b2b"),
            QPalette.ColorRole.WindowText: QColor("#eeeeee"),
            QPalette.ColorRole.Base: QColor("#1f1f1f"),
            QPalette.ColorRole.AlternateBase: QColor("#333333"),
            QPalette.ColorRole.ToolTipBase: QColor("#353535"),
            QPalette.ColorRole.ToolTipText: QColor("#ffffff"),
            QPalette.ColorRole.Text: QColor("#eeeeee"),
            QPalette.ColorRole.Button: QColor("#383838"),
            QPalette.ColorRole.ButtonText: QColor("#eeeeee"),
            QPalette.ColorRole.BrightText: QColor("#ff6666"),
            QPalette.ColorRole.Link: QColor("#66aaff"),
            QPalette.ColorRole.Highlight: QColor("#356f9f"),
            QPalette.ColorRole.HighlightedText: QColor("#ffffff"),
            QPalette.ColorRole.PlaceholderText: QColor("#a0a0a0"),
        }
        for role, color in colors.items():
            palette.setColor(role, color)
        for role in (
            QPalette.ColorRole.WindowText,
            QPalette.ColorRole.Text,
            QPalette.ColorRole.ButtonText,
        ):
            palette.setColor(
                QPalette.ColorGroup.Disabled, role, QColor("#888888")
            )
        application.setPalette(palette)
        application.setStyleSheet(
            "QToolTip { color: #ffffff; background-color: #353535; "
            "border: 1px solid #666666; }"
        )
    elif mode == "light":
        application.setStyleSheet("")
        application.setPalette(application.style().standardPalette())
    else:
        application.setStyleSheet("")
        application.setPalette(application.style().standardPalette())
    application.setProperty("sbppTheme", mode)
    return mode


def classify_type(value: Any, path: Path, role_hint: str = "") -> tuple[str, bool]:
    """Return (frame_type, is_master) using header first and path fallback."""
    if "REJMAP" in normalized_text(path.name):
        # Siril rejection maps inherit LIGHT headers but are diagnostic stack
        # products, never source exposures for a new preprocessing run.
        return FRAME_UNKNOWN, True
    raw = normalized_text(value)
    is_master = "MASTER" in raw or "MASTER" in normalized_text(path.name)
    if "DARKFLAT" in raw or "FLATDARK" in raw:
        return FRAME_DARKFLAT, is_master
    if "BIAS" in raw or "OFFSET" == raw:
        return FRAME_BIAS, is_master
    if "FLAT" in raw:
        return FRAME_FLAT, is_master
    if "DARK" in raw:
        return FRAME_DARK, is_master
    if raw in {"LIGHT", "OBJECT", "SCIENCE", "TARGET"} or "MASTERLIGHT" in raw:
        return FRAME_LIGHT, is_master

    parts = [normalized_text(part) for part in path.parts[-5:]]
    joined = "_".join(parts)
    is_master = is_master or "MASTER" in joined
    if "DARKFLAT" in joined or "FLATDARK" in joined:
        return FRAME_DARKFLAT, is_master
    if "BIAS" in joined:
        return FRAME_BIAS, is_master
    if "FLAT" in joined:
        return FRAME_FLAT, is_master
    if "DARK" in joined:
        return FRAME_DARK, is_master
    if "LIGHT" in joined:
        return FRAME_LIGHT, is_master
    if role_hint == "input":
        return FRAME_LIGHT, is_master
    return FRAME_UNKNOWN, is_master


@dataclass
class FrameMeta:
    path: str
    frame_type: str = FRAME_UNKNOWN
    is_master: bool = False
    source_format: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    channels: int = 1
    instrument: str = ""
    telescope: str = ""
    object_name: str = ""
    filter_name: str = ""
    exposure: Optional[float] = None
    gain: Optional[float] = None
    offset: Optional[float] = None
    temperature: Optional[float] = None
    set_temperature: Optional[float] = None
    bin_x: Optional[int] = None
    bin_y: Optional[int] = None
    bayer: str = ""
    date_obs: str = ""
    pointing_ra_deg: Optional[float] = None
    pointing_dec_deg: Optional[float] = None
    pointing_source: str = ""
    focal_length_mm: Optional[float] = None
    pixel_size_x_um: Optional[float] = None
    pixel_size_y_um: Optional[float] = None
    stack_count: Optional[int] = None
    size: int = 0
    mtime_ns: int = 0
    warning: str = ""
    position_angle_deg: Optional[float] = None

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    @property
    def effective_temperature(self) -> Optional[float]:
        return self.set_temperature if self.set_temperature is not None else self.temperature

    @property
    def camera_key(self) -> tuple[Any, ...]:
        return (
            normalized_text(self.instrument), self.width, self.height, self.channels,
            self.bin_x, self.bin_y, round(self.gain, 3) if self.gain is not None else None,
            round(self.offset, 3) if self.offset is not None else None,
            normalized_text(self.bayer),
        )

    @property
    def geometry_key(self) -> tuple[Any, ...]:
        return (
            normalized_text(self.instrument), self.width, self.height, self.channels,
            self.bin_x, self.bin_y, normalized_text(self.bayer),
        )

    def summary(self) -> str:
        label = Path(self.path).name
        return f"{label} ({self.frame_type}{' master' if self.is_master else ''})"


def meaningful_object_name(value: Any) -> bool:
    """Reject capture placeholders that are not astronomical target names."""
    ignored = {"", "UNKNOWN", "UNNAMED", "NONE", "LIGHT", "TARGET", "OBJECT"}
    return normalized_text(value) not in ignored


def infer_target_name_from_headers(
    frames: Iterable[FrameMeta], minimum_agreement: int = 3
) -> str:
    """Infer a target only when at least three light OBJECT headers agree.

    A folder holding only master lights (a master merge) uses the masters'
    OBJECT headers instead, requiring two agreeing values.
    """
    frames = list(frames)
    use_masters = not any(
        frame.frame_type == FRAME_LIGHT and not frame.is_master for frame in frames
    )
    if use_masters:
        minimum_agreement = min(minimum_agreement, 2)
    names = [
        frame.object_name.strip()
        for frame in frames
        if frame.frame_type == FRAME_LIGHT
        and frame.is_master == use_masters
        and meaningful_object_name(frame.object_name)
    ]
    if len(names) < minimum_agreement:
        return ""
    normalized_names = {normalized_text(name) for name in names}
    if len(normalized_names) != 1:
        return ""
    return Counter(names).most_common(1)[0][0]


def metadata_from_header(
    path: Path,
    header: dict[str, Any],
    source_format: str,
    image_type: str = "",
    geometry: tuple[Optional[int], Optional[int], int] = (None, None, 1),
    role_hint: str = "",
) -> FrameMeta:
    width = safe_int(first_header(header, "NAXIS1")) or geometry[0]
    height = safe_int(first_header(header, "NAXIS2")) or geometry[1]
    channels = safe_int(first_header(header, "NAXIS3")) or geometry[2] or 1
    bayer = str(first_header(header, "BAYERPAT", "BAYERPATTERN") or "").strip()
    kind_value = first_header(
        header,
        "IMAGETYP", "IMAGETYPE", "FRAMETYP", "FRAMETYPE", "FRAME-TYP",
        "OBSTYPE", "CALTYPE",
    ) or image_type
    frame_type, is_master = classify_type(kind_value, path, role_hint)
    stack_count = safe_int(first_header(header, "NCOMBINE", "STACKCNT"))
    if stack_count and stack_count > 1:
        is_master = True
    pointing_ra, pointing_dec, pointing_source = pointing_from_header(header)
    stat = path.stat()
    return FrameMeta(
        path=str(path.resolve()),
        frame_type=frame_type,
        is_master=is_master,
        source_format=source_format,
        width=width,
        height=height,
        channels=channels,
        instrument=str(first_header(header, "INSTRUME", "CAMERA") or "").strip(),
        telescope=str(first_header(header, "TELESCOP") or "").strip(),
        object_name=str(first_header(header, "OBJECT") or "").strip(),
        filter_name=normalize_filter(first_header(header, "FILTER", "FILT-1"), bayer),
        exposure=safe_float(first_header(header, "EXPTIME", "EXPOSURE")),
        gain=safe_float(first_header(header, "GAIN", "CAM-GAIN")),
        offset=safe_float(first_header(header, "OFFSET", "BLKLEVEL")),
        temperature=safe_float(first_header(header, "CCD-TEMP", "CCD_TEMP", "CCDTEMP", "TEMPERAT")),
        set_temperature=safe_float(first_header(header, "SET-TEMP", "SET_TEMP")),
        bin_x=safe_int(first_header(header, "XBINNING", "BINX")),
        bin_y=safe_int(first_header(header, "YBINNING", "BINY")),
        bayer=bayer,
        date_obs=str(first_header(header, "DATE-OBS") or "").strip(),
        pointing_ra_deg=pointing_ra,
        pointing_dec_deg=pointing_dec,
        pointing_source=pointing_source,
        position_angle_deg=position_angle_from_header(header),
        focal_length_mm=safe_float(first_header(
            header, "FOCALLEN", "FOCAL-LEN", "FOCLEN", "FOCAL"
        )),
        pixel_size_x_um=safe_float(first_header(
            header, "XPIXSZ", "XPIXELSZ", "PIXSIZE1", "PIXSIZE"
        )),
        pixel_size_y_um=safe_float(first_header(
            header, "YPIXSZ", "YPIXELSZ", "PIXSIZE2", "PIXSIZE"
        )),
        stack_count=stack_count,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def read_fits_meta(path: Path, role_hint: str = "") -> FrameMeta:
    with fits.open(path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
        combined: dict[str, Any] = dict(hdul[0].header)
        chosen = hdul[0]
        for hdu in hdul:
            if safe_int(hdu.header.get("NAXIS")) and safe_int(hdu.header.get("NAXIS")) >= 2:
                chosen = hdu
                combined.update(dict(hdu.header))
                break
        geometry = (
            safe_int(chosen.header.get("NAXIS1")),
            safe_int(chosen.header.get("NAXIS2")),
            safe_int(chosen.header.get("NAXIS3")) or 1,
        )
    return metadata_from_header(path, combined, "FITS", geometry=geometry, role_hint=role_hint)


def read_xisf_meta(path: Path, role_hint: str = "") -> FrameMeta:
    with path.open("rb") as handle:
        if handle.read(8) != b"XISF0100":
            raise ValueError("invalid XISF signature")
        header_length = struct.unpack("<Q", handle.read(8))[0]
        if header_length <= 0 or header_length > 128 * 1024 * 1024:
            raise ValueError(f"invalid XISF XML header length {header_length}")
        xml_data = handle.read(header_length).rstrip(b"\x00")
    root = ET.fromstring(xml_data)
    image = next((node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Image"), None)
    if image is None:
        raise ValueError("XISF contains no Image element")
    geometry_text = image.attrib.get("geometry", "")
    geometry_parts = [safe_int(part) for part in geometry_text.split(":")]
    geometry = (
        geometry_parts[0] if len(geometry_parts) > 0 else None,
        geometry_parts[1] if len(geometry_parts) > 1 else None,
        geometry_parts[2] if len(geometry_parts) > 2 and geometry_parts[2] else 1,
    )
    header: dict[str, Any] = {}
    for node in image.iter():
        if node.tag.rsplit("}", 1)[-1] == "FITSKeyword":
            name = node.attrib.get("name", "").upper()
            if name and name not in header:
                header[name] = clean_header_value(node.attrib.get("value", ""))
    return metadata_from_header(
        path,
        header,
        "XISF",
        image_type=image.attrib.get("imageType", ""),
        geometry=geometry,
        role_hint=role_hint,
    )


def read_fallback_meta(path: Path, role_hint: str = "") -> FrameMeta:
    frame_type, is_master = classify_type("", path, role_hint)
    stat = path.stat()
    is_raw = file_suffix(path) in RAW_SUFFIXES
    return FrameMeta(
        path=str(path.resolve()),
        frame_type=frame_type,
        is_master=is_master,
        source_format=file_suffix(path).lstrip(".").upper(),
        filter_name="OSC" if is_raw else "L",
        bayer="UNKNOWN_CFA" if is_raw else "",
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        warning="No readable FITS/XISF metadata; classification used the folder path.",
    )


class MetadataCache:
    """Persistent header cache keyed by path, size, and nanosecond mtime."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_version") == METADATA_CACHE_VERSION:
                self.entries = payload.get("entries", {})
        except (OSError, ValueError, TypeError):
            self.entries = {}

    def get(self, path: Path) -> Optional[FrameMeta]:
        key = str(path.resolve())
        entry = self.entries.get(key)
        if not entry:
            return None
        try:
            stat = path.stat()
            if entry.get("size") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
                return None
            return FrameMeta(**entry["meta"])
        except (OSError, KeyError, TypeError):
            return None

    def put(self, meta: FrameMeta) -> None:
        self.entries[meta.path] = {
            "size": meta.size,
            "mtime_ns": meta.mtime_ns,
            "meta": asdict(meta),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, {
            "cache_version": METADATA_CACHE_VERSION,
            "updated_utc": utc_now(),
            "entries": self.entries,
        })


class HeaderScanner:
    def __init__(self, cache: MetadataCache):
        self.cache = cache

    @staticmethod
    def iter_files(root: Path, recursive: bool) -> Iterable[Path]:
        if root.is_file():
            if is_supported(root):
                yield root
            return
        if not root.is_dir():
            return
        if not recursive:
            for path in root.iterdir():
                if is_supported(path):
                    yield path
            return
        for current, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if not should_skip_dir(Path(current) / name)]
            for name in files:
                path = Path(current) / name
                if is_supported(path):
                    yield path

    def read(self, path: Path, role_hint: str) -> FrameMeta:
        cached = self.cache.get(path)
        if cached is not None:
            if "REJMAP" in normalized_text(path.name):
                cached.frame_type = FRAME_UNKNOWN
                cached.is_master = True
            # A cached unknown from a calibration scan must still become a light
            # when the same path is later selected as the input root.
            if role_hint == "input" and cached.frame_type == FRAME_UNKNOWN:
                cached.frame_type = FRAME_LIGHT
            return cached
        suffix = file_suffix(path)
        if suffix in FITS_SUFFIXES:
            meta = read_fits_meta(path, role_hint)
        elif suffix in XISF_SUFFIXES:
            meta = read_xisf_meta(path, role_hint)
        else:
            meta = read_fallback_meta(path, role_hint)
        self.cache.put(meta)
        return meta

    def scan(
        self,
        input_root: Path,
        calibration_roots: list[Path],
        recursive: bool = True,
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> tuple[list[FrameMeta], list[str]]:
        sources: list[tuple[Path, str]] = [(input_root, "input")]
        sources.extend((root, "calibration") for root in calibration_roots)
        candidates: list[tuple[Path, str]] = []
        seen: set[str] = set()
        for root, hint in sources:
            for path in self.iter_files(root, recursive):
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append((path, hint))
        frames: list[FrameMeta] = []
        errors: list[str] = []
        total = max(1, len(candidates))
        for index, (path, hint) in enumerate(candidates, start=1):
            try:
                frames.append(self.read(path, hint))
            except Exception as exc:  # input corruption must not abort discovery
                errors.append(f"{path}: {exc}")
            if progress and (index == total or index % 20 == 0):
                progress(int(index * 100 / total), path.name)
        self.cache.save()
        return frames, errors


@dataclass
class PlanOptions:
    recursive: bool = True
    prefer_masters: bool = True
    minimum_calibration_frames: int = 5
    dark_policy: str = "exact_or_optimized"
    exposure_tolerance: float = 0.25
    temperature_tolerance: float = 3.0
    fwhm_max: float = 6.0
    roundness_min: float = 0.70
    stars_min: int = 40
    small_stack_threshold: int = MIN_LINEAR_STACK_COUNT
    large_stack_threshold: int = 50
    max_stars_registration: int = 2000
    min_pairs_registration: int = 20
    solve_final_images: bool = True
    create_rejection_maps: bool = False
    debayer_cfa: bool = True
    drizzle_scale: float = 1.0
    drizzle_pixfrac: float = 0.5
    drizzle_kernel: str = "square"
    keep_intermediates: bool = True
    rolling_cleanup: bool = True
    mosaic_enabled: bool = False
    mosaic_cluster_percent: float = 8.0
    mosaic_feather: int = 100
    mosaic_overlap_normalization: bool = True
    mosaic_gradient_correction: bool = True
    graxpert_executable: str = ""
    graxpert_smoothing: float = 0.5
    graxpert_gpu: bool = True
    mosaic_background_matching: bool = True
    mosaic_downsample_solve: bool = True
    mosaic_blind_fallback: bool = True
    merge_blend_large_scales: bool = True
    merge_transition_pixels: int = 0
    merge_detail_sigma: float = 0.0
    merge_output_grid: str = "auto"


def retains_intermediates(options: PlanOptions) -> bool:
    """Whether generated process trees should survive a successful run."""
    return options.keep_intermediates and not options.rolling_cleanup


@dataclass
class CalibrationChoice:
    kind: str
    mode: str = "none"  # master, build, none, bpm_master, bpm_build
    files: list[str] = field(default_factory=list)
    description: str = "None"
    warning: str = ""

    @property
    def available(self) -> bool:
        return self.mode != "none" and bool(self.files)

    @property
    def optimized(self) -> bool:
        return self.mode in {"optimized_master", "optimized_build"}

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return self.kind, self.mode, tuple(sorted(self.files))


@dataclass
class LightSubgroup:
    subgroup_id: str
    panel_id: str
    filter_name: str
    frames: list[FrameMeta]
    bias: CalibrationChoice
    dark: CalibrationChoice
    flat: CalibrationChoice
    darkflat: CalibrationChoice
    bpm: CalibrationChoice
    warnings: list[str] = field(default_factory=list)

    @property
    def representative(self) -> FrameMeta:
        return self.frames[0]

    @property
    def integration_seconds(self) -> float:
        return sum(
            frame.exposure for frame in self.frames
            if frame.exposure is not None and frame.exposure >= 0
        )

    @property
    def unknown_exposure_count(self) -> int:
        return sum(
            1 for frame in self.frames
            if frame.exposure is None or frame.exposure < 0
        )


@dataclass
class DatasetPlan:
    dataset_id: str
    geometry_label: str
    channels: dict[str, list[LightSubgroup]]
    panel_detection: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return sum(len(group.frames) for groups in self.channels.values() for group in groups)

    @property
    def integration_seconds(self) -> float:
        return sum(
            group.integration_seconds
            for groups in self.channels.values()
            for group in groups
        )

    @property
    def unknown_exposure_count(self) -> int:
        return sum(
            group.unknown_exposure_count
            for groups in self.channels.values()
            for group in groups
        )

    @property
    def panels(self) -> list[str]:
        return sorted(
            {group.panel_id for groups in self.channels.values() for group in groups},
            key=natural_key,
        )

    @property
    def panel_count(self) -> int:
        return len(self.panels)


def _dataclass_from_dict(cls: type, data: Any) -> Any:
    """Construct ``cls`` from the dataclass fields present in ``data``."""
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in dict(data).items() if key in names})


@dataclass
class MergeMaster:
    """One finished master light entering a multi-scale master merge."""

    path: str
    filter_name: str
    layer: int
    scale_arcsec: Optional[float]
    width: Optional[int] = None
    height: Optional[int] = None
    telescope: str = ""
    focal_length_mm: Optional[float] = None
    exposure: Optional[float] = None
    role: str = ""


@dataclass
class MasterMergePlan:
    """Masters from several plate scales merged into one canvas per filter."""

    masters: list[MergeMaster]
    layer_scales: list[Optional[float]]
    filters: list[str]
    estimated_canvas: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)

    @property
    def layer_count(self) -> int:
        return len(self.layer_scales)


@dataclass
class ProcessingPlan:
    target_name: str
    input_root: str
    calibration_roots: list[str]
    output_root: str
    options: PlanOptions
    datasets: list[DatasetPlan]
    drizzle_channels: list[str] = field(default_factory=list)
    scan_errors: list[str] = field(default_factory=list)
    ignored_count: int = 0
    warnings: list[str] = field(default_factory=list)
    estimated_bytes: int = 0
    estimated_final_bytes: int = 0
    free_bytes: int = 0
    master_merge: Optional[MasterMergePlan] = None

    @property
    def output_prefix(self) -> str:
        """Filesystem-safe run name derived exclusively from the target name."""
        return slugify(self.target_name, "stack")

    @property
    def light_count(self) -> int:
        return sum(dataset.frame_count for dataset in self.datasets)

    @property
    def integration_seconds(self) -> float:
        return sum(dataset.integration_seconds for dataset in self.datasets)

    @property
    def unknown_exposure_count(self) -> int:
        return sum(dataset.unknown_exposure_count for dataset in self.datasets)

    @property
    def valid(self) -> bool:
        if self.master_merge is not None:
            return (
                bool(self.master_merge.masters)
                and bool(self.output_root)
                and not any(
                    message.startswith("BLOCKING:")
                    for message in [*self.warnings, *self.master_merge.warnings]
                )
            )
        return (
            self.light_count > 0
            and bool(self.output_root)
            and not any(message.startswith("BLOCKING:") for message in self.warnings)
            and not any(
                message.startswith("BLOCKING:")
                for dataset in self.datasets
                for message in dataset.warnings
            )
            and not any(
                message.startswith("BLOCKING:")
                for dataset in self.datasets
                for groups in dataset.channels.values()
                for subgroup in groups
                for message in subgroup.warnings
            )
            and all(
                subgroup.flat.available
                for dataset in self.datasets
                for groups in dataset.channels.values()
                for subgroup in groups
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_prefix"] = self.output_prefix
        payload["integration_seconds"] = self.integration_seconds
        payload["integration_time"] = integration_summary(
            self.integration_seconds, self.unknown_exposure_count
        )
        payload["unknown_exposure_lights"] = self.unknown_exposure_count
        for dataset_payload, dataset in zip(payload["datasets"], self.datasets):
            dataset_payload["integration_seconds"] = dataset.integration_seconds
            dataset_payload["unknown_exposure_lights"] = dataset.unknown_exposure_count
            for filter_name, groups in dataset.channels.items():
                for group_payload, group in zip(
                    dataset_payload["channels"][filter_name], groups
                ):
                    group_payload["integration_seconds"] = group.integration_seconds
                    group_payload["unknown_exposure_lights"] = (
                        group.unknown_exposure_count
                    )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessingPlan":
        """Rebuild a plan saved by ``to_dict`` (e.g. a run's manifest).

        Unknown keys are ignored and missing keys take their defaults, so
        manifests from older and newer SBPP versions both load.
        """
        def subgroup(data: dict[str, Any]) -> LightSubgroup:
            values = dict(data)
            values["frames"] = [
                _dataclass_from_dict(FrameMeta, frame) for frame in data.get("frames", [])
            ]
            for kind in ("bias", "dark", "flat", "darkflat", "bpm"):
                choice = data.get(kind) or {"kind": kind.upper()}
                values[kind] = _dataclass_from_dict(CalibrationChoice, choice)
            return _dataclass_from_dict(LightSubgroup, values)

        datasets = []
        for data in payload.get("datasets", []):
            values = dict(data)
            values["channels"] = {
                filter_name: [subgroup(group) for group in groups]
                for filter_name, groups in data.get("channels", {}).items()
            }
            datasets.append(_dataclass_from_dict(DatasetPlan, values))
        merge = None
        if payload.get("master_merge"):
            values = dict(payload["master_merge"])
            values["masters"] = [
                _dataclass_from_dict(MergeMaster, master)
                for master in values.get("masters", [])
            ]
            if "estimated_canvas" in values:
                values["estimated_canvas"] = tuple(values["estimated_canvas"])
            merge = _dataclass_from_dict(MasterMergePlan, values)
        values = dict(payload)
        values["options"] = _dataclass_from_dict(PlanOptions, payload.get("options", {}))
        values["datasets"] = datasets
        values["master_merge"] = merge
        return _dataclass_from_dict(cls, values)

    def channel_uses_drizzle(self, dataset: DatasetPlan, filter_name: str) -> bool:
        return channel_key(dataset.dataset_id, filter_name) in set(self.drizzle_channels)


def compatible_camera(light: FrameMeta, calibration: FrameMeta) -> bool:
    if light.width and calibration.width and light.width != calibration.width:
        return False
    if light.height and calibration.height and light.height != calibration.height:
        return False
    if light.channels and calibration.channels and light.channels != calibration.channels:
        return False
    if light.bin_x and calibration.bin_x and light.bin_x != calibration.bin_x:
        return False
    if light.bin_y and calibration.bin_y and light.bin_y != calibration.bin_y:
        return False
    # Camera labels vary between native, ASCOM, capture-application, and driver
    # versions. The user explicitly selected the calibration library, so name
    # equality ranks candidates but never rejects otherwise compatible data.
    if not almost_equal(light.gain, calibration.gain, 0.51):
        return False
    if not almost_equal(light.offset, calibration.offset, 0.51):
        return False
    if light.bayer and calibration.bayer:
        left = normalized_text(light.bayer)
        right = normalized_text(calibration.bayer)
        if "UNKNOWN" not in {left, right} and left != right:
            return False
    return True


def calibration_score(light: FrameMeta, calibration: FrameMeta, options: PlanOptions) -> float:
    score = 0.0
    light_instrument = comparable_instrument(light.instrument)
    calibration_instrument = comparable_instrument(calibration.instrument)
    if (
        light_instrument and calibration_instrument
        and light_instrument == calibration_instrument
    ):
        score += 30.0
    if light.width and calibration.width:
        score += 20.0
    if light.gain is not None and calibration.gain is not None:
        score += 10.0
    if light.offset is not None and calibration.offset is not None:
        score += 10.0
    # Only flats are optically filter-dependent. A FILTER card on a covered
    # dark or bias records the wheel position used during capture, not a
    # calibration constraint, and must not influence which detector frame wins.
    if (
        calibration.frame_type == FRAME_FLAT
        and light.filter_name and calibration.filter_name
    ):
        if normalized_text(light.filter_name) == normalized_text(calibration.filter_name):
            score += 25.0
    if light.exposure is not None and calibration.exposure is not None:
        score -= abs(light.exposure - calibration.exposure) / max(options.exposure_tolerance, 0.01)
    if light.effective_temperature is not None and calibration.effective_temperature is not None:
        score -= abs(light.effective_temperature - calibration.effective_temperature)
    if calibration.is_master:
        score += 5.0
    return score


def calibration_mismatch_warning(
    kind: str,
    light: FrameMeta,
    candidates: list[FrameMeta],
    options: PlanOptions,
    *,
    require_filter: bool,
    require_exact_exposure: bool,
    require_temperature: bool,
    masters: bool,
) -> str:
    """Describe the first actual compatibility constraint that rejected a set."""
    pool = [
        candidate for candidate in candidates
        if candidate.frame_type == kind and candidate.is_master == masters
    ]
    if not pool:
        return ""
    count = len(pool)
    source = f"{kind.lower()} master(s)" if masters else f"raw {kind.lower()} frame(s)"
    compatible = [candidate for candidate in pool if compatible_camera(light, candidate)]
    if not compatible:
        return (
            f"Found {count} {source}, but none matched sensor geometry, channel count, "
            "binning, gain/offset, or CFA pattern."
        )
    if require_filter:
        filter_matches = [
            candidate for candidate in compatible
            if normalized_text(light.filter_name) == normalized_text(candidate.filter_name)
        ]
        if not filter_matches:
            available = sorted({candidate.filter_name or "unknown" for candidate in compatible})
            return (
                f"Found {count} {source}, but none matched filter "
                f"{light.filter_name or 'unknown'} (available: {', '.join(available)})."
            )
        compatible = filter_matches
    if require_exact_exposure:
        exposure_matches = [
            candidate for candidate in compatible
            if almost_equal(light.exposure, candidate.exposure, options.exposure_tolerance)
        ]
        if not exposure_matches:
            available_exposures = sorted({
                candidate.exposure for candidate in compatible
                if candidate.exposure is not None
            })
            available = ", ".join(
                f"{display_number(exposure, 3)}s" for exposure in available_exposures
            ) or "unknown"
            return (
                f"Found {count} {source}, but none matched the required "
                f"{display_number(light.exposure, 3)}s exposure within "
                f"+/-{display_number(options.exposure_tolerance, 3)}s "
                f"(available: {available})."
            )
        compatible = exposure_matches
    if require_temperature:
        temperature_matches = [
            candidate for candidate in compatible
            if almost_equal(
                light.effective_temperature,
                candidate.effective_temperature,
                options.temperature_tolerance,
            )
        ]
        if not temperature_matches:
            available_temperatures = sorted({
                candidate.effective_temperature for candidate in compatible
                if candidate.effective_temperature is not None
            })
            available = ", ".join(
                f"{display_number(temperature, 1)} C"
                for temperature in available_temperatures
            ) or "unknown"
            return (
                f"Found {count} {source}, but none matched the required "
                f"{display_number(light.effective_temperature, 1)} C temperature within "
                f"+/-{display_number(options.temperature_tolerance, 1)} C "
                f"(available: {available})."
            )
    return f"Found {count} {source}, but none met the required calibration metadata."


def choose_master(
    kind: str,
    light: FrameMeta,
    candidates: list[FrameMeta],
    options: PlanOptions,
    require_filter: bool = False,
    require_exact_exposure: bool = False,
) -> CalibrationChoice:
    candidates_of_kind = [
        candidate for candidate in candidates
        if candidate.frame_type == kind and candidate.is_master
    ]
    matches: list[FrameMeta] = []
    for candidate in candidates_of_kind:
        if not compatible_camera(light, candidate):
            continue
        if require_filter and normalized_text(light.filter_name) != normalized_text(candidate.filter_name):
            continue
        if require_exact_exposure and not almost_equal(
            light.exposure, candidate.exposure, options.exposure_tolerance
        ):
            continue
        if kind in {FRAME_DARK, FRAME_DARKFLAT} and light.effective_temperature is not None and candidate.effective_temperature is not None:
            if abs(light.effective_temperature - candidate.effective_temperature) > options.temperature_tolerance:
                continue
        matches.append(candidate)
    if not matches:
        warning = calibration_mismatch_warning(
            kind,
            light,
            candidates,
            options,
            require_filter=require_filter,
            require_exact_exposure=require_exact_exposure,
            require_temperature=kind in {FRAME_DARK, FRAME_DARKFLAT},
            masters=True,
        )
        return CalibrationChoice(kind=kind, warning=warning)
    matches.sort(key=lambda item: calibration_score(light, item, options), reverse=True)
    selected = matches[0]
    warning = ""
    if len(matches) > 1:
        warning = f"Selected best of {len(matches)} compatible {kind.lower()} masters."
    return CalibrationChoice(
        kind=kind,
        mode="master",
        files=[selected.path],
        description=f"Master: {Path(selected.path).name}",
        warning=warning,
    )


def choose_raw_group(
    kind: str,
    light: FrameMeta,
    candidates: list[FrameMeta],
    options: PlanOptions,
    require_filter: bool = False,
    require_exact_exposure: bool = False,
) -> CalibrationChoice:
    candidates_of_kind = [
        candidate for candidate in candidates
        if candidate.frame_type == kind and not candidate.is_master
    ]
    matches: list[FrameMeta] = []
    for candidate in candidates_of_kind:
        if not compatible_camera(light, candidate):
            continue
        if require_filter and normalized_text(light.filter_name) != normalized_text(candidate.filter_name):
            continue
        if require_exact_exposure and not almost_equal(
            light.exposure, candidate.exposure, options.exposure_tolerance
        ):
            continue
        if kind in {FRAME_DARK, FRAME_DARKFLAT} and require_exact_exposure:
            if light.effective_temperature is not None and candidate.effective_temperature is not None:
                if abs(light.effective_temperature - candidate.effective_temperature) > options.temperature_tolerance:
                    continue
        matches.append(candidate)
    if len(matches) < options.minimum_calibration_frames:
        warning = ""
        if matches:
            warning = (
                f"Only {len(matches)} compatible raw {kind.lower()} frames; "
                f"minimum is {options.minimum_calibration_frames}."
            )
        elif candidates_of_kind:
            warning = calibration_mismatch_warning(
                kind,
                light,
                candidates,
                options,
                require_filter=require_filter,
                require_exact_exposure=require_exact_exposure,
                require_temperature=(
                    kind in {FRAME_DARK, FRAME_DARKFLAT}
                    and require_exact_exposure
                ),
                masters=False,
            )
        return CalibrationChoice(kind=kind, warning=warning)
    return CalibrationChoice(
        kind=kind,
        mode="build",
        files=[item.path for item in sorted(matches, key=lambda item: item.path)],
        description=f"Build from {len(matches)} raw {kind.lower()} frames",
    )


def choose_calibration(
    kind: str,
    light: FrameMeta,
    candidates: list[FrameMeta],
    options: PlanOptions,
    require_filter: bool = False,
    require_exact_exposure: bool = False,
) -> CalibrationChoice:
    master = choose_master(
        kind, light, candidates, options, require_filter, require_exact_exposure
    )
    raw = choose_raw_group(
        kind, light, candidates, options, require_filter, require_exact_exposure
    )
    if options.prefer_masters:
        selected, alternate = master, raw
    else:
        selected, alternate = raw, master
    if selected.available:
        return selected
    if alternate.available:
        return alternate
    warnings = [warning for warning in (selected.warning, alternate.warning) if warning]
    return CalibrationChoice(kind=kind, warning=" ".join(warnings))


def choose_bpm_dark(
    light: FrameMeta,
    candidates: list[FrameMeta],
    options: PlanOptions,
) -> CalibrationChoice:
    compatible = [
        item for item in candidates
        if item.frame_type == FRAME_DARK and compatible_camera(light, item)
    ]
    if not compatible:
        return CalibrationChoice(kind="BPM")
    masters = [item for item in compatible if item.is_master]
    if masters and options.prefer_masters:
        masters.sort(key=lambda item: calibration_score(light, item, options), reverse=True)
        selected = masters[0]
        return CalibrationChoice(
            kind="BPM", mode="bpm_master", files=[selected.path],
            description=f"BPM from {Path(selected.path).name}",
        )
    raw = [item for item in compatible if not item.is_master]
    buckets: dict[tuple[Any, ...], list[FrameMeta]] = defaultdict(list)
    for item in raw:
        temp = item.effective_temperature
        temp_bucket = None if temp is None else round(temp / max(options.temperature_tolerance, 0.1))
        exposure_bucket = None if item.exposure is None else round(item.exposure, 2)
        buckets[(exposure_bucket, temp_bucket)].append(item)
    eligible = [items for items in buckets.values() if len(items) >= options.minimum_calibration_frames]
    if not eligible:
        if masters:
            selected = max(masters, key=lambda item: calibration_score(light, item, options))
            return CalibrationChoice(
                kind="BPM", mode="bpm_master", files=[selected.path],
                description=f"BPM from {Path(selected.path).name}",
            )
        return CalibrationChoice(kind="BPM")
    eligible.sort(
        key=lambda items: (
            len(items),
            -abs((items[0].exposure or 0) - (light.exposure or 0)),
        ),
        reverse=True,
    )
    selected_group = eligible[0]
    return CalibrationChoice(
        kind="BPM", mode="bpm_build",
        files=[item.path for item in sorted(selected_group, key=lambda item: item.path)],
        description=f"BPM from {len(selected_group)} raw darks ({display_number(selected_group[0].exposure)}s)",
    )


def choose_optimized_dark(
    light: FrameMeta,
    candidates: list[FrameMeta],
    options: PlanOptions,
) -> CalibrationChoice:
    """Choose a temperature-compatible non-exact dark for Siril's -opt mode."""
    darks = [item for item in candidates if item.frame_type == FRAME_DARK]
    compatible = [
        item for item in darks
        if compatible_camera(light, item)
        and almost_equal(
            light.effective_temperature,
            item.effective_temperature,
            options.temperature_tolerance,
        )
    ]

    master_choice = CalibrationChoice(kind=FRAME_DARK)
    masters = [item for item in compatible if item.is_master]
    if masters:
        selected = max(
            masters, key=lambda item: calibration_score(light, item, options)
        )
        master_choice = CalibrationChoice(
            kind=FRAME_DARK,
            mode="optimized_master",
            files=[selected.path],
            description=f"Optimize: {Path(selected.path).name}",
        )
    else:
        master_choice.warning = calibration_mismatch_warning(
            FRAME_DARK,
            light,
            candidates,
            options,
            require_filter=False,
            require_exact_exposure=False,
            require_temperature=True,
            masters=True,
        )

    raw_choice = CalibrationChoice(kind=FRAME_DARK)
    raw = [item for item in compatible if not item.is_master]
    buckets: dict[tuple[Any, ...], list[FrameMeta]] = defaultdict(list)
    for item in raw:
        temperature = item.effective_temperature
        temperature_bucket = (
            None if temperature is None
            else round(temperature / max(options.temperature_tolerance, 0.1))
        )
        exposure_bucket = None if item.exposure is None else round(item.exposure, 3)
        buckets[(exposure_bucket, temperature_bucket)].append(item)
    eligible = [
        items for items in buckets.values()
        if len(items) >= options.minimum_calibration_frames
    ]
    if eligible:
        eligible.sort(
            key=lambda items: (
                calibration_score(light, items[0], options),
                len(items),
            ),
            reverse=True,
        )
        selected_group = eligible[0]
        raw_choice = CalibrationChoice(
            kind=FRAME_DARK,
            mode="optimized_build",
            files=[
                item.path for item in sorted(selected_group, key=lambda item: item.path)
            ],
            description=(
                f"Optimize from {len(selected_group)} raw darks "
                f"({display_number(selected_group[0].exposure)}s)"
            ),
        )
    elif raw:
        largest = max((len(items) for items in buckets.values()), default=0)
        raw_choice.warning = (
            f"Largest compatible raw dark group has {largest} frame(s); "
            f"minimum is {options.minimum_calibration_frames}."
        )
    else:
        raw_choice.warning = calibration_mismatch_warning(
            FRAME_DARK,
            light,
            candidates,
            options,
            require_filter=False,
            require_exact_exposure=False,
            require_temperature=True,
            masters=False,
        )

    preferred, alternate = (
        (master_choice, raw_choice)
        if options.prefer_masters
        else (raw_choice, master_choice)
    )
    if preferred.available:
        return preferred
    if alternate.available:
        return alternate
    warnings = [
        warning for warning in (preferred.warning, alternate.warning) if warning
    ]
    return CalibrationChoice(kind=FRAME_DARK, warning=" ".join(warnings))


def subgroup_key(meta: FrameMeta, options: PlanOptions) -> tuple[Any, ...]:
    temperature = meta.effective_temperature
    temperature_bucket = None
    if temperature is not None:
        temperature_bucket = round(temperature / max(options.temperature_tolerance, 0.1))
    exposure = None if meta.exposure is None else round(meta.exposure, 3)
    return (
        normalize_filter(meta.filter_name, meta.bayer), exposure,
        round(meta.gain, 3) if meta.gain is not None else None,
        round(meta.offset, 3) if meta.offset is not None else None,
        temperature_bucket,
    )


def geometry_label(meta: FrameMeta) -> str:
    dimensions = f"{meta.width or '?'}x{meta.height or '?'}"
    camera = meta.instrument or "Unknown camera"
    cfa = f", CFA {meta.bayer}" if meta.bayer else ", mono"
    return f"{camera}, {dimensions}, bin {meta.bin_x or '?'}{cfa}"


def dataset_uses_true_cfa_drizzle(
    dataset: DatasetPlan,
    drizzle_channels: Iterable[str],
    options: PlanOptions,
) -> bool:
    """Whether a complete CFA setup can stay mosaiced until Bayer drizzle."""
    keys = {channel_key(dataset.dataset_id, name) for name in dataset.channels}
    selected = set(drizzle_channels)
    representatives = [groups[0].representative for groups in dataset.channels.values() if groups]
    return bool(
        options.debayer_cfa
        and keys
        and keys.issubset(selected)
        and representatives
        and all(frame.bayer for frame in representatives)
    )


def estimate_plan_storage(
    datasets: list[DatasetPlan],
    options: PlanOptions,
    drizzle_channels: Iterable[str],
) -> tuple[int, int]:
    """Estimate peak workspace and retained size from the actual output modes.

    Siril's quality-preserving workflow writes 32-bit calibrated images and a
    second 32-bit registered generation. Drizzle scales area by scale squared.
    For mixed native/drizzle output, SEQAPPLYREG must see the complete globally
    registered sequence in each mode so all channels retain one reference and
    common crop.
    """
    selected = set(drizzle_channels)
    scale_area = max(0.01, options.drizzle_scale**2)
    calibrated_total = 0.0
    conversion_total = 0.0
    registered_peak = 0.0
    registered_retained = 0.0
    final_masters = 0.0
    removable_mosaic_tiles = 0.0
    mosaic_workspace = 0.0
    mosaic_final_masters = 0.0
    rolling_panel_workspace = 0.0
    rolling_mosaic_workspace = 0.0

    for dataset in datasets:
        dataset_keys = {channel_key(dataset.dataset_id, name) for name in dataset.channels}
        has_drizzle = bool(dataset_keys & selected)
        has_native = bool(dataset_keys - selected)
        true_cfa = dataset_uses_true_cfa_drizzle(dataset, selected, options)
        calibrated_dataset = 0.0
        drizzle_registered_dataset = 0.0
        native_master_bytes: dict[str, float] = {}
        drizzle_master_bytes: dict[str, float] = {}
        panel_calibrated: dict[str, float] = defaultdict(float)
        panel_conversion: dict[str, float] = defaultdict(float)
        panel_drizzle_registered: dict[str, float] = defaultdict(float)

        for filter_name, groups in dataset.channels.items():
            channel_native_master = 0.0
            channel_drizzle_master = 0.0
            for subgroup in groups:
                for frame in subgroup.frames:
                    if frame.width and frame.height:
                        pixels = frame.width * frame.height
                        source_channels = max(1, frame.channels)
                        calibrated_channels = source_channels
                        if frame.bayer and options.debayer_cfa and not true_cfa:
                            calibrated_channels = 3
                        calibrated_bytes = pixels * calibrated_channels * 4
                        calibrated_dataset += calibrated_bytes
                        panel_calibrated[subgroup.panel_id] += calibrated_bytes
                        if file_suffix(frame.path_obj) not in FITS_SUFFIXES:
                            conversion_bytes = pixels * source_channels * 4
                            conversion_total += conversion_bytes
                            panel_conversion[subgroup.panel_id] += conversion_bytes
                        drizzle_channels_out = 3 if frame.bayer and true_cfa else calibrated_channels
                        drizzle_bytes = pixels * drizzle_channels_out * 4 * scale_area
                        drizzle_registered_dataset += drizzle_bytes
                        panel_drizzle_registered[subgroup.panel_id] += drizzle_bytes
                        channel_native_master = max(channel_native_master, calibrated_bytes)
                        channel_drizzle_master = max(
                            channel_drizzle_master,
                            pixels * drizzle_channels_out * 4 * scale_area,
                        )
                    else:
                        fallback = max(frame.size, 100 * 1024 * 1024)
                        calibrated_bytes = fallback * (3 if frame.bayer else 1)
                        drizzle_bytes = fallback * scale_area
                        calibrated_dataset += calibrated_bytes
                        panel_calibrated[subgroup.panel_id] += calibrated_bytes
                        drizzle_registered_dataset += drizzle_bytes
                        panel_drizzle_registered[subgroup.panel_id] += drizzle_bytes
                        if file_suffix(frame.path_obj) not in FITS_SUFFIXES:
                            conversion_total += fallback
                            panel_conversion[subgroup.panel_id] += fallback
                        channel_native_master = max(channel_native_master, fallback)
                        channel_drizzle_master = max(channel_drizzle_master, fallback * scale_area)
            native_master_bytes[filter_name] = channel_native_master
            drizzle_master_bytes[filter_name] = channel_drizzle_master

        calibrated_total += calibrated_dataset
        native_registered_dataset = calibrated_dataset
        # Siril drizzle also keeps an 8-bit per-pixel weight sequence in
        # drizztmp beside the 32-bit registered images (about 25% extra).
        drizzle_workspace_dataset = drizzle_registered_dataset * 1.25
        mode_sizes = []
        if has_native:
            mode_sizes.append(native_registered_dataset)
        if has_drizzle:
            mode_sizes.append(drizzle_workspace_dataset)
        if retains_intermediates(options):
            registered_peak += sum(mode_sizes)
            registered_retained += sum(mode_sizes)
        else:
            # Mixed modes are generated/stacked one at a time and discarded.
            registered_peak += max(mode_sizes, default=0.0)

        dataset_tile_bytes = 0.0
        dataset_native_tiles = 0.0
        dataset_drizzle_tiles = 0.0
        for filter_name, groups in dataset.channels.items():
            panel_count = len({group.panel_id for group in groups})
            if channel_key(dataset.dataset_id, filter_name) in selected:
                tile_bytes = drizzle_master_bytes[filter_name] * panel_count
                dataset_drizzle_tiles += tile_bytes
            else:
                tile_bytes = native_master_bytes[filter_name] * panel_count
                dataset_native_tiles += tile_bytes
            final_masters += tile_bytes
            dataset_tile_bytes += tile_bytes
            if dataset.panel_count > 1 and options.mosaic_enabled:
                # The mosaic union is at most the sum of its tile areas; the
                # transformed mosaic tiles average about 1.5x their source
                # bounding boxes when arbitrary rotation is present.
                mosaic_final_masters += tile_bytes

        if dataset.panel_count > 1 and options.mosaic_enabled:
            removable_mosaic_tiles += dataset_tile_bytes
            mosaic_multiplier = 2.5
            if options.mosaic_gradient_correction:
                mosaic_multiplier += 1.0
            if options.mosaic_background_matching:
                mosaic_multiplier += 1.0
            mosaic_workspace += dataset_tile_bytes * mosaic_multiplier
            rolling_mosaic_workspace = max(
                rolling_mosaic_workspace,
                max(dataset_native_tiles, dataset_drizzle_tiles) * mosaic_multiplier,
            )

        if options.rolling_cleanup:
            for panel_id in dataset.panels:
                panel_modes: list[float] = []
                if has_native:
                    panel_modes.append(panel_calibrated[panel_id])
                if has_drizzle:
                    panel_modes.append(panel_drizzle_registered[panel_id] * 1.25)
                rolling_panel_workspace = max(
                    rolling_panel_workspace,
                    panel_calibrated[panel_id]
                    + panel_conversion[panel_id]
                    + max(panel_modes, default=0.0),
                )

    # Rejection maps can add two master-sized files per output channel.
    if options.create_rejection_maps:
        final_masters *= 3
        removable_mosaic_tiles *= 3
    final_products = final_masters + mosaic_final_masters
    safety = 1.08
    base_overhead = 2 * 1024**3
    if options.rolling_cleanup:
        peak = int(
            (
                rolling_panel_workspace + rolling_mosaic_workspace
                + final_masters + mosaic_final_masters
            ) * safety
        ) + base_overhead
    else:
        peak = int(
            (
                calibrated_total + conversion_total + registered_peak
                + final_masters + mosaic_workspace + mosaic_final_masters
            ) * safety
        ) + base_overhead
    if retains_intermediates(options):
        retained = int(
            (
                calibrated_total + conversion_total + registered_retained
                + final_products + mosaic_workspace
            ) * safety
        ) + base_overhead
    else:
        retained = int(
            max(0.0, final_products - removable_mosaic_tiles) * safety
        ) + base_overhead
    return peak, retained


def refresh_plan_storage(plan: ProcessingPlan) -> None:
    """Refresh estimates and the disk-space blocking warning after UI changes."""
    for dataset in plan.datasets:
        dataset.warnings = [
            message for message in dataset.warnings
            if not re.match(r"^Detected \d+ (?:mosaic )?panels;", message)
        ]
        if dataset.panel_count > 1:
            if plan.options.mosaic_enabled:
                dataset.warnings.insert(
                    0,
                    f"Detected {dataset.panel_count} mosaic panels; SBPP will integrate each "
                    "panel independently and stitch one astrometric mosaic per filter.",
                )
            else:
                dataset.warnings.insert(
                    0,
                    f"Detected {dataset.panel_count} panels; mosaic stitching is disabled, "
                    "so panel masters will be retained separately.",
                )
    if plan.master_merge is not None:
        plan.estimated_bytes, plan.estimated_final_bytes = (
            estimate_master_merge_storage(plan.master_merge)
        )
    else:
        plan.estimated_bytes, plan.estimated_final_bytes = estimate_plan_storage(
            plan.datasets, plan.options, plan.drizzle_channels
        )
    plan.warnings = [
        message for message in plan.warnings
        if not message.startswith("BLOCKING: estimated peak workspace")
        and not message.startswith("BLOCKING: GraXpert panel correction")
    ]
    needs_graxpert = (
        plan.options.mosaic_enabled
        and plan.options.mosaic_gradient_correction
        and any(dataset.panel_count > 1 for dataset in plan.datasets)
    )
    if needs_graxpert and not resolve_graxpert_executable(
        plan.options.graxpert_executable
    ):
        plan.warnings.append(
            "BLOCKING: GraXpert panel correction is enabled, but the standalone "
            "GraXpert executable was not found. Select it in Advanced or disable "
            "panel gradient correction."
        )
    if plan.free_bytes and plan.estimated_bytes > plan.free_bytes:
        plan.warnings.append(
            f"BLOCKING: estimated peak workspace {human_bytes(plan.estimated_bytes)} "
            f"exceeds {human_bytes(plan.free_bytes)} free."
        )


MERGE_SCALE_TOLERANCE = 0.03
MERGE_GRID_RETENTION = 0.90
MERGE_TRANSITION_FRACTION = 0.08
MERGE_MAX_SIRIL_SCALE = 3.0


def plate_scale_arcsec(meta: FrameMeta) -> Optional[float]:
    """Header plate scale in arcsec/pixel.

    Like the mosaic field-of-view estimate, XPIXSZ is taken as the effective
    (already binned) pixel size, which is the common FITS convention.
    """
    if not meta.focal_length_mm or meta.focal_length_mm <= 0:
        return None
    if not meta.pixel_size_x_um or meta.pixel_size_x_um <= 0:
        return None
    return 206.264806 * meta.pixel_size_x_um / meta.focal_length_mm


def cluster_plate_scales(
    scales: list[Optional[float]], tolerance: float = MERGE_SCALE_TOLERANCE
) -> list[int]:
    """Assign 1-based layers, finest first; scales within tolerance share one.

    Unknown scales each receive a provisional layer after every known one;
    the run re-clusters them from their solved scales.
    """
    order = sorted(
        (index for index, scale in enumerate(scales) if scale),
        key=lambda index: float(scales[index]),  # type: ignore[arg-type]
    )
    layers = [0] * len(scales)
    layer_start: Optional[float] = None
    layer = 0
    for index in order:
        scale = float(scales[index])  # type: ignore[arg-type]
        if layer_start is None or scale > layer_start * (1.0 + tolerance):
            layer += 1
            layer_start = scale
        layers[index] = layer
    for index, scale in enumerate(scales):
        if not scale:
            layer += 1
            layers[index] = layer
    return layers


def merge_layer_role(layer: int, layer_count: int) -> str:
    if layer == 1:
        return "detail"
    return "wide fill" if layer == layer_count else "intermediate fill"


def build_master_merge_plan(masters: list[FrameMeta]) -> MasterMergePlan:
    """Group finished masters into plate-scale layers for a multi-scale merge."""
    scales = [plate_scale_arcsec(meta) for meta in masters]
    layers = cluster_plate_scales(scales)
    layer_count = max(layers, default=0)
    entries = [
        MergeMaster(
            path=meta.path,
            filter_name=normalize_filter(meta.filter_name, meta.bayer),
            layer=layer,
            scale_arcsec=scale,
            width=meta.width,
            height=meta.height,
            telescope=meta.telescope,
            focal_length_mm=meta.focal_length_mm,
            exposure=meta.exposure,
            role=merge_layer_role(layer, layer_count),
        )
        for meta, scale, layer in zip(masters, scales, layers)
    ]
    entries.sort(key=lambda entry: (
        entry.layer,
        FILTER_ORDER.get(normalized_text(entry.filter_name), 100),
        normalized_text(entry.filter_name),
    ))
    layer_scales: list[Optional[float]] = []
    for layer in range(1, layer_count + 1):
        members = [entry.scale_arcsec for entry in entries if entry.layer == layer]
        known = [value for value in members if value]
        layer_scales.append(min(known) if known else None)

    filters: list[str] = []
    for entry in entries:
        if entry.filter_name not in filters:
            filters.append(entry.filter_name)
    filters.sort(key=lambda name: (
        FILTER_ORDER.get(normalized_text(name), 100), normalized_text(name)
    ))

    warnings: list[str] = []
    if len({max(1, meta.channels) for meta in masters}) > 1:
        warnings.append(
            "BLOCKING: master merge needs either all monochrome or all colour masters."
        )
    seen: dict[tuple[int, str], str] = {}
    for entry in entries:
        key = (entry.layer, normalized_text(entry.filter_name))
        if key in seen:
            warnings.append(
                f"BLOCKING: layer {entry.layer} has two {entry.filter_name} masters at the "
                f"same plate scale ({Path(seen[key]).name} and {Path(entry.path).name}). "
                "Merging same-scale masters (a mosaic of masters) is not supported yet; "
                "keep one per filter and scale."
            )
        else:
            seen[key] = entry.path
    if any(not meta.width or not meta.height for meta in masters):
        warnings.append("BLOCKING: one or more masters have unreadable image dimensions.")
    if layer_count < 2:
        warnings.append(
            "BLOCKING: every master has the same plate scale, so there is nothing to "
            "merge. Master merge combines data from telescopes with different focal "
            "lengths."
        )
    if any(scale is None for scale in scales):
        warnings.append(
            "Some masters lack focal length or pixel size; their layer will be "
            "determined from the plate solve."
        )

    by_layer: dict[int, list[FrameMeta]] = defaultdict(list)
    for meta, layer in zip(masters, layers):
        by_layer[layer].append(meta)
    # Field overlap is checked from pointing headers when every layer has them.
    if layer_count >= 2:
        fine = by_layer[1][0]
        for layer in range(2, layer_count + 1):
            wide = by_layer[layer][0]
            fov_fine = short_field_of_view_deg(fine)
            fov_wide = None
            if wide.focal_length_mm and wide.pixel_size_x_um and wide.width and wide.height:
                fov_wide = math.degrees(math.hypot(
                    wide.width * wide.pixel_size_x_um, wide.height * wide.pixel_size_x_um
                ) / 1000.0 / wide.focal_length_mm)
            if (
                fov_fine and fov_wide
                and None not in (
                    fine.pointing_ra_deg, fine.pointing_dec_deg,
                    wide.pointing_ra_deg, wide.pointing_dec_deg,
                )
            ):
                separation = angular_separation_deg(
                    fine.pointing_ra_deg, fine.pointing_dec_deg,  # type: ignore[arg-type]
                    wide.pointing_ra_deg, wide.pointing_dec_deg,  # type: ignore[arg-type]
                )
                if separation > (fov_wide + fov_fine) / 2.0:
                    warnings.append(
                        f"BLOCKING: layer {layer} points {separation:.2f}° away from layer 1 "
                        "and cannot overlap it."
                    )

    for name in filters:
        present = sorted({entry.layer for entry in entries if entry.filter_name == name})
        if 1 not in present:
            warnings.append(
                f"Filter {name} exists only in a wider layer; its output is that data "
                "upsampled to the finest scale without added detail."
            )
        elif present == [1] and layer_count > 1:
            warnings.append(
                f"Filter {name} exists only in the finest layer; outside that field its "
                "output will be empty."
            )

    estimated_canvas = (0, 0)
    finest_scale = next((scale for scale in layer_scales if scale), None)
    widest = max(
        (entry for entry in entries if entry.scale_arcsec and entry.width and entry.height),
        key=lambda entry: (
            float(entry.scale_arcsec) * float(entry.width) * float(entry.scale_arcsec)  # type: ignore[arg-type]
            * float(entry.height)  # type: ignore[arg-type]
        ),
        default=None,
    )
    if finest_scale and widest is not None:
        ratio = float(widest.scale_arcsec) / finest_scale  # type: ignore[arg-type]
        estimated_canvas = (
            int(round(float(widest.width) * ratio)),  # type: ignore[arg-type]
            int(round(float(widest.height) * ratio)),  # type: ignore[arg-type]
        )
    return MasterMergePlan(
        masters=entries,
        layer_scales=layer_scales,
        filters=filters,
        estimated_canvas=estimated_canvas,
        warnings=warnings,
    )


def estimate_master_merge_storage(merge: MasterMergePlan) -> tuple[int, int]:
    """Peak disk use (solved copies, registered canvases, outputs) and retained size."""
    canvas_pixels = merge.estimated_canvas[0] * merge.estimated_canvas[1]
    native = sum(
        (entry.width or 0) * (entry.height or 0) * 4 for entry in merge.masters
    )
    registered = canvas_pixels * 4 * len(merge.masters)
    outputs = canvas_pixels * 4 * len(merge.filters)
    overhead = 2 * 1024**3
    return int((native + registered + outputs) * 1.08) + overhead, int(outputs * 1.08) + overhead


def build_plan(
    frames: list[FrameMeta],
    input_root: Path,
    calibration_roots: list[Path],
    output_root: Path,
    target_name: str,
    options: PlanOptions,
    scan_errors: Optional[list[str]] = None,
) -> ProcessingPlan:
    lights = [
        item for item in frames
        if item.frame_type == FRAME_LIGHT and not item.is_master
    ]
    master_lights = [
        item for item in frames
        if item.frame_type == FRAME_LIGHT and item.is_master
    ]
    if not lights and master_lights:
        return build_master_merge_processing_plan(
            master_lights, frames, input_root, calibration_roots, output_root,
            target_name, options, scan_errors,
        )
    calibrations = [
        item for item in frames
        if item.frame_type in {FRAME_BIAS, FRAME_DARK, FRAME_FLAT, FRAME_DARKFLAT}
    ]
    calibration_by_path = {item.path: item for item in calibrations}
    ignored_count = len(frames) - len(lights) - len(calibrations)
    datasets_by_geometry: dict[tuple[Any, ...], list[FrameMeta]] = defaultdict(list)
    for light in lights:
        datasets_by_geometry[light.geometry_key].append(light)

    dataset_plans: list[DatasetPlan] = []
    global_warnings: list[str] = []
    for dataset_index, dataset_lights in enumerate(
        sorted(datasets_by_geometry.values(), key=lambda values: (-len(values), geometry_label(values[0]))),
        start=1,
    ):
        representative = dataset_lights[0]
        dataset_id = f"setup_{dataset_index:02d}"
        panel_assignments, panel_detection, panel_detection_warnings = assign_panel_ids(
            dataset_lights, input_root, options
        )
        grouped: dict[tuple[Any, ...], list[FrameMeta]] = defaultdict(list)
        for light in dataset_lights:
            grouped[(panel_assignments[light.path], *subgroup_key(light, options))].append(light)

        channels: dict[str, list[LightSubgroup]] = defaultdict(list)
        for group_index, (group_key, group_frames) in enumerate(
            sorted(
                grouped.items(),
                key=lambda item: (
                    natural_key(str(item[0][0])),
                    FILTER_ORDER.get(normalized_text(item[0][1]), 100),
                    str(item[0]),
                ),
            ),
            start=1,
        ):
            sample = group_frames[0]
            panel_id = str(group_key[0])
            filter_name = normalize_filter(sample.filter_name, sample.bayer)
            bias = choose_calibration(FRAME_BIAS, sample, calibrations, options)
            flat = choose_calibration(
                FRAME_FLAT, sample, calibrations, options, require_filter=True
            )
            flat_sample = calibration_by_path.get(flat.files[0]) if flat.available else None
            darkflat = CalibrationChoice(kind=FRAME_DARKFLAT)
            if flat_sample is not None and flat.mode == "build":
                darkflat = choose_calibration(
                    FRAME_DARKFLAT,
                    flat_sample,
                    calibrations,
                    options,
                    require_exact_exposure=True,
                )
            dark = CalibrationChoice(kind=FRAME_DARK)
            bpm = CalibrationChoice(kind="BPM")
            if options.dark_policy in {
                "exact", "exact_or_bpm", "exact_or_optimized"
            }:
                dark = choose_calibration(
                    FRAME_DARK, sample, calibrations, options, require_exact_exposure=True
                )
            if options.dark_policy == "exact_or_optimized" and not dark.available:
                dark = choose_optimized_dark(sample, calibrations, options)
            if options.dark_policy == "bpm_only" or (
                options.dark_policy == "exact_or_bpm" and not dark.available
            ):
                bpm = choose_bpm_dark(sample, calibrations, options)

            warnings: list[str] = []
            if not flat.available:
                warnings.append("BLOCKING: no compatible filter-matched flat.")
                if flat.warning:
                    warnings.append(flat.warning)
            elif flat.mode == "build":
                if not darkflat.available and not bias.available:
                    warnings.append(
                        "BLOCKING: raw flats require a compatible bias or exact-exposure "
                        "flat-dark before they can be integrated safely."
                    )
            if not dark.available and not bias.available:
                warnings.append(
                    "BLOCKING: lights require either an exact dark or compatible bias "
                    "before flat-field division; a BPM does not remove the camera pedestal."
                )
            if dark.optimized and not bias.available:
                warnings.append(
                    "BLOCKING: optimized dark subtraction requires a compatible bias master."
                )
            if options.dark_policy == "exact" and not dark.available:
                warnings.append("No exact compatible dark; dark subtraction will be skipped.")
            if options.dark_policy == "exact_or_optimized" and not dark.available:
                warnings.append(
                    "No exact or optimizable dark; dark subtraction will be skipped."
                )
            if options.dark_policy == "exact_or_bpm" and not dark.available:
                if not bpm.available:
                    warnings.append("No exact dark or BPM source; dark correction will be skipped.")
            if options.dark_policy == "bpm_only" and not bpm.available:
                warnings.append("No compatible dark is available to construct a BPM.")
            diagnostic_choices = [bias, flat, darkflat, bpm]
            # An exposure-mismatched dark is expected under Exact-or-BPM when a
            # compatible BPM source was found. Its exact-dark diagnostic would
            # otherwise contradict the selected fallback and alarm the user.
            if dark.available or not bpm.available:
                diagnostic_choices.append(dark)
            for choice in diagnostic_choices:
                if choice.warning:
                    warnings.append(choice.warning)
            subgroup = LightSubgroup(
                subgroup_id=(
                    f"group_{group_index:03d}_{slugify(panel_id)}_{slugify(filter_name)}"
                ),
                panel_id=panel_id,
                filter_name=filter_name,
                frames=sorted(group_frames, key=lambda item: item.path),
                bias=bias,
                dark=dark,
                flat=flat,
                darkflat=darkflat,
                bpm=bpm,
                warnings=warnings,
            )
            channels[filter_name].append(subgroup)

        sorted_channels = dict(sorted(
            channels.items(),
            key=lambda item: (FILTER_ORDER.get(normalized_text(item[0]), 100), normalized_text(item[0])),
        ))
        # The clustering summary is displayed in the Run Log and panel column;
        # only actionable panel-detection problems belong in the warning list.
        dataset_warnings: list[str] = list(panel_detection_warnings)
        if not representative.width or not representative.height:
            dataset_warnings.append("Image dimensions are unknown; disk estimate is conservative.")
        panel_ids = sorted(
            {group.panel_id for groups in sorted_channels.values() for group in groups},
            key=natural_key,
        )
        if len(panel_ids) > 1:
            if options.mosaic_enabled:
                dataset_warnings.append(
                    f"Detected {len(panel_ids)} mosaic panels; SBPP will integrate each panel "
                    "independently and stitch one astrometric mosaic per filter."
                )
            else:
                dataset_warnings.append(
                    f"Detected {len(panel_ids)} panels; mosaic stitching is disabled, so panel "
                    "masters will be retained separately."
                )
            for filter_name, groups in sorted_channels.items():
                present = {group.panel_id for group in groups}
                if len(present) != len(panel_ids):
                    missing = ", ".join(panel for panel in panel_ids if panel not in present)
                    dataset_warnings.append(
                        f"Filter {filter_name} is missing {missing}; its mosaic will contain only "
                        f"{len(present)} of {len(panel_ids)} panels."
                    )
        dataset_plans.append(DatasetPlan(
            dataset_id=dataset_id,
            geometry_label=geometry_label(representative),
            channels=sorted_channels,
            panel_detection=panel_detection,
            warnings=dataset_warnings,
        ))

    if not lights:
        global_warnings.append("BLOCKING: no non-master LIGHT frames were found.")
    if lights and master_lights:
        global_warnings.append(
            f"Ignored {len(master_lights)} master light(s) beside the raw lights. Merging "
            "your stacks with external masters is planned for a later release; for now, "
            "place the masters alone in a folder to merge them."
        )
    if not target_name.strip():
        global_warnings.append(
            "BLOCKING: target name is blank. Enter it manually and rebuild the plan; "
            "SBPP only infers a target when at least three usable light OBJECT headers "
            "exist and all discovered values agree."
        )
    if len(dataset_plans) > 1:
        global_warnings.append(
            f"Detected {len(dataset_plans)} incompatible camera/geometry setups; each will be processed independently."
        )
    try:
        free_bytes = shutil.disk_usage(output_root).free
    except OSError:
        free_bytes = 0
    plan = ProcessingPlan(
        target_name=target_name,
        input_root=str(input_root),
        calibration_roots=[str(path) for path in calibration_roots],
        output_root=str(output_root),
        options=options,
        datasets=dataset_plans,
        scan_errors=list(scan_errors or []),
        ignored_count=ignored_count,
        warnings=global_warnings,
        free_bytes=free_bytes,
    )
    refresh_plan_storage(plan)
    return plan


def build_master_merge_processing_plan(
    master_lights: list[FrameMeta],
    frames: list[FrameMeta],
    input_root: Path,
    calibration_roots: list[Path],
    output_root: Path,
    target_name: str,
    options: PlanOptions,
    scan_errors: Optional[list[str]] = None,
) -> ProcessingPlan:
    """Plan a merge of finished masters; calibration and stacking are skipped."""
    merge = build_master_merge_plan(master_lights)
    global_warnings: list[str] = []
    if not target_name.strip():
        global_warnings.append(
            "BLOCKING: target name is blank. Enter it manually and rebuild the plan."
        )
    try:
        free_bytes = shutil.disk_usage(output_root).free
    except OSError:
        free_bytes = 0
    plan = ProcessingPlan(
        target_name=target_name,
        input_root=str(input_root),
        calibration_roots=[str(path) for path in calibration_roots],
        output_root=str(output_root),
        options=options,
        datasets=[],
        scan_errors=list(scan_errors or []),
        ignored_count=len(frames) - len(master_lights),
        warnings=global_warnings,
        free_bytes=free_bytes,
        master_merge=merge,
    )
    refresh_plan_storage(plan)
    return plan


def human_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{number:.1f} TiB"


def human_duration(seconds: float) -> str:
    """WBPP-style elapsed time without losing long-run day information."""
    total = max(0, int(round(seconds)))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def integration_summary(seconds: float, unknown_count: int = 0) -> str:
    """Format selected exposure with hours unbounded for astronomy totals."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        duration = f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes:
        duration = f"{minutes}m {secs:02d}s"
    else:
        duration = f"{secs}s"
    if unknown_count:
        suffix = "light" if unknown_count == 1 else "lights"
        if total:
            return f"at least {duration} + {unknown_count} {suffix} with unknown exposure"
        return f"unknown ({unknown_count} {suffix} missing exposure metadata)"
    return duration


def integration_folder_token(seconds: float, unknown_count: int = 0) -> str:
    """Return a compact, filesystem-safe integration label for a run folder."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        duration = f"{hours}h{minutes:02d}m"
        if secs:
            duration += f"{secs:02d}s"
    elif minutes:
        duration = f"{minutes}m"
        if secs:
            duration += f"{secs:02d}s"
    else:
        duration = f"{secs}s"
    if unknown_count:
        return f"{duration}_plus_unknown" if total else "unknown_integration"
    return duration


def final_filter_stem(filter_name: str) -> str:
    """Use a compact PixInsight-friendly name for a final filter master."""
    return slugify(normalize_filter(filter_name), "stack").lower()


def light_rejection_arguments(
    frame_count: int, options: PlanOptions
) -> list[str]:
    """Choose a rejection model appropriate for the channel sample size."""
    if frame_count <= TINY_STACK_MEDIAN_MAX:
        # With only a handful of samples, sigma estimators cannot reliably
        # distinguish detector defects from the background. A median is
        # deterministic and prevents one resampled hot pixel from becoming a
        # black/white integration artifact.
        return ["median"]
    linear_start = max(MIN_LINEAR_STACK_COUNT, options.small_stack_threshold)
    if frame_count < linear_start:
        # On the same 27 accepted NGC 6543 O frames, Winsorized 4/3 left
        # about nine times fewer isolated bright residuals than linear 5/5.
        # Keep the established 43- and 48-frame paths above this boundary.
        return ["rej", "winsorized", "4", "3"]
    if frame_count >= options.large_stack_threshold:
        # Linear 10/10 is too permissive for residual hot pixels and transient
        # bright events. On the exact 53-frame IC 1318 H dataset it left more
        # than twenty times as many isolated >8-sigma bright residuals as
        # Winsorized 4/3, while Winsorized preserved matched-star flux and FWHM.
        return ["rej", "winsorized", "4", "3"]
    return ["rej", "linear", "5", "5"]


def light_cosmetic_hot_sigma(frame_count: int) -> str:
    """Use a modestly stronger detector-defect map only for tiny stacks."""
    if frame_count <= TINY_STACK_MEDIAN_MAX:
        return TINY_STACK_HOT_SIGMA
    return STANDARD_HOT_SIGMA


def aligned_mosaic_tile_origins(
    paths: Iterable[Path],
    maximum_grid_error: float = 0.25,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float]:
    """Locate registered, differently sized tiles on their shared WCS grid."""
    image_paths = list(paths)
    headers = [fits.getheader(path, ext=0) for path in image_paths]
    shapes = [
        (int(header.get("NAXIS2", 0)), int(header.get("NAXIS1", 0)))
        for header in headers
    ]
    if not image_paths or any(height <= 0 or width <= 0 for height, width in shapes):
        raise RuntimeError("Unable to read aligned mosaic tile geometry.")
    try:
        reference = WCS(headers[0]).celestial
        if not reference.has_celestial:
            raise ValueError("missing celestial WCS")
        origins: list[tuple[int, int]] = []
        worst_error = 0.0
        for header, (height, width) in zip(headers, shapes):
            candidate = WCS(header).celestial
            if not candidate.has_celestial:
                raise ValueError("missing celestial WCS")
            points = np.asarray([
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [0.0, height - 1.0],
                [width - 1.0, height - 1.0],
                [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
            ])
            world = candidate.all_pix2world(points, 0)
            mapped = reference.all_world2pix(world, 0)
            translations = mapped - points
            translation = np.median(translations, axis=0)
            residuals = np.hypot(
                translations[:, 0] - translation[0],
                translations[:, 1] - translation[1],
            )
            integer_translation = np.rint(translation)
            error = max(
                float(np.max(residuals)),
                float(np.hypot(*(translation - integer_translation))),
            )
            if not math.isfinite(error) or error > maximum_grid_error:
                raise RuntimeError(
                    "Registered mosaic tiles are not on one integer-aligned WCS grid."
                )
            worst_error = max(worst_error, error)
            origins.append((
                int(integer_translation[0]), int(integer_translation[1])
            ))
        return origins, shapes, worst_error
    except RuntimeError:
        raise
    except Exception as exc:
        if len(set(shapes)) == 1:
            return [(0, 0)] * len(image_paths), shapes, 0.0
        raise RuntimeError(
            "Differently sized aligned mosaic tiles require a shared celestial WCS."
        ) from exc


def robust_overlap_background_offsets(
    paths: Iterable[Path],
    maximum_samples: int = 350_000,
    minimum_overlap_samples: int = 1_000,
) -> tuple[list[float], dict[str, Any]]:
    """Solve additive tile offsets from robust differences in shared sky areas."""
    image_paths = list(paths)
    count = len(image_paths)
    if count < 2:
        return [0.0] * count, {"pairs": [], "components": count}

    origins, shapes, grid_error = aligned_mosaic_tile_origins(image_paths)
    largest_area = max(height * width for height, width in shapes)
    stride = max(1, int(math.ceil(math.sqrt(largest_area / maximum_samples))))

    def sampled_intensity(
        data: np.ndarray,
        y_start: int,
        y_stop: int,
        x_start: int,
        x_stop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sampled = np.asarray(
            data[..., y_start:y_stop:stride, x_start:x_stop:stride],
            dtype=np.float64,
        )
        if sampled.ndim == 2:
            valid = np.isfinite(sampled) & (sampled != 0)
            intensity = sampled
        else:
            finite = np.all(np.isfinite(sampled), axis=0)
            supported = np.any(sampled != 0, axis=0)
            valid = finite & supported
            intensity = np.mean(sampled, axis=0)
        return intensity, valid

    equations: list[tuple[int, int, float, int, float]] = []
    pair_details: list[dict[str, Any]] = []
    adjacency: dict[int, set[int]] = {index: set() for index in range(count)}
    for left in range(count):
        for right in range(left + 1, count):
            left_x, left_y = origins[left]
            right_x, right_y = origins[right]
            left_height, left_width = shapes[left]
            right_height, right_width = shapes[right]
            overlap_left = max(left_x, right_x)
            overlap_top = max(left_y, right_y)
            overlap_right = min(left_x + left_width, right_x + right_width)
            overlap_bottom = min(left_y + left_height, right_y + right_height)
            if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
                continue
            with fits.open(
                image_paths[left], mode="readonly", memmap=True, lazy_load_hdus=True
            ) as left_hdul, fits.open(
                image_paths[right], mode="readonly", memmap=True, lazy_load_hdus=True
            ) as right_hdul:
                left_array = np.squeeze(left_hdul[0].data)
                right_array = np.squeeze(right_hdul[0].data)
                if left_array.ndim not in {2, 3} or right_array.ndim not in {2, 3}:
                    raise RuntimeError("Unsupported mosaic tile dimensions.")
                left_data, left_valid = sampled_intensity(
                    left_array,
                    overlap_top - left_y,
                    overlap_bottom - left_y,
                    overlap_left - left_x,
                    overlap_right - left_x,
                )
                right_data, right_valid = sampled_intensity(
                    right_array,
                    overlap_top - right_y,
                    overlap_bottom - right_y,
                    overlap_left - right_x,
                    overlap_right - right_x,
                )
            overlap = left_valid & right_valid
            overlap_count = int(np.count_nonzero(overlap))
            if overlap_count < minimum_overlap_samples:
                continue
            differences = left_data[overlap] - right_data[overlap]
            differences = differences[np.isfinite(differences)]
            if differences.size < minimum_overlap_samples:
                continue
            center = float(np.median(differences))
            sigma = 0.0
            for _iteration in range(3):
                mad = float(np.median(np.abs(differences - center)))
                sigma = 1.4826 * mad
                if not math.isfinite(sigma) or sigma <= 0:
                    break
                kept = differences[np.abs(differences - center) <= 4.0 * sigma]
                if kept.size < minimum_overlap_samples:
                    break
                differences = kept
                center = float(np.median(differences))
            sample_count = int(differences.size)
            equations.append((left, right, center, sample_count, sigma))
            adjacency[left].add(right)
            adjacency[right].add(left)
            pair_details.append({
                "left": left + 1,
                "right": right + 1,
                "difference": center,
                "samples": sample_count,
                "sigma": sigma,
                "overlap_width": overlap_right - overlap_left,
                "overlap_height": overlap_bottom - overlap_top,
            })

    if not equations:
        return [0.0] * count, {
            "pairs": [], "components": count, "sample_stride": stride,
            "grid_error_pixels": grid_error,
            "warning": "No sufficiently large aligned panel overlaps were found.",
        }

    largest_count = max(equation[3] for equation in equations)
    matrix = np.zeros((len(equations), count), dtype=np.float64)
    target = np.zeros(len(equations), dtype=np.float64)
    for row, (left, right, difference, sample_count, _sigma) in enumerate(equations):
        weight = math.sqrt(sample_count / largest_count)
        matrix[row, left] = weight
        matrix[row, right] = -weight
        # (left + offset_left) - (right + offset_right) must be zero.
        target[row] = -difference * weight
    solution = np.linalg.lstsq(matrix, target, rcond=None)[0]

    unseen = set(range(count))
    components = 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            current = pending.pop()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            pending.extend(neighbors)
    details: dict[str, Any] = {
        "pairs": pair_details,
        "components": components,
        "sample_stride": stride,
        "grid_error_pixels": grid_error,
        "tile_origins": [
            {"x": origin[0], "y": origin[1]}
            for origin in origins
        ],
    }
    if components > 1:
        details["warning"] = (
            "The overlap graph is disconnected; levels were matched only within "
            "connected panel groups."
        )
    return [float(value) for value in solution], details


def copy_fits_with_background_offset(
    source: Path, destination: Path, offset: float
) -> None:
    """Copy a float FITS tile and add an offset only inside its valid footprint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    apply_fits_background_offset(destination, offset)


def apply_fits_background_offset(path: Path, offset: float) -> None:
    """Add an offset to valid pixels while preserving a registered tile in place."""
    if abs(offset) <= np.finfo(np.float32).eps:
        return
    with fits.open(path, mode="update", memmap=True, lazy_load_hdus=True) as hdul:
        data = hdul[0].data
        if data is None or not np.issubdtype(data.dtype, np.floating):
            raise RuntimeError(f"Expected a floating-point mosaic tile: {path}")
        height = int(data.shape[-2])
        for start in range(0, height, 256):
            stop = min(height, start + 256)
            block = np.array(data[..., start:stop, :], copy=True)
            if block.ndim == 2:
                valid = np.isfinite(block) & (block != 0)
            else:
                spatial = np.all(np.isfinite(block), axis=0) & np.any(
                    block != 0, axis=0
                )
                valid = np.broadcast_to(spatial, block.shape) & np.isfinite(block)
            block[valid] += offset
            data[..., start:stop, :] = block
        for hdu in hdul:
            hdu.header.pop("CHECKSUM", None)
            hdu.header.pop("DATASUM", None)
        hdul.flush(output_verify="silentfix")


def shared_wcs_alignment_error(paths: Iterable[Path]) -> Optional[float]:
    """Return the worst same-pixel WCS displacement, or None without valid WCS."""
    image_paths = list(paths)
    if len(image_paths) < 2:
        return 0.0
    try:
        headers = [fits.getheader(path, ext=0) for path in image_paths]
        reference = WCS(headers[0]).celestial
        if not reference.has_celestial:
            return None
        width = min(int(header.get("NAXIS1", 0)) for header in headers)
        height = min(int(header.get("NAXIS2", 0)) for header in headers)
        if width <= 1 or height <= 1:
            return None
        x = np.asarray([0.0, width - 1.0, 0.0, width - 1.0, (width - 1.0) / 2.0])
        y = np.asarray([0.0, 0.0, height - 1.0, height - 1.0, (height - 1.0) / 2.0])
        worst = 0.0
        for header in headers[1:]:
            candidate = WCS(header).celestial
            if not candidate.has_celestial:
                return None
            world = candidate.all_pix2world(x, y, 0)
            reference_pixels = reference.all_world2pix(*world, 0)
            offsets = np.hypot(reference_pixels[0] - x, reference_pixels[1] - y)
            if not np.all(np.isfinite(offsets)):
                return None
            worst = max(worst, float(np.max(offsets)))
        return worst
    except Exception:
        return None


def vertical_mirror_wcs_error(
    before_header: fits.Header, after_header: fits.Header
) -> Optional[float]:
    """Return the worst sky error after a lossless top/bottom row reversal."""
    try:
        width = int(before_header.get("NAXIS1", 0))
        height = int(before_header.get("NAXIS2", 0))
        if (
            width <= 1
            or height <= 1
            or int(after_header.get("NAXIS1", 0)) != width
            or int(after_header.get("NAXIS2", 0)) != height
        ):
            return None
        before = WCS(before_header).celestial
        after = WCS(after_header).celestial
        if not before.has_celestial or not after.has_celestial:
            return None
        x = np.asarray([
            0.0, width - 1.0, 0.0, width - 1.0, (width - 1.0) / 2.0,
        ])
        y = np.asarray([
            0.0, 0.0, height - 1.0, height - 1.0, (height - 1.0) / 2.0,
        ])
        before_world = before.all_pix2world(x, y, 0)
        after_world = after.all_pix2world(x, height - 1.0 - y, 0)
        declination = np.deg2rad(
            (np.asarray(before_world[1]) + np.asarray(after_world[1])) / 2.0
        )
        delta_ra = (
            (np.asarray(after_world[0]) - np.asarray(before_world[0]) + 180.0)
            % 360.0
            - 180.0
        ) * np.cos(declination)
        delta_dec = np.asarray(after_world[1]) - np.asarray(before_world[1])
        offsets = np.hypot(delta_ra, delta_dec) * 3600.0
        if not np.all(np.isfinite(offsets)):
            return None
        return float(np.max(offsets))
    except Exception:
        return None


def _row_support_interval(
    row: np.ndarray, maximum_internal_gap: int = 8
) -> Optional[tuple[int, int]]:
    """Return the widest supported interval, tolerating isolated zero pixels."""
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(indices) > maximum_internal_gap + 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    widths = indices[ends] - indices[starts] + 1
    best = int(np.argmax(widths))
    return int(indices[starts[best]]), int(indices[ends[best]]) + 1


class MosaicCropTooLossy(RuntimeError):
    """The best clean rectangle keeps too little of an irregular footprint."""

    def __init__(self, kept_fraction: float, valid_fraction: float):
        super().__init__("The safe mosaic crop would discard more than 75% of the image.")
        self.kept_fraction = kept_fraction
        self.valid_fraction = valid_fraction


def largest_valid_mask_crop(
    valid_mask: np.ndarray, edge_inset: int = 2
) -> tuple[int, int, int, int]:
    """Find a large axis-aligned rectangle fully inside a supplied footprint."""
    if valid_mask.ndim != 2 or not valid_mask.size:
        raise ValueError("A non-empty two-dimensional validity mask is required.")
    height, width = valid_mask.shape
    left = np.full(height, width, dtype=np.int32)
    right = np.zeros(height, dtype=np.int32)
    for row_index in range(height):
        interval = _row_support_interval(valid_mask[row_index])
        if interval is not None:
            left[row_index], right[row_index] = interval

    best_area = 0
    best = (0, 0, width, height)
    best_width = 0
    for top in range(height):
        if left[top] >= right[top]:
            continue
        running_left = 0
        running_right = width
        for bottom in range(top, height):
            if left[bottom] >= right[bottom]:
                break
            running_left = max(running_left, int(left[bottom]))
            running_right = min(running_right, int(right[bottom]))
            rectangle_width = running_right - running_left
            if rectangle_width <= 0:
                break
            rectangle_height = bottom - top + 1
            area = rectangle_width * rectangle_height
            if area > best_area or (area == best_area and rectangle_width > best_width):
                best_area = area
                best_width = rectangle_width
                best = (running_left, top, rectangle_width, rectangle_height)

    if best_area == 0:
        raise RuntimeError("No common valid mosaic area was found.")
    x, y, crop_width, crop_height = best
    inset = max(0, int(edge_inset))
    if crop_width > 2 * inset and crop_height > 2 * inset:
        x += inset
        y += inset
        crop_width -= 2 * inset
        crop_height -= 2 * inset
    if crop_width * crop_height < width * height * 0.25:
        raise MosaicCropTooLossy(
            kept_fraction=crop_width * crop_height / float(width * height),
            valid_fraction=float(np.count_nonzero(valid_mask)) / float(width * height),
        )
    return x, y, crop_width, crop_height


def largest_common_valid_crop(
    paths: Iterable[Path], edge_inset: int = 2
) -> tuple[int, int, int, int]:
    """Find the largest low-waste rectangle inside every image's valid footprint."""
    image_paths = list(paths)
    if not image_paths:
        raise ValueError("No images were supplied for common cropping.")

    common: Optional[np.ndarray] = None
    shape: Optional[tuple[int, int]] = None
    for path in image_paths:
        with fits.open(path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
            data = np.squeeze(hdul[0].data)
            if data.ndim != 2:
                raise RuntimeError(f"Expected a monochrome mosaic image: {path}")
            current_shape = (int(data.shape[0]), int(data.shape[1]))
            if shape is None:
                shape = current_shape
                common = np.ones(shape, dtype=bool)
            elif current_shape != shape:
                raise RuntimeError(
                    "Final mosaic registration produced incompatible image dimensions."
                )
            assert common is not None
            for start in range(0, shape[0], 256):
                block = np.asarray(data[start:start + 256])
                common[start:start + block.shape[0]] &= (
                    np.isfinite(block) & (block != 0)
                )

    assert common is not None
    return largest_valid_mask_crop(common, edge_inset)


# ---------------------------------------------------------------------------
# Multi-scale master merge: numerical core
#
# Every master is first reprojected by Siril onto one canvas at the finest
# plate scale. Layers are then merged finest -> coarsest per filter:
#   detail      (< detail sigma)         from the finer layer only;
#   mid         (detail .. background)   inverse-variance blend of both;
#   background  (> background sigma)     from the coarser, wider layer.
# A smooth coverage ramp hands over to the coarser layer outside the finer
# field. All operations are linear, so outputs remain valid linear masters.
# ---------------------------------------------------------------------------


def _box_radii_for_gaussian(sigma: float, passes: int = 3) -> list[int]:
    """Box half-widths whose repeated convolution approximates a Gaussian."""
    if sigma <= 0.5 or passes <= 0:
        return []
    ideal = math.sqrt(12.0 * sigma * sigma / passes + 1.0)
    lower = int(math.floor(ideal))
    if lower % 2 == 0:
        lower -= 1
    lower = max(1, lower)
    upper = lower + 2
    lower_count = int(round(
        (12.0 * sigma * sigma - passes * lower * lower - 4 * passes * lower - 3 * passes)
        / (-4.0 * lower - 4.0)
    ))
    lower_count = min(passes, max(0, lower_count))
    widths = [lower] * lower_count + [upper] * (passes - lower_count)
    return [(width - 1) // 2 for width in widths if width > 1]


_STRIP_POOL: Optional[Any] = None


def _for_strips(count: int, step: int, work: Callable[[int, int], None]) -> None:
    """Run ``work(start, stop)`` over independent strips on a thread pool.

    NumPy releases the GIL inside the cumulative sums and arithmetic used by
    the filters, so strips of one image are processed in parallel.
    """
    global _STRIP_POOL
    starts = list(range(0, count, step))
    if len(starts) < 2:
        for start in starts:
            work(start, min(count, start + step))
        return
    if _STRIP_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _STRIP_POOL = ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1)))
    list(_STRIP_POOL.map(lambda start: work(start, min(count, start + step)), starts))


def _box_filter_axis(data: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Zero-padded moving average of width 2r+1, in place, float64 accumulation.

    With E[k] = C[clip(k - r, 0, n)] for the prefix sums C, every window sum
    is E[i + 2r + 1] - E[i], so the filter needs only contiguous slices.
    """
    if radius <= 0:
        return data
    length = data.shape[axis]
    width = float(2 * radius + 1)
    other = data.shape[1 - axis]

    def work(start: int, stop: int) -> None:
        block = data[start:stop, :] if axis == 1 else data[:, start:stop].T
        rows = block.shape[0]
        extended = np.empty((rows, length + 2 * radius + 1), dtype=np.float64)
        extended[:, :radius + 1] = 0.0
        np.cumsum(block, axis=1, dtype=np.float64, out=extended[:, radius + 1:radius + 1 + length])
        extended[:, radius + 1 + length:] = extended[:, radius + length:radius + 1 + length]
        result = extended[:, 2 * radius + 1:2 * radius + 1 + length]
        result -= extended[:, :length]
        result /= width
        if axis == 1:
            data[start:stop, :] = result
        else:
            data[:, start:stop] = result.T

    _for_strips(other, 128, work)
    return data


def _three_tap_axis(data: np.ndarray, weight: float, axis: int) -> np.ndarray:
    """In-place zero-padded [w, 1-2w, w] filter (variance 2w) along one axis."""
    if weight <= 0:
        return data
    centre = 1.0 - 2.0 * weight
    other = data.shape[1 - axis]

    def work(start: int, stop: int) -> None:
        block = data[start:stop, :] if axis == 1 else data[:, start:stop].T
        original = np.array(block, dtype=np.float32, copy=True)
        filtered = original * centre
        filtered[:, 1:] += weight * original[:, :-1]
        filtered[:, :-1] += weight * original[:, 1:]
        if axis == 1:
            data[start:stop, :] = filtered
        else:
            data[:, start:stop] = filtered.T

    _for_strips(other, 256, work)
    return data


def approximate_gaussian_blur(
    data: np.ndarray, sigma: float, passes: int = 3
) -> np.ndarray:
    """Zero-padded Gaussian blur with exactly the requested variance.

    Repeated box filters provide the bulk of the variance cheaply. Their
    integer widths undershoot the target slightly, and the remainder is made
    up with small three-tap passes, so blurs of different sigma subtract
    without residual PSF mismatch.
    """
    result = np.array(data, dtype=np.float32, copy=True)
    if sigma <= 0:
        return result
    target = sigma * sigma
    radii = _box_radii_for_gaussian(sigma, passes)
    def box_variance(values: list[int]) -> float:
        return sum(((2 * radius + 1) ** 2 - 1) / 12.0 for radius in values)
    while radii and box_variance(radii) > target:
        largest = radii.index(max(radii))
        radii[largest] -= 1
        radii = [radius for radius in radii if radius > 0]
    for radius in radii:
        _box_filter_axis(result, radius, 1)
        _box_filter_axis(result, radius, 0)
    remainder = target - box_variance(radii)
    if remainder > 1e-6:
        tap_passes = int(math.ceil(remainder / 0.5))
        weight = remainder / (2.0 * tap_passes)
        for _pass in range(tap_passes):
            _three_tap_axis(result, weight, 1)
            _three_tap_axis(result, weight, 0)
    return result


def _block_sum(data: np.ndarray, factor: int) -> np.ndarray:
    height, width = data.shape
    small_height = -(-height // factor)
    small_width = -(-width // factor)
    padded = np.zeros((small_height * factor, small_width * factor), dtype=np.float32)
    padded[:height, :width] = data
    return padded.reshape(small_height, factor, small_width, factor).sum(
        axis=(1, 3), dtype=np.float64
    ).astype(np.float32)


def _block_all(mask: np.ndarray, factor: int) -> np.ndarray:
    """Conservative downsample: a block is valid only when all its pixels are."""
    if factor <= 1:
        return mask.copy()
    height, width = mask.shape
    small_height = -(-height // factor)
    small_width = -(-width // factor)
    padded = np.zeros((small_height * factor, small_width * factor), dtype=bool)
    padded[:height, :width] = mask
    return padded.reshape(small_height, factor, small_width, factor).all(axis=(1, 3))


def _upsample_bilinear(
    small: np.ndarray, shape: tuple[int, int], factor: int
) -> np.ndarray:
    """Bilinearly expand a block-downsampled image back to full resolution."""
    def coordinates(full: int, reduced: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = (np.arange(full, dtype=np.float64) + 0.5) / factor - 0.5
        positions = np.clip(positions, 0.0, max(0, reduced - 1))
        first = np.floor(positions).astype(np.intp)
        second = np.minimum(first + 1, reduced - 1)
        return first, second, (positions - first).astype(np.float32)

    x0, x1, wx = coordinates(shape[1], small.shape[1])
    rows = small[:, x0] * (1.0 - wx) + small[:, x1] * wx
    y0, y1, wy = coordinates(shape[0], small.shape[0])
    output = np.empty(shape, dtype=np.float32)
    for start in range(0, shape[0], 512):
        stop = min(shape[0], start + 512)
        weight = wy[start:stop, None]
        output[start:stop] = rows[y0[start:stop]] * (1.0 - weight) + rows[y1[start:stop]] * weight
    return output


def _masked_blur_lowres(
    data: np.ndarray, valid: np.ndarray, sigma: float, factor: int = 0
) -> tuple[np.ndarray, int]:
    """Normalized (valid-only) Gaussian blur, computed at reduced resolution."""
    factor = factor or max(1, int(sigma // 4))
    values = np.where(valid, data, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    if factor > 1:
        values = _block_sum(values, factor)
        weights = _block_sum(weights, factor)
    reduced_sigma = sigma / factor
    numerator = approximate_gaussian_blur(values, reduced_sigma)
    denominator = approximate_gaussian_blur(weights, reduced_sigma)
    blurred = np.zeros_like(numerator)
    usable = denominator > 1e-6
    blurred[usable] = numerator[usable] / denominator[usable]
    return blurred, factor


def masked_gaussian_blur(
    data: np.ndarray, valid: np.ndarray, sigma: float
) -> np.ndarray:
    """Gaussian blur of valid pixels only, so footprint edges are not darkened."""
    if sigma <= 12.0:
        values = np.where(valid, data, 0.0).astype(np.float32)
        numerator = approximate_gaussian_blur(values, sigma)
        del values
        denominator = approximate_gaussian_blur(valid.astype(np.float32), sigma)
        usable = denominator > 1e-6
        numerator[usable] /= denominator[usable]
        numerator[~usable] = 0.0
        return numerator
    blurred, factor = _masked_blur_lowres(data, valid, sigma)
    return _upsample_bilinear(blurred, data.shape, factor)


def erode_mask(valid: np.ndarray, radius: int) -> np.ndarray:
    """Square erosion: keep pixels whose (2r+1)^2 neighbourhood is fully valid."""
    if radius <= 0:
        return valid.copy()
    fraction = valid.astype(np.float32)
    _box_filter_axis(fraction, radius, 1)
    _box_filter_axis(fraction, radius, 0)
    return fraction >= 1.0 - 1e-5


def coverage_transition_weight(valid: np.ndarray, width: float) -> np.ndarray:
    """Smooth 0..1 weight: 0 at the coverage edge, 1 once `width` px inside."""
    if width <= 1.0:
        return valid.astype(np.float32)
    factor = max(1, int(width // 32))
    small_valid = _block_all(valid, factor)
    eroded = erode_mask(small_valid, int(round(width / 2.0 / factor)))
    ramp = approximate_gaussian_blur(eroded.astype(np.float32), width / 4.0 / factor)
    ramp = np.clip((ramp - 0.023) / 0.954, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    full = _upsample_bilinear(ramp.astype(np.float32), valid.shape, factor)
    full[~valid] = 0.0
    return full


def robust_noise_sigma(
    data: np.ndarray, valid: np.ndarray, maximum_rows: int = 400
) -> Optional[float]:
    """White-noise sigma from second differences along rows (robust to signal)."""
    height = data.shape[0]
    if height < 3 or data.shape[1] < 3:
        return None
    step = max(1, height // maximum_rows)
    samples: list[np.ndarray] = []
    for row in range(1, height - 1, step):
        values = np.asarray(data[row], dtype=np.float64)
        good = np.asarray(valid[row])
        both = good[1:-1] & good[:-2] & good[2:]
        difference = values[1:-1] - 0.5 * (values[:-2] + values[2:])
        samples.append(difference[both])
    if not samples:
        return None
    combined = np.concatenate(samples)
    if combined.size < 100:
        return None
    deviation = np.median(np.abs(combined - np.median(combined)))
    sigma = float(1.4826 * deviation / math.sqrt(1.5))
    return sigma if math.isfinite(sigma) and sigma > 0 else None


def estimate_star_fwhm(
    data: np.ndarray, valid: np.ndarray, maximum_stars: int = 400
) -> Optional[float]:
    """Median star FWHM (px) from Gaussian-weighted adaptive second moments."""
    image = np.where(valid, np.asarray(data, dtype=np.float32), 0.0)
    background = masked_gaussian_blur(image, valid, 16.0)
    detail = image - background
    detail[~valid] = 0.0
    noise = robust_noise_sigma(image, valid)
    if noise is None:
        return None
    finite = image[valid]
    if finite.size == 0:
        return None
    saturation = float(np.max(finite)) * 0.8
    height, width = detail.shape
    radius = 2
    candidates: list[tuple[float, int, int]] = []
    for start in range(radius, height - radius, 512):
        stop = min(height - radius, start + 512)
        block = detail[start - radius:stop + radius]
        center = block[radius:-radius, radius:-radius]
        peak = center > 20.0 * noise
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = block[
                    radius + dy:block.shape[0] - radius + dy,
                    radius + dx:block.shape[1] - radius + dx,
                ]
                peak &= center >= shifted
        ys, xs = np.nonzero(peak)
        for y, x in zip(ys + start, xs + radius):
            candidates.append((float(detail[y, x]), int(y), int(x)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    stamp_radius = 10
    grid_y, grid_x = np.mgrid[-stamp_radius:stamp_radius + 1, -stamp_radius:stamp_radius + 1]
    measurements: list[float] = []
    for _value, y, x in candidates:
        if len(measurements) >= maximum_stars:
            break
        if (
            y < stamp_radius or x < stamp_radius
            or y >= height - stamp_radius or x >= width - stamp_radius
        ):
            continue
        window = (slice(y - stamp_radius, y + stamp_radius + 1),
                  slice(x - stamp_radius, x + stamp_radius + 1))
        if not valid[window].all() or float(image[window].max()) >= saturation:
            continue
        stamp = np.clip(detail[window], 0.0, None).astype(np.float64)
        ring = np.hypot(grid_y, grid_x) >= stamp_radius - 1
        if stamp[ring].max() > 0.25 * stamp[stamp_radius, stamp_radius]:
            continue  # crowded or extended source
        sigma_w = 2.0
        converged = False
        radius_sq = grid_x * grid_x + grid_y * grid_y
        for _iteration in range(12):
            weight = np.exp(-radius_sq / (2.0 * sigma_w * sigma_w))
            total = float(np.sum(stamp * weight))
            if total <= 0:
                break
            # Per-axis weighted variance; for a Gaussian star of sigma s
            # under a Gaussian window w it equals s^2 w^2 / (s^2 + w^2).
            moment = float(np.sum(stamp * weight * radius_sq)) / total / 2.0
            if moment <= 0 or moment >= sigma_w * sigma_w:
                break
            star_sigma = math.sqrt(moment * sigma_w * sigma_w / (sigma_w * sigma_w - moment))
            if not math.isfinite(star_sigma) or star_sigma > stamp_radius / 2.5:
                break
            if abs(star_sigma - sigma_w) < 0.005:
                sigma_w = star_sigma
                converged = True
                break
            sigma_w = max(0.4, star_sigma)
        if converged:
            measurements.append(2.35482 * sigma_w)
    if len(measurements) < 10:
        return None
    return float(np.median(measurements))


def robust_gain_offset(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    maximum_samples: int = 400_000,
) -> Optional[dict[str, float]]:
    """Fit reference ~= gain * candidate + offset over the masked overlap.

    Both images should already be low-pass filtered to a common resolution.
    A reduced-major-axis regression (the geometric mean of the two
    least-squares slopes) treats noise in both layers symmetrically. It is
    deliberately not sigma-clipped: most overlap pixels are nearly flat
    background, so clipping on the residual spread discards the bright
    nebulosity that actually constrains the gain. Only the brightest 0.1%
    (saturated cores) is excluded. Quality is the relative residual
    sqrt(1 - r^2).
    """
    indices = np.flatnonzero(mask)
    if indices.size < 2_000:
        return None
    if indices.size > maximum_samples:
        indices = indices[:: indices.size // maximum_samples]
    ref = np.asarray(reference).ravel()[indices].astype(np.float64)
    cand = np.asarray(candidate).ravel()[indices].astype(np.float64)
    keep = np.isfinite(ref) & np.isfinite(cand)
    if np.count_nonzero(keep) < 2_000:
        return None
    keep &= (ref < np.percentile(ref[keep], 99.9)) & (cand < np.percentile(cand[keep], 99.9))
    x, y = cand[keep], ref[keep]
    dx, dy = x - x.mean(), y - y.mean()
    sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    if sxx <= 0 or syy <= 0 or sxy <= 0:
        return None
    correlation = sxy / math.sqrt(sxx * syy)
    gain = math.sqrt(syy / sxx)
    offset = float(np.median(y - gain * x))
    return {
        "gain": gain,
        "offset": offset,
        "samples": float(x.size),
        "correlation": correlation,
        "residual_fraction": math.sqrt(max(0.0, 1.0 - correlation * correlation)),
    }


def composite_layer_pair(
    fine: np.ndarray,
    fine_valid: np.ndarray,
    coarse: np.ndarray,
    coarse_valid: np.ndarray,
    *,
    gain: float,
    offset: float,
    fine_weight: float,
    coarse_weight: float,
    detail_sigma: float,
    background_sigma: float,
    transition: float,
    blend_large_scales: bool = True,
    psf_match_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a finer layer into a coarser one on a shared canvas.

    Inside the fine field:
        detail_F + (wF*mid_F + wC*mid_C)/(wF+wC) + background_C
      = F + aC*(G_d C - G_d F') + aF*(G_b C - G_b F')
    with aF, aC the normalized inverse-variance weights. F' is F blurred by
    ``psf_match_sigma`` to the coarse PSF, so stars cancel in the correction
    terms instead of gaining halos; F's own detail is never blurred. Outside,
    the matched coarse layer is used, with smooth ramps at both coverage edges.
    """
    fine_values = np.where(fine_valid, fine, 0.0).astype(np.float32)
    coarse_values = np.where(coarse_valid, coarse * gain + offset, 0.0).astype(np.float32)
    if blend_large_scales:
        total = fine_weight + coarse_weight
        alpha_coarse = coarse_weight / total if total > 0 else 0.5
        alpha_fine = 1.0 - alpha_coarse
        # Both layers are averaged over the same (shared) support, so the
        # correction terms vanish wherever the layers agree, even beside an
        # edge where a gradient would bias a one-sided average.
        shared = fine_valid & coarse_valid
        matched_detail = math.hypot(detail_sigma, psf_match_sigma)
        matched_background = math.hypot(background_sigma, psf_match_sigma)
        inner = masked_gaussian_blur(coarse_values, shared, detail_sigma)
        inner -= masked_gaussian_blur(fine_values, shared, matched_detail)
        inner *= alpha_coarse
        inner += fine_values
        coarse_background, factor = _masked_blur_lowres(
            coarse_values, shared, background_sigma
        )
        fine_background, _factor = _masked_blur_lowres(
            fine_values, shared, matched_background, factor
        )
        background_difference = _upsample_bilinear(
            (coarse_background - fine_background).astype(np.float32),
            fine.shape, factor,
        )
        background_difference *= alpha_fine
        inner += background_difference
        del background_difference
    else:
        inner = fine_values.copy()

    fine_ramp = coverage_transition_weight(fine_valid, transition)
    coarse_ramp = coverage_transition_weight(coarse_valid, transition)
    # Near the coarse edge, fall back to the fine data alone.
    inner -= fine_values
    inner *= coarse_ramp
    inner += fine_values
    # Fine weight: the ramp where the coarse layer exists; 1 where it does not.
    fine_share = 1.0 - (1.0 - fine_ramp) * coarse_ramp
    fine_share[~fine_valid] = 0.0
    del fine_ramp, coarse_ramp
    inner -= coarse_values
    inner *= fine_share
    inner += coarse_values
    output_valid = fine_valid | coarse_valid
    inner[~output_valid] = 0.0
    return inner, output_valid


def solved_pixel_scale_arcsec(header: fits.Header) -> Optional[float]:
    try:
        celestial = WCS(header).celestial
        if not celestial.has_celestial:
            return None
        scale = math.sqrt(abs(float(np.linalg.det(celestial.pixel_scale_matrix)))) * 3600.0
        return scale if math.isfinite(scale) and scale > 0 else None
    except Exception:
        return None


def choose_merge_grid(
    layers: list[int],
    scales: list[float],
    preference: str = "auto",
) -> dict[str, Any]:
    """Pick the Siril registration reference and scale for the merge canvas.

    Siril's astrometric registration removes each master's distortion, so
    every master, including the finest, is resampled once in any case. The
    widest layer's grid at the finest plate scale therefore costs nothing in
    sharpness and keeps that layer's entire field and orientation. The finest
    layer's grid (extended to cover every footprint) is used on request, or
    when the upscale exceeds Siril's limit.
    """
    finest_layer = min(layers)
    widest_layer = max(layers)
    fine_index = layers.index(finest_layer)
    wide_index = layers.index(widest_layer)
    finest_scale = min(
        scale for scale, layer in zip(scales, layers) if layer == finest_layer
    )
    upscale = scales[wide_index] / finest_scale
    if preference == "finest":
        choice, reason = "finest", "selected in Advanced"
    elif upscale > MERGE_MAX_SIRIL_SCALE:
        choice, reason = "finest", (
            f"the {upscale:.2f}x upscale exceeds Siril's {MERGE_MAX_SIRIL_SCALE:g}x limit"
        )
    else:
        choice, reason = "widest", (
            "keeps the entire wide field in its own orientation"
            if preference != "widest" else "selected in Advanced"
        )
    if choice == "widest":
        return {
            "grid": "widest", "reference_index": wide_index, "scale": upscale,
            "reason": reason,
        }
    return {
        "grid": "finest", "reference_index": fine_index, "scale": 1.0,
        "reason": reason,
    }


def pad_reference_to_union(
    reference: Path, others: list[Path], destination: Path, margin: int = 8
) -> tuple[int, int]:
    """Write a Siril-solved image padded to cover every footprint.

    Only CRPIX changes, so Siril still treats the copy as plate solved and
    `-framing=current` then yields one canvas covering every master.
    Returns the (x, y) position of the original image on the padded canvas.
    """
    with fits.open(reference, mode="readonly") as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))
        header = hdul[0].header.copy()
    reference_wcs = WCS(header).celestial
    min_x, min_y = 0.0, 0.0
    max_x, max_y = data.shape[1] - 1.0, data.shape[0] - 1.0
    for path in others:
        other = fits.getheader(path, ext=0)
        width, height = int(other["NAXIS1"]), int(other["NAXIS2"])
        edge = np.linspace(0.0, 1.0, 9)
        xs = np.concatenate([edge * (width - 1), np.full(9, width - 1.0),
                             edge * (width - 1), np.zeros(9)])
        ys = np.concatenate([np.zeros(9), edge * (height - 1),
                             np.full(9, height - 1.0), edge * (height - 1)])
        world = WCS(other).celestial.all_pix2world(xs, ys, 0)
        px, py = reference_wcs.all_world2pix(world[0], world[1], 0)
        min_x, min_y = min(min_x, float(px.min())), min(min_y, float(py.min()))
        max_x, max_y = max(max_x, float(px.max())), max(max_y, float(py.max()))
    left = int(math.ceil(-min_x)) + margin if min_x < 0 else 0
    bottom = int(math.ceil(-min_y)) + margin if min_y < 0 else 0
    width = int(math.ceil(max_x)) + 1 + left + margin
    height = int(math.ceil(max_y)) + 1 + bottom + margin
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas[bottom:bottom + data.shape[0], left:left + data.shape[1]] = data
    header["CRPIX1"] = float(header["CRPIX1"]) + left
    header["CRPIX2"] = float(header["CRPIX2"]) + bottom
    for key in ("CHECKSUM", "DATASUM"):
        header.pop(key, None)
    fits.PrimaryHDU(canvas, header=header).writeto(destination, overwrite=True)
    return left, bottom


def _load_float_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, mode="readonly", memmap=False) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise RuntimeError(f"Master merge supports monochrome masters only: {path}")
    return data, header


def valid_footprint(data: np.ndarray, erosion: int = 3) -> np.ndarray:
    return erode_mask(np.isfinite(data) & (data != 0), erosion)


def _load_on_canvas(
    record: dict[str, Any], canvas_shape: tuple[int, int]
) -> tuple[np.ndarray, fits.Header]:
    """Place a registered image on the shared canvas by its integer offset."""
    data, header = _load_float_image(Path(record["aligned"]))
    x, y = record["canvas_offset"]
    canvas = np.zeros(canvas_shape, dtype=np.float32)
    canvas[y:y + data.shape[0], x:x + data.shape[1]] = data
    return canvas, header


def merge_master_layers(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    blend_large_scales: bool = True,
    transition_pixels: int = 0,
    detail_sigma: float = 0.0,
    log: Callable[[str], None] = lambda _message: None,
) -> tuple[list[Path], dict[str, Any]]:
    """Composite registered masters per filter and write one master per filter.

    Each record holds ``filter``, ``layer`` (1 = finest), ``native`` (the solved
    master at its own scale) and ``aligned`` (Siril's reprojection onto the
    shared canvas). Returns the output paths and a manifest report.
    """
    if not records:
        raise ValueError("No registered masters were supplied.")
    canvas_scale: Optional[float] = None
    for record in records:
        header = fits.getheader(record["aligned"], ext=0)
        record["canvas_scale"] = solved_pixel_scale_arcsec(header)
        native_header = fits.getheader(record["native"], ext=0)
        record["native_scale"] = solved_pixel_scale_arcsec(native_header)
        record["native_shape"] = (
            int(native_header.get("NAXIS2", 0)), int(native_header.get("NAXIS1", 0))
        )
        if record["canvas_scale"] and canvas_scale is None:
            canvas_scale = record["canvas_scale"]
    if canvas_scale is None:
        raise RuntimeError("Registered masters carry no celestial WCS.")

    # Siril may store registered images in individual bounding boxes on one
    # shared pixel grid; place them on a common canvas without resampling.
    shapes = []
    for record in records:
        header = fits.getheader(record["aligned"], ext=0)
        shapes.append((int(header["NAXIS2"]), int(header["NAXIS1"])))
    if len(set(shapes)) == 1:
        # `-framing=current` writes every master on the reference canvas.
        origins, grid_error = [(0, 0)] * len(records), 0.0
    else:
        origins, shapes, grid_error = aligned_mosaic_tile_origins(
            [Path(record["aligned"]) for record in records]
        )
    min_x = min(origin[0] for origin in origins)
    min_y = min(origin[1] for origin in origins)
    max_x = max(origin[0] + shape[1] for origin, shape in zip(origins, shapes))
    max_y = max(origin[1] + shape[0] for origin, shape in zip(origins, shapes))
    canvas_shape = (max_y - min_y, max_x - min_x)
    for record, origin in zip(records, origins):
        record["canvas_offset"] = (origin[0] - min_x, origin[1] - min_y)

    # Per-master native measurements: noise for weights, FWHM for the split.
    for record in records:
        native, _header = _load_float_image(Path(record["native"]))
        native_valid = valid_footprint(native)
        record["noise"] = robust_noise_sigma(native, native_valid)
        record["fwhm_native_px"] = estimate_star_fwhm(native, native_valid)
        del native, native_valid
        log(
            f"Measured {Path(record['source']).name}: noise "
            f"{record['noise'] if record['noise'] is None else format(record['noise'], '.3g')}, "
            f"FWHM {record['fwhm_native_px'] if record['fwhm_native_px'] is None else format(record['fwhm_native_px'], '.2f')} px."
        )

    finest_records = [record for record in records if record["layer"] == 1]
    reference = finest_records[0] if finest_records else records[0]
    fine_height, fine_width = reference["native_shape"]
    fine_native_scale = reference["native_scale"] or canvas_scale
    transition = float(transition_pixels) if transition_pixels > 0 else max(
        32.0,
        MERGE_TRANSITION_FRACTION * min(fine_width, fine_height)
        * fine_native_scale / canvas_scale,
    )

    filters: list[str] = []
    for record in records:
        if record["filter"] not in filters:
            filters.append(record["filter"])

    # One safe crop for every filter with (near-)complete coverage.
    footprints: dict[str, np.ndarray] = {}
    for name in filters:
        union: Optional[np.ndarray] = None
        for record in records:
            if record["filter"] != name:
                continue
            coverage = np.zeros(canvas_shape, dtype=bool)
            x, y = record["canvas_offset"]
            with fits.open(record["aligned"], mode="readonly", memmap=True) as hdul:
                data = np.squeeze(hdul[0].data)
                coverage[y:y + data.shape[0], x:x + data.shape[1]] = (
                    np.isfinite(data) & (data != 0)
                )
            union = coverage if union is None else (union | coverage)
        assert union is not None
        footprints[name] = _block_all(union, 4)
    largest = max(int(mask.sum()) for mask in footprints.values())
    common: Optional[np.ndarray] = None
    for name, mask in footprints.items():
        if mask.sum() >= 0.9 * largest:
            common = mask if common is None else (common & mask)
    assert common is not None
    small_x, small_y, small_width, small_height = largest_valid_mask_crop(common, 1)
    crop = (small_x * 4, small_y * 4, small_width * 4, small_height * 4)
    del footprints, common

    report: dict[str, Any] = {
        "canvas_scale_arcsec": canvas_scale,
        "canvas": {"width": canvas_shape[1], "height": canvas_shape[0]},
        "registration_grid_error_px": round(grid_error, 4),
        "transition_pixels": round(transition, 1),
        "crop": {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3]},
        "filters": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name in filters:
        layered = sorted(
            (record for record in records if record["filter"] == name),
            key=lambda record: record["layer"],
        )
        base_record = layered[0]
        composite, header = _load_on_canvas(base_record, canvas_shape)
        composite_valid = valid_footprint(composite)
        base_ratio = (base_record["native_scale"] or canvas_scale) / canvas_scale
        composite_fwhm = (
            base_record["fwhm_native_px"] * base_ratio
            if base_record["fwhm_native_px"] else None
        )
        composite_weight = 1.0
        if base_record["noise"] and base_record["native_scale"]:
            composite_weight = 1.0 / (base_record["noise"] * base_record["native_scale"]) ** 2
        filter_report: dict[str, Any] = {
            "layers": [
                {
                    "layer": record["layer"],
                    "source": str(record["source"]),
                    "native_scale_arcsec": record["native_scale"],
                    "fwhm_native_px": record["fwhm_native_px"],
                    "noise": record["noise"],
                }
                for record in layered
            ],
            "merges": [],
        }
        for record in layered[1:]:
            coarse, _coarse_header = _load_on_canvas(record, canvas_shape)
            coarse_valid = valid_footprint(coarse)
            ratio = (record["native_scale"] or canvas_scale) / canvas_scale
            fwhm_canvas = (
                record["fwhm_native_px"] * ratio
                if record["fwhm_native_px"] else 3.0 * ratio
            )
            split = detail_sigma if detail_sigma > 0 else max(1.0, fwhm_canvas)
            # Blur the finer layer to the coarser PSF inside the correction
            # terms only, so stars cancel there instead of gaining halos.
            psf_match = 0.0
            if composite_fwhm and record["fwhm_native_px"]:
                psf_match = math.sqrt(max(
                    0.0, (fwhm_canvas / 2.35482) ** 2 - (composite_fwhm / 2.35482) ** 2
                ))
            background_sigma = max(4.0 * split, transition / 2.0)
            overlap = composite_valid & coarse_valid
            # The fit compares smooth images only, so it runs at reduced
            # resolution: the low-pass values are unchanged and it is ~16x
            # cheaper on a large canvas.
            fit_sigma = 2.0 * split
            fit_factor = max(1, int(fit_sigma // 2.5))
            fine_smooth, _factor = _masked_blur_lowres(
                composite, composite_valid, fit_sigma, fit_factor
            )
            coarse_smooth, _factor = _masked_blur_lowres(
                coarse, coarse_valid, fit_sigma, fit_factor
            )
            fit = robust_gain_offset(
                fine_smooth,
                coarse_smooth,
                erode_mask(
                    _block_all(overlap, fit_factor),
                    int(math.ceil(2 * fit_sigma / fit_factor)),
                ),
            )
            del fine_smooth, coarse_smooth
            warnings: list[str] = []
            if fit is None or not (0.02 <= fit["gain"] <= 50.0):
                coarse_median = float(np.median(coarse[overlap])) if overlap.any() else 0.0
                fine_median = float(np.median(composite[overlap])) if overlap.any() else 0.0
                gain, offset = 1.0, fine_median - coarse_median
                warnings.append(
                    "Photometric gain fit was unreliable; used an offset-only match."
                )
            else:
                gain, offset = fit["gain"], fit["offset"]
                if fit["residual_fraction"] > 0.25:
                    warnings.append(
                        f"Layers agree poorly after matching (residual "
                        f"{fit['residual_fraction']:.0%}); check that both masters are "
                        "linear and artifact-free."
                    )
            coarse_weight = 1.0
            if record["noise"] and record["native_scale"]:
                coarse_weight = 1.0 / (gain * record["noise"] * record["native_scale"]) ** 2
            composite, composite_valid = composite_layer_pair(
                composite, composite_valid, coarse, coarse_valid,
                gain=gain, offset=offset,
                fine_weight=composite_weight, coarse_weight=coarse_weight,
                detail_sigma=split, background_sigma=background_sigma,
                transition=transition, blend_large_scales=blend_large_scales,
                psf_match_sigma=psf_match,
            )
            merge_record = {
                "coarse_layer": record["layer"],
                "gain": gain,
                "offset": offset,
                "fit": fit,
                "detail_sigma_px": round(split, 3),
                "psf_match_sigma_px": round(psf_match, 3),
                "background_sigma_px": round(background_sigma, 1),
                "fine_weight_share": round(
                    composite_weight / (composite_weight + coarse_weight), 4
                ),
                "warnings": warnings,
            }
            filter_report["merges"].append(merge_record)
            for message in warnings:
                log(f"WARNING: {name}: {message}")
            log(
                f"Merged {name} layer {record['layer']}: gain {gain:.4g}, offset "
                f"{offset:.3g}, detail split {split:.2f} px, fine mid-band weight "
                f"{merge_record['fine_weight_share']:.0%}."
            )
            composite_weight += coarse_weight
            del coarse, coarse_valid, overlap

        x, y, width, height = crop
        cropped = np.ascontiguousarray(composite[y:y + height, x:x + width])
        del composite, composite_valid
        header = header.copy()
        for key in ("CHECKSUM", "DATASUM", "BZERO", "BSCALE"):
            header.pop(key, None)
        offset_x, offset_y = base_record["canvas_offset"]
        if "CRPIX1" in header:
            header["CRPIX1"] = float(header["CRPIX1"]) + offset_x - x
        if "CRPIX2" in header:
            header["CRPIX2"] = float(header["CRPIX2"]) + offset_y - y
        header["FILTER"] = name
        # Header-guided solvers read scale and centre from these keywords;
        # they must describe the merged canvas, not the source telescope.
        pixel_size = safe_float(header.get("XPIXSZ"))
        if pixel_size:
            header["FOCALLEN"] = round(206.264806 * pixel_size / canvas_scale, 3)
        try:
            centre = WCS(header).celestial.all_pix2world(
                [[(width - 1) / 2.0, (height - 1) / 2.0]], 0
            )[0]
            header["RA"] = float(centre[0])
            header["DEC"] = float(centre[1])
            for key in ("OBJCTRA", "OBJCTDEC"):
                header.pop(key, None)
        except Exception:
            pass
        # Mark the product as a master whatever its first source said, so
        # SBPP never mistakes it for a raw sub in a later scan.
        header["IMAGETYP"] = "Master Light"
        header["SBPPMERG"] = (True, "SBPP multi-scale master merge")
        header["MRGLAYRS"] = (len(layered), "Plate-scale layers merged")
        header["HISTORY"] = (
            f"SBPP {VERSION} multi-scale master merge of {len(layered)} layer(s)"
        )
        output = output_dir / f"{final_filter_stem(name)}.fit"
        fits.PrimaryHDU(cropped, header=header).writeto(output, overwrite=True)
        outputs.append(output)
        report["filters"][name] = filter_report
        log(f"Wrote {output.name} ({width}x{height}).")
    return outputs, report


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def default_config_dir() -> Path:
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    base = Path(location) if location else Path.home() / ".config"
    return base / APP_SLUG


class ProfileStore:
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or default_config_dir()
        self.settings_path = self.config_dir / "profiles.json"
        self.cache_path = self.config_dir / "header-index.json"
        self.payload = self._load()

    @staticmethod
    def _default_profile() -> dict[str, Any]:
        return {
            "target_name": "",
            "input_root": "",
            "output_root": "",
            "calibration_roots": [],
            "drizzle_channels": [],
            "options": asdict(PlanOptions()),
        }

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == SCHEMA_VERSION:
                payload.setdefault("profiles", {"Default": self._default_profile()})
                payload.setdefault("selected_profile", "Default")
                # The built-in profile tracks application defaults. Named
                # profiles are user presets and must retain their choices.
                defaults_revision = int(payload.get("defaults_revision", 1))
                if defaults_revision < 2:
                    default_options = payload["profiles"].setdefault(
                        "Default", self._default_profile()
                    ).setdefault("options", {})
                    default_options["create_rejection_maps"] = False
                    default_options["drizzle_scale"] = 1.0
                if defaults_revision < 3:
                    default_options = payload["profiles"].setdefault(
                        "Default", self._default_profile()
                    ).setdefault("options", {})
                    default_options["mosaic_enabled"] = False
                if defaults_revision < 4:
                    default_options = payload["profiles"].setdefault(
                        "Default", self._default_profile()
                    ).setdefault("options", {})
                    default_options["solve_final_images"] = True
                    default_options["mosaic_gradient_correction"] = True
                    default_options["graxpert_smoothing"] = 0.5
                    default_options["graxpert_gpu"] = True
                    default_options["mosaic_background_matching"] = True
                if defaults_revision < 5:
                    default_options = payload["profiles"].setdefault(
                        "Default", self._default_profile()
                    ).setdefault("options", {})
                    default_options["small_stack_threshold"] = MIN_LINEAR_STACK_COUNT
                if defaults_revision < 6:
                    default_options = payload["profiles"].setdefault(
                        "Default", self._default_profile()
                    ).setdefault("options", {})
                    default_options["dark_policy"] = "exact_or_optimized"
                if defaults_revision < 7:
                    # Older releases kept project paths in one global `last`
                    # object. Preserve those values in the profile that was
                    # active when the user last closed SBPP.
                    selected = str(payload.get("selected_profile", "Default"))
                    profile = payload["profiles"].setdefault(
                        selected, self._default_profile()
                    )
                    legacy = payload.get("last", {})
                    for key in ("target_name", "input_root", "output_root"):
                        if not profile.get(key) and legacy.get(key):
                            profile[key] = legacy[key]
                for profile in payload["profiles"].values():
                    profile.setdefault("target_name", "")
                    profile.setdefault("input_root", "")
                    profile.setdefault("output_root", "")
                    profile.setdefault("calibration_roots", [])
                    profile.setdefault("drizzle_channels", [])
                    profile.setdefault("options", asdict(PlanOptions()))
                    profile.pop("output_prefix", None)
                payload.pop("last", None)
                if defaults_revision < 8:
                    # The former default of 25 left hot-pixel residuals in
                    # 25–39 frame stacks. Preserve thresholds users raised.
                    for profile in payload["profiles"].values():
                        options = profile.setdefault("options", {})
                        if options.get("small_stack_threshold") == 25:
                            options["small_stack_threshold"] = MIN_LINEAR_STACK_COUNT
                if defaults_revision < DEFAULTS_REVISION:
                    payload["defaults_revision"] = DEFAULTS_REVISION
                machine = payload.setdefault("machine", {})
                if not machine.get("graxpert_executable"):
                    # GraXpert's location belongs to the machine, not a target.
                    # Adopt one saved in the selected profile, else any profile.
                    profiles = payload["profiles"]
                    selected = profiles.get(str(payload.get("selected_profile")), {})
                    for profile in [selected, *profiles.values()]:
                        location = str(
                            profile.get("options", {}).get("graxpert_executable") or ""
                        ).strip()
                        if location:
                            machine["graxpert_executable"] = location
                            break
                return payload
        except (OSError, ValueError, TypeError):
            pass
        return {
            "schema_version": SCHEMA_VERSION,
            "defaults_revision": DEFAULTS_REVISION,
            "selected_profile": "Default",
            "profiles": {"Default": self._default_profile()},
        }

    def save(self) -> None:
        self.payload["schema_version"] = SCHEMA_VERSION
        self.payload["defaults_revision"] = DEFAULTS_REVISION
        self.payload["updated_utc"] = utc_now()
        atomic_write_json(self.settings_path, self.payload)

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        return self.payload["profiles"]

    def profile(self, name: str) -> dict[str, Any]:
        return self.profiles.setdefault(name, self._default_profile())

    @property
    def graxpert_executable(self) -> str:
        """Machine-wide GraXpert location shared by every target profile."""
        return str(self.payload.setdefault("machine", {}).get("graxpert_executable") or "")

    @graxpert_executable.setter
    def graxpert_executable(self, location: str) -> None:
        self.payload.setdefault("machine", {})["graxpert_executable"] = location.strip()

    def delete_profile(self, name: str) -> None:
        if name != "Default":
            self.profiles.pop(name, None)
            if self.payload.get("selected_profile") == name:
                self.payload["selected_profile"] = "Default"
            self.save()


@dataclass
class SequenceRef:
    base: str
    count: int
    filter_name: str


def sequence_image_ids(directory: Path, base: str) -> list[int]:
    """Read file identities in Siril's one-based sequence-position order."""
    candidates = [directory / f"{base}_.seq", directory / f"{base}.seq"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(f"Cannot identify sequence metadata for {base} in {directory}")
    count = None
    identities = []
    for line in existing[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("S "):
            count = int(shlex.split(line)[3])
        elif line.startswith("I "):
            identities.append(int(line.split()[1]))
    if not identities or len(identities) != count or len(set(identities)) != count:
        raise RuntimeError(f"Incomplete or ambiguous image identities in {existing[0]}")
    return identities


def registration_quality(
    directory: Path, base: str
) -> Optional[list[tuple[float, float, int]]]:
    """Per-position (FWHM, roundness, star count) from Siril's sequence file.

    Siril writes one ``R0`` line per image, in sequence-position order, after
    registration. Returns None when the statistics are absent or ambiguous.
    """
    candidates = [directory / f"{base}_.seq", directory / f"{base}.seq"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        return None
    count = None
    stats: list[tuple[float, float, int]] = []
    try:
        for line in existing[0].read_text(encoding="utf-8").splitlines():
            if line.startswith("S "):
                count = int(shlex.split(line)[3])
            elif line.startswith("R0 "):
                fields = line.split()
                stats.append((float(fields[1]), float(fields[3]), int(float(fields[6]))))
    except (OSError, ValueError, IndexError):
        return None
    if count is None or len(stats) != count:
        return None
    return stats


QUALITY_GATE_MINIMUM_FRAMES = 3


def quality_gate_filters(
    stats: Optional[list[tuple[float, float, int]]],
    start: int,
    end: int,
    options: PlanOptions,
) -> tuple[list[str], Optional[str]]:
    """Siril stack filters for positions start..end, relaxed only when needed.

    The absolute FWHM/roundness/star gates suit normal stacks but can leave
    a small channel with fewer frames than Siril will stack (it needs two),
    as with a few subs per panel in a mosaic. When fewer than three frames
    (or all frames, for smaller channels) would pass, the FWHM and roundness
    gates are dropped for this channel, then the star-count gate if needed.
    """
    full = [
        "-filter-included",
        f"-filter-fwhm={options.fwhm_max}",
        f"-filter-round={options.roundness_min}",
        f"-filter-nbstars={options.stars_min}",
    ]
    if stats is None or start < 1 or end > len(stats) or end < start:
        return full, None
    frames = stats[start - 1:end]
    minimum = min(len(frames), QUALITY_GATE_MINIMUM_FRAMES)
    passing = sum(
        1 for fwhm, roundness, stars in frames
        if fwhm <= options.fwhm_max
        and roundness >= options.roundness_min
        and stars >= options.stars_min
    )
    if passing >= minimum:
        return full, None
    starred = sum(1 for _fwhm, _roundness, stars in frames if stars >= options.stars_min)
    summary = (
        f"only {passing} of {len(frames)} frames pass FWHM <= {options.fwhm_max:g} px, "
        f"roundness >= {options.roundness_min:g} and >= {options.stars_min} stars"
    )
    if starred >= minimum:
        return (
            ["-filter-included", f"-filter-nbstars={options.stars_min}"],
            f"{summary}; stacked the {starred} frame(s) with >= {options.stars_min} "
            "stars instead (FWHM and roundness gates relaxed for this channel).",
        )
    return (
        ["-filter-included"],
        f"{summary}; stacked all {len(frames)} frames without quality gates.",
    )


def registered_channel_ranges(
    source_ids: list[int], registered_ids: list[int], channels: list[SequenceRef]
) -> dict[str, tuple[int, int]]:
    """Translate source channel membership into exported sequence positions."""
    if len(source_ids) != sum(channel.count for channel in channels):
        raise RuntimeError("Source sequence count does not match the channel plan.")
    if len(set(source_ids)) != len(source_ids) or len(set(registered_ids)) != len(registered_ids):
        raise RuntimeError("Duplicate registration image identities; cannot safely select filters.")
    membership = {}
    offset = 0
    for channel in channels:
        for identity in source_ids[offset:offset + channel.count]:
            membership[identity] = channel.filter_name
        offset += channel.count
    positions: dict[str, list[int]] = {}
    for position, identity in enumerate(registered_ids, start=1):
        if identity not in membership:
            raise RuntimeError("Registered sequence contains an unknown image; cannot safely select filters.")
        positions.setdefault(membership[identity], []).append(position)
    ranges = {}
    for channel in channels:
        selected = positions.get(channel.filter_name, [])
        if not selected:
            raise RuntimeError(f"No {channel.filter_name} frames survived registration.")
        if selected != list(range(selected[0], selected[-1] + 1)):
            raise RuntimeError(f"Registered {channel.filter_name} frames are interleaved; refusing a mixed-filter stack.")
        ranges[channel.filter_name] = (selected[0], selected[-1])
    return ranges


class PipelineCancelled(RuntimeError):
    pass


def siril_path_text(value: Any) -> str:
    """Use one path spelling that Siril accepts on every supported OS."""
    return str(value).replace("\\", "/")


def quoted(value: Any) -> str:
    text = siril_path_text(value).replace('"', '\\"')
    return f'"{text}"'


def option_path(name: str, path: Path) -> str:
    return quoted(f"-{name}={path}")


PLAIN_FITS_SUFFIXES = (".fit", ".fits", ".fts")
_SYMLINK_SUPPORT: dict[str, bool] = {}


def symlinks_usable(directory: Path) -> bool:
    """Whether symlinks created in `directory` can actually be followed.

    Some Windows 11 builds allow creating symlinks but then refuse to traverse
    them ("untrusted mount point", WinError 448). Siril's convert still reports
    success on such links, so the failure only appears later as "Sequence not
    found". Probe once per drive with a real read through a link.
    """
    key = Path(directory).anchor or str(directory)
    if key not in _SYMLINK_SUPPORT:
        usable = False
        try:
            with tempfile.TemporaryDirectory(
                prefix=".sbpp-link-probe-", dir=directory, ignore_cleanup_errors=True
            ) as temporary:
                target = Path(temporary) / "probe.txt"
                link = Path(temporary) / "probe-link.txt"
                target.write_bytes(b"ok")
                try:
                    link.symlink_to(target)
                    usable = link.read_bytes() == b"ok"
                except OSError:
                    usable = False
                try:
                    link.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            usable = False
        _SYMLINK_SUPPORT[key] = usable
    return _SYMLINK_SUPPORT[key]


def link_or_copy(source: Path, target: Path, allow_symlink: bool = True) -> str:
    """Place `source` at `target` as cheaply as possible; return the method used."""
    if allow_symlink:
        try:
            target.symlink_to(source)
            return "symlink"
        except OSError:
            pass
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        pass
    shutil.copy2(source, target)
    return "copy"


def resolve_graxpert_executable(configured: str = "") -> Optional[str]:
    """Resolve the standalone GraXpert CLI on Windows, macOS, or Linux."""
    requested = os.path.expandvars(os.path.expanduser(configured.strip()))
    if requested:
        candidate = Path(requested)
        if candidate.suffix.lower() == ".app":
            candidate = candidate / "Contents" / "MacOS" / "GraXpert"
        if candidate.is_file():
            return str(candidate.resolve())
        located = shutil.which(requested)
        if located:
            return located
        # A host path is not always visible inside Siril's Flatpak sandbox.
        if (
            candidate.is_absolute()
            and Path("/.flatpak-info").is_file()
            and shutil.which("flatpak-spawn")
        ):
            return requested
        return None

    for name in (
        "GraXpert", "graxpert", "GraXpert-linux", "GraXpert.exe",
        "GraXpert-win64.exe",
    ):
        located = shutil.which(name)
        if located:
            return located

    candidates: list[Path] = []
    if sys.platform == "darwin":
        candidates.append(
            Path("/Applications/GraXpert.app/Contents/MacOS/GraXpert")
        )
    elif os.name == "nt":
        for variable in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
            root = os.environ.get(variable)
            if root:
                candidates.extend((
                    Path(root) / "GraXpert" / "GraXpert.exe",
                    Path(root) / "Programs" / "GraXpert" / "GraXpert.exe",
                ))
    else:
        candidates.extend((
            Path.home() / ".local" / "bin" / "GraXpert",
            Path.home() / ".local" / "bin" / "GraXpert-linux",
            Path.home() / "Downloads" / "GraXpert-linux",
            Path("/opt/GraXpert/GraXpert"),
            Path("/usr/local/bin/GraXpert"),
            Path("/usr/bin/GraXpert"),
        ))
    return next((str(path) for path in candidates if path.is_file()), None)


def graxpert_background_command(
    executable: str,
    source: Path,
    output_base: Path,
    smoothing: float,
    use_gpu: bool,
) -> list[str]:
    """Build the documented standalone GraXpert background-extraction command."""
    command = [
        executable,
        "-cmd", "background-extraction",
        str(source),
        "-cli",
        "-output", str(output_base),
        "-gpu", str(bool(use_gpu)).lower(),
        "-correction", "Subtraction",
        "-smoothing", f"{smoothing:g}",
    ]
    if (
        os.name != "nt"
        and Path("/.flatpak-info").is_file()
        and not siril_path_text(executable).startswith("/app/")
        and shutil.which("flatpak-spawn")
    ):
        command = ["flatpak-spawn", "--host", *command]
    return command


class PipelineEngine:
    """Translate a reviewed ProcessingPlan into Siril commands."""

    def __init__(
        self,
        siril: s.SirilInterface,
        plan: ProcessingPlan,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
        cancel_check: Callable[[], bool],
    ):
        self.siril = siril
        self.plan = plan
        self._ui_log = log_callback
        self.log_callback = self._log
        self._run_log: Optional[TextIO] = None
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.run_dir: Optional[Path] = None
        self.manifest: dict[str, Any] = {}
        self.commands: list[str] = []
        self.assets: dict[tuple[str, str, tuple[str, ...]], Path] = {}
        self.asset_bpm: dict[tuple[Any, ...], Path] = {}
        self.steps_total = self._estimate_steps()
        self.steps_done = 0
        self.started_monotonic: Optional[float] = None
        self.started_utc = ""

    def _log(self, message: str) -> None:
        """Show a line in the UI and append it to the run's sbpp.log.

        The file is written from the processing thread and flushed per line, so
        it keeps everything up to a crash even if the UI never displayed it.
        """
        self._write_run_log(message)
        self._ui_log(message)

    def _write_run_log(self, message: str) -> None:
        if self._run_log is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._run_log.write(f"[{stamp}] {message}\n")
            self._run_log.flush()
        except OSError:
            self._run_log = None

    def _open_run_log(self) -> None:
        assert self.run_dir is not None
        try:
            self._run_log = open(self.run_dir / "sbpp.log", "a", encoding="utf-8")
        except OSError as exc:
            self._ui_log(f"Could not create sbpp.log in the run folder: {exc}")
            return
        self._write_run_log(
            f"{APP_NAME} {VERSION} on {platform.platform()} / Python {platform.python_version()}"
        )

    def _close_run_log(self) -> None:
        if self._run_log is not None:
            try:
                self._run_log.close()
            except OSError:
                pass
            self._run_log = None

    def _estimate_steps(self) -> int:
        merge = getattr(self.plan, "master_merge", None)
        if isinstance(merge, MasterMergePlan):
            outputs = len(merge.filters)
            return max(1, len(merge.masters) + 3 + outputs * (
                2 + int(self.plan.options.solve_final_images)
            ))
        subgroup_count = sum(
            len(groups) for dataset in self.plan.datasets for groups in dataset.channels.values()
        )
        asset_keys = {
            choice.key
            for dataset in self.plan.datasets
            for groups in dataset.channels.values()
            for subgroup in groups
            for choice in (
                subgroup.bias,
                subgroup.dark,
                subgroup.flat,
                subgroup.darkflat,
                subgroup.bpm,
            )
            if choice.available
        }
        output_modes = sum(
            1 + int(
                any(self.plan.channel_uses_drizzle(dataset, name) for name in dataset.channels)
                and any(not self.plan.channel_uses_drizzle(dataset, name) for name in dataset.channels)
            )
            for dataset in self.plan.datasets
        )
        mosaic_steps = 0
        final_solve_steps = 0
        final_orientation_steps = 0
        for dataset in self.plan.datasets:
            if dataset.panel_count <= 1 or not self.plan.options.mosaic_enabled:
                output_count = sum(
                    len({group.panel_id for group in groups})
                    for groups in dataset.channels.values()
                )
                final_orientation_steps += output_count
                if self.plan.options.solve_final_images:
                    final_solve_steps += output_count
                continue
            final_orientation_steps += len(dataset.channels)
            has_drizzle = any(
                self.plan.channel_uses_drizzle(dataset, name)
                for name in dataset.channels
            )
            has_native = any(
                not self.plan.channel_uses_drizzle(dataset, name)
                for name in dataset.channels
            )
            mosaic_steps += 4 * (int(has_native) + int(has_drizzle))
            mosaic_steps += len(dataset.channels)
            if self.plan.options.mosaic_gradient_correction:
                mosaic_steps += sum(
                    len({group.panel_id for group in groups})
                    for groups in dataset.channels.values()
                )
            if self.plan.options.mosaic_background_matching:
                mosaic_steps += int(has_native) + int(has_drizzle)
        return max(
            1,
            subgroup_count * 2 + len(asset_keys) * 2
            + len(self.plan.datasets) * 3 + output_modes * 2
            + mosaic_steps + final_solve_steps + final_orientation_steps,
        )

    def _check_cancel(self) -> None:
        if self.cancel_check():
            raise PipelineCancelled("Stopped by user after the current Siril command.")

    def _tick(self, message: str) -> None:
        self.steps_done += 1
        self.progress_callback(min(100, int(self.steps_done * 100 / self.steps_total)))
        self.log_callback(message)
        self._write_manifest()
        self._check_cancel()

    def _cmd(self, *args: str) -> None:
        self._check_cancel()
        rendered = " ".join(args)
        self.commands.append(rendered)
        self.log_callback(f"> {rendered}")
        self.siril.log(f"SBPP: {rendered}")
        self.siril.cmd(*args)
        if self.run_dir:
            (self.run_dir / "commands.ssf").write_text(
                "# Generated by Starloch Batch Preprocessor\n" + "\n".join(self.commands) + "\n",
                encoding="utf-8",
            )

    def _write_manifest(self) -> None:
        if self.run_dir:
            self.manifest["updated_utc"] = utc_now()
            self.manifest["commands"] = self.commands
            if self.started_monotonic is not None:
                elapsed = time.monotonic() - self.started_monotonic
                self.manifest.setdefault("timing", {})["elapsed_seconds"] = round(elapsed, 3)
                self.manifest["timing"]["elapsed"] = human_duration(elapsed)
            atomic_write_json(self.run_dir / "manifest.json", self.manifest)

    def _remove_generated_paths(self, paths: Iterable[Path], reason: str) -> None:
        """Remove verified, reproducible data strictly inside this run directory."""
        assert self.run_dir is not None
        run_root = self.run_dir.absolute()
        candidates: list[tuple[Path, Path]] = []
        for target in paths:
            absolute = target.absolute()
            try:
                relative = absolute.relative_to(run_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing cleanup outside the active run: {target}"
                ) from exc
            if not relative.parts:
                raise RuntimeError("Refusing to remove the active run directory itself.")
            if target.exists() or target.is_symlink():
                candidates.append((target, relative))
        if not candidates:
            return

        # Siril maintains a working directory independently of Python. Windows
        # will not remove that directory until Siril has moved somewhere safe.
        try:
            self._cmd("cd", quoted(run_root))
        except PipelineCancelled:
            raise
        except Exception as exc:
            deferred = [str(relative) for _target, relative in candidates]
            detail = self._cleanup_path_summary(deferred)
            self.manifest.setdefault("cleanup_deferred", []).append(
                f"{reason}: {detail} ({type(exc).__name__}: {exc})"
            )
            self.log_callback(
                f"Cleanup deferred — {detail}. Processing will continue."
            )
            self._write_manifest()
            return

        removed: list[str] = []
        deferred: list[tuple[str, OSError]] = []
        for target, relative in candidates:
            error = self._delete_generated_path(target)
            if error is None:
                removed.append(str(relative))
            else:
                deferred.append((str(relative), error))
        if removed:
            detail = f"{reason}: {self._cleanup_path_summary(removed)}"
            self.manifest.setdefault("cleanup", []).append(detail)
            self.log_callback(f"Rolling cleanup — {detail}")
        if deferred:
            names = [name for name, _error in deferred]
            errors = sorted({
                f"{type(error).__name__}: {error}" for _name, error in deferred
            })
            detail = f"{reason}: {self._cleanup_path_summary(names)} ({'; '.join(errors)})"
            self.manifest.setdefault("cleanup_deferred", []).append(detail)
            self.log_callback(
                f"Cleanup deferred — {self._cleanup_path_summary(names)}. "
                "Processing will continue."
            )
        if removed or deferred:
            self._write_manifest()

    @staticmethod
    def _cleanup_path_summary(paths: list[str]) -> str:
        if len(paths) <= 4:
            return ", ".join(paths)
        return f"{len(paths)} generated paths"

    @staticmethod
    def _delete_generated_path(target: Path) -> Optional[OSError]:
        """Delete one generated path, tolerating short-lived Windows locks."""
        last_error: Optional[OSError] = None
        for attempt in range(len(CLEANUP_RETRY_DELAYS) + 1):
            try:
                if target.is_symlink() or target.is_file():
                    target.unlink(missing_ok=True)
                elif target.is_dir():
                    shutil.rmtree(target)
                return None
            except OSError as exc:
                if not target.exists() and not target.is_symlink():
                    return None
                last_error = exc
                if attempt < len(CLEANUP_RETRY_DELAYS):
                    time.sleep(CLEANUP_RETRY_DELAYS[attempt])
        return last_error

    @staticmethod
    def _verify_outputs(paths: Iterable[Path], stage: str) -> None:
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"{stage} completed without expected output(s): "
                + ", ".join(str(path) for path in missing)
            )

    def _convert_sequence(
        self,
        source: Path,
        output: Path,
        base: str,
        expected: int,
        staging_modes: Optional[dict[str, int]] = None,
        timeout: float = 30.0,
    ) -> int:
        """Convert every image in `source` into a `base` sequence in `output`.

        A conversion that silently produces nothing otherwise surfaces only as
        SELECT's bare "Sequence not found", so wait for the converted frames and
        write a diagnostic report describing both folders when they never appear.
        """
        frames = sorted(path for path in source.iterdir() if is_supported(path))
        if (
            frames
            and not symlinks_usable(output)
            and all(file_suffix(path) in PLAIN_FITS_SUFFIXES for path in frames)
        ):
            # Siril's convert would only link these FITS files, and links are
            # unreadable here. Lay out the sequence directly with hard links or
            # copies in Siril's naming; SELECT then indexes it.
            for index, path in enumerate(frames, start=1):
                link_or_copy(path, output / f"{base}_{index:05d}.fit", allow_symlink=False)
        else:
            self._cmd("cd", quoted(source))
            try:
                self._cmd("convert", base, option_path("out", output))
            except s.CommandError as exc:
                raise self._conversion_failure(
                    source, output, base, expected, staging_modes,
                    f"Siril could not convert the {base} frames: {exc}",
                ) from exc
        deadline = time.monotonic() + timeout
        while not any(output.glob(f"{base}_*.fit*")):
            if time.monotonic() >= deadline:
                raise self._conversion_failure(
                    source, output, base, expected, staging_modes,
                    f"Siril's convert produced no {base} frames in {timeout:.0f} seconds.",
                )
            time.sleep(0.1)
        self._cmd("cd", quoted(output))
        try:
            self._cmd("select", base, "1", str(expected))
        except s.CommandError as exc:
            raise self._conversion_failure(
                source, output, base, expected, staging_modes,
                f"Siril converted frames but could not select the {base} sequence: {exc}",
            ) from exc
        return self._seq_count(output, base)

    def _conversion_failure(
        self,
        source: Path,
        output: Path,
        base: str,
        expected: int,
        staging_modes: Optional[dict[str, int]],
        reason: str,
    ) -> RuntimeError:
        """Write a shareable report for a failed conversion and build the error."""

        def describe(path: Path) -> str:
            try:
                if path.is_symlink():
                    target = os.readlink(path)
                    state = "ok" if path.exists() else "BROKEN, target missing"
                    return f"{path.name} -> {target} (symlink, {state})"
                if path.is_dir():
                    return f"{path.name}/ (folder)"
                return f"{path.name} ({path.stat().st_size} bytes)"
            except OSError as exc:
                return f"{path.name} (unreadable: {exc})"

        def listing(directory: Path, limit: int = 12) -> list[str]:
            if not directory.is_dir():
                return ["  (folder does not exist)"]
            try:
                entries = sorted(directory.iterdir())
            except OSError as exc:
                return [f"  (could not list folder: {exc})"]
            lines = [f"  {len(entries)} entries"]
            lines += [f"  {describe(entry)}" for entry in entries[:limit]]
            if len(entries) > limit:
                lines.append(f"  … and {len(entries) - limit} more")
            return lines

        lines = [
            f"SBPP {VERSION} conversion failure report",
            f"Time: {utc_now()}",
            f"System: {platform.platform()} / Python {platform.python_version()}",
            f"Reason: {reason}",
            f"Sequence: {base} (expected {expected} frames)",
            f"Staging method: {staging_modes or 'n/a'}",
            "",
            f"Source folder: {source}",
            *listing(source),
            "",
            f"Output folder: {output}",
            *listing(output),
        ]
        report = output / f"{base}_conversion.txt"
        if report.is_file():
            try:
                text = report.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                text = []
            lines += ["", "Siril conversion report (first 20 lines):"]
            lines += [f"  {line}" for line in text[:20]]
        lines += [
            "",
            "Please send this file together with the Siril Console log from the",
            "same run (the lines around the 'convert' command).",
        ]
        text = "\n".join(lines) + "\n"
        destination = (self.run_dir or output) / "diagnostics" / f"convert_{base}.txt"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            where = f" Diagnostic report: {destination}"
        except OSError:
            where = ""
        self.log_callback(text.rstrip())
        return RuntimeError(f"{reason}{where}")

    @staticmethod
    def _seq_count(directory: Path, base: str, timeout: float = 30.0) -> int:
        """Wait for Siril's post-conversion sequence index and read its count.

        FITS conversion can finish linking every image just before the `.seq`
        index is committed. sirilpy returns at that boundary on some builds, so
        an immediate filesystem check races the final index write.
        """
        deadline = time.monotonic() + timeout
        candidates: list[Path] = []
        while time.monotonic() < deadline:
            candidates = sorted(directory.glob(f"{base}*.seq"))
            if candidates:
                try:
                    lines = candidates[0].read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                except OSError:
                    lines = []
                for line in lines:
                    if line.startswith("S "):
                        fields = shlex.split(line)
                        if len(fields) >= 5:
                            return int(fields[3])
            time.sleep(0.05)

        linked_count = sum(1 for _ in directory.glob(f"{base}_*.fit*"))
        report = directory / f"{base}_conversion.txt"
        details = (
            f" Found {linked_count} converted FITS files and "
            f"{'a conversion report' if report.exists() else 'no conversion report'}."
        )
        if candidates:
            raise RuntimeError(
                f"Siril created but did not finish a readable sequence index for {base} "
                f"in {directory}.{details}"
            )
        raise RuntimeError(
            f"Siril did not create a sequence index for {base} in {directory} "
            f"after {timeout:.0f} seconds.{details}"
        )

    @staticmethod
    def _stage_files(files: list[str], destination: Path) -> dict[str, int]:
        destination.mkdir(parents=True, exist_ok=False)
        allow_symlink = symlinks_usable(destination)
        modes: Counter[str] = Counter()
        for index, source_text in enumerate(files, start=1):
            source = Path(source_text)
            detected_suffix = file_suffix(source)
            suffix = source.name[-len(detected_suffix):] if detected_suffix else source.suffix
            stem = source.name[:-len(detected_suffix)] if detected_suffix else source.stem
            name = f"{index:06d}_{slugify(stem, 'frame')}{suffix}"
            modes[link_or_copy(source, destination / name, allow_symlink)] += 1
        return dict(modes)

    def _materialize_master(self, choice: CalibrationChoice, label: str) -> Path:
        if choice.key in self.assets:
            return self.assets[choice.key]
        assert self.run_dir is not None
        source = Path(choice.files[0])
        if file_suffix(source) in FITS_SUFFIXES:
            self.assets[choice.key] = source
            return source
        output = self.run_dir / "masters" / f"imported_{slugify(label)}.fit"
        self._cmd("load", quoted(source))
        self._cmd("save", quoted(output))
        self._cmd("close")
        self.assets[choice.key] = output
        self._tick(f"Converted external master {source.name} to FITS.")
        return output

    def _build_master(self, choice: CalibrationChoice, label: str) -> Path:
        if choice.key in self.assets:
            return self.assets[choice.key]
        assert self.run_dir is not None
        job = slugify(label)
        staging = self.run_dir / "inputs" / "calibration" / job
        process = self.run_dir / "process" / "calibration" / job
        process.mkdir(parents=True, exist_ok=True)
        modes = self._stage_files(choice.files, staging)
        base = slugify(choice.kind.lower())
        count = self._convert_sequence(
            staging, process, base, len(choice.files), modes
        )
        output_base = self.run_dir / "masters" / f"master_{job}"
        if choice.kind in {FRAME_BIAS, FRAME_DARK, FRAME_DARKFLAT, "BPM"}:
            if count >= 5:
                self._cmd(
                    "stack", base, "rej", "winsorized", "4", "3", "-nonorm", "-32b",
                    option_path("out", output_base),
                )
            else:
                self._cmd(
                    "stack", base, "rej", "none", "-nonorm", "-32b",
                    option_path("out", output_base),
                )
        else:
            raise RuntimeError(f"Unsupported generic master kind {choice.kind}")
        output = output_base.with_suffix(".fit")
        self._verify_outputs([output], f"Building {choice.kind.lower()} master")
        self.assets[choice.key] = output
        self.manifest.setdefault("staging", {})[job] = modes
        self._tick(f"Built {choice.kind.lower()} master from {count} frames.")
        if self.plan.options.rolling_cleanup:
            self._cmd("close")
            self._remove_generated_paths(
                [staging, process],
                f"Released completed {choice.kind.lower()} master workspace",
            )
        return output

    def _resolve_simple_asset(self, choice: CalibrationChoice, label: str) -> Optional[Path]:
        if not choice.available:
            return None
        if choice.mode in {"master", "bpm_master", "optimized_master"}:
            return self._materialize_master(choice, label)
        if choice.mode in {"build", "bpm_build", "optimized_build"} and choice.kind != FRAME_FLAT:
            return self._build_master(choice, label)
        return None

    def _resolve_bpm(
        self, choice: CalibrationChoice, label: str, frame_count: int
    ) -> Optional[Path]:
        if not choice.available:
            return None
        hot_sigma = light_cosmetic_hot_sigma(frame_count)
        cache_key = (*choice.key, hot_sigma)
        if cache_key in self.asset_bpm:
            return self.asset_bpm[cache_key]
        dark_master = self._resolve_simple_asset(choice, f"{label}_dark_for_bpm")
        if dark_master is None:
            return None
        assert self.run_dir is not None
        bpm_dir = self.run_dir / "masters" / "bpm"
        bpm_dir.mkdir(parents=True, exist_ok=True)
        bpm_base = bpm_dir / f"bad_pixels_{slugify(label)}"
        self._cmd("cd", quoted(bpm_dir))
        self._cmd("load", quoted(dark_master))
        self._cmd("find_hot", quoted(bpm_base), "5", hot_sigma)
        self._cmd("close")
        bpm_path = bpm_base.with_suffix(".lst")
        self.asset_bpm[cache_key] = bpm_path
        self._tick(f"Created BPM from {dark_master.name}.")
        return bpm_path

    def _resolve_flat(
        self,
        choice: CalibrationChoice,
        bias_master: Optional[Path],
        darkflat_master: Optional[Path],
        label: str,
    ) -> Optional[Path]:
        if not choice.available:
            return None
        if choice.key in self.assets:
            return self.assets[choice.key]
        if choice.mode == "master":
            return self._materialize_master(choice, label)
        assert self.run_dir is not None
        job = slugify(label)
        staging = self.run_dir / "inputs" / "calibration" / job
        process = self.run_dir / "process" / "calibration" / job
        process.mkdir(parents=True, exist_ok=True)
        modes = self._stage_files(choice.files, staging)
        base = "flat"
        count = self._convert_sequence(
            staging, process, base, len(choice.files), modes
        )
        stack_base = base
        if darkflat_master is not None:
            # A conventional flat-dark contains the sensor bias pedestal too.
            self._cmd("calibrate", base, option_path("dark", darkflat_master))
            stack_base = f"pp_{base}"
        elif bias_master is not None:
            self._cmd("calibrate", base, option_path("bias", bias_master))
            stack_base = f"pp_{base}"
        output_base = self.run_dir / "masters" / f"master_{job}"
        if count >= 5:
            self._cmd(
                "stack", stack_base, "rej", "winsorized", "3", "3", "-norm=mul", "-32b",
                option_path("out", output_base),
            )
        else:
            self._cmd(
                "stack", stack_base, "rej", "none", "-norm=mul", "-32b",
                option_path("out", output_base),
            )
        output = output_base.with_suffix(".fit")
        self._verify_outputs([output], f"Building {label}")
        self.assets[choice.key] = output
        self.manifest.setdefault("staging", {})[job] = modes
        self._tick(f"Built {label} from {count} raw flats.")
        if self.plan.options.rolling_cleanup:
            self._cmd("close")
            self._remove_generated_paths(
                [staging, process],
                f"Released completed {label} workspace",
            )
        return output

    def _calibrate_subgroup(self, dataset: DatasetPlan, subgroup: LightSubgroup) -> SequenceRef:
        assert self.run_dir is not None
        label = f"{dataset.dataset_id}_{subgroup.subgroup_id}"
        bias = self._resolve_simple_asset(subgroup.bias, f"{label}_bias")
        dark = self._resolve_simple_asset(subgroup.dark, f"{label}_dark")
        darkflat = self._resolve_simple_asset(subgroup.darkflat, f"{label}_darkflat")
        flat = self._resolve_flat(subgroup.flat, bias, darkflat, f"{label}_flat")
        frame_count = len(subgroup.frames)
        bpm = self._resolve_bpm(subgroup.bpm, f"{label}_bpm", frame_count)
        if flat is None:
            raise RuntimeError(f"No flat is available for {label}")

        staging = (
            self.run_dir / "inputs" / dataset.dataset_id / "panels"
            / subgroup.panel_id / subgroup.subgroup_id
        )
        process = (
            self.run_dir / "process" / dataset.dataset_id / "panels"
            / subgroup.panel_id / subgroup.subgroup_id
        )
        process.mkdir(parents=True, exist_ok=True)
        modes = self._stage_files([item.path for item in subgroup.frames], staging)
        base = f"light_{slugify(subgroup.filter_name)}"
        count = self._convert_sequence(
            staging, process, base, len(subgroup.frames), modes
        )
        args = ["calibrate", base]
        # Conventional exact master darks contain the bias pedestal, so the
        # light uses dark OR bias. Siril's optimized subtraction instead needs
        # both masters so it can estimate the coefficient applied to the dark.
        if bias is not None and (dark is None or subgroup.dark.optimized):
            args.append(option_path("bias", bias))
        if dark is not None:
            args.append(option_path("dark", dark))
        args.append(option_path("flat", flat))
        if subgroup.dark.optimized:
            args.append("-opt")
        if bpm is not None:
            args.extend(["-cc=bpm", quoted(bpm)])
        elif dark is not None:
            args.extend(["-cc=dark", "5", light_cosmetic_hot_sigma(frame_count)])
        representative = subgroup.representative
        if representative.bayer:
            args.append("-cfa")
            # A fully drizzled CFA setup stays mosaiced for true Bayer drizzle.
            # Partial channel drizzle is still supported, but all frames are
            # debayered first so the global sequence has one channel geometry.
            if self.plan.options.debayer_cfa and not dataset_uses_true_cfa_drizzle(
                dataset, self.plan.drizzle_channels, self.plan.options
            ):
                args.extend(["-equalize_cfa", "-debayer"])
        self._cmd(*args)
        self.manifest.setdefault("staging", {})[label] = modes
        self._tick(f"Calibrated {count} {subgroup.filter_name} lights ({label}).")
        relative_base = siril_path_text(
            process.relative_to(self.run_dir / "process") / f"pp_{base}"
        )
        return SequenceRef(relative_base, count, subgroup.filter_name)

    def _merge_sequences(
        self,
        cwd: Path,
        sequences: list[SequenceRef],
        output_base: str,
        filter_name: str,
    ) -> SequenceRef:
        if len(sequences) == 1:
            return sequences[0]
        self._cmd("cd", quoted(cwd))
        self._cmd("merge", *(sequence.base for sequence in sequences), output_base)
        count = sum(sequence.count for sequence in sequences)
        self._tick(f"Merged {len(sequences)} sequences into {output_base} ({count} frames).")
        return SequenceRef(output_base, count, filter_name)

    def _stack_channel(
        self,
        dataset: DatasetPlan,
        panel_id: str,
        process_root: Path,
        registered: SequenceRef,
        start: int,
        end: int,
        total: int,
        channel_count: int,
        filter_name: str,
        drizzle: bool,
    ) -> Path:
        assert self.run_dir is not None
        self._cmd("cd", quoted(process_root))
        if total != channel_count:
            self._cmd("unselect", registered.base, "1", str(total))
            self._cmd("select", registered.base, str(start), str(end))
        output_name = final_filter_stem(filter_name)
        if dataset.panel_count > 1:
            output_directory = (
                self.run_dir / "masters" / "panels" / dataset.dataset_id / panel_id
            )
            output_directory.mkdir(parents=True, exist_ok=True)
        else:
            output_directory = self.run_dir / "masters"
        if len(self.plan.datasets) > 1 and dataset.panel_count == 1:
            output_name += f"_{slugify(dataset.dataset_id).lower()}"
        output = output_directory / output_name
        filters, gate_warning = quality_gate_filters(
            registration_quality(process_root, registered.base),
            start, end, self.plan.options,
        )
        if gate_warning:
            message = f"{dataset.dataset_id}/{panel_id}/{filter_name}: {gate_warning}"
            self.manifest.setdefault("quality_gate_relaxations", []).append(message)
            self.log_callback(f"WARNING: {message}")
        stack_method = light_rejection_arguments(channel_count, self.plan.options)
        common = ["-norm=addscale"]
        if stack_method != ["median"]:
            common.append("-weight=noise")
        common.extend([*filters, "-32b", option_path("out", output)])
        rejection_description = (
            "median combination"
            if stack_method == ["median"]
            else f"{stack_method[1]} {stack_method[2]}/{stack_method[3]} rejection"
        )
        if (
            self.plan.options.create_rejection_maps
            and stack_method != ["median"]
        ):
            stack_method.append("-rejmaps")
        self._cmd("stack", registered.base, *stack_method, *common)
        output_fit = output.with_suffix(".fit")
        self._verify_outputs([output_fit], f"Stacking {panel_id}/{filter_name}")
        mode = f"{self.plan.options.drizzle_scale:g}x drizzle" if drizzle else "native-scale"
        self._tick(
            f"Stacked {panel_id}/{filter_name}: source range {start}-{end}, "
            f"{rejection_description}, {mode} output."
        )
        return output_fit

    def _apply_registration_mode(
        self,
        dataset: DatasetPlan,
        panel_id: str,
        process_root: Path,
        global_ref: SequenceRef,
        drizzle: bool,
        prefix: str,
        channel_counts: Iterable[int],
    ) -> SequenceRef:
        assert self.run_dir is not None
        self._cmd("cd", quoted(process_root))
        args = [
            "seqapplyreg", global_ref.base, f"-prefix={prefix}", "-framing=min",
        ]
        if drizzle:
            args.extend([
                f"-scale={self.plan.options.drizzle_scale:g}",
                "-drizzle",
                f"-pixfrac={self.plan.options.drizzle_pixfrac:g}",
                f"-kernel={self.plan.options.drizzle_kernel}",
            ])
        else:
            # Bicubic avoids the spatially varying ringing/correlation that
            # Lanczos-4 can imprint when rotated frames land at changing
            # sub-pixel phases. Controlled tests also measured a slightly
            # smaller stellar FWHM with bicubic on the same registered data.
            args.append("-interp=cubic")
        self._cmd(*args)
        mode = (
            f"{self.plan.options.drizzle_scale:g}x drizzle "
            f"(pixfrac {self.plan.options.drizzle_pixfrac:g}, "
            f"{self.plan.options.drizzle_kernel})"
            if drizzle else "Bicubic native"
        )
        self._tick(
            f"Applied {mode} registration and common crop for "
            f"{dataset.dataset_id}/{panel_id}."
        )
        global_path = Path(global_ref.base)
        registered_base = siril_path_text(
            global_path.parent / f"{prefix}{global_path.name}"
        )
        return SequenceRef(
            registered_base, len(sequence_image_ids(process_root, registered_base)), "ALL"
        )

    def _discard_sequence(self, process_root: Path, sequence: SequenceRef) -> None:
        """Remove only a generated registered mode after its stacks are complete."""
        assert self.run_dir is not None
        base_path = process_root / sequence.base
        candidates = [
            candidate for candidate in base_path.parent.glob(f"{base_path.name}*")
            if candidate.is_file() or candidate.is_symlink()
        ]
        for auxiliary in (process_root / "cache", process_root / "drizztmp"):
            if auxiliary.is_dir():
                candidates.extend(
                    candidate for candidate in auxiliary.glob(f"{base_path.name}*")
                    if candidate.is_file() or candidate.is_symlink()
                )
        self._remove_generated_paths(
            candidates,
            f"Discarded {sequence.base} after stacking",
        )

    def _process_panel(self, dataset: DatasetPlan, panel_id: str) -> dict[str, Path]:
        assert self.run_dir is not None
        process_root = (
            self.run_dir / "process" / dataset.dataset_id / "panels" / panel_id
        )
        process_root.mkdir(parents=True, exist_ok=True)
        channel_refs: list[SequenceRef] = []
        for filter_name, subgroups in dataset.channels.items():
            panel_subgroups = [
                subgroup for subgroup in subgroups if subgroup.panel_id == panel_id
            ]
            if not panel_subgroups:
                continue
            refs = [
                self._calibrate_subgroup(dataset, subgroup)
                for subgroup in panel_subgroups
            ]
            # Calibration references are relative to the run process root;
            # merges execute from this panel's own process directory.
            normalized_refs = []
            for ref in refs:
                absolute = self.run_dir / "process" / ref.base
                relative = siril_path_text(os.path.relpath(absolute, process_root))
                normalized_refs.append(SequenceRef(relative, ref.count, ref.filter_name))
            channel_refs.append(self._merge_sequences(
                process_root,
                normalized_refs,
                f"channel_{slugify(filter_name)}",
                filter_name,
            ))

        # Siril's MERGE command accepts input sequences in child directories,
        # but ordinary sequence commands search the current working directory.
        # With exactly one subgroup there is nothing to merge, so run directly
        # from that sequence's directory instead of handing REGISTER a nested
        # path that it cannot resolve.
        registration_root = process_root
        if len(channel_refs) == 1:
            single_path = Path(channel_refs[0].base)
            if single_path.parent != Path("."):
                registration_root = process_root / single_path.parent
                channel_refs = [SequenceRef(
                    single_path.name,
                    channel_refs[0].count,
                    channel_refs[0].filter_name,
                )]

        global_ref = self._merge_sequences(
            registration_root, channel_refs, "all_calibrated", "ALL"
        )
        self._cmd("cd", quoted(registration_root))
        self._cmd(
            "setfindstar", "reset", "-sigma=1.0", "-roundness=0.5",
            "-convergence=2", "-gaussian",
        )
        self._cmd(
            "register", global_ref.base, "-2pass", "-transf=homography",
            f"-minpairs={self.plan.options.min_pairs_registration}",
            f"-maxstars={self.plan.options.max_stars_registration}",
        )
        self._tick(
            f"Computed two-pass homography registration for {dataset.dataset_id}/{panel_id}."
        )
        source_ids = sequence_image_ids(registration_root, global_ref.base)
        outputs: dict[str, Path] = {}
        channel_ranges: list[tuple[SequenceRef, int, int]] = []
        start = 1
        for channel in channel_refs:
            end = start + channel.count - 1
            channel_ranges.append((channel, start, end))
            start = end + 1

        native_ranges = [
            item for item in channel_ranges
            if not self.plan.channel_uses_drizzle(dataset, item[0].filter_name)
        ]
        drizzle_ranges = [
            item for item in channel_ranges
            if self.plan.channel_uses_drizzle(dataset, item[0].filter_name)
        ]
        modes: list[tuple[bool, list[tuple[SequenceRef, int, int]], str]] = []
        if native_ranges:
            modes.append((False, native_ranges, "native_" if drizzle_ranges else "r_"))
        if drizzle_ranges:
            modes.append((True, drizzle_ranges, "drizzle_"))

        for mode_index, (drizzle, ranges, prefix) in enumerate(modes):
            registered = self._apply_registration_mode(
                dataset,
                panel_id,
                registration_root,
                global_ref,
                drizzle,
                prefix,
                (channel.count for channel, _start, _end in ranges),
            )
            surviving_ranges = registered_channel_ranges(
                source_ids,
                sequence_image_ids(registration_root, registered.base),
                channel_refs,
            )
            for channel, _range_start, _range_end in ranges:
                range_start, range_end = surviving_ranges[channel.filter_name]
                surviving_count = range_end - range_start + 1
                if surviving_count != channel.count:
                    self.log_callback(
                        f"{panel_id}/{channel.filter_name}: {surviving_count} of "
                        f"{channel.count} frames survived registration."
                    )
                outputs[channel.filter_name] = self._stack_channel(
                    dataset,
                    panel_id,
                    registration_root,
                    registered,
                    range_start,
                    range_end,
                    registered.count,
                    surviving_count,
                    channel.filter_name,
                    drizzle,
                )
            discard_mode = self.plan.options.rolling_cleanup or (
                not self.plan.options.keep_intermediates
                and mode_index < len(modes) - 1
            )
            if discard_mode:
                self._cmd("close")
                self._discard_sequence(registration_root, registered)
        return outputs

    def _mosaic_output_name(
        self, dataset: DatasetPlan, filter_name: str, drizzle: bool
    ) -> str:
        del drizzle  # one final product exists per filter, regardless of its scale
        name = final_filter_stem(filter_name)
        if len(self.plan.datasets) > 1:
            name += f"_{slugify(dataset.dataset_id).lower()}"
        return name

    def _finalize_mosaic_geometry(
        self,
        dataset: DatasetPlan,
        outputs: list[Path],
        mode_name: str,
        process_root: Path,
    ) -> list[Path]:
        """Verify or align filter mosaics and apply one shared safe crop."""
        if not outputs:
            return outputs
        process = process_root / "final_geometry"
        process.mkdir(parents=True, exist_ok=True)
        base = f"mosaic_products_{mode_name}"

        # Per-filter mosaic stacks can differ by one or more border pixels.
        # Siril cannot construct a sequence until their dimensions agree, so
        # first place them on the largest size already shared by every output.
        # All filter mosaics originate in the same astrometric tile sequence;
        # preserving pixel origin (crop x=0, y=0) preserves that shared grid.
        shapes: list[tuple[int, int]] = []
        for output in outputs:
            header = fits.getheader(output, ext=0)
            width = int(header.get("NAXIS1", 0))
            height = int(header.get("NAXIS2", 0))
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Unable to read mosaic geometry from {output}.")
            shapes.append((width, height))
        wcs_error = shared_wcs_alignment_error(outputs)
        sequence_width = min(width for width, _height in shapes)
        sequence_height = min(height for _width, height in shapes)
        normalized_paths: list[Path] = []
        for index, (output, shape) in enumerate(zip(outputs, shapes), start=1):
            normalized = process / f"{base}_{index:05d}.fit"
            normalized.unlink(missing_ok=True)
            self._cmd("load", quoted(output))
            if shape != (sequence_width, sequence_height):
                self._cmd(
                    "crop", "0", "0", str(sequence_width), str(sequence_height)
                )
            self._cmd("save", quoted(normalized))
            self._cmd("close")
            normalized_paths.append(normalized)
        self._verify_outputs(
            normalized_paths,
            f"Normalizing {dataset.dataset_id} {mode_name} mosaic geometry",
        )

        self._cmd("cd", quoted(process))
        self._cmd("select", base, "1", str(len(outputs)))
        count = self._seq_count(process, base)
        if count != len(outputs):
            raise RuntimeError(
                f"Final mosaic alignment expected {len(outputs)} images but found {count}."
            )

        source_base = base
        # These products are selected stacks of the same aligned mosaic-tile
        # sequence. Their shared pixel grid is therefore stronger evidence
        # than the independently propagated WCS headers on the stacks. Those
        # headers can disagree by many pixels even though the image arrays are
        # already registered; registering the products again needlessly
        # resamples every filter and can create spatially varying softness.
        shared_grid_verified = True
        resampled_for_alignment = False

        aligned_paths = [
            process / f"{source_base}_{index:05d}.fit"
            for index in range(1, count + 1)
        ]
        self._verify_outputs(
            aligned_paths, f"Aligning {dataset.dataset_id} {mode_name} filter mosaics"
        )
        mode_manifest = self.manifest.setdefault("mosaics", {}).setdefault(
            dataset.dataset_id, {"panels": dataset.panels, "modes": {}}
        ).setdefault("modes", {}).setdefault(mode_name, {})
        try:
            crop = largest_common_valid_crop(
                aligned_paths,
                edge_inset=max(2, self.plan.options.mosaic_feather),
            )
            crop_source = "registered_mosaic_output_footprint"
        except MosaicCropTooLossy as exc:
            # Irregular panel layouts (staggered rows, missing corners, an
            # L-shape) have no rectangle that keeps most of the data. Keep
            # the full shared canvas instead of failing a finished run.
            crop = (0, 0, sequence_width, sequence_height)
            crop_source = "uncropped_irregular_footprint"
            self.log_callback(
                f"Warning: {dataset.dataset_id} {mode_name} mosaic left uncropped. "
                f"Every filter covers {exc.valid_fraction:.0%} of the canvas, but the "
                f"largest clean rectangle would keep only {exc.kept_fraction:.0%}. "
                "The edges outside the panel footprint stay black; crop them by hand."
            )
            mode_manifest["crop_warning"] = {
                "valid_fraction": round(exc.valid_fraction, 4),
                "largest_rectangle_fraction": round(exc.kept_fraction, 4),
            }
        x, y, width, height = crop
        temporary_outputs: list[tuple[Path, Path]] = []
        try:
            for source, output in zip(aligned_paths, outputs):
                temporary = output.with_name(
                    f"{output.stem}_final_geometry_tmp.fit"
                )
                temporary.unlink(missing_ok=True)
                self._cmd("load", quoted(source))
                self._cmd("crop", str(x), str(y), str(width), str(height))
                self._cmd("save", quoted(temporary))
                self._cmd("close")
                temporary_outputs.append((temporary, output))
            self._verify_outputs(
                [temporary for temporary, _output in temporary_outputs],
                f"Cropping {dataset.dataset_id} {mode_name} filter mosaics",
            )
            for temporary, output in temporary_outputs:
                os.replace(temporary, output)
        finally:
            for temporary, _output in temporary_outputs:
                temporary.unlink(missing_ok=True)

        mode_manifest["final_geometry"] = {
            "star_aligned": True,
            "alignment_method": (
                "shared_astrometric_grid"
                if shared_grid_verified
                else "star_registration"
            ),
            "alignment_error_pixels": (
                round(wcs_error, 6) if wcs_error is not None else None
            ),
            "resampled_for_alignment": resampled_for_alignment,
            "crop_source": crop_source,
            "crop": {"x": x, "y": y, "width": width, "height": height},
            "source_shapes": [
                {"path": str(path), "width": shape[0], "height": shape[1]}
                for path, shape in zip(outputs, shapes)
            ],
            "normalized_sequence": {
                "width": sequence_width, "height": sequence_height,
            },
        }
        self._verify_outputs(
            outputs, f"Finalizing {dataset.dataset_id} {mode_name} mosaic geometry"
        )
        alignment_text = (
            "Verified shared astrometric grid and cropped"
            if shared_grid_verified
            else "Star-aligned and cropped"
        )
        if crop_source == "uncropped_irregular_footprint":
            alignment_text = alignment_text.replace("cropped", "kept full canvas for")
        self._tick(
            f"{alignment_text} {count} {mode_name} filter mosaic(s) "
            f"to {width}×{height}."
        )
        return outputs

    def _solve_mosaic_tiles(
        self,
        dataset: DatasetPlan,
        base: str,
        count: int,
        mode_name: str,
    ) -> None:
        """Solve each tile independently, then compute their shared WCS registration."""
        for index in range(1, count + 1):
            self._cmd("unselect", base, "1", str(count))
            self._cmd("select", base, str(index), str(index))
            solve_args = [
                "seqplatesolve", base, "-force", "-noreg", "-order=3",
                "-catalog=gaia",
            ]
            if self.plan.options.mosaic_downsample_solve:
                solve_args.append("-downscale")
            try:
                self._cmd(*solve_args)
            except Exception as first_error:
                solved = False
                last_error = first_error
                if self.plan.options.mosaic_downsample_solve:
                    self.log_callback(
                        f"Mosaic tile {index}/{count} needs a full-resolution solve; retrying."
                    )
                    try:
                        self._cmd(
                            "seqplatesolve", base, "-force", "-noreg",
                            "-order=3", "-catalog=gaia",
                        )
                        solved = True
                    except Exception as retry_error:
                        last_error = retry_error
                if not solved and self.plan.options.mosaic_blind_fallback:
                    self.log_callback(
                        f"Mosaic tile {index}/{count} needs the local blind solver; retrying."
                    )
                    try:
                        self._cmd(
                            "seqplatesolve", base, "-force", "-noreg",
                            "-order=3", "-localasnet", "-blindpos", "-blindres",
                        )
                        solved = True
                    except Exception as fallback_error:
                        last_error = fallback_error
                if not solved:
                    raise RuntimeError(
                        f"Could not plate-solve mosaic tile {index}/{count} "
                        f"for {dataset.dataset_id} ({mode_name})."
                    ) from last_error

        self._cmd("select", base, "1", str(count))
        self._cmd("seqplatesolve", base, "-order=3", "-catalog=gaia")

    def _run_graxpert_background(
        self, source: Path, output_base: Path, log_path: Path
    ) -> Path:
        """Run one standalone GraXpert subtraction pass and locate its output."""
        executable = resolve_graxpert_executable(
            self.plan.options.graxpert_executable
        )
        if not executable:
            raise RuntimeError(
                "GraXpert panel correction is enabled, but its standalone "
                "executable could not be found."
            )
        command = graxpert_background_command(
            executable,
            source,
            output_base,
            self.plan.options.graxpert_smoothing,
            self.plan.options.graxpert_gpu,
        )
        rendered = " ".join(shlex.quote(argument) for argument in command)
        self.log_callback(f"> {rendered}")
        self.manifest.setdefault("external_commands", []).append(rendered)
        self._write_manifest()
        creation_flags = (
            subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            if os.name == "nt" else 0
        )
        with log_path.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(output_base.parent),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                raise RuntimeError(f"Could not start GraXpert: {exc}") from exc
            while process.poll() is None:
                if self.cancel_check():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise PipelineCancelled("Stopped while GraXpert was processing a panel.")
                time.sleep(0.25)
        if process.returncode:
            try:
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-20:]
            except OSError:
                tail = []
            detail = "\n".join(tail)
            raise RuntimeError(
                f"GraXpert failed for {source.name} with exit code "
                f"{process.returncode}." + (f"\n{detail}" if detail else "")
            )

        candidates = [
            output_base.with_suffix(suffix)
            for suffix in (".fit", ".fits", ".fts", ".xisf")
        ]
        candidates.extend(sorted(
            (
                path for path in output_base.parent.glob(f"{output_base.name}*")
                if path.is_file() and is_supported(path)
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        ))
        output = next(
            (path for path in candidates if path.is_file() and path.stat().st_size > 0),
            None,
        )
        if output is None:
            raise RuntimeError(
                f"GraXpert completed for {source.name} but no corrected image was found."
            )
        return output

    def _prepare_mosaic_tiles(
        self,
        dataset: DatasetPlan,
        ordered_tiles: list[tuple[str, str, Path]],
        mode_name: str,
        process_root: Path,
    ) -> list[tuple[str, str, Path]]:
        """Apply optional per-panel gradient correction before astrometric stitching."""
        if not self.plan.options.mosaic_gradient_correction:
            return ordered_tiles
        prepared_root = process_root / "graxpert"
        prepared_root.mkdir(parents=True, exist_ok=True)
        prepared: list[tuple[str, str, Path]] = []
        records: list[dict[str, Any]] = []
        for index, (filter_name, panel_id, source) in enumerate(
            ordered_tiles, start=1
        ):
            self._check_cancel()
            output_base = prepared_root / f"tile_{index:05d}"
            corrected = self._run_graxpert_background(
                source, output_base, prepared_root / f"tile_{index:05d}.log"
            )
            prepared.append((filter_name, panel_id, corrected))
            records.append({
                "filter": filter_name,
                "panel": panel_id,
                "source": str(source),
                "corrected": str(corrected),
            })
            self._tick(
                f"Corrected {panel_id}/{filter_name} gradient with GraXpert."
            )
        mode_manifest = self.manifest.setdefault("mosaics", {}).setdefault(
            dataset.dataset_id, {"panels": dataset.panels, "modes": {}}
        ).setdefault("modes", {}).setdefault(mode_name, {})
        mode_manifest["gradient_correction"] = {
            "engine": "GraXpert",
            "correction": "Subtraction",
            "smoothing": self.plan.options.graxpert_smoothing,
            "gpu": self.plan.options.graxpert_gpu,
            "tiles": records,
        }
        return prepared

    def _match_mosaic_backgrounds(
        self,
        dataset: DatasetPlan,
        mode_name: str,
        process_root: Path,
        aligned: str,
        count: int,
        ranges: dict[str, tuple[int, int]],
        ordered_tiles: list[tuple[str, str, Path]],
    ) -> str:
        """Match additive tile levels from robust registered overlap differences."""
        if not self.plan.options.mosaic_background_matching:
            return aligned
        # The caller closes Siril before Astropy updates these files, avoiding
        # Windows sharing locks while retaining the sequence metadata on disk.
        aligned_paths = [
            process_root / f"{aligned}_{index:05d}.fit"
            for index in range(1, count + 1)
        ]
        self._verify_outputs(
            aligned_paths, f"Reading {dataset.dataset_id} {mode_name} panel overlaps"
        )
        offsets = [0.0] * count
        filter_records: dict[str, Any] = {}
        for filter_name, (start, end) in ranges.items():
            local_offsets, details = robust_overlap_background_offsets(
                aligned_paths[start - 1:end]
            )
            offsets[start - 1:end] = local_offsets
            details["tiles"] = [
                {
                    "panel": ordered_tiles[index - 1][1],
                    "offset": local_offsets[index - start],
                }
                for index in range(start, end + 1)
            ]
            filter_records[filter_name] = details
            if details.get("warning"):
                self.log_callback(
                    f"Background matching {filter_name}: {details['warning']}"
                )

        # Preserve the SEQAPPLYREG sequence and its per-tile registration
        # metadata. Siril needs those footprints for -maximize, overlap
        # normalization, and feathering; rebuilding the sequence from padded
        # files makes every tile appear to cover the entire canvas.
        for source, offset in zip(aligned_paths, offsets):
            self._check_cancel()
            apply_fits_background_offset(source, offset)
        mode_manifest = self.manifest.setdefault("mosaics", {}).setdefault(
            dataset.dataset_id, {"panels": dataset.panels, "modes": {}}
        ).setdefault("modes", {}).setdefault(mode_name, {})
        mode_manifest["background_matching"] = {
            "method": "robust additive overlap differences",
            "filters": filter_records,
            "preserved_registered_sequence": True,
        }
        self._tick(
            f"Matched {dataset.dataset_id} {mode_name} panel background levels from overlaps."
        )
        return aligned

    def _stitch_mosaic_mode(
        self,
        dataset: DatasetPlan,
        panel_outputs: dict[str, dict[str, Path]],
        filter_names: list[str],
        drizzle_mode: bool,
    ) -> list[Path]:
        """Astrometrically align equal-scale tiles once, then stitch each filter."""
        assert self.run_dir is not None
        ordered_tiles: list[tuple[str, str, Path]] = []
        ranges: dict[str, tuple[int, int]] = {}
        for filter_name in filter_names:
            start = len(ordered_tiles) + 1
            for panel_id in dataset.panels:
                path = panel_outputs.get(panel_id, {}).get(filter_name)
                if path is not None:
                    ordered_tiles.append((filter_name, panel_id, path))
            if len(ordered_tiles) >= start:
                ranges[filter_name] = (start, len(ordered_tiles))
        if not ordered_tiles:
            raise RuntimeError(f"No panel masters were created for {dataset.dataset_id}")

        mode_name = "drizzle" if drizzle_mode else "native"
        staging = (
            self.run_dir / "inputs" / dataset.dataset_id / f"mosaic_tiles_{mode_name}"
        )
        process_root = (
            self.run_dir / "process" / dataset.dataset_id / f"mosaic_{mode_name}"
        )
        process_root.mkdir(parents=True, exist_ok=True)
        original_tiles = ordered_tiles
        if self.plan.options.mosaic_gradient_correction:
            self._cmd("close")
        ordered_tiles = self._prepare_mosaic_tiles(
            dataset, ordered_tiles, mode_name, process_root
        )
        modes = self._stage_files(
            [str(path) for _filter, _panel, path in ordered_tiles], staging
        )
        base = f"mosaic_tiles_{mode_name}"
        count = self._convert_sequence(
            staging, process_root, base, len(ordered_tiles), modes
        )
        if count != len(ordered_tiles):
            raise RuntimeError(
                f"Mosaic staging expected {len(ordered_tiles)} tile masters but Siril "
                f"created a {count}-image sequence."
            )
        self.manifest.setdefault("staging", {})[
            f"{dataset.dataset_id}_mosaic_tiles_{mode_name}"
        ] = modes
        mosaic_manifest = self.manifest.setdefault("mosaics", {}).setdefault(
            dataset.dataset_id,
            {"panels": dataset.panels, "modes": {}},
        )
        mode_manifest = mosaic_manifest["modes"].setdefault(mode_name, {})
        mode_manifest.update({
            "tiles": [
                {
                    "filter": filter_name,
                    "panel": panel_id,
                    "path": str(path),
                    "source": str(original_tiles[index][2]),
                }
                for index, (filter_name, panel_id, path) in enumerate(ordered_tiles)
            ],
            "feather_pixels": self.plan.options.mosaic_feather,
            "normalize_on_overlaps": self.plan.options.mosaic_overlap_normalization,
        })
        self._tick(
            f"Created {dataset.dataset_id} {mode_name} astrometric mosaic sequence "
            f"from {count} panel masters."
        )

        self._solve_mosaic_tiles(dataset, base, count, mode_name)
        self._tick(
            f"Plate-solved every {mode_name} mosaic tile for {dataset.dataset_id}."
        )

        aligned_prefix = "aligned_"
        self._cmd(
            "seqapplyreg", base, f"-prefix={aligned_prefix}",
            "-framing=max", "-interp=cubic",
        )
        aligned = f"{aligned_prefix}{base}"
        self._tick(
            f"Applied shared astrometric maximum framing to {dataset.dataset_id} "
            f"{mode_name} tiles."
        )
        # Release the sequence before optional in-place background offsets;
        # its registration metadata remains intact on disk for stitching.
        self._cmd("close")
        aligned = self._match_mosaic_backgrounds(
            dataset,
            mode_name,
            process_root,
            aligned,
            count,
            ranges,
            ordered_tiles,
        )

        outputs: list[Path] = []
        for filter_name, (start, end) in ranges.items():
            tile_count = end - start + 1
            self._cmd("unselect", aligned, "1", str(count))
            self._cmd("select", aligned, str(start), str(end))
            output = self.run_dir / "masters" / self._mosaic_output_name(
                dataset, filter_name, drizzle_mode
            )
            args = [
                "stack", aligned, "rej", "none", "-norm=addscale",
                "-filter-included", "-maximize", "-32b",
            ]
            if tile_count > 1 and self.plan.options.mosaic_overlap_normalization:
                args.append("-overlap_norm")
            if tile_count > 1 and self.plan.options.mosaic_feather > 0:
                args.append(f"-feather={self.plan.options.mosaic_feather}")
            args.append(option_path("out", output))
            self._cmd(*args)
            output_fit = output.with_suffix(".fit")
            outputs.append(output_fit)
            self._tick(
                f"Stitched {filter_name} mosaic from {tile_count} panel master(s)."
            )
        self._verify_outputs(
            outputs, f"Stitching {dataset.dataset_id} {mode_name} mosaics"
        )
        outputs = self._finalize_mosaic_geometry(
            dataset, outputs, mode_name, process_root
        )
        if self.plan.options.rolling_cleanup:
            self._cmd("close")
            self._remove_generated_paths(
                [staging, process_root],
                f"Released verified {dataset.dataset_id} {mode_name} mosaic workspace",
            )
        return outputs

    def _stitch_dataset_mosaics(
        self,
        dataset: DatasetPlan,
        panel_outputs: dict[str, dict[str, Path]],
    ) -> list[Path]:
        """Stitch native and drizzle filters without mixing their pixel scales."""
        native_filters = [
            filter_name for filter_name in dataset.channels
            if not self.plan.channel_uses_drizzle(dataset, filter_name)
        ]
        drizzle_filters = [
            filter_name for filter_name in dataset.channels
            if self.plan.channel_uses_drizzle(dataset, filter_name)
        ]
        outputs: list[Path] = []
        if native_filters:
            outputs.extend(self._stitch_mosaic_mode(
                dataset, panel_outputs, native_filters, False
            ))
        if drizzle_filters:
            outputs.extend(self._stitch_mosaic_mode(
                dataset, panel_outputs, drizzle_filters, True
            ))
        self._verify_outputs(outputs, f"Completing {dataset.dataset_id} mosaics")
        if self.plan.options.rolling_cleanup:
            assert self.run_dir is not None
            self._remove_generated_paths(
                [self.run_dir / "masters" / "panels" / dataset.dataset_id],
                f"Released panel masters consumed by {dataset.dataset_id} mosaics",
            )
        return outputs

    def _solve_final_master(
        self, dataset: DatasetPlan, output: Path
    ) -> tuple[bool, str]:
        """Plate-solve one completed ordinary master without risking the stack."""
        return self._platesolve_to(output, output)

    def _platesolve_to(self, source: Path, destination: Path) -> tuple[bool, str]:
        """Plate-solve one image into ``destination`` using the configured fallbacks."""
        attempts: list[tuple[str, list[str]]] = []
        guided = ["-force", "-order=3", "-catalog=gaia"]
        if self.plan.options.mosaic_downsample_solve:
            attempts.append(("header-guided downsampled", [*guided, "-downscale"]))
            attempts.append(("header-guided full resolution", guided))
        else:
            attempts.append(("header-guided full resolution", guided))
        if self.plan.options.mosaic_blind_fallback:
            attempts.append((
                "local Astrometry.net blind",
                ["-force", "-order=3", "-localasnet", "-blindpos", "-blindres"],
            ))

        last_error = ""
        temporary = destination.with_name(f"{destination.stem}_solved_tmp.fit")
        temporary.unlink(missing_ok=True)
        for attempt_index, (label, arguments) in enumerate(attempts):
            self._check_cancel()
            try:
                self._cmd("load", quoted(source))
                self._cmd("platesolve", *arguments)
                self._cmd("save", quoted(temporary))
                self._cmd("close")
                self._verify_outputs(
                    [temporary], f"Saving solved image {destination.name}"
                )
                os.replace(temporary, destination)
                return True, label
            except Exception as exc:
                last_error = str(exc)
                temporary.unlink(missing_ok=True)
                try:
                    self._cmd("close")
                except Exception:
                    pass
                if attempt_index + 1 < len(attempts):
                    self.log_callback(
                        f"Plate solve for {source.name} failed with {label}; retrying."
                    )
        return False, last_error

    def _solve_non_mosaic_outputs(
        self, dataset: DatasetPlan, outputs: list[Path]
    ) -> list[Path]:
        """Attach celestial WCS to ordinary stacks for SPCC and catalog tools."""
        return self._solve_outputs(dataset.dataset_id, outputs)

    def _solve_outputs(self, owner_id: str, outputs: list[Path]) -> list[Path]:
        if not self.plan.options.solve_final_images:
            return outputs
        records: list[dict[str, Any]] = []
        for output in outputs:
            solved, detail = self._platesolve_to(output, output)
            records.append({
                "path": str(output),
                "solved": solved,
                "method" if solved else "error": detail,
            })
            if solved:
                self._tick(f"Plate-solved final master {output.name}.")
            else:
                self._tick(
                    f"WARNING: retained {output.name}, but its final plate solve failed: "
                    f"{detail}"
                )
        self.manifest.setdefault("final_plate_solving", {})[owner_id] = records
        return outputs

    def _normalize_output_orientation(
        self, owner_id: str, outputs: list[Path]
    ) -> list[Path]:
        """Store every final product bottom-up without resampling its pixels."""
        records: list[dict[str, Any]] = []
        for output in outputs:
            self._check_cancel()
            before_header = fits.getheader(output, ext=0)
            before_order = str(before_header.get("ROWORDER", "")).strip()
            if normalized_text(before_order) == "BOTTOMUP":
                records.append({
                    "path": str(output),
                    "changed": False,
                    "row_order_before": before_order,
                    "row_order_after": before_order,
                    "wcs_error_arcsec": 0.0,
                })
                self._tick(f"Verified bottom-up orientation for {output.name}.")
                continue

            temporary = output.with_name(f"{output.stem}_orientation_tmp.fit")
            temporary.unlink(missing_ok=True)
            loaded = False
            try:
                self._cmd("load", quoted(output))
                loaded = True
                self._cmd("mirrorx", "-bottomup")
                self._cmd("save", quoted(temporary))
                self._cmd("close")
                loaded = False
                self._verify_outputs(
                    [temporary], f"Normalizing final orientation for {output.name}"
                )
                after_header = fits.getheader(temporary, ext=0)
                after_order = str(after_header.get("ROWORDER", "")).strip()
                if normalized_text(after_order) != "BOTTOMUP":
                    raise RuntimeError(
                        f"Siril did not store {output.name} in bottom-up FITS order."
                    )
                wcs_error = vertical_mirror_wcs_error(before_header, after_header)
                if (
                    wcs_error is not None
                    and wcs_error > ORIENTATION_WCS_TOLERANCE_ARCSEC
                ):
                    raise RuntimeError(
                        f"Orientation normalization changed the WCS of {output.name} "
                        f"by {wcs_error:.4f} arcsec."
                    )
                os.replace(temporary, output)
                records.append({
                    "path": str(output),
                    "changed": True,
                    "row_order_before": before_order or "unspecified",
                    "row_order_after": after_order,
                    "wcs_error_arcsec": (
                        None if wcs_error is None else round(wcs_error, 6)
                    ),
                })
                self._tick(
                    f"Stored {output.name} bottom-up with its WCS preserved."
                )
            finally:
                if loaded:
                    try:
                        self._cmd("close")
                    except Exception:
                        pass
                temporary.unlink(missing_ok=True)
        self.manifest.setdefault("orientation_normalization", {})[owner_id] = records
        return outputs

    def _process_master_merge(self) -> list[Path]:
        """Solve finished masters, reproject them onto one canvas, and merge them."""
        assert self.run_dir is not None
        merge = self.plan.master_merge
        assert merge is not None
        options = self.plan.options
        staging = self.run_dir / "inputs" / "master_merge"
        process_root = self.run_dir / "process" / "master_merge"
        solved_root = process_root / "solved"
        registered_root = process_root / "registered"
        solved_root.mkdir(parents=True, exist_ok=True)
        modes = self._stage_files([entry.path for entry in merge.masters], staging)
        staged = sorted(staging.iterdir(), key=lambda path: path.name)
        self.manifest.setdefault("staging", {})["master_merge"] = modes
        report = self.manifest.setdefault("master_merge", {})
        self._tick(f"Staged {len(staged)} masters for the multi-scale merge.")

        # Each master is solved on its own: a sequence solve would apply the
        # first image's focal length and centre to every telescope.
        solved: list[Path] = []
        for index, (entry, source) in enumerate(zip(merge.masters, staged), start=1):
            destination = solved_root / f"merge_{index:05d}.fit"
            succeeded, detail = self._platesolve_to(source, destination)
            if not succeeded:
                raise RuntimeError(
                    f"Could not plate-solve {Path(entry.path).name}; a master merge needs "
                    f"every master's sky position. Last error: {detail}"
                )
            solved.append(destination)
            self._tick(f"Plate-solved {Path(entry.path).name} ({detail}).")

        headers = [fits.getheader(path, ext=0) for path in solved]
        scales = [solved_pixel_scale_arcsec(header) for header in headers]
        if any(scale is None for scale in scales):
            raise RuntimeError("A solved master carries no usable celestial WCS.")
        solved_scales = [float(scale) for scale in scales]  # type: ignore[arg-type]
        layers = cluster_plate_scales(solved_scales)  # type: ignore[arg-type]
        planned = [entry.layer for entry in merge.masters]
        if layers != planned:
            self.log_callback(
                "Solved plate scales re-ranked the layers: "
                + ", ".join(
                    f"{Path(entry.path).name} {old}→{new}"
                    for entry, old, new in zip(merge.masters, planned, layers)
                    if old != new
                )
            )
        if max(layers) < 2:
            raise RuntimeError("Every solved master has the same plate scale; nothing to merge.")
        seen: set[tuple[int, str]] = set()
        for entry, layer in zip(merge.masters, layers):
            key = (layer, normalized_text(entry.filter_name))
            if key in seen:
                raise RuntimeError(
                    f"Two {entry.filter_name} masters share solved layer {layer}; "
                    "same-scale masters cannot be merged yet."
                )
            seen.add(key)
        grid = choose_merge_grid(layers, solved_scales, options.merge_output_grid)
        report.update({
            "masters": [
                {
                    "source": entry.path,
                    "filter": entry.filter_name,
                    "layer": layer,
                    "solved_scale_arcsec": round(scale, 5),
                }
                for entry, layer, scale in zip(merge.masters, layers, solved_scales)
            ],
            "grid": grid,
        })
        self.log_callback(
            f"Merge canvas uses the {grid['grid']} layer's grid at "
            f"{min(solved_scales):.3f}\"/px ({grid['reason']})."
        )

        # The registration input mirrors the solved masters. On the finest
        # grid its reference is padded to cover every footprint, because
        # Siril's per-image bounding boxes under maximum framing carry WCS
        # headers that do not match their pixels.
        registration_input = process_root / "registration_input"
        registration_input.mkdir(parents=True, exist_ok=True)
        reference_index = int(grid["reference_index"])
        for index, path in enumerate(solved):
            target = registration_input / path.name
            if grid["grid"] == "finest" and index == reference_index:
                pad_reference_to_union(
                    path, [other for other in solved if other != path], target
                )
            else:
                link_or_copy(path, target, symlinks_usable(registration_input))
        count = self._convert_sequence(
            registration_input, registered_root, "merge", len(solved), None
        )
        if count != len(solved):
            raise RuntimeError(
                f"Master merge expected {len(solved)} images but Siril created {count}."
            )
        self._cmd("setref", "merge", str(reference_index + 1))
        # Every image is already solved, so this only computes the shared
        # astrometric registration (including distortion) from their WCS.
        self._cmd("seqplatesolve", "merge", "-order=3", "-catalog=gaia")
        # The reference footprint is the canvas: the widest master itself,
        # or the padded finest master.
        register = [
            "seqapplyreg", "merge", "-prefix=r_", "-framing=current",
            "-interp=lanczos4",
        ]
        if abs(float(grid["scale"]) - 1.0) > 1e-6:
            register.append(f"-scale={float(grid['scale']):.6f}")
        self._cmd(*register)
        self._cmd("close")
        aligned = [
            registered_root / f"r_merge_{index:05d}.fit" for index in range(1, count + 1)
        ]
        self._verify_outputs(aligned, "Reprojecting masters onto the merge canvas")
        self._tick(
            f"Reprojected {count} masters onto one canvas at the finest plate scale."
        )

        records = [
            {
                "filter": entry.filter_name,
                "layer": layer,
                "native": native,
                "aligned": registered,
                "source": entry.path,
            }
            for entry, layer, native, registered in zip(
                merge.masters, layers, solved, aligned
            )
        ]
        outputs, merge_report = merge_master_layers(
            records,
            self.run_dir / "masters",
            blend_large_scales=options.merge_blend_large_scales,
            transition_pixels=options.merge_transition_pixels,
            detail_sigma=options.merge_detail_sigma,
            log=self.log_callback,
        )
        report.update(merge_report)
        self._verify_outputs(outputs, "Writing merged masters")
        for output in outputs:
            self._tick(f"Merged {output.name}.")
        if options.rolling_cleanup:
            self._remove_generated_paths(
                [staging, process_root], "Released verified master-merge workspace"
            )
        return self._solve_outputs("master_merge", outputs)

    def _process_dataset(self, dataset: DatasetPlan) -> list[Path]:
        assert self.run_dir is not None
        panel_outputs: dict[str, dict[str, Path]] = {}
        for panel_id in dataset.panels:
            outputs = self._process_panel(dataset, panel_id)
            self._verify_outputs(
                outputs.values(), f"Integrating {dataset.dataset_id}/{panel_id}"
            )
            panel_outputs[panel_id] = outputs
            if self.plan.options.rolling_cleanup:
                self._cmd("close")
                self._remove_generated_paths(
                    [
                        self.run_dir / "inputs" / dataset.dataset_id / "panels" / panel_id,
                        self.run_dir / "process" / dataset.dataset_id / "panels" / panel_id,
                    ],
                    f"Released verified {dataset.dataset_id}/{panel_id} light workspace",
                )
        if dataset.panel_count > 1 and self.plan.options.mosaic_enabled:
            return self._stitch_dataset_mosaics(dataset, panel_outputs)
        outputs = [
            output
            for panel_id in dataset.panels
            for output in panel_outputs[panel_id].values()
        ]
        return self._solve_non_mosaic_outputs(dataset, outputs)

    def _finish_timing(self, status: str) -> float:
        assert self.started_monotonic is not None
        elapsed = time.monotonic() - self.started_monotonic
        timing = self.manifest.setdefault("timing", {})
        timing["started_utc"] = self.started_utc
        timing["finished_utc"] = utc_now()
        timing["elapsed_seconds"] = round(elapsed, 3)
        timing["elapsed"] = human_duration(elapsed)
        timing["result"] = status
        return elapsed

    def _cleanup_intermediates(self) -> None:
        """Final sweep for reproducible workspaces not already released."""
        assert self.run_dir is not None
        targets = [self.run_dir / name for name in ("inputs", "process")]
        panel_tiles = self.run_dir / "masters" / "panels"
        if self.plan.options.mosaic_enabled:
            targets.append(panel_tiles)
        self._remove_generated_paths(
            targets, "Removed generated workspaces after verified stacks"
        )

    def _cleanup_calibration_products(self, outputs: Iterable[Path]) -> None:
        """Leave only requested science products in masters after a clean run."""
        assert self.run_dir is not None
        masters_root = (self.run_dir / "masters").absolute()
        retained = {path.absolute() for path in outputs}
        candidates: set[Path] = {self.run_dir / "masters" / "bpm"}
        for path in [*self.assets.values(), *self.asset_bpm.values()]:
            absolute = path.absolute()
            try:
                absolute.relative_to(masters_root)
            except ValueError:
                continue
            if absolute not in retained:
                candidates.add(path)
        self._remove_generated_paths(
            sorted(candidates, key=lambda path: len(path.parts), reverse=True),
            "Removed calibration working masters and BPM files",
        )

    def run(self) -> Path:
        if not self.plan.valid:
            raise RuntimeError("The processing plan contains blocking calibration errors.")
        self.started_monotonic = time.monotonic()
        self.started_utc = utc_now()
        output_root = Path(self.plan.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        integration = integration_folder_token(
            self.plan.integration_seconds, self.plan.unknown_exposure_count
        )
        if self.plan.master_merge is not None:
            integration = "master_merge"
        self.run_dir = output_root / f"{self.plan.output_prefix}_{integration}_{stamp}"
        self.run_dir.mkdir(parents=False, exist_ok=False)
        (self.run_dir / "masters").mkdir()
        self._open_run_log()
        self.log_callback(f"Run folder: {self.run_dir}")
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "application": APP_NAME,
            "application_version": VERSION,
            "status": "running",
            "created_utc": self.started_utc,
            "timing": {
                "started_utc": self.started_utc,
                "elapsed_seconds": 0.0,
                "elapsed": "00:00:00",
            },
            "plan": self.plan.to_dict(),
            "commands": [],
            "outputs": [],
        }
        self._write_manifest()
        try:
            self._cmd("requires", "1.4.4")
            self._cmd("setext", "fit")
            self._cmd("set32bits")
            outputs: list[Path] = []
            if self.plan.master_merge is not None:
                outputs.extend(self._normalize_output_orientation(
                    "master_merge", self._process_master_merge()
                ))
            for dataset in self.plan.datasets:
                dataset_outputs = self._process_dataset(dataset)
                outputs.extend(
                    self._normalize_output_orientation(dataset.dataset_id, dataset_outputs)
                )
            self._cmd("close")
            missing = [path for path in outputs if not path.is_file()]
            if missing:
                raise RuntimeError(
                    "Siril reported completion but did not create expected stack(s): "
                    + ", ".join(str(path) for path in missing)
                )
            if not retains_intermediates(self.plan.options):
                self._cleanup_intermediates()
            if self.plan.options.rolling_cleanup:
                self._cleanup_calibration_products(outputs)
            self.manifest["status"] = "complete"
            self.manifest["completed_utc"] = utc_now()
            self.manifest["outputs"] = [str(path) for path in outputs]
            elapsed = self._finish_timing("complete")
            self.progress_callback(100)
            self._write_manifest()
            self.log_callback(
                f"Complete in {human_duration(elapsed)}. Outputs are in {self.run_dir / 'masters'}"
            )
            return self.run_dir
        except PipelineCancelled:
            self.manifest["status"] = "cancelled"
            self.manifest["cancelled_utc"] = utc_now()
            self._finish_timing("cancelled")
            self._write_manifest()
            raise
        except Exception as exc:
            self.manifest["status"] = "failed"
            self.manifest["failed_utc"] = utc_now()
            self.manifest["error"] = str(exc)
            self.manifest["traceback"] = traceback.format_exc()
            self._write_run_log(f"Processing failed:\n{self.manifest['traceback'].rstrip()}")
            self._finish_timing("failed")
            self._write_manifest()
            raise
        finally:
            self._close_run_log()


# --- Crash-safe session log ----------------------------------------------------


class SessionLog:
    """Append-only log of one SBPP session, flushed per line to survive crashes.

    Python's fault handler writes here too, so even a hard crash of the
    interpreter leaves a stack trace behind.
    """

    KEEP_SESSIONS = 20

    def __init__(self, directory: Path):
        self.directory = directory
        self.path: Optional[Path] = None
        self._handle: Optional[TextIO] = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = directory / f"sbpp_{stamp}_{os.getpid()}.log"
            self._handle = open(self.path, "a", encoding="utf-8")
        except OSError:
            self._handle = None
            return
        for old in sorted(directory.glob("sbpp_*.log"))[:-self.KEEP_SESSIONS]:
            try:
                old.unlink()
            except OSError:
                pass
        try:
            faulthandler.enable(file=self._handle, all_threads=True)
        except (OSError, RuntimeError, ValueError):
            pass
        self.write(
            f"{APP_NAME} {VERSION} session started on {platform.platform()} / "
            f"Python {platform.python_version()}"
        )

    def write(self, message: str) -> None:
        if self._handle is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._handle.write(f"[{stamp}] {message}\n")
            self._handle.flush()
        except (OSError, ValueError):
            self._handle = None

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            faulthandler.disable()
            self._handle.close()
        except (OSError, ValueError):
            pass
        self._handle = None


def install_exception_logging(record: Callable[[str], None]) -> None:
    """Record uncaught exceptions instead of letting PyQt abort the process."""

    def hook(kind: type[BaseException], value: BaseException, trace: Any) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, trace)
            return
        text = "".join(traceback.format_exception(kind, value, trace)).rstrip()
        try:
            record(f"Unexpected error:\n{text}")
        except Exception:  # never let the hook itself fail
            sys.__excepthook__(kind, value, trace)

    sys.excepthook = hook


# --- Self-update ---------------------------------------------------------------


@dataclass
class ReleaseInfo:
    version: str
    notes: str
    asset_url: str
    page_url: str


def version_tuple(text: str) -> tuple[int, int, int]:
    match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", text.strip())
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _github_request(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"SBPP/{VERSION}",
    })
    context = None
    try:  # some Python builds (notably on macOS) ship without CA certificates
        import certifi
        import ssl

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def fetch_latest_release(timeout: float = 10.0) -> ReleaseInfo:
    payload = json.loads(_github_request(UPDATE_API_URL, timeout).decode("utf-8"))
    asset = next(
        (item for item in payload.get("assets", []) if item.get("name") == SCRIPT_FILENAME),
        None,
    )
    if asset is None:
        raise RuntimeError(f"The latest release does not include {SCRIPT_FILENAME}.")
    return ReleaseInfo(
        version=str(payload.get("tag_name", "")).lstrip("v"),
        notes=str(payload.get("body") or "").strip(),
        asset_url=str(asset["browser_download_url"]),
        page_url=str(payload.get("html_url") or f"https://github.com/{GITHUB_REPO}/releases"),
    )


def script_version(text: str) -> Optional[str]:
    match = re.search(r'^VERSION = "([^"]+)"$', text, re.MULTILINE)
    return match.group(1) if match else None


def validate_update(data: bytes, expected_version: str) -> None:
    """Refuse anything that is not a complete SBPP script of the expected version."""
    try:
        text = data.decode("utf-8")
        ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError(f"The downloaded update is damaged ({exc}).") from exc
    found = script_version(text)
    if found != expected_version or "class MainWindow" not in text:
        raise RuntimeError(
            f"The downloaded file is not SBPP {expected_version} (found {found or 'no version'})."
        )


def download_release(release: ReleaseInfo, timeout: float = 60.0) -> bytes:
    data = _github_request(release.asset_url, timeout)
    validate_update(data, release.version)
    return data


def find_installed_scripts(script_dirs: Iterable[str], running_file: Path) -> list[Path]:
    """Locate SBPP installs Siril can launch.

    Siril may run a temporary copy of the script, so the running file is used
    only when it carries the real name; otherwise search Siril's script folders.
    """
    found: list[Path] = []
    if running_file.name == SCRIPT_FILENAME and running_file.is_file():
        found.append(running_file)
    for directory in script_dirs:
        root = Path(directory) if directory else None
        if root is None or not root.is_dir():
            continue
        for current, folders, files in os.walk(root):
            depth = len(Path(current).relative_to(root).parts)
            folders[:] = [] if depth >= 3 else [
                name for name in folders if not name.startswith(".") and name != "__pycache__"
            ]
            if SCRIPT_FILENAME in files:
                found.append(Path(current) / SCRIPT_FILENAME)
    unique: dict[str, Path] = {}
    for path in found:
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            key = os.path.normcase(str(path))
        unique.setdefault(key, path)
    return list(unique.values())


def install_update(target: Path, data: bytes, backup_dir: Path) -> Path:
    """Atomically replace `target` with `data`, keeping a backup of the old file."""
    target = target.resolve()
    current = script_version(target.read_text(encoding="utf-8", errors="replace"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{target.stem}-{current or 'unknown'}.py"
    shutil.copy2(target, backup)
    temporary = target.with_name(f".{target.name}.update")
    temporary.write_bytes(data)
    try:
        shutil.copymode(target, temporary)
    except OSError:
        pass
    os.replace(temporary, target)
    return backup


class BackgroundTask(QThread):
    """Run one blocking call off the GUI thread."""

    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, function: Callable[[], Any]):
        super().__init__()
        self.function = function

    def run(self) -> None:
        try:
            self.done.emit(self.function())
        except Exception as exc:  # surfaced to the user by the caller
            self.error.emit(str(exc) or type(exc).__name__)


class PipelineThread(QThread):
    log_line = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)

    def __init__(self, siril: s.SirilInterface, plan: ProcessingPlan):
        super().__init__()
        self.siril = siril
        self.plan = plan
        self._cancel_requested = False
        self.engine: Optional[PipelineEngine] = None

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.log_line.emit("Stop requested; SBPP will stop after the current Siril command completes.")

    def run(self) -> None:
        engine = PipelineEngine(
            self.siril,
            self.plan,
            self.log_line.emit,
            self.progress_value.emit,
            lambda: self._cancel_requested,
        )
        self.engine = engine
        try:
            run_dir = engine.run()
            self.succeeded.emit(str(run_dir))
        except PipelineCancelled as exc:
            partial = str(engine.run_dir or "")
            self.cancelled.emit(f"{exc}\nPartial run: {partial}")
        except Exception as exc:  # report complete traceback in the UI and manifest
            self.failed.emit(str(exc), traceback.format_exc())


def panel_tooltip_text(panel: PanelOutline) -> str:
    lines = [f"{name}: {count} frames" for name, count in panel.filter_counts.items()]
    lines.append(f"Centre RA {panel.center_ra_deg:.4f}°, Dec {panel.center_dec_deg:+.4f}°")
    for other, fraction in sorted(panel.overlaps.items(), key=lambda item: natural_key(item[0])):
        lines.append(f"Overlap with {other}: {fraction:.0%}")
    if panel.missing_filters:
        lines.append("Missing: " + ", ".join(panel.missing_filters))
    if panel.isolated:
        lines.append("No overlap with any other panel")
    return "\n".join(lines)


class _PanelCanvas(QWidget):
    WARNING_COLOR = QColor("#ff7a3c")

    def __init__(self, owner: "PanelMapWindow"):
        super().__init__(owner)
        self.owner = owner
        self.setMouseTracking(True)
        self.setMinimumSize(320, 260)

    def _transform(self) -> Optional[tuple[float, float, float]]:
        layout = self.owner.current_layout()
        if layout is None or not layout.drawable or not layout.panels:
            return None
        xs = [x for panel in layout.panels for x, _ in panel.corners]
        ys = [y for panel in layout.panels for _, y in panel.corners]
        span_x = max(max(xs) - min(xs), 1e-9)
        span_y = max(max(ys) - min(ys), 1e-9)
        margin = 0.05
        scale = min(
            self.width() * (1 - 2 * margin) / span_x,
            self.height() * (1 - 2 * margin) / span_y,
        )
        offset_x = self.width() / 2 - scale * (min(xs) + max(xs)) / 2
        # Widget y grows downwards; north (larger y) is drawn up.
        offset_y = self.height() / 2 + scale * (min(ys) + max(ys)) / 2
        return scale, offset_x, offset_y

    def to_widget(self, x_deg: float, y_deg: float) -> tuple[float, float]:
        transform = self._transform()
        if transform is None:
            return (0.0, 0.0)
        scale, offset_x, offset_y = transform
        return offset_x + scale * x_deg, offset_y - scale * y_deg

    def _polygon(self, panel: PanelOutline) -> QPolygonF:
        return QPolygonF([QPointF(*self.to_widget(x, y)) for x, y in panel.corners])

    def panel_at(self, x: float, y: float) -> Optional[PanelOutline]:
        layout = self.owner.current_layout()
        if self._transform() is None or layout is None:
            return None
        point = QPointF(x, y)
        for panel in reversed(layout.panels):
            if self._polygon(panel).containsPoint(point, Qt.FillRule.OddEvenFill):
                return panel
        return None

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt API name
        if self._transform() is None:
            return
        layout = self.owner.current_layout()
        assert layout is not None
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        normal = self.palette().color(QPalette.ColorRole.Highlight)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(max(9, font.pointSize() + 2))
        painter.setFont(font)
        for panel in layout.panels:
            warning = panel.isolated or bool(panel.missing_filters)
            color = self.WARNING_COLOR if warning else normal
            fill = QColor(color)
            fill.setAlpha(64)
            pen = QPen(color, 2)
            if warning:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            polygon = self._polygon(panel)
            painter.drawPolygon(polygon)
            label = str(panel.number) if panel.number is not None else panel.panel_id
            painter.drawText(
                polygon.boundingRect(), Qt.AlignmentFlag.AlignCenter, label
            )
        painter.end()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        position = event.position()
        panel = self.panel_at(position.x(), position.y())
        if panel is None:
            QToolTip.hideText()
        else:
            QToolTip.showText(event.globalPosition().toPoint(), panel_tooltip_text(panel), self)


class PanelMapWindow(QWidget):
    """Non-modal outline map of mosaic panels, refreshed in place on rescan."""

    ROTATION_NOTE = "Camera rotation not in headers; panels drawn north-up."

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(640, 520)
        self._plan: Optional[ProcessingPlan] = None
        self._layouts: list[PanelLayout] = []
        layout = QVBoxLayout(self)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        top = QHBoxLayout()
        top.addWidget(self.dataset_combo)
        top.addStretch(1)
        layout.addLayout(top)
        self.canvas = _PanelCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

    def set_plan(self, plan: Optional[ProcessingPlan]) -> None:
        self._plan = plan
        datasets = [
            dataset for dataset in (plan.datasets if plan else [])
            if dataset.panel_count > 1
        ]
        self._layouts = []
        for dataset in datasets:
            # A display-only feature must never break scanning.
            try:
                self._layouts.append(panel_layout(dataset))
            except Exception as exc:
                self._layouts.append(PanelLayout(
                    dataset.dataset_id, [], False,
                    f"Layout could not be computed: {exc}", False, dataset.panels,
                ))
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems([dataset.dataset_id for dataset in datasets])
        self.dataset_combo.blockSignals(False)
        self.dataset_combo.setVisible(len(datasets) > 1)
        self._dataset_changed()

    def current_layout(self) -> Optional[PanelLayout]:
        index = self.dataset_combo.currentIndex()
        if 0 <= index < len(self._layouts):
            return self._layouts[index]
        return None

    def _dataset_changed(self, *_args: Any) -> None:
        layout = self.current_layout()
        name = (self._plan.target_name if self._plan else "") or (
            layout.dataset_id if layout else ""
        )
        self.setWindowTitle(f"Panel layout: {name}")
        notes = []
        if layout is not None:
            if not layout.drawable:
                notes.append(layout.message)
                if layout.unplaced:
                    notes.append("Panels found: " + ", ".join(layout.unplaced))
            else:
                if not layout.rotation_known:
                    notes.append(self.ROTATION_NOTE)
                if layout.unplaced:
                    notes.append("Not drawn (no coordinates): " + ", ".join(layout.unplaced))
                if layout.far:
                    notes.append(
                        "Too far from the rest of the mosaic to draw: " + ", ".join(layout.far)
                    )
        self.note_label.setText("\n".join(notes))
        self.canvas.update()

    def to_widget(self, x_deg: float, y_deg: float) -> tuple[float, float]:
        return self.canvas.to_widget(x_deg, y_deg)

    def panel_at(self, x: float, y: float) -> Optional[PanelOutline]:
        return self.canvas.panel_at(x, y)


class MainWindow(QMainWindow):
    def __init__(self, siril: s.SirilInterface):
        super().__init__()
        self.siril = siril
        self.store = ProfileStore()
        self.session_log = SessionLog(self.store.config_dir / "logs")
        self.last_run_dir: Optional[Path] = None
        self.cache = MetadataCache(self.store.cache_path)
        self.current_plan: Optional[ProcessingPlan] = None
        self.pipeline_thread: Optional[PipelineThread] = None
        self._loading_profile = False
        self._active_profile_name = ""
        self._auto_inferred_target = ""
        self._updating_drizzle = False
        self._default_reset_buttons: dict[int, QPushButton] = {}
        self.option_defaults = PlanOptions()
        self.run_started_monotonic: Optional[float] = None
        self.last_run_elapsed = 0.0
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._update_elapsed_display)
        self.update_task: Optional[BackgroundTask] = None
        self.panel_map_window: Optional[PanelMapWindow] = None

        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1320, 820)
        self._build_ui()
        self._populate_profiles()
        selected = self.store.payload.get("selected_profile", "Default")
        index = self.profile_combo.findText(selected)
        self.profile_combo.setCurrentIndex(max(0, index))
        self._load_profile(self.profile_combo.currentText())
        QTimer.singleShot(1500, self._startup_update_check)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        title = QLabel("Starloch Batch Preprocessor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle = QLabel(
            "Dynamic calibration, two-pass homography registration, common crop, "
            "quality gating, optional per-channel drizzle, and automatic per-filter "
            "astrometric mosaics through Siril."
        )
        subtitle.setWordWrap(True)
        kofi = QLabel(f'<a href="{KOFI_URL}">☕ Support on Ko-fi</a>')
        kofi.setTextFormat(Qt.TextFormat.RichText)
        kofi.setOpenExternalLinks(True)
        kofi.setToolTip(f"Enjoying the preprocessor? Buy me a coffee: {KOFI_URL}")
        kofi.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.update_button = QPushButton("Check for updates")
        self.update_button.setToolTip(f"Check GitHub for a newer {APP_NAME} and install it.")
        self.update_button.clicked.connect(lambda: self._check_for_updates(manual=True))
        self.update_on_startup = QCheckBox("Check at startup")
        self.update_on_startup.setToolTip("Look for a new version once a day when SBPP opens.")
        self.update_on_startup.setChecked(bool(self._update_settings()["check_on_startup"]))
        self.update_on_startup.toggled.connect(self._update_on_startup_toggled)
        title_row.addWidget(self.update_on_startup)
        title_row.addWidget(self.update_button)
        title_row.addSpacing(12)
        title_row.addWidget(kofi)
        layout.addLayout(title_row)
        layout.addWidget(subtitle)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Target profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._profile_changed)
        profile_row.addWidget(self.profile_combo, 1)
        new_profile = QPushButton("New")
        new_profile.clicked.connect(self._new_profile)
        save_profile = QPushButton("Save profile")
        save_profile.clicked.connect(self._save_profile)
        self.delete_profile_button = QPushButton("Delete")
        self.delete_profile_button.clicked.connect(self._delete_profile)
        profile_row.addWidget(new_profile)
        profile_row.addWidget(save_profile)
        profile_row.addWidget(self.delete_profile_button)
        layout.addLayout(profile_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_setup_tab(), "1. Setup")
        self.tabs.addTab(self._build_plan_tab(), "2. Calibration plan")
        self.tabs.addTab(self._build_advanced_tab(), "3. Advanced")
        self.tabs.addTab(self._build_log_tab(), "4. Run log")
        layout.addWidget(self.tabs, 1)

        controls = QHBoxLayout()
        self.scan_button = QPushButton("Scan and build plan")
        self.scan_button.clicked.connect(self._scan)
        self.save_plan_button = QPushButton("Save dry-run JSON")
        self.save_plan_button.clicked.connect(self._save_dry_run)
        self.save_plan_button.setEnabled(False)
        self.start_button = QPushButton("Start processing")
        self.start_button.clicked.connect(self._start_processing)
        self.start_button.setEnabled(False)
        self.stop_button = QPushButton("Stop after current step")
        self.stop_button.clicked.connect(self._stop_processing)
        self.stop_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.elapsed_label = QLabel("Elapsed 00:00:00")
        self.elapsed_label.setMinimumWidth(130)
        controls.addWidget(self.scan_button)
        self.panel_layout_button = QPushButton("Panel layout…")
        self.panel_layout_button.setToolTip("Show how the mosaic panels line up on the sky.")
        self.panel_layout_button.setEnabled(False)
        self.panel_layout_button.clicked.connect(lambda: self._show_panel_map())
        controls.addWidget(self.panel_layout_button)
        controls.addWidget(self.save_plan_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.elapsed_label)
        controls.addWidget(self.progress_bar, 1)
        layout.addLayout(controls)

    def _path_row(self, edit: QLineEdit, callback: Callable[[], None]) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        row.addWidget(button)
        return widget

    def _option_control(
        self,
        control: QWidget,
        default: Any,
        browse_callback: Optional[Callable[[], None]] = None,
    ) -> QWidget:
        """Pair a saved option with a reset button shown only off-default."""
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(control, 1)
        if browse_callback is not None:
            browse = QPushButton("Browse…")
            browse.clicked.connect(browse_callback)
            row.addWidget(browse)
        reset = QPushButton("Default")
        reset.setToolTip(f"Restore the application default: {default}")
        row.addWidget(reset)
        self._default_reset_buttons[id(control)] = reset

        def current_value() -> Any:
            if isinstance(control, QCheckBox):
                return control.isChecked()
            if isinstance(control, QComboBox):
                return control.currentData()
            if isinstance(control, QDoubleSpinBox):
                return control.value()
            if isinstance(control, QSpinBox):
                return control.value()
            if isinstance(control, QLineEdit):
                return control.text()
            raise TypeError(f"Unsupported defaultable control: {type(control).__name__}")

        def is_default() -> bool:
            value = current_value()
            if isinstance(control, QDoubleSpinBox):
                return math.isclose(float(value), float(default), abs_tol=1e-9)
            return value == default

        def update_visibility(*_args: Any) -> None:
            reset.setHidden(is_default())

        def restore_default() -> None:
            if isinstance(control, QCheckBox):
                control.setChecked(bool(default))
            elif isinstance(control, QComboBox):
                index = control.findData(default)
                if index >= 0:
                    control.setCurrentIndex(index)
            elif isinstance(control, QDoubleSpinBox):
                control.setValue(float(default))
            elif isinstance(control, QSpinBox):
                control.setValue(int(default))
            elif isinstance(control, QLineEdit):
                control.setText(str(default))
            update_visibility()

        if isinstance(control, QCheckBox):
            control.toggled.connect(update_visibility)
        elif isinstance(control, QComboBox):
            control.currentIndexChanged.connect(update_visibility)
        elif isinstance(control, (QDoubleSpinBox, QSpinBox)):
            control.valueChanged.connect(update_visibility)
        elif isinstance(control, QLineEdit):
            control.textChanged.connect(update_visibility)
        reset.clicked.connect(restore_default)
        update_visibility()
        return widget

    def _default_button_for(self, control: QWidget) -> QPushButton:
        """Return a control's reset button for UI tests and accessibility hooks."""
        return self._default_reset_buttons[id(control)]

    def _build_setup_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        paths = QGroupBox("Project")
        form = QFormLayout(paths)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Folder containing lights, or a complete project root")
        self.input_edit.editingFinished.connect(self._infer_target_from_input)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Existing folder that will receive a timestamped SBPP run")
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Auto-detected from agreeing light OBJECT headers")
        self.target_edit.textEdited.connect(self._target_name_edited)
        form.addRow("Input root:", self._path_row(self.input_edit, self._browse_input))
        form.addRow("Output root:", self._path_row(self.output_edit, self._browse_output))
        form.addRow("Target name:", self.target_edit)
        layout.addWidget(paths)

        libraries = QGroupBox("Reusable calibration libraries")
        library_layout = QVBoxLayout(libraries)
        help_label = QLabel(
            "Optional. Add one or more folders containing master or raw bias, dark, "
            "dark-flat, and flat frames. These paths are stored in the selected target profile."
        )
        help_label.setWordWrap(True)
        library_layout.addWidget(help_label)
        self.calibration_list = QListWidget()
        library_layout.addWidget(self.calibration_list)
        buttons = QHBoxLayout()
        add_button = QPushButton("Add folder…")
        add_button.clicked.connect(self._add_calibration_root)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_calibration_root)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        library_layout.addLayout(buttons)
        layout.addWidget(libraries, 1)

        behavior = QGroupBox("Discovery")
        behavior_layout = QHBoxLayout(behavior)
        self.recursive_check = QCheckBox("Scan folders recursively")
        self.recursive_check.setChecked(True)
        self.prefer_masters_check = QCheckBox("Prefer compatible existing masters")
        self.prefer_masters_check.setChecked(True)
        behavior_layout.addWidget(self._option_control(
            self.recursive_check, self.option_defaults.recursive
        ))
        behavior_layout.addWidget(self._option_control(
            self.prefer_masters_check, self.option_defaults.prefer_masters
        ))
        behavior_layout.addStretch(1)
        layout.addWidget(behavior)
        return tab

    def _build_plan_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.plan_summary = QLabel("Choose inputs and click “Scan and build plan”.")
        self.plan_summary.setWordWrap(True)
        layout.addWidget(self.plan_summary)
        self.plan_table = QTableWidget(0, 15)
        self.plan_table.setHorizontalHeaderLabels([
            "Setup", "Panel", "Filter", "Lights", "Exposure", "Integration",
            "Gain/offset", "Temp", "Bias", "Dark", "Flat", "Flat-dark", "BPM",
            "Output", "Status",
        ])
        self.plan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.plan_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plan_table.verticalHeader().setVisible(False)
        header = self.plan_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.plan_table, 1)
        self.plan_warnings = QPlainTextEdit()
        self.plan_warnings.setReadOnly(True)
        self.plan_warnings.setMaximumHeight(150)
        layout.addWidget(self.plan_warnings)
        return tab

    @staticmethod
    def _configured_double(minimum: float, maximum: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    @staticmethod
    def _configured_int(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _build_advanced_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QGridLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        calibration = QGroupBox("Calibration matching")
        calibration_form = QFormLayout(calibration)
        self.dark_policy_combo = QComboBox()
        self.dark_policy_combo.addItem(
            "Exact dark, otherwise optimize best dark (WBPP-style, recommended)",
            "exact_or_optimized",
        )
        self.dark_policy_combo.addItem("Exact dark, otherwise BPM", "exact_or_bpm")
        self.dark_policy_combo.addItem("Exact dark only", "exact")
        self.dark_policy_combo.addItem("BPM only", "bpm_only")
        self.dark_policy_combo.addItem("No dark/BPM correction", "none")
        self.minimum_cal_spin = self._configured_int(1, 500, 5)
        self.exposure_tolerance_spin = self._configured_double(0.0, 60.0, 0.25, 3)
        self.temperature_tolerance_spin = self._configured_double(0.0, 30.0, 3.0, 1)
        self.debayer_check = QCheckBox("Debayer CFA lights after calibration")
        self.debayer_check.setChecked(True)
        calibration_form.addRow("Dark policy:", self._option_control(
            self.dark_policy_combo, self.option_defaults.dark_policy
        ))
        calibration_form.addRow("Minimum raw calibration frames:", self._option_control(
            self.minimum_cal_spin, self.option_defaults.minimum_calibration_frames
        ))
        calibration_form.addRow("Dark exposure tolerance (s):", self._option_control(
            self.exposure_tolerance_spin, self.option_defaults.exposure_tolerance
        ))
        calibration_form.addRow("Dark temperature tolerance (°C):", self._option_control(
            self.temperature_tolerance_spin, self.option_defaults.temperature_tolerance
        ))
        calibration_form.addRow(self._option_control(
            self.debayer_check, self.option_defaults.debayer_cfa
        ))
        layout.addWidget(calibration, 0, 0)

        quality = QGroupBox("Registration and quality gates")
        quality_form = QFormLayout(quality)
        self.fwhm_spin = self._configured_double(0.1, 100.0, 6.0, 2)
        self.roundness_spin = self._configured_double(0.0, 0.99, 0.70, 2)
        self.stars_spin = self._configured_int(0, 100000, 40)
        self.min_pairs_spin = self._configured_int(3, 1000, 20)
        self.max_stars_spin = self._configured_int(100, 2000, 2000)
        self.solve_final_check = QCheckBox("Plate-solve final non-mosaic masters")
        self.solve_final_check.setChecked(True)
        self.solve_final_check.toggled.connect(self._sync_output_options)
        quality_form.addRow("Maximum FWHM (px):", self._option_control(
            self.fwhm_spin, self.option_defaults.fwhm_max
        ))
        quality_form.addRow("Minimum roundness:", self._option_control(
            self.roundness_spin, self.option_defaults.roundness_min
        ))
        quality_form.addRow("Minimum detected stars:", self._option_control(
            self.stars_spin, self.option_defaults.stars_min
        ))
        quality_form.addRow("Registration minimum pairs:", self._option_control(
            self.min_pairs_spin, self.option_defaults.min_pairs_registration
        ))
        quality_form.addRow("Registration maximum stars:", self._option_control(
            self.max_stars_spin, self.option_defaults.max_stars_registration
        ))
        quality_form.addRow(self._option_control(
            self.solve_final_check, self.option_defaults.solve_final_images
        ))
        layout.addWidget(quality, 0, 1)

        integration = QGroupBox("Integration")
        integration_form = QFormLayout(integration)
        self.small_stack_spin = self._configured_int(
            MIN_LINEAR_STACK_COUNT, 10000, MIN_LINEAR_STACK_COUNT
        )
        self.large_stack_spin = self._configured_int(5, 10000, 50)
        self.rejection_maps_check = QCheckBox("Create low/high rejection maps")
        self.rejection_maps_check.setChecked(False)
        integration_form.addRow(
            "Linear-fit starts at frame count:", self._option_control(
                self.small_stack_spin, self.option_defaults.small_stack_threshold
            )
        )
        integration_form.addRow(
            "High-count Winsorized 4/3 at frame count:",
            self._option_control(
                self.large_stack_spin, self.option_defaults.large_stack_threshold
            ),
        )
        integration_form.addRow(self._option_control(
            self.rejection_maps_check, self.option_defaults.create_rejection_maps
        ))
        layout.addWidget(integration, 1, 0, 1, 2)

        drizzle = QGroupBox("Per-channel drizzle")
        drizzle_layout = QVBoxLayout(drizzle)
        drizzle_help = QLabel(
            "After scanning, check any combination of discovered channels. Unchecked "
            "channels remain native-scale; checked channels use Siril drizzle."
        )
        drizzle_help.setWordWrap(True)
        drizzle_layout.addWidget(drizzle_help)
        self.drizzle_list = QListWidget()
        self.drizzle_list.setMaximumHeight(150)
        self.drizzle_list.itemChanged.connect(self._sync_output_options)
        self.drizzle_list.itemChanged.connect(self._update_drizzle_default_visibility)
        drizzle_layout.addWidget(self.drizzle_list)
        preset_row = QHBoxLayout()
        for label, preset in (("None", "none"), ("All", "all"), ("SHO", "sho"), ("RGB", "rgb")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=preset: self._set_drizzle_preset(value))
            preset_row.addWidget(button)
        self.drizzle_default_button = QPushButton("Default")
        self.drizzle_default_button.setToolTip("Restore the default: no drizzled channels")
        self.drizzle_default_button.clicked.connect(
            lambda _checked=False: self._set_drizzle_preset("none")
        )
        self.drizzle_default_button.setHidden(True)
        preset_row.addWidget(self.drizzle_default_button)
        preset_row.addStretch(1)
        drizzle_layout.addLayout(preset_row)
        drizzle_form = QFormLayout()
        self.drizzle_scale_spin = self._configured_double(1.0, 3.0, 1.0, 2)
        self.drizzle_pixfrac_spin = self._configured_double(0.01, 1.0, 0.5, 2)
        self.drizzle_kernel_combo = QComboBox()
        for kernel in ("square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"):
            self.drizzle_kernel_combo.addItem(kernel, kernel)
        self.drizzle_scale_spin.valueChanged.connect(self._sync_output_options)
        self.drizzle_pixfrac_spin.valueChanged.connect(self._sync_output_options)
        self.drizzle_kernel_combo.currentIndexChanged.connect(self._sync_output_options)
        drizzle_form.addRow("Scale:", self._option_control(
            self.drizzle_scale_spin, self.option_defaults.drizzle_scale
        ))
        drizzle_form.addRow("Pixel fraction:", self._option_control(
            self.drizzle_pixfrac_spin, self.option_defaults.drizzle_pixfrac
        ))
        drizzle_form.addRow("Kernel:", self._option_control(
            self.drizzle_kernel_combo, self.option_defaults.drizzle_kernel
        ))
        drizzle_layout.addLayout(drizzle_form)
        layout.addWidget(drizzle, 2, 0)

        storage = QGroupBox("Workspace retention")
        storage_layout = QVBoxLayout(storage)
        self.rolling_cleanup_check = QCheckBox(
            "Delete completed process files as processing advances (recommended)"
        )
        self.rolling_cleanup_check.setChecked(True)
        self.rolling_cleanup_check.toggled.connect(self._rolling_cleanup_changed)
        storage_layout.addWidget(self._option_control(
            self.rolling_cleanup_check, self.option_defaults.rolling_cleanup
        ))
        self.keep_intermediates_check = QCheckBox(
            "When rolling cleanup is off, keep intermediates after success"
        )
        self.keep_intermediates_check.setChecked(True)
        self.keep_intermediates_check.setEnabled(False)
        self.keep_intermediates_check.toggled.connect(self._sync_output_options)
        storage_layout.addWidget(self._option_control(
            self.keep_intermediates_check, self.option_defaults.keep_intermediates
        ))
        storage_note = QLabel(
            "Rolling cleanup applies to ordinary stacks and mosaics. Raw-calibration "
            "workspaces, light staging, calibrated sequences, registered native/drizzle "
            "generations, and mosaic workspaces are released only after their dependent "
            "masters have been verified. Disable it to preserve process files for inspection."
        )
        storage_note.setWordWrap(True)
        storage_layout.addWidget(storage_note)
        storage_layout.addStretch(1)
        layout.addWidget(storage, 2, 1)

        mosaic = QGroupBox("Automatic mosaic assembly")
        mosaic_form = QFormLayout(mosaic)
        self.mosaic_enabled_check = QCheckBox(
            "Detect sky pointings and stitch one registered mosaic per filter"
        )
        self.mosaic_enabled_check.setChecked(False)
        self.mosaic_cluster_spin = self._configured_double(0.5, 50.0, 8.0, 1)
        self.mosaic_cluster_spin.setSuffix(" %")
        self.mosaic_feather_spin = self._configured_int(0, 4096, 100)
        self.mosaic_feather_spin.setSuffix(" px")
        self.mosaic_overlap_check = QCheckBox("Normalize panel intensity on overlaps")
        self.mosaic_overlap_check.setChecked(True)
        self.mosaic_gradient_check = QCheckBox(
            "Correct panel gradients with GraXpert before stitching"
        )
        self.mosaic_gradient_check.setChecked(True)
        self.graxpert_edit = QLineEdit()
        self.graxpert_edit.setPlaceholderText("Auto-detect, or select standalone GraXpert")
        self.graxpert_smoothing_spin = self._configured_double(0.0, 1.0, 0.5, 2)
        self.graxpert_gpu_check = QCheckBox("Use GPU for GraXpert")
        self.graxpert_gpu_check.setChecked(True)
        self.mosaic_background_match_check = QCheckBox(
            "Match additive background levels from panel overlaps"
        )
        self.mosaic_background_match_check.setChecked(True)
        self.mosaic_downsample_check = QCheckBox("Downsample during panel plate solving")
        self.mosaic_downsample_check.setChecked(True)
        self.mosaic_blind_fallback_check = QCheckBox(
            "Try local Astrometry.net blind solve if header-guided solving fails"
        )
        self.mosaic_blind_fallback_check.setChecked(True)
        self.mosaic_enabled_check.toggled.connect(self._mosaic_detection_changed)
        self.mosaic_cluster_spin.valueChanged.connect(self._mosaic_detection_changed)
        self.mosaic_feather_spin.valueChanged.connect(self._sync_output_options)
        self.mosaic_overlap_check.toggled.connect(self._sync_output_options)
        self.mosaic_gradient_check.toggled.connect(self._sync_output_options)
        self.graxpert_edit.textChanged.connect(self._sync_output_options)
        self.graxpert_edit.editingFinished.connect(self._graxpert_location_edited)
        self.graxpert_smoothing_spin.valueChanged.connect(self._sync_output_options)
        self.graxpert_gpu_check.toggled.connect(self._sync_output_options)
        self.mosaic_background_match_check.toggled.connect(self._sync_output_options)
        self.mosaic_downsample_check.toggled.connect(self._sync_output_options)
        self.mosaic_blind_fallback_check.toggled.connect(self._sync_output_options)
        mosaic_form.addRow(self._option_control(
            self.mosaic_enabled_check, self.option_defaults.mosaic_enabled
        ))
        mosaic_form.addRow(
            "Pointing cluster tolerance (short FOV):", self._option_control(
                self.mosaic_cluster_spin, self.option_defaults.mosaic_cluster_percent
            )
        )
        mosaic_form.addRow("Border feathering:", self._option_control(
            self.mosaic_feather_spin, self.option_defaults.mosaic_feather
        ))
        mosaic_form.addRow(self._option_control(
            self.mosaic_overlap_check, self.option_defaults.mosaic_overlap_normalization
        ))
        mosaic_form.addRow(self._option_control(
            self.mosaic_gradient_check, self.option_defaults.mosaic_gradient_correction
        ))
        mosaic_form.addRow(
            "GraXpert executable:",
            self._option_control(
                self.graxpert_edit,
                self.option_defaults.graxpert_executable,
                self._browse_graxpert,
            ),
        )
        mosaic_form.addRow("GraXpert smoothing:", self._option_control(
            self.graxpert_smoothing_spin, self.option_defaults.graxpert_smoothing
        ))
        mosaic_form.addRow(self._option_control(
            self.graxpert_gpu_check, self.option_defaults.graxpert_gpu
        ))
        mosaic_form.addRow(self._option_control(
            self.mosaic_background_match_check,
            self.option_defaults.mosaic_background_matching,
        ))
        mosaic_form.addRow(self._option_control(
            self.mosaic_downsample_check, self.option_defaults.mosaic_downsample_solve
        ))
        mosaic_form.addRow(self._option_control(
            self.mosaic_blind_fallback_check,
            self.option_defaults.mosaic_blind_fallback,
        ))
        layout.addWidget(mosaic, 3, 0, 1, 2)

        merge = QGroupBox("Multi-scale master merge")
        merge_form = QFormLayout(merge)
        merge_note = QLabel(
            "Used automatically when the input folder holds only finished master "
            "lights from telescopes with different focal lengths. The finest data "
            "supplies the detail inside its field; wider data extends the field and "
            "lowers large-scale noise."
        )
        merge_note.setWordWrap(True)
        merge_form.addRow(merge_note)
        self.merge_blend_check = QCheckBox(
            "Blend wide-field data into large scales inside the detailed field"
        )
        self.merge_blend_check.setChecked(True)
        self.merge_transition_spin = self._configured_int(0, 8192, 0)
        self.merge_transition_spin.setSuffix(" px")
        self.merge_transition_spin.setSpecialValueText("Automatic")
        self.merge_detail_spin = self._configured_double(0.0, 64.0, 0.0, 1)
        self.merge_detail_spin.setSuffix(" px")
        self.merge_detail_spin.setSpecialValueText("Automatic")
        self.merge_grid_combo = QComboBox()
        self.merge_grid_combo.addItem("Automatic (keep the most field)", "auto")
        self.merge_grid_combo.addItem("Finest layer's pixel grid", "finest")
        self.merge_grid_combo.addItem("Widest layer's orientation", "widest")
        self.merge_blend_check.toggled.connect(self._sync_output_options)
        self.merge_transition_spin.valueChanged.connect(self._sync_output_options)
        self.merge_detail_spin.valueChanged.connect(self._sync_output_options)
        self.merge_grid_combo.currentIndexChanged.connect(self._sync_output_options)
        merge_form.addRow(self._option_control(
            self.merge_blend_check, self.option_defaults.merge_blend_large_scales
        ))
        merge_form.addRow("Transition width:", self._option_control(
            self.merge_transition_spin, self.option_defaults.merge_transition_pixels
        ))
        merge_form.addRow("Detail split scale (sigma):", self._option_control(
            self.merge_detail_spin, self.option_defaults.merge_detail_sigma
        ))
        merge_form.addRow("Output grid:", self._option_control(
            self.merge_grid_combo, self.option_defaults.merge_output_grid
        ))
        layout.addWidget(merge, 4, 0, 1, 2)
        layout.setRowStretch(5, 1)
        return tab

    def _rolling_cleanup_changed(self, checked: bool) -> None:
        self.keep_intermediates_check.setEnabled(not checked)
        self._sync_output_options()

    def _build_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)
        buttons = QHBoxLayout()
        open_session_logs = QPushButton("Open session logs")
        open_session_logs.setToolTip(
            "Every SBPP session is logged here, including scans and crashes."
        )
        open_session_logs.clicked.connect(
            lambda: self._open_folder(self.session_log.directory)
        )
        self.open_run_folder_button = QPushButton("Open last run folder")
        self.open_run_folder_button.setToolTip(
            "Contains sbpp.log, manifest.json, commands.ssf and any diagnostics."
        )
        self.open_run_folder_button.setEnabled(False)
        self.open_run_folder_button.clicked.connect(
            lambda: self.last_run_dir and self._open_folder(self.last_run_dir)
        )
        buttons.addWidget(open_session_logs)
        buttons.addWidget(self.open_run_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return tab

    def record_unexpected_error(self, text: str) -> None:
        self._append_log(text)
        try:
            self.siril.log(f"SBPP: {text.splitlines()[-1]}")
        except Exception:  # Siril may already be gone
            pass

    def _open_folder(self, folder: Path) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        folder.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            QMessageBox.information(self, APP_NAME, f"The folder is:\n{folder}")

    def _populate_drizzle_channels(self, plan: ProcessingPlan) -> None:
        self._updating_drizzle = True
        self.drizzle_list.clear()
        selected = set(plan.drizzle_channels)
        for dataset in plan.datasets:
            for filter_name in dataset.channels:
                item = QListWidgetItem(f"{dataset.dataset_id} — {filter_name}")
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    {
                        "key": channel_key(dataset.dataset_id, filter_name),
                        "filter": filter_name,
                    },
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if channel_key(dataset.dataset_id, filter_name) in selected
                    else Qt.CheckState.Unchecked
                )
                self.drizzle_list.addItem(item)
        self._updating_drizzle = False
        self._update_drizzle_default_visibility()

    def _update_drizzle_default_visibility(self, *_args: Any) -> None:
        any_selected = any(
            self.drizzle_list.item(index).checkState() == Qt.CheckState.Checked
            for index in range(self.drizzle_list.count())
        )
        self.drizzle_default_button.setHidden(not any_selected)

    def _set_drizzle_preset(self, preset: str) -> None:
        self._updating_drizzle = True
        sho = {"H", "HA", "HALPHA", "S", "SII", "O", "OIII"}
        rgb = {"R", "RED", "G", "GREEN", "B", "BLUE"}
        for index in range(self.drizzle_list.count()):
            item = self.drizzle_list.item(index)
            payload = item.data(Qt.ItemDataRole.UserRole) or {}
            name = normalized_text(payload.get("filter", ""))
            checked = (
                preset == "all"
                or (preset == "sho" and name in sho)
                or (preset == "rgb" and name in rgb)
            )
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self._updating_drizzle = False
        self._update_drizzle_default_visibility()
        self._sync_output_options()

    def _mosaic_detection_changed(self, *_args: Any) -> None:
        """Require a fresh header plan when pointing-cluster inputs change."""
        if self._loading_profile or self.current_plan is None:
            return
        self.current_plan = None
        self._refresh_panel_map()
        self.plan_table.setRowCount(0)
        self.plan_warnings.clear()
        self.drizzle_list.clear()
        self._update_drizzle_default_visibility()
        self.plan_summary.setText(
            "Mosaic detection settings changed. Select “Scan and build plan” to "
            "recompute sky-pointing clusters."
        )
        self.save_plan_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.statusBar().showMessage(
            "Mosaic settings changed; rebuild the calibration plan."
        )

    def _sync_output_options(self, *_args: Any) -> None:
        if self._updating_drizzle or self._loading_profile or self.current_plan is None:
            return
        selected: list[str] = []
        for index in range(self.drizzle_list.count()):
            item = self.drizzle_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                payload = item.data(Qt.ItemDataRole.UserRole) or {}
                if payload.get("key"):
                    selected.append(str(payload["key"]))
        self.current_plan.drizzle_channels = selected
        self.current_plan.options = self._current_options()
        self.current_plan.warnings = [
            message for message in self.current_plan.warnings
            if not message.startswith("BLOCKING: Lanczos drizzle kernels")
            and not message.startswith("Partial CFA drizzle:")
        ]
        if (
            selected
            and self.current_plan.options.drizzle_kernel.startswith("lanczos")
            and (
                abs(self.current_plan.options.drizzle_scale - 1.0) > 1e-6
                or abs(self.current_plan.options.drizzle_pixfrac - 1.0) > 1e-6
            )
        ):
            self.current_plan.warnings.append(
                "BLOCKING: Lanczos drizzle kernels require scale 1.0 and pixel fraction 1.0; "
                "use square or Gaussian for upscaled drizzle."
            )
        selected_set = set(selected)
        for dataset in self.current_plan.datasets:
            keys = {channel_key(dataset.dataset_id, name) for name in dataset.channels}
            representatives = [
                groups[0].representative
                for groups in dataset.channels.values() if groups
            ]
            if (
                keys & selected_set
                and keys - selected_set
                and representatives
                and all(frame.bayer for frame in representatives)
            ):
                self.current_plan.warnings.append(
                    f"Partial CFA drizzle: {dataset.dataset_id} will be debayered before both "
                    "output modes; select every channel in that setup for true Bayer drizzle."
                )
        refresh_plan_storage(self.current_plan)
        self._display_plan(self.current_plan)
        self.start_button.setEnabled(self.current_plan.valid)

    def _populate_profiles(self) -> None:
        current = self.profile_combo.currentText()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(self.store.profiles, key=str.lower))
        index = self.profile_combo.findText(current)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)

    def _profile_changed(self, name: str) -> None:
        if not self._loading_profile and name:
            previous = self._active_profile_name
            if previous and previous != name:
                self._save_profile(
                    profile_name=previous, announce=False, select_profile=False
                )
            self._load_profile(name)
            self.store.save()

    def _load_profile(self, name: str) -> None:
        if not name:
            return
        self._loading_profile = True
        profile = self.store.profile(name)
        self.calibration_list.clear()
        self.calibration_list.addItems(profile.get("calibration_roots", []))
        options = PlanOptions(**{
            **asdict(PlanOptions()),
            **profile.get("options", {}),
            "graxpert_executable": self.store.graxpert_executable,
        })
        self._set_option_widgets(options)
        self.input_edit.setText(profile.get("input_root", ""))
        self.output_edit.setText(profile.get("output_root", ""))
        self._auto_inferred_target = ""
        self.target_edit.setText(profile.get("target_name", ""))
        self.store.payload["selected_profile"] = name
        self._active_profile_name = name
        self.delete_profile_button.setEnabled(name != "Default")
        self.current_plan = None
        self._refresh_panel_map()
        self.drizzle_list.clear()
        self._update_drizzle_default_visibility()
        self.start_button.setEnabled(False)
        self.save_plan_button.setEnabled(False)
        self._loading_profile = False

    def _current_options(self) -> PlanOptions:
        return PlanOptions(
            recursive=self.recursive_check.isChecked(),
            prefer_masters=self.prefer_masters_check.isChecked(),
            minimum_calibration_frames=self.minimum_cal_spin.value(),
            dark_policy=str(self.dark_policy_combo.currentData()),
            exposure_tolerance=self.exposure_tolerance_spin.value(),
            temperature_tolerance=self.temperature_tolerance_spin.value(),
            fwhm_max=self.fwhm_spin.value(),
            roundness_min=self.roundness_spin.value(),
            stars_min=self.stars_spin.value(),
            small_stack_threshold=self.small_stack_spin.value(),
            large_stack_threshold=self.large_stack_spin.value(),
            max_stars_registration=self.max_stars_spin.value(),
            min_pairs_registration=self.min_pairs_spin.value(),
            solve_final_images=self.solve_final_check.isChecked(),
            create_rejection_maps=self.rejection_maps_check.isChecked(),
            debayer_cfa=self.debayer_check.isChecked(),
            drizzle_scale=self.drizzle_scale_spin.value(),
            drizzle_pixfrac=self.drizzle_pixfrac_spin.value(),
            drizzle_kernel=str(self.drizzle_kernel_combo.currentData()),
            keep_intermediates=self.keep_intermediates_check.isChecked(),
            rolling_cleanup=self.rolling_cleanup_check.isChecked(),
            mosaic_enabled=self.mosaic_enabled_check.isChecked(),
            mosaic_cluster_percent=self.mosaic_cluster_spin.value(),
            mosaic_feather=self.mosaic_feather_spin.value(),
            mosaic_overlap_normalization=self.mosaic_overlap_check.isChecked(),
            mosaic_gradient_correction=self.mosaic_gradient_check.isChecked(),
            graxpert_executable=self.graxpert_edit.text().strip(),
            graxpert_smoothing=self.graxpert_smoothing_spin.value(),
            graxpert_gpu=self.graxpert_gpu_check.isChecked(),
            mosaic_background_matching=self.mosaic_background_match_check.isChecked(),
            mosaic_downsample_solve=self.mosaic_downsample_check.isChecked(),
            mosaic_blind_fallback=self.mosaic_blind_fallback_check.isChecked(),
            merge_blend_large_scales=self.merge_blend_check.isChecked(),
            merge_transition_pixels=self.merge_transition_spin.value(),
            merge_detail_sigma=self.merge_detail_spin.value(),
            merge_output_grid=str(self.merge_grid_combo.currentData()),
        )

    def _set_option_widgets(self, options: PlanOptions) -> None:
        self.recursive_check.setChecked(options.recursive)
        self.prefer_masters_check.setChecked(options.prefer_masters)
        self.minimum_cal_spin.setValue(options.minimum_calibration_frames)
        index = self.dark_policy_combo.findData(options.dark_policy)
        self.dark_policy_combo.setCurrentIndex(max(0, index))
        self.exposure_tolerance_spin.setValue(options.exposure_tolerance)
        self.temperature_tolerance_spin.setValue(options.temperature_tolerance)
        self.fwhm_spin.setValue(options.fwhm_max)
        self.roundness_spin.setValue(options.roundness_min)
        self.stars_spin.setValue(options.stars_min)
        self.small_stack_spin.setValue(options.small_stack_threshold)
        self.large_stack_spin.setValue(options.large_stack_threshold)
        self.max_stars_spin.setValue(options.max_stars_registration)
        self.min_pairs_spin.setValue(options.min_pairs_registration)
        self.solve_final_check.setChecked(options.solve_final_images)
        self.rejection_maps_check.setChecked(options.create_rejection_maps)
        self.debayer_check.setChecked(options.debayer_cfa)
        self.drizzle_scale_spin.setValue(options.drizzle_scale)
        self.drizzle_pixfrac_spin.setValue(options.drizzle_pixfrac)
        index = self.drizzle_kernel_combo.findData(options.drizzle_kernel)
        self.drizzle_kernel_combo.setCurrentIndex(max(0, index))
        self.keep_intermediates_check.setChecked(options.keep_intermediates)
        self.rolling_cleanup_check.setChecked(options.rolling_cleanup)
        self.keep_intermediates_check.setEnabled(not options.rolling_cleanup)
        self.mosaic_enabled_check.setChecked(options.mosaic_enabled)
        self.mosaic_cluster_spin.setValue(options.mosaic_cluster_percent)
        self.mosaic_feather_spin.setValue(options.mosaic_feather)
        self.mosaic_overlap_check.setChecked(options.mosaic_overlap_normalization)
        self.mosaic_gradient_check.setChecked(options.mosaic_gradient_correction)
        self.graxpert_edit.setText(options.graxpert_executable)
        self.graxpert_smoothing_spin.setValue(options.graxpert_smoothing)
        self.graxpert_gpu_check.setChecked(options.graxpert_gpu)
        self.mosaic_background_match_check.setChecked(
            options.mosaic_background_matching
        )
        self.mosaic_downsample_check.setChecked(options.mosaic_downsample_solve)
        self.mosaic_blind_fallback_check.setChecked(options.mosaic_blind_fallback)
        self.merge_blend_check.setChecked(options.merge_blend_large_scales)
        self.merge_transition_spin.setValue(options.merge_transition_pixels)
        self.merge_detail_spin.setValue(options.merge_detail_sigma)
        index = self.merge_grid_combo.findData(options.merge_output_grid)
        self.merge_grid_combo.setCurrentIndex(max(0, index))

    def _selected_drizzle_channels(self) -> list[str]:
        selected: list[str] = []
        for index in range(self.drizzle_list.count()):
            item = self.drizzle_list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            payload = item.data(Qt.ItemDataRole.UserRole) or {}
            if payload.get("key"):
                selected.append(str(payload["key"]))
        return selected

    def _save_profile(
        self,
        _checked: bool = False,
        *,
        profile_name: Optional[str] = None,
        announce: bool = True,
        select_profile: bool = True,
    ) -> None:
        del _checked
        name = profile_name or self.profile_combo.currentText() or "Default"
        profile = self.store.profile(name)
        profile["target_name"] = self.target_edit.text().strip()
        profile["input_root"] = self.input_edit.text().strip()
        profile["output_root"] = self.output_edit.text().strip()
        profile["calibration_roots"] = [
            self.calibration_list.item(index).text()
            for index in range(self.calibration_list.count())
        ]
        profile["options"] = asdict(self._current_options())
        self.store.graxpert_executable = self.graxpert_edit.text()
        if self.current_plan is not None:
            profile["drizzle_channels"] = self._selected_drizzle_channels()
        else:
            profile.setdefault("drizzle_channels", [])
        if select_profile:
            self.store.payload["selected_profile"] = name
        self.store.save()
        if announce:
            self._append_log(
                f"Saved target profile “{name}” to {self.store.settings_path}"
            )

    def _new_profile(self) -> None:
        from PyQt6.QtWidgets import QInputDialog  # local import keeps the main list compact
        suggested = self.target_edit.text().strip()
        name, accepted = QInputDialog.getText(
            self,
            APP_NAME,
            "New target profile name:",
            QLineEdit.EchoMode.Normal,
            suggested,
        )
        name = name.strip()
        if not accepted or not name:
            return
        if name in self.store.profiles:
            QMessageBox.warning(self, APP_NAME, "A profile with that name already exists.")
            return
        source = self.store.profile(
            self._active_profile_name or self.profile_combo.currentText() or "Default"
        )
        drizzle_channels = (
            self._selected_drizzle_channels()
            if self.current_plan is not None
            else list(source.get("drizzle_channels", []))
        )
        self.store.profiles[name] = {
            "target_name": self.target_edit.text().strip(),
            "input_root": self.input_edit.text().strip(),
            "output_root": self.output_edit.text().strip(),
            "calibration_roots": [
                self.calibration_list.item(index).text()
                for index in range(self.calibration_list.count())
            ],
            "drizzle_channels": drizzle_channels,
            "options": asdict(self._current_options()),
        }
        self.store.payload["selected_profile"] = name
        self.store.save()
        self._populate_profiles()
        # This is Save As: the current form belongs to the new profile and
        # should not also overwrite the previously selected profile.
        self._active_profile_name = ""
        self.profile_combo.setCurrentText(name)
        self._append_log(
            f"Created target profile “{name}” from the current setup."
        )

    def _delete_profile(self) -> None:
        name = self.profile_combo.currentText()
        if name == "Default":
            return
        answer = QMessageBox.question(
            self, APP_NAME, f"Delete target profile “{name}”?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.delete_profile(name)
            self._active_profile_name = ""
            self._populate_profiles()
            self.profile_combo.setCurrentText("Default")

    def _target_name_edited(self, _text: str) -> None:
        """A manual edit takes ownership away from automatic inference."""
        self._auto_inferred_target = ""

    def _infer_target_from_input(self) -> None:
        """Quickly auto-fill a blank target from three input-light headers."""
        root = Path(self.input_edit.text().strip())
        if not root.exists():
            return
        current = self.target_edit.text().strip()
        if current and current != self._auto_inferred_target:
            return

        scanner = HeaderScanner(self.cache)
        samples: list[FrameMeta] = []
        headers_read = 0
        for path in scanner.iter_files(root, self.recursive_check.isChecked()):
            if file_suffix(path) not in FITS_SUFFIXES + XISF_SUFFIXES:
                continue
            headers_read += 1
            try:
                frame = scanner.read(path, "input")
            except Exception:
                if headers_read >= TARGET_INFERENCE_HEADER_LIMIT:
                    break
                continue
            if (
                frame.frame_type == FRAME_LIGHT
                and not frame.is_master
                and meaningful_object_name(frame.object_name)
            ):
                samples.append(frame)
                if len(samples) >= 3:
                    break
            if headers_read >= TARGET_INFERENCE_HEADER_LIMIT:
                break
        if headers_read:
            self.cache.save()

        inferred = infer_target_name_from_headers(samples)
        if inferred:
            self._auto_inferred_target = inferred
            self.target_edit.setText(inferred)
            self.statusBar().showMessage(
                f"Target inferred from light OBJECT headers: {inferred}", 5000
            )
        elif current == self._auto_inferred_target:
            self._auto_inferred_target = ""
            self.target_edit.clear()

    def _browse_directory(self, title: str, current: str) -> str:
        start = current if Path(current).is_dir() else str(Path.home())
        return QFileDialog.getExistingDirectory(self, title, start)

    def _browse_input(self) -> None:
        selected = self._browse_directory("Select input/project root", self.input_edit.text())
        if selected:
            self.input_edit.setText(selected)
            self._infer_target_from_input()

    def _browse_output(self) -> None:
        selected = self._browse_directory("Select output root", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _graxpert_location_edited(self) -> None:
        if self._loading_profile:
            return
        self.store.graxpert_executable = self.graxpert_edit.text()
        self.store.save()

    def _browse_graxpert(self) -> None:
        current = Path(os.path.expanduser(self.graxpert_edit.text().strip()))
        start = str(current if current.is_file() else current.parent)
        if not Path(start).is_dir():
            start = str(Path.home())
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select standalone GraXpert executable",
            start,
            "Applications (*.exe *.AppImage);;All files (*)",
        )
        if selected:
            self.graxpert_edit.setText(selected)
            self.store.graxpert_executable = selected
            self.store.save()

    def _add_calibration_root(self) -> None:
        selected = self._browse_directory("Add calibration library", "")
        if not selected:
            return
        existing = {
            self.calibration_list.item(index).text()
            for index in range(self.calibration_list.count())
        }
        if selected not in existing:
            self.calibration_list.addItem(selected)

    def _remove_calibration_root(self) -> None:
        for item in self.calibration_list.selectedItems():
            self.calibration_list.takeItem(self.calibration_list.row(item))

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{stamp}] {message}")
        self.session_log.write(message)

    def _update_elapsed_display(self) -> None:
        if self.run_started_monotonic is None:
            return
        elapsed = time.monotonic() - self.run_started_monotonic
        self.elapsed_label.setText(f"Elapsed {human_duration(elapsed)}")

    def _scan(self) -> None:
        input_root = Path(self.input_edit.text().strip())
        output_root = Path(self.output_edit.text().strip())
        calibration_roots = [
            Path(self.calibration_list.item(index).text())
            for index in range(self.calibration_list.count())
        ]
        if not input_root.is_dir():
            QMessageBox.warning(self, APP_NAME, "Select an existing input directory.")
            return
        if not output_root.is_dir():
            QMessageBox.warning(self, APP_NAME, "Select an existing output directory.")
            return
        missing = [str(path) for path in calibration_roots if not path.is_dir()]
        if missing:
            QMessageBox.warning(self, APP_NAME, "Missing calibration folders:\n" + "\n".join(missing))
            return

        self.scan_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.save_plan_button.setEnabled(False)
        self.progress_bar.setValue(0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._append_log("Scanning FITS/XISF headers and calibration libraries…")
        QApplication.processEvents()
        try:
            scanner = HeaderScanner(self.cache)

            def update_scan(percent: int, name: str) -> None:
                self.progress_bar.setValue(percent)
                self.statusBar().showMessage(f"Scanning {name}")
                QApplication.processEvents()

            frames, errors = scanner.scan(
                input_root,
                calibration_roots,
                recursive=self.recursive_check.isChecked(),
                progress=update_scan,
            )
            inventory = Counter(
                (item.frame_type, item.is_master) for item in frames
                if item.frame_type in {
                    FRAME_BIAS, FRAME_DARK, FRAME_FLAT, FRAME_DARKFLAT
                }
            )
            inventory_parts = []
            for kind in (FRAME_BIAS, FRAME_DARK, FRAME_FLAT, FRAME_DARKFLAT):
                raw_count = inventory[(kind, False)]
                master_count = inventory[(kind, True)]
                inventory_parts.append(
                    f"{kind.lower()} {raw_count} raw/{master_count} master"
                )
            self._append_log("Calibration inventory: " + "; ".join(inventory_parts) + ".")
            target_name = self.target_edit.text().strip()
            target_is_automatic = bool(
                self._auto_inferred_target
                and target_name == self._auto_inferred_target
            )
            if not target_name or target_is_automatic:
                inferred_target = infer_target_name_from_headers(frames)
                if inferred_target:
                    target_name = inferred_target
                    self._auto_inferred_target = inferred_target
                    self.target_edit.setText(inferred_target)
                    self._append_log(
                        f"Inferred target “{inferred_target}” from agreeing light "
                        "OBJECT headers."
                    )
                else:
                    target_name = ""
                    self._auto_inferred_target = ""
                    self.target_edit.clear()
                    self._append_log(
                        "Target name left blank because fewer than three usable light "
                        "OBJECT headers were found or their values disagreed. Enter it "
                        "manually and rebuild the plan."
                    )
            self.current_plan = build_plan(
                frames,
                input_root,
                calibration_roots,
                output_root,
                target_name,
                self._current_options(),
                errors,
            )
            available_channels = {
                channel_key(dataset.dataset_id, filter_name)
                for dataset in self.current_plan.datasets
                for filter_name in dataset.channels
            }
            profile = self.store.profile(
                self._active_profile_name
                or self.profile_combo.currentText()
                or "Default"
            )
            self.current_plan.drizzle_channels = [
                key for key in profile.get("drizzle_channels", [])
                if key in available_channels
            ]
            refresh_plan_storage(self.current_plan)
            self._populate_drizzle_channels(self.current_plan)
            self._display_plan(self.current_plan)
            self._refresh_panel_map()
            if self.panel_layout_button.isEnabled():
                self._show_panel_map(activate=False)
            self._save_profile()
            self.save_plan_button.setEnabled(True)
            self.start_button.setEnabled(self.current_plan.valid)
            self.tabs.setCurrentIndex(1)
            if self.current_plan.master_merge is not None:
                merge = self.current_plan.master_merge
                self._append_log(
                    f"Only master lights were found: planned a multi-scale merge of "
                    f"{len(merge.masters)} masters in {merge.layer_count} plate-scale layer(s)."
                )
            self._append_log(
                f"Plan contains {self.current_plan.light_count} lights in "
                f"{len(self.current_plan.datasets)} compatible setup(s) and "
                f"{sum(dataset.panel_count for dataset in self.current_plan.datasets)} panel(s), "
                f"with {integration_summary(self.current_plan.integration_seconds, self.current_plan.unknown_exposure_count)} "
                "of planned integration."
            )
            if self.current_plan.options.mosaic_enabled:
                for dataset in self.current_plan.datasets:
                    self._append_log(
                        f"{dataset.dataset_id} panel detection: {dataset.panel_detection}"
                    )
        except Exception as exc:
            self.current_plan = None
            self._refresh_panel_map()
            QMessageBox.critical(self, APP_NAME, f"Scan failed:\n{exc}")
            self._append_log(traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            self.scan_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.statusBar().clearMessage()

    def _display_merge_plan(self, plan: ProcessingPlan) -> None:
        merge = plan.master_merge
        assert merge is not None
        application = QApplication.instance()
        dark_theme = bool(
            application and application.property("sbppTheme") == "dark"
        )
        warnings = [*plan.warnings, *merge.warnings]
        blocked = not plan.valid
        self.plan_table.setRowCount(len(merge.masters))
        for row, entry in enumerate(merge.masters):
            if blocked:
                status = "BLOCKED"
                color = QColor("#5a2424" if dark_theme else "#ffd1d1")
                status_text = QColor("#ffdede" if dark_theme else "#000000")
            else:
                status = "Ready"
                color = QColor("#244c2a" if dark_theme else "#d8f5d0")
                status_text = QColor("#dcf7df" if dark_theme else "#000000")
            scale = (
                f"{entry.scale_arcsec:.3f}\"/px" if entry.scale_arcsec else "solve"
            )
            values = [
                "master merge",
                f"layer {entry.layer} ({scale})",
                entry.filter_name,
                "master",
                f"{display_number(entry.exposure, 3)} s",
                Path(entry.path).name,
                "—",
                "—",
                entry.telescope or "—",
                f"{display_number(entry.focal_length_mm)} mm",
                f"{entry.width or '?'}x{entry.height or '?'}",
                "—",
                "—",
                entry.role,
                status,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 14:
                    item.setBackground(color)
                    item.setForeground(status_text)
                self.plan_table.setItem(row, column, item)
        warnings.extend(f"Scan error: {message}" for message in plan.scan_errors[:50])
        self.plan_warnings.setPlainText("\n".join(warnings) if warnings else "No warnings.")
        width, height = merge.estimated_canvas
        self.plan_summary.setText(
            f"Master merge • {len(merge.masters)} masters in {merge.layer_count} plate-scale "
            f"layer(s) • {len(merge.filters)} filter(s) • output canvas ≈ {width}x{height} "
            f"before cropping • peak workspace {human_bytes(plan.estimated_bytes)} • "
            f"free {human_bytes(plan.free_bytes)} • {'READY' if plan.valid else 'BLOCKED'}"
        )

    def _display_plan(self, plan: ProcessingPlan) -> None:
        if plan.master_merge is not None:
            self._display_merge_plan(plan)
            return
        rows = [
            (dataset, subgroup)
            for dataset in plan.datasets
            for groups in dataset.channels.values()
            for subgroup in groups
        ]
        self.plan_table.setRowCount(len(rows))
        warnings = list(plan.warnings)
        for dataset in plan.datasets:
            warnings.extend(f"{dataset.dataset_id}: {message}" for message in dataset.warnings)
        application = QApplication.instance()
        dark_theme = bool(
            application and application.property("sbppTheme") == "dark"
        )
        for row, (dataset, subgroup) in enumerate(rows):
            sample = subgroup.representative
            status = "Ready"
            color = QColor("#244c2a" if dark_theme else "#d8f5d0")
            status_text = QColor("#dcf7df" if dark_theme else "#000000")
            subgroup_blocked = any(
                message.startswith("BLOCKING:") for message in subgroup.warnings
            )
            if not subgroup.flat.available or subgroup_blocked:
                status = "BLOCKED"
                color = QColor("#5a2424" if dark_theme else "#ffd1d1")
                status_text = QColor("#ffdede" if dark_theme else "#000000")
            elif subgroup.warnings:
                status = "Ready with warnings"
                color = QColor("#59491e" if dark_theme else "#fff0bf")
                status_text = QColor("#fff0bf" if dark_theme else "#000000")
            values = [
                dataset.dataset_id,
                subgroup.panel_id,
                subgroup.filter_name,
                str(len(subgroup.frames)),
                f"{display_number(sample.exposure, 3)} s",
                integration_summary(
                    subgroup.integration_seconds, subgroup.unknown_exposure_count
                ),
                f"{display_number(sample.gain)}/{display_number(sample.offset)}",
                f"{display_number(sample.effective_temperature)} °C",
                subgroup.bias.description,
                subgroup.dark.description,
                subgroup.flat.description,
                subgroup.darkflat.description,
                subgroup.bpm.description,
                (
                    f"Drizzle {plan.options.drizzle_scale:g}x"
                    if plan.channel_uses_drizzle(dataset, subgroup.filter_name)
                    else "Native"
                ) + (
                    " → mosaic"
                    if dataset.panel_count > 1 and plan.options.mosaic_enabled
                    else ""
                ),
                status,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 14:
                    item.setBackground(color)
                    item.setForeground(status_text)
                self.plan_table.setItem(row, column, item)
            warnings.extend(
                f"{dataset.dataset_id}/{subgroup.filter_name}: {message}"
                for message in subgroup.warnings
            )
        warnings.extend(f"Scan error: {message}" for message in plan.scan_errors[:50])
        if len(plan.scan_errors) > 50:
            warnings.append(f"…and {len(plan.scan_errors) - 50} additional scan errors.")
        self.plan_warnings.setPlainText("\n".join(warnings) if warnings else "No warnings.")
        self.plan_summary.setText(
            f"{plan.light_count} lights • {len(rows)} calibration subgroup(s) • "
            f"planned integration {integration_summary(plan.integration_seconds, plan.unknown_exposure_count)} • "
            f"{sum(dataset.panel_count for dataset in plan.datasets)} panel(s) • "
            f"{len(plan.datasets)} setup(s) • peak workspace {human_bytes(plan.estimated_bytes)} • "
            f"retained after success {human_bytes(plan.estimated_final_bytes)} • "
            f"rolling cleanup {'on' if plan.options.rolling_cleanup else 'off'} • "
            f"free {human_bytes(plan.free_bytes)} • {'READY' if plan.valid else 'BLOCKED'}"
        )

    def _save_dry_run(self) -> None:
        if self.current_plan is None:
            return
        default = Path(self.current_plan.output_root) / f"{self.current_plan.output_prefix}_SBPP_dry_run.json"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save dry-run plan", str(default), "JSON files (*.json)"
        )
        if filename:
            atomic_write_json(Path(filename), {
                "schema_version": SCHEMA_VERSION,
                "application": APP_NAME,
                "application_version": VERSION,
                "created_utc": utc_now(),
                "plan": self.current_plan.to_dict(),
            })
            self._append_log(f"Saved dry-run plan to {filename}")

    def _start_processing(self) -> None:
        self._sync_output_options()
        if self.current_plan is None or not self.current_plan.valid:
            QMessageBox.warning(self, APP_NAME, "Build a valid calibration plan first.")
            return
        merge = self.current_plan.master_merge
        if merge is not None:
            prompt = (
                f"Merge {len(merge.masters)} masters from {merge.layer_count} plate-scale "
                f"layers into {len(merge.filters)} filter master(s)?\n\n"
                f"Peak workspace: {human_bytes(self.current_plan.estimated_bytes)}\n"
                "A new timestamped run folder will be created; existing data will not be "
                "overwritten."
            )
        else:
            prompt = ""
        answer = QMessageBox.question(
            self,
            APP_NAME,
            prompt or f"Process {self.current_plan.light_count} lights?\n\n"
            f"Planned integration: "
            f"{integration_summary(self.current_plan.integration_seconds, self.current_plan.unknown_exposure_count)}\n"
            f"Peak workspace: {human_bytes(self.current_plan.estimated_bytes)}\n"
            f"Retained after success: {human_bytes(self.current_plan.estimated_final_bytes)}\n"
            f"Drizzled channels: "
            f"{', '.join(self.current_plan.drizzle_channels) or 'none'}\n"
            f"Mosaic assembly: "
            f"{'automatic' if self.current_plan.options.mosaic_enabled else 'disabled'}\n"
            f"Rolling cleanup: "
            f"{'enabled' if self.current_plan.options.rolling_cleanup else 'disabled'}\n"
            "A new timestamped run folder will be created; existing data will not be overwritten.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._save_profile()
        self.tabs.setCurrentIndex(3)
        self.progress_bar.setValue(0)
        self.scan_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.save_plan_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_started_monotonic = time.monotonic()
        self.last_run_elapsed = 0.0
        self.elapsed_label.setText("Elapsed 00:00:00")
        self.elapsed_timer.start(1000)
        self._append_log("Processing timer started.")
        self.pipeline_thread = PipelineThread(self.siril, self.current_plan)
        self.pipeline_thread.log_line.connect(self._append_log)
        self.pipeline_thread.progress_value.connect(self.progress_bar.setValue)
        self.pipeline_thread.succeeded.connect(self._processing_succeeded)
        self.pipeline_thread.failed.connect(self._processing_failed)
        self.pipeline_thread.cancelled.connect(self._processing_cancelled)
        self.pipeline_thread.start()

    def _stop_processing(self) -> None:
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            self.pipeline_thread.request_cancel()
            self.stop_button.setEnabled(False)

    def _reset_after_processing(self) -> None:
        if self.run_started_monotonic is not None:
            self.last_run_elapsed = time.monotonic() - self.run_started_monotonic
            self.elapsed_label.setText(
                f"Elapsed {human_duration(self.last_run_elapsed)}"
            )
        self.elapsed_timer.stop()
        self.run_started_monotonic = None
        engine = self.pipeline_thread.engine if self.pipeline_thread else None
        if engine is not None and engine.run_dir is not None:
            self.last_run_dir = engine.run_dir
            self.open_run_folder_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.scan_button.setEnabled(True)
        self.save_plan_button.setEnabled(self.current_plan is not None)
        self.start_button.setEnabled(bool(self.current_plan and self.current_plan.valid))

    def _processing_succeeded(self, run_dir: str) -> None:
        self._reset_after_processing()
        self.progress_bar.setValue(100)
        elapsed_text = human_duration(self.last_run_elapsed)
        try:
            manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
            elapsed_text = str(manifest.get("timing", {}).get("elapsed", elapsed_text))
        except (OSError, ValueError, TypeError):
            pass
        QMessageBox.information(
            self,
            APP_NAME,
            f"Processing completed successfully in {elapsed_text}.\n\n{run_dir}/masters",
        )

    def _processing_failed(self, message: str, details: str) -> None:
        self._reset_after_processing()
        self._append_log(details)
        QMessageBox.critical(
            self,
            APP_NAME,
            f"Processing failed: {message}\n\n"
            f"Elapsed: {human_duration(self.last_run_elapsed)}\n\n"
            "The timestamped partial run and manifest were preserved for diagnosis.",
        )

    def _processing_cancelled(self, message: str) -> None:
        self._reset_after_processing()
        QMessageBox.information(
            self,
            APP_NAME,
            f"{message}\nElapsed: {human_duration(self.last_run_elapsed)}",
        )

    # --- Self-update -----------------------------------------------------------

    def _update_settings(self) -> dict[str, Any]:
        settings = self.store.payload.setdefault("updates", {})
        settings.setdefault("check_on_startup", True)
        settings.setdefault("last_check_epoch", 0.0)
        settings.setdefault("skipped_version", "")
        settings.setdefault("install_path", "")
        return settings

    def _update_on_startup_toggled(self, checked: bool) -> None:
        self._update_settings()["check_on_startup"] = bool(checked)
        self.store.save()

    def _startup_update_check(self) -> None:
        settings = self._update_settings()
        if not settings["check_on_startup"]:
            return
        if time.time() - float(settings["last_check_epoch"] or 0) < UPDATE_CHECK_INTERVAL_SECONDS:
            return
        self._check_for_updates(manual=False)

    def _run_update_task(
        self,
        function: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        busy_text: str,
    ) -> None:
        self.update_button.setEnabled(False)
        self.update_button.setText(busy_text)

        def finish() -> None:
            self.update_button.setEnabled(True)
            self.update_button.setText("Check for updates")
            self.update_task = None

        task = BackgroundTask(function)
        task.done.connect(lambda result: (finish(), on_done(result)))
        task.error.connect(lambda message: (finish(), on_error(message)))
        self.update_task = task
        task.start()

    def _check_for_updates(self, manual: bool) -> None:
        if self.update_task is not None:
            return

        def failed(message: str) -> None:
            if manual:
                QMessageBox.warning(
                    self, "Update check failed",
                    f"Could not reach GitHub to check for updates.\n\n{message}",
                )
            else:
                self._append_log(f"Update check skipped: {message}")

        self._run_update_task(
            fetch_latest_release,
            lambda release: self._update_check_finished(release, manual),
            failed,
            "Checking…",
        )

    def _update_check_finished(self, release: ReleaseInfo, manual: bool) -> None:
        settings = self._update_settings()
        settings["last_check_epoch"] = time.time()
        self.store.save()
        if version_tuple(release.version) <= version_tuple(VERSION):
            if manual:
                QMessageBox.information(
                    self, "No update available",
                    f"You are running the latest version ({VERSION}).",
                )
            return
        if not manual and settings["skipped_version"] == release.version:
            return
        notes = release.notes or "No release notes."
        if len(notes) > 1500:
            notes = notes[:1500].rstrip() + "…"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Update available")
        box.setText(f"{APP_NAME} {release.version} is available (you have {VERSION}).")
        box.setInformativeText(f"What's new:\n{notes}")
        update = box.addButton("Update now", QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton("Skip this version", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(update)
        box.exec()
        clicked = box.clickedButton()
        if clicked is skip:
            settings["skipped_version"] = release.version
            self.store.save()
        elif clicked is update:
            self._start_update(release)

    def _script_search_dirs(self) -> list[str]:
        try:
            value = self.siril.get_siril_config("gui", "script_path")
        except Exception:  # older Siril or a closed connection: fall back to a manual pick
            return []
        if isinstance(value, str):
            return [part for part in value.split(";") if part.strip()]
        return [str(part) for part in value or [] if str(part).strip()]

    def _locate_install(self) -> Optional[Path]:
        settings = self._update_settings()
        remembered = Path(settings["install_path"]) if settings["install_path"] else None
        if remembered is not None and remembered.is_file():
            return remembered
        candidates = find_installed_scripts(self._script_search_dirs(), Path(__file__))
        if len(candidates) == 1:
            choice: Optional[Path] = candidates[0]
        elif candidates:
            from PyQt6.QtWidgets import QInputDialog  # local import keeps the main list compact

            labels = [str(path) for path in candidates]
            picked, accepted = QInputDialog.getItem(
                self, "Choose the copy to update",
                "Several copies of SBPP were found in Siril's script folders.\n"
                "Which one do you launch from Siril?",
                labels, 0, False,
            )
            choice = Path(picked) if accepted else None
        else:
            QMessageBox.information(
                self, "Locate SBPP",
                f"SBPP could not find {SCRIPT_FILENAME} in Siril's script folders.\n"
                "Please select the copy you launch from Siril.",
            )
            picked, _filter = QFileDialog.getOpenFileName(
                self, "Locate SBPP", str(Path.home()), f"SBPP script ({SCRIPT_FILENAME});;Python (*.py)",
            )
            choice = Path(picked) if picked else None
        if choice is not None:
            settings["install_path"] = str(choice)
            self.store.save()
        return choice

    def _start_update(self, release: ReleaseInfo) -> None:
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            QMessageBox.information(
                self, "Processing in progress",
                "SBPP can't update while processing. Use Check for updates again "
                "once the run has finished.",
            )
            return
        target = self._locate_install()
        if target is None:
            return

        def downloaded(data: bytes) -> None:
            try:
                backup = install_update(target, data, self.store.config_dir / "backups")
            except OSError as exc:
                QMessageBox.warning(
                    self, "Update failed",
                    f"SBPP could not replace {target}:\n{exc}\n\n"
                    f"You can download the new version manually from:\n{release.page_url}",
                )
                return
            self._append_log(f"Updated SBPP {VERSION} → {release.version} at {target}.")
            QMessageBox.information(
                self, "Update installed",
                f"SBPP {release.version} is installed.\n\n"
                "Close this window and launch SBPP again from Siril's Scripts menu "
                f"to use the new version.\n\nThe previous version was saved to:\n{backup}",
            )

        def failed(message: str) -> None:
            QMessageBox.warning(
                self, "Update failed",
                f"The update could not be downloaded:\n{message}\n\n"
                f"Your current version was not changed. Manual download:\n{release.page_url}",
            )

        self._run_update_task(lambda: download_release(release), downloaded, failed, "Updating…")

    def _update_panel_layout_button(self) -> None:
        plan = self.current_plan
        self.panel_layout_button.setEnabled(bool(
            plan and any(dataset.panel_count > 1 for dataset in plan.datasets)
        ))

    def _show_panel_map(self, activate: bool = True) -> None:
        if self.panel_map_window is None:
            self.panel_map_window = PanelMapWindow()
        self.panel_map_window.set_plan(self.current_plan)
        # Opened automatically after a scan, it must not steal keyboard focus.
        self.panel_map_window.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating, not activate
        )
        self.panel_map_window.show()
        self.panel_map_window.raise_()
        if activate:
            self.panel_map_window.activateWindow()

    def _refresh_panel_map(self) -> None:
        self._update_panel_layout_button()
        if self.panel_map_window is not None and self.panel_map_window.isVisible():
            self.panel_map_window.set_plan(self.current_plan)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        if self.update_task is not None:
            # Destroying a running QThread aborts the process; a check or
            # download is short, so let it finish.
            self.update_task.wait(15000)
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            QMessageBox.warning(
                self, APP_NAME,
                "Processing is still active. Request a stop and wait for the current Siril command to finish."
            )
            event.ignore()
            return
        self._save_profile()
        if self.panel_map_window is not None:
            self.panel_map_window.close()
        event.accept()


def run_self_test(input_root: Path, calibration_roots: list[Path]) -> int:
    """Read-only classifier/planner test used during development and support."""
    with tempfile.TemporaryDirectory(prefix="sbpp-self-test-") as temp_dir:
        cache = MetadataCache(Path(temp_dir) / "header-index.json")
        frames, errors = HeaderScanner(cache).scan(input_root, calibration_roots, recursive=True)
        plan = build_plan(
            frames,
            input_root,
            calibration_roots,
            input_root.parent,
            infer_target_name_from_headers(frames),
            PlanOptions(),
            errors,
        )
        print(json.dumps({
            "valid": plan.valid,
            "lights": plan.light_count,
            "integration_seconds": plan.integration_seconds,
            "integration_time": integration_summary(
                plan.integration_seconds, plan.unknown_exposure_count
            ),
            "unknown_exposure_lights": plan.unknown_exposure_count,
            "datasets": len(plan.datasets),
            "panels": {
                dataset.dataset_id: dataset.panels for dataset in plan.datasets
            },
            "channels": {
                dataset.dataset_id: {
                    name: sum(len(group.frames) for group in groups)
                    for name, groups in dataset.channels.items()
                }
                for dataset in plan.datasets
            },
            "calibration_plan": [
                {
                    "setup": dataset.dataset_id,
                    "panel": subgroup.panel_id,
                    "filter": subgroup.filter_name,
                    "lights": len(subgroup.frames),
                    "bias": subgroup.bias.description,
                    "dark": subgroup.dark.description,
                    "flat": subgroup.flat.description,
                    "darkflat": subgroup.darkflat.description,
                    "bpm": subgroup.bpm.description,
                    "notes": subgroup.warnings,
                }
                for dataset in plan.datasets
                for groups in dataset.channels.values()
                for subgroup in groups
            ],
            "warnings": plan.warnings,
            "scan_errors": plan.scan_errors[:10],
            "estimated_bytes": plan.estimated_bytes,
            "estimated_final_bytes": plan.estimated_final_bytes,
        }, indent=2))
        return 0 if plan.valid else 2


def main() -> int:
    if "--self-test" in sys.argv:
        index = sys.argv.index("--self-test")
        if len(sys.argv) <= index + 1:
            print("usage: Starloch_Batch_Preprocessor.py --self-test INPUT_ROOT [CALIBRATION_ROOT ...]")
            return 64
        return run_self_test(Path(sys.argv[index + 1]), [Path(arg) for arg in sys.argv[index + 2:]])

    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd("requires", "1.4.4")
    except s.SirilError as exc:
        print(f"Unable to connect to Siril 1.4.4+: {exc}", file=sys.stderr)
        return 1

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)
    application.setApplicationName(APP_NAME)
    application.setOrganizationName("Starloch")
    theme_mode = apply_siril_theme(application, siril)
    siril.log(f"SBPP: Qt interface theme follows Siril ({theme_mode}).")
    window = MainWindow(siril)
    install_exception_logging(window.record_unexpected_error)
    window.show()
    if owns_application:
        return application.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
