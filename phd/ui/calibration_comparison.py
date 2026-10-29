"""Offline, unnormalised comparison across recorded nodes, sensors and runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QScrollArea, QShortcut, QSizePolicy, QVBoxLayout, QWidget,
)

from phd.ui import components, theme
from phd.ui.calibration_analysis import (
    APPROACH_PHASES, RETURN_PHASE, surface_referenced_rows,
)
from phd.ui.calibration_result_dialog import (
    CalibrationCurveWidget, SIGNAL_LABELS, SURFACE_DISTANCE_LABEL,
)
from phd.ui.calibration_recording_catalog import (
    dataset_axis_bounds, dataset_rows, recorded_taxels,
)
from phd.ui.calibration_sequence import sequence_segments, stage_at_time
from phd.ui.calibration_inspection import TimelineInspection


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def original_taxel(dataset):
    return int((dataset.get("metadata") or {}).get("taxel_index", 0) or 0)


def comparison_group(dataset):
    """Never silently share units or grids between different recorded sensors."""
    meta = dataset.get("metadata") or {}
    quality = meta.get("measurement_quality") or {}
    shape = meta.get("sensor_shape") or (8, 10)
    try:
        shape = [int(shape[0]), int(shape[1])]
        if min(shape) < 1:
            raise ValueError("Invalid sensor grid")
    except (TypeError, ValueError, IndexError):
        shape = [8, 10]
    return json.dumps({
        "shape": shape,
        "signal": dataset.get("signal_field") or "diff_ave",
        "sensor": quality.get("sensor_identity") or meta.get("sensor_identity"),
    }, sort_keys=True)


def comparison_source(dataset):
    """Identify a run from its actual CSV location, not a stale saved path."""
    meta = dataset.get("metadata") or {}
    quality = meta.get("measurement_quality") or {}
    path = Path(dataset.get("source_path") or ".").expanduser().resolve()
    directory = path.parent
    if re.fullmatch(r"taxel_\d+", directory.name):
        directory = directory.parent
    sensor = (meta.get("sensor_name") or quality.get("sensor_name")
              or quality.get("sensor_identity") or meta.get("sensor_identity") or "Unknown sensor")
    stamp = re.search(r"(?:^|_)(\d{8})_(\d{6})(?:_|$)", directory.name)
    if stamp:
        day, clock = stamp.groups()
        run = f"{day[:4]}-{day[4:6]}-{day[6:]} {clock[:2]}:{clock[2:4]}:{clock[4:]}"
    else:
        run = directory.name or "Recording"
        run = run if len(run) <= 48 else run[:45] + "…"
    setup = quality.get("setup") or {}
    probe = setup.get("probe_material") or ""
    note = setup.get("probe_material_note") or ""
    if probe and note:
        probe = f"{probe} ({note})"
    if not probe and "_probe-" in directory.name:
        probe = directory.name.split("_probe-", 1)[1]
    detail = f"{run} · {probe}" if probe else run
    return {"key": (str(directory), comparison_group(dataset)),
            "path": str(directory), "sensor": str(sensor), "run": run,
            "probe": probe, "detail": detail, "label": f"{sensor} · {detail}"}


def recording_key(dataset):
    """Deduplicate CSV/JSON/PNG selections of the same saved trial."""
    source = str(dataset.get("source_path") or "")
    if not source:
        return ("in_memory", id(dataset))
    return (str(Path(source).expanduser().resolve()), dataset.get("signal_field") or "diff_ave")


def _force(row):
    force = finite(row.get("force_delta_abs_g"))
    newtons = finite(row.get("force_delta_abs_n"))
    return abs(newtons) / .00980665 if force is None and newtons is not None else force


def shared_axis_ranges(datasets):
    """Scan native columns once, not one expensive fit per taxel.

    Call with matching signal fields. Main analysis uses one comparison_group;
    the cross-experiment workspace supplies only its pinned snapshots. All
    signed excursions contribute. No normalisation or baseline shifting.
    """
    bounds = {}

    def add(field, value):
        value = finite(value)
        if value is not None:
            low, high = bounds.get(field, (value, value))
            bounds[field] = min(low, value), max(high, value)

    for dataset in datasets:
        field = dataset.get("signal_field") or "diff_ave"
        if dataset.get("deferred_csv"):
            for key, pair in dataset_axis_bounds(dataset).items():
                for value in pair:
                    add(key, value)
            continue
        indices = recorded_taxels(dataset)
        columns = [f"taxel_{index:03d}_{field}" for index in indices]
        rows = dataset.get("rows") or []
        native = any(column in row for row in rows[:3] for column in columns)
        for row in rows:
            for physical in ("elapsed_s", "travel_down_mm"):
                add(physical, row.get(physical))
            add("force_delta_abs_g", _force(row))
            for column in columns if native else (field,):
                add(field, row.get(column))
    ranges = {}
    for field, pair in bounds.items():
        physical = field in {"elapsed_s", "travel_down_mm", "force_delta_abs_g"}
        ranges[field] = CalibrationCurveWidget._range(
            pair, padding=0.0 if field == "elapsed_s" else .08,
            include_zero=True, nonnegative=physical,
        )
        # A mean ± SD can exceed the individual extrema. This conservative
        # envelope bounds it across ANY subset of recorded matching trials.
        span = pair[1] - pair[0]
        ranges[f"repeatability:{field}"] = CalibrationCurveWidget._range(
            (pair[0] - span / 2, pair[1] + span / 2),
            include_zero=True, nonnegative=physical,
        )
    if "travel_down_mm" in bounds:
        low, high = bounds["travel_down_mm"]
        # Contact must lie within the recorded travel span. Use a fixed common
        # envelope without pretending missing contact is a measured zero.
        span = max(high - low, 1.0)
        ranges["contact_relative_travel_mm"] = (-span * 1.08, span * 1.08)
        ranges["surface_distance_mm"] = (-span * 1.08, span * 1.08)
    return ranges


def comparison_entry(dataset, taxel, dataset_index):
    """Snapshot only plotted fields; do not duplicate 400 taxel columns/frame."""
    taxel = int(taxel)
    if taxel not in recorded_taxels(dataset):
        raise ValueError(f"Taxel {taxel} has no recorded signal in this trial.")
    signal = dataset.get("signal_field") or "diff_ave"
    column = f"taxel_{taxel:03d}_{signal}"
    source_rows = dataset_rows(dataset)
    native = any(column in row for row in source_rows)
    rows = [{
        "elapsed_s": row.get("elapsed_s"), "phase": row.get("phase"),
        "travel_down_mm": row.get("travel_down_mm"),
        "force_delta_abs_g": _force(row),
        signal: row.get(column if native else signal),
        **{key: row[key] for key in (
            "frame_sequence", "sensor_monotonic_s", "step_state", "step_mode", "force_delta_g",
        ) if key in row},
    } for row in source_rows]
    # Imported lazily to keep helpers usable without loading the main dialog.
    from phd.ui.calibration_analysis import analyze_calibration_rows
    analysis = analyze_calibration_rows(rows, signal)
    rows = surface_referenced_rows(rows, signal, analysis=analysis, in_place=True)
    name = dataset.get("name") or f"Trial {dataset_index + 1}"
    source = comparison_source(dataset)
    return {
        "key": (int(dataset_index), taxel), "taxel_index": taxel,
        "physical_target_taxel_index": original_taxel(dataset),
        "name": name,
        "label": f"{source['sensor']} · Node {taxel} · {name}\n{source['detail']}",
        "sensor_name": source["sensor"], "run_label": source["run"],
        "probe_label": source["probe"],
        "source_path": str(dataset.get("source_path") or ""),
        "signal_field": signal, "group": comparison_group(dataset),
        "metadata": dict(dataset.get("metadata") or {}),
        "rows": rows, "analysis": analysis, "sequence_segments": sequence_segments(rows),
    }


class CalibrationComparisonDialog(QDialog):
    """Pin trial/node snapshots; linked equal-sized axes or branch-aware overlay."""

    MAX_ENTRIES = 6
    COLORS = ("#086B78", "#BD4B18", "#7750A3", "#2D728F", "#847018", "#B03A6A")
    MODES = (
        ("Signal vs Distance", "signal_distance"),
        ("Force vs Distance", "force_distance"),
        ("Force vs Signal", "force_signal"),
        ("Full Sequence", "process"),
        ("Contact-referenced Signal", "contact_signal"),
    )

    def __init__(self, datasets, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        self.datasets = list(datasets)
        self.entries = []
        self._profile_cache = {}
        self.plot_axes = []
        self._curves = []
        self._cursors = []
        self._markers = []
        self._last_hover = None
        self._inspectors = {}
        self._event_targets = []
        self._windowed_view = None
        self.setWindowTitle("Compare Sensors & Nodes — Shared Physical Scales")
        self.setModal(False)
        components.size_to_screen(self, 1380, 900)
        self.setStyleSheet(f"QDialog {{background:{theme.WINDOW_BG}; color:{theme.TEXT_PRIMARY};}}")
        layout = QVBoxLayout(self)
        title = QLabel("Compare sensors and nodes")
        title.setStyleSheet(f"font-size:18px; font-weight:600; color:{theme.TEXT_PRIMARY};")
        header = QHBoxLayout()
        header.addWidget(title, 1)
        self.controls_button = QPushButton("Hide Controls")
        self.controls_button.setCheckable(True)
        self.controls_button.setChecked(True)
        self.controls_button.setAccessibleName("Show or hide comparison setup controls")
        self.controls_button.setToolTip("Hide the source selectors and pinned list to give the graphs more space.")
        self.fullscreen_button = QPushButton("Full Screen (F11)")
        self.fullscreen_button.setAccessibleName("Toggle full-screen comparison")
        self.fullscreen_button.setToolTip("Use the whole screen, hide setup controls and fit all plots. Esc returns to the previous view.")
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        header.addWidget(self.controls_button)
        header.addWidget(self.fullscreen_button)
        layout.addLayout(header)
        self.controls_panel = QWidget()
        controls = QVBoxLayout(self.controls_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        self.context = QLabel("Import saved experiments, then pin up to 6 node/trial pairs. No hardware is needed.")
        self.context.setTextFormat(Qt.PlainText)
        self.context.setWordWrap(True)
        controls.addWidget(self.context)
        sources = QHBoxLayout()
        sources.addWidget(QLabel("Sensor / experiment"))
        self.source_selector = QComboBox()
        self.source_selector.setAccessibleName("Sensor and experiment to compare")
        self.source_selector.setMinimumContentsLength(35)
        self.source_selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        sources.addWidget(self.source_selector, 1)
        self.import_button = QPushButton("Import Experiment…")
        self.import_button.setToolTip(
            "Select a saved calibration CSV, JSON or graph from another experiment. "
            "Full Taxel runs add their saved trials without replacing pinned graphs."
        )
        self.import_button.clicked.connect(self._import_recordings)
        sources.addWidget(self.import_button)
        controls.addLayout(sources)
        select = QHBoxLayout()
        select.addWidget(QLabel("Trial"))
        self.trial_selector = QComboBox()
        self.trial_selector.setAccessibleName("Recorded trial")
        self.trial_selector.setMinimumContentsLength(25)
        self.trial_selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        select.addWidget(self.trial_selector, 1)
        select.addWidget(QLabel("Node"))
        self.node_selector = QComboBox()
        self.node_selector.setAccessibleName("Recorded node to pin")
        select.addWidget(self.node_selector)
        self.add_button = QPushButton("Add Node")
        self.add_button.clicked.connect(self._add_selected)
        select.addWidget(self.add_button)
        controls.addLayout(select)
        self.selection_list = QListWidget()
        self.selection_list.setMaximumHeight(110)
        self.selection_list.setAccessibleName("Pinned node and trial comparisons")
        pinned = QHBoxLayout()
        pinned.addWidget(self.selection_list, 1)
        pin_actions = QVBoxLayout()
        for text, callback in (("Remove Selected", self._remove_selected), ("Clear", self._clear)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            pin_actions.addWidget(button)
        pin_actions.addStretch()
        pinned.addLayout(pin_actions)
        controls.addLayout(pinned)
        layout.addWidget(self.controls_panel)
        self.controls_button.toggled.connect(self._set_controls_visible)
        tools = QHBoxLayout()
        self.mode_selector = QComboBox()
        for label, mode in self.MODES:
            self.mode_selector.addItem(label, mode)
        tools.addWidget(self.mode_selector)
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["Side by side", "Overlay"])
        tools.addWidget(self.layout_selector)
        self.smooth_check = QCheckBox("Show smoothed curves")
        self.smooth_check.setChecked(True)
        self.smooth_check.setToolTip("Faint lines retain the raw values; smoothing is visual, not a fitted equation.")
        tools.addWidget(self.smooth_check)
        tools.addStretch()
        self.fit_check = QCheckBox("Fit All Plots")
        self.fit_check.setToolTip("Fit the entire figure in the available area. Uncheck for taller, scrollable plots, especially with many nodes.")
        self.fit_check.toggled.connect(self._resize_plot_area)
        tools.addWidget(self.fit_check)
        layout.addLayout(tools)
        self.figure = Figure(figsize=(12, 5), facecolor=theme.SURFACE)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll, 1)
        self.readout = QLabel("Move the pointer over a graph to compare values at the same x.")
        self.readout.setWordWrap(True)
        self.readout.setMinimumHeight(32)
        self.readout.setTextFormat(Qt.PlainText)
        # Many pinned nodes can produce a long hover readout. Keep it scrollable
        # instead of allowing it to squeeze the graphs out of the window.
        self.readout.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.readout_scroll = QScrollArea()
        self.readout_scroll.setFrameShape(QScrollArea.NoFrame)
        self.readout_scroll.setWidgetResizable(True)
        self.readout_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.readout_scroll.setMinimumHeight(36)
        self.readout_scroll.setMaximumHeight(76)
        self.readout_scroll.setWidget(self.readout)
        layout.addWidget(self.readout_scroll)
        bottom = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setWordWrap(True)
        bottom.addWidget(self.status_label, 1)
        reset = QPushButton("Reset Shared View")
        reset.clicked.connect(self.redraw)
        export = QPushButton("Export Comparison…")
        export.clicked.connect(self._export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for button in (reset, export, close):
            bottom.addWidget(button)
        layout.addLayout(bottom)
        self.source_selector.currentIndexChanged.connect(self._select_source)
        self.trial_selector.currentIndexChanged.connect(self._select_trial)
        self.mode_selector.currentIndexChanged.connect(self.redraw)
        self.layout_selector.currentIndexChanged.connect(self.redraw)
        self.smooth_check.toggled.connect(self.redraw)
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self.canvas.mpl_connect("figure_leave_event", self._leave)
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        # Esc must leave full screen, not close the comparison and lose context.
        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.setEnabled(False)
        self.escape_shortcut.activated.connect(self._toggle_fullscreen)
        self._refresh_sources()
        self.redraw()

    def _set_controls_visible(self, visible):
        self.controls_panel.setVisible(visible)
        self.controls_button.setText("Hide Controls" if visible else "Show Controls")

    def _toggle_fullscreen(self):
        if self._windowed_view is None:
            self._windowed_view = (
                self.saveGeometry(), self.isMaximized(),
                self.controls_button.isChecked(), self.fit_check.isChecked(),
            )
            self.controls_button.setChecked(False)
            self.fit_check.setChecked(True)
            self.fullscreen_button.setText("Exit Full Screen")
            self.escape_shortcut.setEnabled(True)
            self.showFullScreen()
        else:
            geometry, maximized, controls, fit = self._windowed_view
            self._windowed_view = None
            self.showNormal()
            self.restoreGeometry(geometry)
            if maximized:
                self.showMaximized()
            self.controls_button.setChecked(controls)
            self.fit_check.setChecked(fit)
            self.fullscreen_button.setText("Full Screen (F11)")
            self.escape_shortcut.setEnabled(False)
        # Resizing the existing canvas preserves linked zoom, pins and data.
        self.canvas.draw_idle()

    def reject(self):
        if self._windowed_view is not None:
            self._toggle_fullscreen()
        else:
            super().reject()

    def _resize_plot_area(self, *_args):
        if self.fit_check.isChecked():
            self.canvas.setMinimumHeight(1)
        else:
            overlay = self.layout_selector.currentIndex() == 1
            groups = 1 if overlay else max(1, math.ceil(len(self.entries) / 2))
            process = self.mode_selector.currentData() == "process"
            self.canvas.setMinimumHeight(max(350, groups * (620 if process else 390)))
        self.canvas.draw_idle()

    def _refresh_sources(self, preferred_dataset=0):
        self._sources = {}
        self.source_selector.blockSignals(True)
        self.source_selector.clear()
        preferred = None
        for index, dataset in enumerate(self.datasets):
            source = comparison_source(dataset)
            key = source["key"]
            if key not in self._sources:
                self._sources[key] = []
                self.source_selector.addItem(source["label"], key)
                self.source_selector.setItemData(self.source_selector.count() - 1, source["path"], Qt.ToolTipRole)
            self._sources[key].append(index)
            if index == preferred_dataset:
                preferred = key
        keys = list(self._sources)
        self.source_selector.setCurrentIndex(keys.index(preferred) if preferred in keys else 0)
        self.source_selector.blockSignals(False)
        self._select_source(self.source_selector.currentIndex())
        self.select_dataset(preferred_dataset)

    def _select_source(self, index):
        keys = list(self._sources)
        self.trial_selector.blockSignals(True)
        self.trial_selector.clear()
        if 0 <= index < len(keys):
            for dataset_index in self._sources[keys[index]]:
                dataset = self.datasets[dataset_index]
                self.trial_selector.addItem(
                    f"{dataset.get('name') or 'Trial'} · probed node {original_taxel(dataset)}",
                    dataset_index,
                )
                self.trial_selector.setItemData(
                    self.trial_selector.count() - 1, str(dataset.get("source_path") or ""), Qt.ToolTipRole,
                )
        self.trial_selector.blockSignals(False)
        self.source_selector.setToolTip(self.source_selector.currentData(Qt.ToolTipRole) or "")
        self._select_trial(self.trial_selector.currentIndex())

    def select_dataset(self, dataset_index):
        """Keep stable dataset IDs when the source picker filters the trial list."""
        if not 0 <= dataset_index < len(self.datasets):
            return
        key = comparison_source(self.datasets[dataset_index])["key"]
        self.source_selector.setCurrentIndex(list(self._sources).index(key))
        self.trial_selector.setCurrentIndex(self.trial_selector.findData(dataset_index))

    def _select_trial(self, index):
        self.node_selector.clear()
        dataset_index = self.trial_selector.itemData(index) if index >= 0 else None
        if dataset_index is None:
            self.add_button.setEnabled(False)
            return
        dataset = self.datasets[dataset_index]
        for taxel in recorded_taxels(dataset):
            self.node_selector.addItem(str(taxel), taxel)
        selected = self.node_selector.findData(original_taxel(dataset))
        if selected >= 0:
            self.node_selector.setCurrentIndex(selected)
        self.add_button.setEnabled(self.node_selector.count() > 0)

    def _add_selected(self):
        if self.node_selector.currentData() is not None:
            self.add_node(self.trial_selector.currentData(), self.node_selector.currentData())

    def import_recordings(self, paths):
        """Read saved files only; retain existing snapshots on cancellation/error."""
        from phd.ui.calibration_graph_analysis import load_calibration_recording
        known = {recording_key(dataset): i for i, dataset in enumerate(self.datasets)}
        result = {"added": 0, "duplicates": 0, "errors": [], "warnings": []}
        preferred = None
        for path in paths:
            try:
                loaded = load_calibration_recording(path, browse_full_run=True)
                imported = loaded["datasets"]
                if not imported:
                    raise ValueError("No recorded trials were found.")
                # Validate selectors before appending anything from this file.
                for dataset in imported:
                    comparison_source(dataset)
                    recorded_taxels(dataset)
                    recording_key(dataset)
                initial = int(loaded.get("initial_dataset_index", 0))
                ids = []
                for dataset in imported:
                    key = recording_key(dataset)
                    if key in known:
                        result["duplicates"] += 1
                    else:
                        known[key] = len(self.datasets)
                        self.datasets.append(dataset)
                        result["added"] += 1
                    ids.append(known[key])
                preferred = ids[max(0, min(initial, len(ids) - 1))]
                result["warnings"].extend(loaded.get("catalog_warnings") or [])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                result["errors"].append(f"{Path(path).name}: {exc}")
        if preferred is not None:
            self._refresh_sources(preferred)
        message = f"Added {result['added']} trial(s); {result['duplicates']} already loaded."
        if result["errors"] or result["warnings"]:
            message += f" {len(result['errors'])} file(s) failed; {len(result['warnings'])} trial(s) skipped. See tooltip."
        if preferred is not None:
            message += " Choose a trial/node, then Add Node. Existing pinned graphs are unchanged."
        self.status_label.setText(message)
        self.status_label.setToolTip("\n".join(result["errors"] + result["warnings"]))
        return result

    def _import_recordings(self):
        from phd.dependence.paths import experiment_path
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import experiments for comparison (saved files only)",
            experiment_path("calibration_recordings"),
            "Calibration data or graph (*.csv *.json *.png)",
        )
        if not paths:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.import_recordings(paths)
        finally:
            QApplication.restoreOverrideCursor()

    def add_node(self, dataset_index, taxel):
        if dataset_index is None or not 0 <= int(dataset_index) < len(self.datasets):
            self.status_label.setText("Choose a recorded trial first.")
            return False
        key = int(dataset_index), int(taxel)
        if any(tuple(entry["key"]) == key for entry in self.entries):
            self.status_label.setText("This node/trial is already pinned. Choose another node or trial.")
            return False
        if len(self.entries) >= self.MAX_ENTRIES:
            self.status_label.setText("Six comparisons are open. Remove one before adding another.")
            return False
        dataset = self.datasets[dataset_index]
        if self.entries and (dataset.get("signal_field") or "diff_ave") != self.entries[0]["signal_field"]:
            self.status_label.setText(
                "Different signal types/units cannot share axes. Choose the same recorded signal "
                "(for example Difference vs Difference, not Difference vs %), or clear the comparison."
            )
            return False
        try:
            entry = comparison_entry(dataset, taxel, dataset_index)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not load this trial: {exc}")
            return False
        if not any(finite(row.get(entry["signal_field"])) is not None for row in entry["rows"]):
            self.status_label.setText("This node has no finite signal samples.")
            return False
        self.entries.append(entry)
        self._profile_cache.clear()
        self.selection_list.addItem(
            f"{len(self.entries)}. {entry['label']}  |  physically probed node "
            f"{entry['physical_target_taxel_index']}"
        )
        self.selection_list.item(len(self.entries) - 1).setToolTip(entry["source_path"])
        self.selection_list.setCurrentRow(len(self.entries) - 1)
        self.redraw()
        return True

    def _remove_selected(self):
        index = self.selection_list.currentRow()
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            self._profile_cache.clear()
            self.selection_list.takeItem(index)
            for number, entry in enumerate(self.entries):
                self.selection_list.item(number).setText(
                    f"{number + 1}. {entry['label']}  |  physically probed node "
                    f"{entry['physical_target_taxel_index']}"
                )
            self.redraw()

    def _clear(self):
        self.entries.clear()
        self._profile_cache.clear()
        self.selection_list.clear()
        self.redraw()

    def _profile(self):
        if not self.entries:
            return {}
        key = tuple(entry["key"] for entry in self.entries)
        if key not in self._profile_cache:
            # Snapshot-only: importing a 400-node run must not scan every CSV
            # or let an unrelated/missing trial break the pinned comparison.
            self._profile_cache[key] = shared_axis_ranges(self.entries)
        return self._profile_cache[key]

    def _specs(self):
        signal = self.entries[0]["signal_field"] if self.entries else "diff_ave"
        label = SIGNAL_LABELS.get(signal, signal)
        return {
            "signal_distance": [("travel_down_mm", signal, "TCP travel (mm)", label)],
            "force_distance": [("travel_down_mm", "force_delta_abs_g", "TCP travel (mm)", "Force change (g)")],
            "force_signal": [(signal, "force_delta_abs_g", label, "Force change (g)")],
            "contact_signal": [("contact_relative_travel_mm", signal, "TCP travel relative to detected contact (mm)", label)],
            "process": [
                ("elapsed_s", "surface_distance_mm", "Elapsed time (s)", SURFACE_DISTANCE_LABEL),
                ("elapsed_s", signal, "Elapsed time (s)", label),
                ("elapsed_s", "force_delta_abs_g", "Elapsed time (s)", "Force change (g)"),
            ],
        }[self.mode_selector.currentData()]

    def redraw(self, *_args):
        self.figure.clear()
        self.plot_axes, self._curves, self._cursors, self._markers = [], [], [], []
        self._event_targets = []
        self._inspectors = {
            entry["key"]: TimelineInspection(entry["rows"], entry["analysis"], entry["sequence_segments"])
            for entry in self.entries
        }
        self._last_hover = None
        self.readout.setText("Move the pointer over a graph to compare values at the same x.")
        if not self.entries:
            self._resize_plot_area()
            self.figure.text(.5, .5, "Add nodes to compare their recorded responses", ha="center", color=theme.TEXT_MUTED)
            self.canvas.draw_idle()
            self.status_label.setText("No nodes pinned. No hardware connection is needed.")
            return
        overlay = self.layout_selector.currentIndex() == 1
        process = self.mode_selector.currentData() == "process"
        specs = self._specs()
        columns = 1 if overlay else min(2, len(self.entries))
        groups = 1 if overlay else math.ceil(len(self.entries) / columns)
        nrows = groups * len(specs)
        self._resize_plot_area()
        grid = self.figure.add_gridspec(nrows, columns, left=.105, right=.97, bottom=.10,
                                       top=.85 if process else .89, hspace=.85, wspace=.42)
        shared_x, shared_y = {}, {}
        ranges = self._profile()
        for slot in range(1 if overlay else len(self.entries)):
            entries = list(enumerate(self.entries)) if overlay else [(slot, self.entries[slot])]
            for spec_index, (xfield, yfield, xlabel, ylabel) in enumerate(specs):
                row = (slot // columns) * len(specs) + spec_index
                column = slot % columns
                axis = self.figure.add_subplot(
                    grid[row, column], sharex=shared_x.get(xfield), sharey=shared_y.get(yfield),
                )
                shared_x[xfield], shared_y[yfield] = axis, axis
                self.plot_axes.append(axis)
                # Long signal names otherwise overlap neighbouring panels in
                # the three-row sequence view. Keep their units visible.
                shown_ylabel = ylabel.replace(" (", "\n(") if len(ylabel) > 32 else ylabel
                axis.set(xlabel=xlabel, ylabel=shown_ylabel)
                axis.tick_params(labelsize=8)
                axis.xaxis.label.set_size(9)
                axis.yaxis.label.set_size(9)
                axis.grid(True, color=theme.BORDER_SUBTLE, linewidth=.7)
                for spine in axis.spines.values():
                    spine.set_color(theme.BORDER)
                if xfield in ranges:
                    axis.set_xlim(ranges[xfield])
                if yfield in ranges:
                    axis.set_ylim(ranges[yfield])
                title = "Compared nodes" if overlay else self.entries[slot]["label"]
                axis.set_title(title if spec_index == 0 else "", fontsize=10, color=theme.TEXT_PRIMARY, pad=12)
                for entry_index, entry in entries:
                    self._draw_entry(axis, entry_index, entry, xfield, yfield)
                if process and not overlay:
                    inspector = self._inspectors[entry["key"]]
                    self._event_targets.append((axis, entry, inspector))
                    for marker in inspector.markers:
                        for event in marker["events"]:
                            axis.axvline(marker["time"], color=CalibrationCurveWidget.EVENT_COLORS[event["key"]],
                                         linestyle=":", linewidth=1)
                if xfield == "contact_relative_travel_mm":
                    axis.axvline(0, color=theme.TEXT_MUTED, linestyle=":", linewidth=1)
                if yfield == "surface_distance_mm":
                    axis.axhline(0, color=theme.TEXT_MUTED, linestyle=":", linewidth=1)
                cursor = axis.axvline(0, color=theme.TEXT_MUTED, linestyle=":", visible=False)
                self._cursors.append((axis, xfield, cursor))
                if overlay:
                    axis.legend(fontsize=8, loc="best", framealpha=.9)
        self.figure.suptitle(
            "Full sequence · shared scales\n"
            "Surface estimate: + above · 0 contact · − below (TCP travel, not compression)"
            if process else
            "Shared scales across pinned nodes · solid: approach · dashed: return",
            fontsize=10, color=theme.TEXT_PRIMARY, y=.975,
        )
        # Toolbar Home must refer to THIS comparison, not a deleted axes stack.
        self.toolbar.update()
        self.canvas.draw_idle()
        self._update_context()

    def _draw_entry(self, axis, entry_index, entry, xfield, yfield):
        color = self.COLORS[entry_index]
        process = xfield == "elapsed_s"
        branches = [(None, "-", "sequence")] if process else [
            (APPROACH_PHASES, "-", "approach"), ({RETURN_PHASE}, "--", "return"),
        ]
        any_points = False
        for phases, style, branch in branches:
            points = CalibrationCurveWidget._finite_points(entry["rows"], xfield, yfield, phases)
            if not points:
                continue
            any_points = True
            smooth = self.smooth_check.isChecked() and yfield != "surface_distance_mm"
            if smooth:
                axis.plot(*zip(*points), color=color, alpha=.25, linewidth=.7, linestyle=style)
                shown = CalibrationCurveWidget._smooth_points(points)
            else:
                shown = points
            axis.plot(*zip(*shown), color=color, linewidth=1.6, linestyle=style,
                      label=f"{entry['label']} · {branch}")
            marker, = axis.plot([], [], "o", color=color, markersize=4, visible=False)
            self._markers.append(marker)
            self._curves.append((axis, xfield, yfield, shown, entry, branch, marker))
        if not any_points:
            missing_contact = xfield == "contact_relative_travel_mm" or yfield == "surface_distance_mm"
            axis.text(.5, .5, "No detected contact reference" if missing_contact else "No recorded samples", transform=axis.transAxes, ha="center", color=theme.TEXT_MUTED)

    def _update_context(self):
        from phd.ui.characterisation_analysis import _condition_key
        conditions = set()
        cross_node = False
        unknown_identity = False
        legacy = False
        groups = set()
        partial = False
        for entry in self.entries:
            metadata = dict(entry["metadata"])
            quality = metadata.get("measurement_quality") or {}
            groups.add(entry["group"])
            partial |= quality.get("trial_completed") is False or quality.get("interrupted") is True
            unknown_identity |= not bool(quality.get("sensor_identity") or metadata.get("sensor_identity"))
            legacy |= not bool(quality)
            metadata.pop("taxel_index", None)
            metadata.pop("physical_target_taxel_index", None)
            conditions.add(_condition_key({"metadata": metadata, "signal_field": entry["signal_field"]}))
            cross_node |= entry["taxel_index"] != entry["physical_target_taxel_index"]
        messages = [f"{len(self.entries)} node/trial pairs · linked pan/zoom · Reset Shared View restores common limits."]
        if cross_node:
            messages.append("Some curves show another node's response, not a probe pressing that node; see physically probed IDs above.")
        if len(conditions) > 1:
            messages.append("Acquisition settings differ: descriptive comparison only, not matched-condition repeatability.")
        if len(groups) > 1:
            messages.append("Different sensors/grids: node IDs are local to each sensor. Equal digital units do not guarantee equal sensitivity.")
        if partial:
            messages.append("Includes partial/interrupted trials; compare only the recorded portions.")
        if legacy or unknown_identity:
            messages.append("Legacy/unverified metadata: confirm sensor identity and setup; force accuracy is not certified.")
        self.context.setText(" ".join(messages))
        self.status_label.setText("Original units and signs preserved. Faint: raw; bold: smoothed display." if self.smooth_check.isChecked() else "Original raw values and signs; no smoothing or amplitude normalisation.")
        if self.mode_selector.currentData() == "process":
            self.status_label.setText(self.status_label.text() + " Hover to see curve values in the readout below. Side-by-side markers snap to exact event times; overlay compares each node at the same time.")

    def _hover(self, event):
        if event.inaxes not in self.plot_axes or event.xdata is None or self.toolbar.mode:
            self._leave()
            return
        source = next((field for axis, field, _ in self._cursors if axis is event.inaxes), None)
        xvalue = float(event.xdata)
        snapped = None
        for axis, _entry, inspector in self._event_targets:
            if axis is event.inaxes:
                xvalue, snapped = inspector.snap(
                    xvalue, axis.get_xlim(), axis.get_window_extent().width / self.canvas.devicePixelRatioF(),
                )
                break
        for _axis, field, cursor in self._cursors:
            cursor.set_visible(field == source)
            if field == source:
                cursor.set_xdata([xvalue, xvalue])
        labels = []
        for _axis, xfield, yfield, points, entry, branch, marker in self._curves:
            sample = CalibrationCurveWidget._sample_at_x(points, xvalue) if xfield == source else None
            marker.set_visible(sample is not None)
            if sample is not None:
                value = sample[1]
                marker.set_data([xvalue], [value])
                stage = stage_at_time(entry["sequence_segments"], xvalue) if xfield == "elapsed_s" else None
                label = stage["label"] if stage else branch
                labels.append(
                    f"{entry['sensor_name']} · {entry['run_label']} · "
                    f"Node {entry['taxel_index']} {entry['name']} {label}: {yfield}={value:+.4g}"
                )
        event_label = " / ".join(item["label"] for item in snapped["events"]) + " · " if snapped else ""
        self.readout.setText(f"{event_label}{source}={xvalue:.6g} | " + " · ".join(labels))
        self._last_hover = (source, xvalue)
        self.canvas.draw_idle()

    def _leave(self, _event=None):
        if self._last_hover is None:
            return
        for _axis, _field, cursor in self._cursors:
            cursor.set_visible(False)
        for marker in self._markers:
            marker.set_visible(False)
        self._last_hover = None
        self.canvas.draw_idle()

    def export_comparison(self, path):
        """Export a clean image plus axis/provenance sidecar, never source data."""
        if not self.entries:
            raise ValueError("Add at least one node before exporting.")
        path = Path(path).expanduser().with_suffix(".png")
        sidecar = path.with_suffix(".json")
        sources = {Path(entry["source_path"]).resolve() for entry in self.entries if entry["source_path"]}
        if path.resolve() in sources or sidecar.resolve() in sources:
            raise ValueError("Choose a new output name, not an input recording.")
        if path.exists() or sidecar.exists():
            raise ValueError("That image or metadata file already exists. Choose a new name.")
        self._leave()
        self.canvas.draw()
        # Match displayed equal-sized panels, including scrollable rows.
        width = max(self.canvas.width(), 800) / self.figure.dpi
        height = max(self.canvas.height(), self.canvas.minimumHeight()) / self.figure.dpi
        old_size = self.figure.get_size_inches().copy()
        try:
            self.figure.set_size_inches(width, height, forward=False)
            self.figure.savefig(path, dpi=160, facecolor=theme.SURFACE)
        finally:
            self.figure.set_size_inches(old_size, forward=False)
            self.canvas.draw_idle()
        payload = {
            "schema_version": 2, "mode": self.mode_selector.currentData(),
            "layout": self.layout_selector.currentText(),
            "smoothing_display_only": self.smooth_check.isChecked(),
            "amplitude_normalisation": False, "shared_scale_scope": "pinned_nodes",
            "axes": [{"x": list(axis.get_xlim()), "y": list(axis.get_ylim()),
                      "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel()} for axis in self.plot_axes],
            "entries": [{key: value for key, value in entry.items() if key not in {"rows", "analysis"}} for entry in self.entries],
        }
        sidecar.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path, sidecar

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export shared-scale comparison", "node_comparison.png", "PNG image (*.png)")
        if not path:
            return
        try:
            image, sidecar = self.export_comparison(path)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not export: {exc}")
            return
        self.status_label.setText(f"Saved {image.name} and axis/source metadata {sidecar.name}.")
