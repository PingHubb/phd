from __future__ import annotations

import numpy as np
import pyvista as pv
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkPointPicker

from phd.dependence.sensor_geometry import (
    compute_grid_normals,
    grid_selection_weights,
    heatmap_corner_points_from_centres,
    heatmap_surface_from_corner_lattice,
    smooth_grid_points,
)
from phd.ui import theme


class SensorShapeEditorDialog(QDialog):
    """Interactive editor for a structured tactile-sensor point grid."""

    MAX_UNDO_STATES = 50

    @staticmethod
    def _point_vertex_cells(point_count):
        point_count = max(0, int(point_count))
        return np.column_stack(
            [np.ones(point_count, dtype=np.int_), np.arange(point_count)]
        ).reshape(-1)

    def __init__(self, editor_data, geometry_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Sensor Shape Editor")
        self.resize(1180, 760)
        self.setModal(True)

        self.n_row = int(editor_data["n_row"])
        self.n_col = int(editor_data["n_col"])
        self.edges = np.array(editor_data["edges"], dtype=int, copy=True)
        self.geometry_config = dict(geometry_config or {})
        self.normal_flip = bool(self.geometry_config.get("normal_flip", False))
        sensor_points = np.array(editor_data["points"], dtype=float, copy=True)
        heatmap_corners = np.array(
            editor_data["heatmap_corners"],
            dtype=float,
            copy=True,
        )
        heatmap_default_corners = np.array(
            editor_data["heatmap_default_corners"],
            dtype=float,
            copy=True,
        )
        heatmap_edge_pairs = np.array(
            editor_data["heatmap_edge_pairs"], dtype=int, copy=True
        )
        self._heatmap_edge_pairs = heatmap_edge_pairs
        self._heatmap_edge_offsets = np.array(
            editor_data["heatmap_edge_offsets"], dtype=float, copy=True
        )
        curve_handles = (
            0.5
            * (
                heatmap_corners[heatmap_edge_pairs[:, 0]]
                + heatmap_corners[heatmap_edge_pairs[:, 1]]
            )
            + self._heatmap_edge_offsets
        )
        no_edges = np.empty(0, dtype=np.int_)
        self._edit_target = "sensor"
        self._datasets = {
            "sensor": {
                "points": sensor_points,
                "initial_points": np.array(sensor_points, copy=True),
                "reset_points": np.array(
                    editor_data["flat_points"], dtype=float, copy=True
                ),
                "undo": [],
                "redo": [],
                "n_row": int(editor_data["n_row"]),
                "n_col": int(editor_data["n_col"]),
                "edges": np.array(editor_data["edges"], dtype=int, copy=True),
            },
            "heatmap": {
                "points": heatmap_corners,
                "initial_points": np.array(heatmap_corners, copy=True),
                "reset_points": heatmap_default_corners,
                "undo": [],
                "redo": [],
                "n_row": int(editor_data["heatmap_n_row"]),
                "n_col": int(editor_data["heatmap_n_col"]),
                "edges": no_edges,
            },
            "curve": {
                "points": curve_handles,
                "initial_points": np.array(curve_handles, copy=True),
                "reset_points": 0.5
                * (
                    heatmap_corners[heatmap_edge_pairs[:, 0]]
                    + heatmap_corners[heatmap_edge_pairs[:, 1]]
                ),
                "undo": [],
                "redo": [],
                "n_row": len(curve_handles),
                "n_col": 1,
                "edges": no_edges,
            },
        }
        self.points = np.array(sensor_points, copy=True)
        self.flat_points = np.array(
            self._datasets["sensor"]["reset_points"], copy=True
        )
        self.normals = compute_grid_normals(
            self.points,
            self.n_row,
            self.n_col,
            normal_flip=self.normal_flip,
        )
        self._initial_points = np.array(self.points, copy=True)
        self._undo_stack = []
        self._redo_stack = []
        self._center_index = None
        self._selected_indices = np.empty(0, dtype=int)
        self._selection_weights = np.empty(0, dtype=float)
        self._dragging = False
        self._drag_changed = False
        self._drag_start_mouse = None
        self._drag_start_points = None
        self._drag_display_depth = 0.0
        self._drag_start_world = None
        self._closed = False

        self._build_ui()
        self._build_scene()
        self._update_undo_buttons()

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(splitter)

        viewport = QWidget(splitter)
        viewport_layout = QVBoxLayout(viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(parent=viewport, auto_update=30.0)
        self.plotter.background_color = theme.VIEWPORT_BG
        viewport_layout.addWidget(self.plotter.interactor)
        splitter.addWidget(viewport)

        controls = QWidget(splitter)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(360)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(8)

        target_form = QFormLayout()
        self.edit_target_combo = QComboBox()
        self.edit_target_combo.addItem("Sensor Point Grid", "sensor")
        self.edit_target_combo.addItem("3D Heatmap Taxels", "heatmap")
        self.edit_target_combo.addItem("Heatmap Edge Curves", "curve")
        self.edit_target_combo.currentIndexChanged.connect(
            self._switch_edit_target
        )
        target_form.addRow("Edit", self.edit_target_combo)
        controls_layout.addLayout(target_form)

        self.independent_heatmap_checkbox = QCheckBox(
            "Use independent 3D heatmap shape"
        )
        self.independent_heatmap_checkbox.setChecked(
            bool(self.geometry_config.get("use_custom_heatmap_shape", False))
        )
        self.independent_heatmap_checkbox.setToolTip(
            "Save heatmap taxel positions separately from the sensor point grid."
        )
        controls_layout.addWidget(self.independent_heatmap_checkbox)

        self.curved_edges_checkbox = QCheckBox("Use curved shared edges")
        self.curved_edges_checkbox.setChecked(
            bool(self.geometry_config.get("use_curved_heatmap_edges", False))
        )
        self.curved_edges_checkbox.setToolTip(
            "Use one shared quadratic curve for each neighboring taxel edge."
        )
        self.curved_edges_checkbox.toggled.connect(
            self._on_curved_edges_toggled
        )
        controls_layout.addWidget(self.curved_edges_checkbox)

        selection_form = QFormLayout()
        self.axis_combo = QComboBox()
        self.axis_combo.addItem("Z axis", "z")
        self.axis_combo.addItem("X axis", "x")
        self.axis_combo.addItem("Y axis", "y")
        self.axis_combo.addItem("Local normal", "normal")
        self.axis_combo.addItem("Screen plane", "screen")
        selection_form.addRow("Drag constraint", self.axis_combo)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(0, max(self.n_row, self.n_col))
        self.radius_spin.setValue(0)
        self.radius_spin.valueChanged.connect(self._rebuild_selection)
        selection_form.addRow("Selection radius", self.radius_spin)

        self.soft_selection_checkbox = QCheckBox("Soft neighborhood falloff")
        self.soft_selection_checkbox.setChecked(True)
        self.soft_selection_checkbox.toggled.connect(self._rebuild_selection)
        selection_form.addRow("", self.soft_selection_checkbox)
        controls_layout.addLayout(selection_form)

        self.selection_label = QLabel("No point selected")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet(theme.INFO_LABEL_STYLE)
        controls_layout.addWidget(self.selection_label)

        coordinate_form = QFormLayout()
        self.coordinate_spins = []
        for axis_name in ("X", "Y", "Z"):
            spin = QDoubleSpinBox()
            spin.setDecimals(5)
            spin.setRange(-1000.0, 1000.0)
            spin.setSingleStep(0.005)
            spin.setEnabled(False)
            coordinate_form.addRow(axis_name, spin)
            self.coordinate_spins.append(spin)
        controls_layout.addLayout(coordinate_form)

        self.apply_position_button = QPushButton("Apply Point Position")
        self.apply_position_button.setEnabled(False)
        self.apply_position_button.clicked.connect(self._apply_numeric_position)
        controls_layout.addWidget(self.apply_position_button)

        edit_row = QHBoxLayout()
        self.flatten_button = QPushButton("Flatten Z")
        self.flatten_button.setEnabled(False)
        self.flatten_button.clicked.connect(self._flatten_selected_z)
        self.smooth_button = QPushButton("Smooth")
        self.smooth_button.setEnabled(False)
        self.smooth_button.clicked.connect(self._smooth_selected)
        edit_row.addWidget(self.flatten_button)
        edit_row.addWidget(self.smooth_button)
        controls_layout.addLayout(edit_row)

        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.05, 1.0)
        self.smoothing_spin.setSingleStep(0.05)
        self.smoothing_spin.setValue(0.5)
        self.smoothing_spin.setDecimals(2)
        smoothing_form = QFormLayout()
        smoothing_form.addRow("Smoothing strength", self.smoothing_spin)
        controls_layout.addLayout(smoothing_form)

        history_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo)
        history_row.addWidget(self.undo_button)
        history_row.addWidget(self.redo_button)
        controls_layout.addLayout(history_row)

        reset_row = QHBoxLayout()
        self.reset_session_button = QPushButton("Reset Session")
        self.reset_session_button.clicked.connect(self._reset_session)
        self.reset_flat_button = QPushButton("Reset to Flat")
        self.reset_flat_button.clicked.connect(self._reset_flat)
        reset_row.addWidget(self.reset_session_button)
        reset_row.addWidget(self.reset_flat_button)
        controls_layout.addLayout(reset_row)

        self.show_labels_checkbox = QCheckBox("Show point labels")
        self.show_labels_checkbox.toggled.connect(self._refresh_labels)
        controls_layout.addWidget(self.show_labels_checkbox)

        self.status_label = QLabel(
            "Left-drag edits points. Right-drag rotates the view; wheel zooms."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(theme.MUTED_LABEL_STYLE)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        action_row = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QPushButton("Apply and Save")
        save_button.clicked.connect(self.accept)
        action_row.addWidget(cancel_button)
        action_row.addWidget(save_button)
        controls_layout.addLayout(action_row)

        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([840, 320])

    def _current_heatmap_corners(self):
        if self._edit_target == "heatmap":
            return np.asarray(self.points, dtype=float)
        return np.asarray(self._datasets["heatmap"]["points"], dtype=float)

    def _on_curved_edges_toggled(self, enabled):
        if bool(enabled):
            self.independent_heatmap_checkbox.setChecked(True)
        self._refresh_heatmap_preview()

    def _curve_midpoints(self, corners=None):
        corner_points = np.asarray(
            self._current_heatmap_corners() if corners is None else corners,
            dtype=float,
        )
        return 0.5 * (
            corner_points[self._heatmap_edge_pairs[:, 0]]
            + corner_points[self._heatmap_edge_pairs[:, 1]]
        )

    def _current_heatmap_edge_offsets(self):
        if self._edit_target == "curve":
            return np.asarray(self.points, dtype=float) - self._curve_midpoints()
        return np.asarray(self._heatmap_edge_offsets, dtype=float)

    def _heatmap_preview_geometry(self, corners=None, edge_offsets=None):
        corner_points = np.asarray(
            self._current_heatmap_corners() if corners is None else corners,
            dtype=float,
        )
        offsets = np.asarray(
            self._current_heatmap_edge_offsets()
            if edge_offsets is None
            else edge_offsets,
            dtype=float,
        )
        if not self.curved_edges_checkbox.isChecked():
            offsets = np.zeros_like(offsets)
        heatmap_dataset = self._datasets["heatmap"]
        return heatmap_surface_from_corner_lattice(
            corner_points,
            heatmap_dataset["n_row"],
            heatmap_dataset["n_col"],
            edge_offsets=offsets,
            curve_resolution=6,
        )

    @staticmethod
    def _curve_grid_geometry(surface):
        line_points = []
        lines = []
        for samples in surface["edge_samples"]:
            samples_np = np.asarray(samples, dtype=float)
            first_index = len(line_points)
            line_points.extend(samples_np)
            lines.extend(
                [
                    len(samples_np),
                    *range(first_index, first_index + len(samples_np)),
                ]
            )
        return np.asarray(line_points, dtype=float), np.asarray(lines, dtype=np.int_)

    def _refresh_heatmap_preview(self, *_args):
        if self._edit_target not in ("heatmap", "curve"):
            return
        surface = self._heatmap_preview_geometry()
        self._heatmap_preview_poly.points = surface["points"]
        self._heatmap_preview_poly.Modified()
        grid_points, _grid_lines = self._curve_grid_geometry(surface)
        self._heatmap_curve_grid_poly.points = grid_points
        self._heatmap_curve_grid_poly.Modified()

    def _build_scene(self):
        heatmap_dataset = self._datasets["heatmap"]
        surface = self._heatmap_preview_geometry(
            corners=heatmap_dataset["points"],
            edge_offsets=self._heatmap_edge_offsets,
        )
        self._heatmap_preview_poly = pv.PolyData(
            surface["points"],
            surface["faces"],
        )
        self._heatmap_preview_actor = self.plotter.add_mesh(
            self._heatmap_preview_poly,
            color="#F4F5F7",
            show_edges=False,
            opacity=0.68,
            lighting=False,
            pickable=False,
            name="editable_heatmap_surface",
        )
        self._heatmap_preview_actor.SetVisibility(False)

        grid_points, grid_lines = self._curve_grid_geometry(surface)
        self._heatmap_curve_grid_poly = pv.PolyData(grid_points)
        self._heatmap_curve_grid_poly.lines = grid_lines
        self._heatmap_curve_grid_actor = self.plotter.add_mesh(
            self._heatmap_curve_grid_poly,
            color="#3F4752",
            line_width=3,
            render_lines_as_tubes=True,
            lighting=False,
            pickable=False,
            name="editable_heatmap_curves",
        )
        self._heatmap_curve_grid_actor.SetVisibility(False)

        self._grid_poly = pv.PolyData(np.array(self.points, copy=True))
        self._grid_poly.verts = self._point_vertex_cells(len(self.points))
        self._grid_poly.lines = self.edges
        self._grid_actor = self.plotter.add_mesh(
            self._grid_poly,
            color="#6EA8FE",
            line_width=3,
            point_size=12,
            render_lines_as_tubes=True,
            render_points_as_spheres=True,
            pickable=True,
            name="editable_sensor_grid",
        )

        placeholder = np.array(self.points[[0]], copy=True)
        self._selected_poly = pv.PolyData(placeholder)
        self._selected_actor = self.plotter.add_mesh(
            self._selected_poly,
            color="#FFD54F",
            style="points",
            point_size=17,
            render_points_as_spheres=True,
            pickable=False,
            name="selected_sensor_points",
        )
        self._selected_actor.SetVisibility(False)
        self._center_poly = pv.PolyData(placeholder)
        self._center_actor = self.plotter.add_mesh(
            self._center_poly,
            color="#FF7043",
            style="points",
            point_size=22,
            render_points_as_spheres=True,
            pickable=False,
            name="selected_sensor_center",
        )
        self._center_actor.SetVisibility(False)
        self.plotter.add_axes(interactive=False)
        self.plotter.show_grid(
            color=theme.TEXT_MUTED,
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        )
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self._picker = vtkPointPicker()
        self._picker.SetTolerance(0.03)
        self.plotter.interactor.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.plotter.interactor:
            event_type = event.type()
            if event_type == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._handle_left_press(event)
                return True
            if event_type == QEvent.MouseMove and self._dragging:
                self._handle_left_move(event)
                return True
            if event_type == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._handle_left_release()
                return True
        return super().eventFilter(watched, event)

    def _vtk_mouse_position(self, event):
        position = event.pos()
        return int(position.x()), int(self.plotter.height() - position.y() - 1)

    def _handle_left_press(self, event):
        mouse_x, mouse_y = self._vtk_mouse_position(event)
        self._picker.Pick(mouse_x, mouse_y, 0.0, self.plotter.renderer)
        point_id = int(self._picker.GetPointId())
        if point_id < 0 or point_id >= len(self.points):
            self._set_center_index(None)
            return

        self._set_center_index(point_id)
        self._dragging = True
        self._drag_changed = False
        self._drag_start_mouse = np.array([mouse_x, mouse_y], dtype=float)
        self._drag_start_points = np.array(self.points, copy=True)
        display = self._world_to_display(self.points[point_id])
        self._drag_display_depth = float(display[2])
        self._drag_start_world = self._display_to_world(
            mouse_x, mouse_y, self._drag_display_depth
        )

    def _handle_left_move(self, event):
        if self._center_index is None or not len(self._selected_indices):
            return
        mouse = np.asarray(self._vtk_mouse_position(event), dtype=float)
        mouse_delta = mouse - self._drag_start_mouse
        if np.linalg.norm(mouse_delta) < 1.0:
            return
        if not self._drag_changed:
            self._push_undo_state(self._drag_start_points)
            self._drag_changed = True

        mode = str(self.axis_combo.currentData() or "z")
        if mode == "screen":
            current_world = self._display_to_world(
                mouse[0], mouse[1], self._drag_display_depth
            )
            delta = current_world - self._drag_start_world
        else:
            axis = self._drag_axis(mode)
            delta = axis * self._axis_drag_distance(axis, mouse_delta)

        selected = self._selected_indices
        weights = self._selection_weights[:, None]
        self.points[selected] = self._drag_start_points[selected] + weights * delta
        self._refresh_geometry(refresh_labels=False)
        self._sync_coordinate_spins()

    def _handle_left_release(self):
        if self._drag_changed:
            self._refresh_labels()
            status = {
                "sensor": "Point-grid shape updated. Apply and Save keeps this geometry.",
                "heatmap": "Heatmap shape updated independently.",
                "curve": "Shared edge curvature updated.",
            }
            self.status_label.setText(status[self._edit_target])
        self._dragging = False
        self._drag_changed = False
        self._drag_start_mouse = None
        self._drag_start_points = None

    def _drag_axis(self, mode):
        if mode == "x":
            return np.array([1.0, 0.0, 0.0])
        if mode == "y":
            return np.array([0.0, 1.0, 0.0])
        if mode == "normal" and self.normals is not None:
            return np.array(self.normals[self._center_index], dtype=float)
        return np.array([0.0, 0.0, 1.0])

    def _axis_drag_distance(self, axis, mouse_delta):
        origin = self.points[self._center_index]
        display_origin = self._world_to_display(origin)
        display_axis = self._world_to_display(origin + axis)
        screen_axis = display_axis[:2] - display_origin[:2]
        pixels_per_unit = float(np.linalg.norm(screen_axis))
        if pixels_per_unit > 1e-6:
            return float(np.dot(mouse_delta, screen_axis / pixels_per_unit)) / pixels_per_unit
        span = max(float(np.ptp(self.points, axis=0).max()), 0.1)
        return float(mouse_delta[1]) * span / 300.0

    def _world_to_display(self, point):
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        return np.asarray(renderer.GetDisplayPoint(), dtype=float)

    def _display_to_world(self, x, y, depth):
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(float(x), float(y), float(depth))
        renderer.DisplayToWorld()
        world = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if world.shape[0] == 4 and abs(world[3]) > 1e-12:
            world = world[:3] / world[3]
        return world[:3]

    def _set_center_index(self, point_id):
        self._center_index = None if point_id is None else int(point_id)
        self._rebuild_selection()

    def _store_active_dataset(self):
        dataset = self._datasets[self._edit_target]
        dataset["points"] = np.array(self.points, copy=True)
        dataset["undo"] = list(self._undo_stack)
        dataset["redo"] = list(self._redo_stack)
        if self._edit_target == "curve":
            self._heatmap_edge_offsets = (
                np.asarray(self.points, dtype=float) - self._curve_midpoints()
            )

    def _switch_edit_target(self, *_args):
        target = str(self.edit_target_combo.currentData() or "sensor")
        if target == self._edit_target:
            return
        self._store_active_dataset()
        self._edit_target = target
        dataset = self._datasets[target]
        if target == "curve":
            curve_midpoints = self._curve_midpoints()
            dataset["points"] = curve_midpoints + self._heatmap_edge_offsets
            dataset["reset_points"] = curve_midpoints
            self.curved_edges_checkbox.blockSignals(True)
            self.curved_edges_checkbox.setChecked(True)
            self.curved_edges_checkbox.blockSignals(False)
            self.independent_heatmap_checkbox.setChecked(True)
        self.n_row = int(dataset["n_row"])
        self.n_col = int(dataset["n_col"])
        self.edges = np.array(dataset["edges"], dtype=int, copy=True)
        self.points = np.array(dataset["points"], copy=True)
        self._initial_points = np.array(dataset["initial_points"], copy=True)
        self.flat_points = np.array(dataset["reset_points"], copy=True)
        self._undo_stack = list(dataset["undo"])
        self._redo_stack = list(dataset["redo"])
        self._set_center_index(None)
        curve_mode = target == "curve"
        self.radius_spin.setMaximum(
            0 if curve_mode else max(self.n_row, self.n_col)
        )
        if curve_mode:
            self.radius_spin.setValue(0)
        self.radius_spin.setEnabled(not curve_mode)
        self.soft_selection_checkbox.setEnabled(not curve_mode)
        self._grid_poly.lines = self.edges
        self._grid_poly.verts = self._point_vertex_cells(len(self.points))
        self._grid_poly.Modified()
        reset_labels = {
            "sensor": "Reset to Flat",
            "heatmap": "Reset to Sensor Shape",
            "curve": "Reset to Straight",
        }
        apply_labels = {
            "sensor": "Apply Point Position",
            "heatmap": "Apply Corner Position",
            "curve": "Apply Curve Handle",
        }
        show_labels = {
            "sensor": "Show point labels",
            "heatmap": "Show corner labels",
            "curve": "Show edge labels",
        }
        status_labels = {
            "sensor": "Left-drag edits points. Right-drag rotates the view; wheel zooms.",
            "heatmap": "Drag shared corners; adjacent taxels remain connected.",
            "curve": "Drag an edge handle to bend its shared taxel boundary.",
        }
        self.reset_flat_button.setText(reset_labels[target])
        self.apply_position_button.setText(apply_labels[target])
        self.show_labels_checkbox.setText(show_labels[target])
        self.status_label.setText(status_labels[target])
        heatmap_mode = target in ("heatmap", "curve")
        self._heatmap_preview_actor.SetVisibility(heatmap_mode)
        self._heatmap_curve_grid_actor.SetVisibility(heatmap_mode)
        self._refresh_geometry()
        self._update_undo_buttons()

    def _rebuild_selection(self, *_args):
        if self._center_index is None:
            self._selected_indices = np.empty(0, dtype=int)
            self._selection_weights = np.empty(0, dtype=float)
        else:
            self._selected_indices, self._selection_weights = grid_selection_weights(
                self._center_index,
                self.n_row,
                self.n_col,
                radius=self.radius_spin.value(),
                soft=self.soft_selection_checkbox.isChecked(),
            )
        self._refresh_selection_actor()
        self._sync_coordinate_spins()

    def _refresh_selection_actor(self):
        span = max(float(np.ptp(self.points, axis=0).max()), 0.1)
        highlight_offset = span * 0.008
        if len(self._selected_indices):
            selected_normals = self.normals[self._selected_indices]
            self._selected_poly.points = (
                self.points[self._selected_indices]
                + selected_normals * highlight_offset
            )
            self._selected_actor.SetVisibility(True)
        else:
            self._selected_actor.SetVisibility(False)
        if self._center_index is not None:
            self._center_poly.points = (
                self.points[[self._center_index]]
                + self.normals[[self._center_index]] * highlight_offset * 1.5
            )
            self._center_actor.SetVisibility(True)
        else:
            self._center_actor.SetVisibility(False)
        self._selected_poly.Modified()
        self._center_poly.Modified()
        self.plotter.render()

    def _sync_coordinate_spins(self):
        enabled = self._center_index is not None
        for spin in self.coordinate_spins:
            spin.setEnabled(enabled)
        self.apply_position_button.setEnabled(enabled)
        self.flatten_button.setEnabled(enabled)
        self.smooth_button.setEnabled(
            enabled and self._edit_target != "curve"
        )
        if not enabled:
            self.selection_label.setText("No point selected")
            return

        point = self.points[self._center_index]
        for spin, value in zip(self.coordinate_spins, point):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        if self._edit_target == "curve":
            self.selection_label.setText(
                f"E{self._center_index} | shared edge curvature handle"
            )
        else:
            col, row = divmod(self._center_index, self.n_row)
            point_prefix = "C" if self._edit_target == "heatmap" else "P"
            self.selection_label.setText(
                f"{point_prefix}{self._center_index} | row {row}, column {col} | "
                f"{len(self._selected_indices)} selected"
            )

    def _apply_numeric_position(self):
        if self._center_index is None:
            return
        target = np.asarray(
            [spin.value() for spin in self.coordinate_spins], dtype=float
        )
        delta = target - self.points[self._center_index]
        self._push_undo_state()
        self.points[self._selected_indices] += (
            self._selection_weights[:, None] * delta
        )
        self._refresh_geometry()

    def _flatten_selected_z(self):
        if not len(self._selected_indices):
            return
        self._push_undo_state()
        target_z = float(np.mean(self.points[self._selected_indices, 2]))
        self.points[self._selected_indices, 2] = target_z
        self._refresh_geometry()

    def _smooth_selected(self):
        if not len(self._selected_indices):
            return
        smoothed = smooth_grid_points(
            self.points,
            self._selected_indices,
            self.n_row,
            self.n_col,
            strength=self.smoothing_spin.value(),
        )
        if smoothed is None:
            return
        self._push_undo_state()
        self.points = smoothed
        self._refresh_geometry()

    def _reset_session(self):
        self._replace_all_points(self._initial_points)

    def _reset_flat(self):
        if self._edit_target == "curve":
            self._replace_all_points(self._curve_midpoints())
            return
        if self._edit_target == "heatmap":
            sensor_dataset = self._datasets["sensor"]
            sensor_normals = compute_grid_normals(
                sensor_dataset["points"],
                sensor_dataset["n_row"],
                sensor_dataset["n_col"],
                normal_flip=self.normal_flip,
            )
            heatmap_corners = heatmap_corner_points_from_centres(
                sensor_dataset["points"],
                sensor_normals,
                sensor_dataset["n_row"],
                sensor_dataset["n_col"],
            )
            self._replace_all_points(heatmap_corners)
            return
        self._replace_all_points(self.flat_points)

    def _replace_all_points(self, points):
        replacement = np.asarray(points, dtype=float)
        if replacement.shape != self.points.shape:
            return
        self._push_undo_state()
        self.points = np.array(replacement, copy=True)
        self._refresh_geometry()

    def _push_undo_state(self, points=None):
        if self._edit_target in ("heatmap", "curve"):
            self.independent_heatmap_checkbox.setChecked(True)
        if self._edit_target == "curve":
            self.curved_edges_checkbox.setChecked(True)
        state = np.array(self.points if points is None else points, copy=True)
        if self._undo_stack and np.array_equal(self._undo_stack[-1], state):
            return
        self._undo_stack.append(state)
        self._undo_stack = self._undo_stack[-self.MAX_UNDO_STATES :]
        self._redo_stack.clear()
        self._update_undo_buttons()

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(np.array(self.points, copy=True))
        self.points = self._undo_stack.pop()
        self._refresh_geometry()
        self._update_undo_buttons()

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(np.array(self.points, copy=True))
        self.points = self._redo_stack.pop()
        self._refresh_geometry()
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        self.undo_button.setEnabled(bool(self._undo_stack))
        self.redo_button.setEnabled(bool(self._redo_stack))

    def _refresh_geometry(self, refresh_labels=True):
        self.normals = compute_grid_normals(
            self.points,
            self.n_row,
            self.n_col,
            normal_flip=self.normal_flip,
        )
        self._grid_poly.points = self.points
        self._grid_poly.Modified()
        self._refresh_heatmap_preview()
        self._refresh_selection_actor()
        self._sync_coordinate_spins()
        if refresh_labels:
            self._refresh_labels()

    def _refresh_labels(self, *_args):
        try:
            self.plotter.remove_actor("sensor_shape_editor_labels", render=False)
        except Exception:
            pass
        if not self.show_labels_checkbox.isChecked():
            self.plotter.render()
            return
        labels = []
        point_prefix = {
            "sensor": "P",
            "heatmap": "C",
            "curve": "E",
        }[self._edit_target]
        for index in range(len(self.points)):
            if self._edit_target == "curve":
                labels.append(f"{point_prefix}{index}")
            else:
                col, row = divmod(index, self.n_row)
                labels.append(f"{point_prefix}{index} r{row} c{col}")
        try:
            self.plotter.add_point_labels(
                self.points,
                labels,
                name="sensor_shape_editor_labels",
                font_size=10,
                text_color="#111111",
                point_size=0,
                shape="rounded_rect",
                shape_color="#FFD54F",
                shape_opacity=0.9,
                always_visible=True,
                pickable=False,
                render=True,
            )
        except Exception as exc:
            self.status_label.setText(f"Could not display point labels: {exc}")

    def result_geometry_config(self):
        self._store_active_dataset()
        geometry = dict(self.geometry_config)
        sensor_dataset = self._datasets["sensor"]
        if not np.array_equal(
            sensor_dataset["points"], sensor_dataset["initial_points"]
        ):
            geometry.update(
                {
                    "use_selected_shape": True,
                    "shape": "custom",
                    "custom_points": np.asarray(
                        sensor_dataset["points"], dtype=float
                    ).tolist(),
                }
            )
        geometry.update(
            {
                "use_custom_heatmap_shape": bool(
                    self.independent_heatmap_checkbox.isChecked()
                ),
                "custom_heatmap_corners": np.asarray(
                    self._datasets["heatmap"]["points"], dtype=float
                ).tolist(),
                "use_curved_heatmap_edges": bool(
                    self.curved_edges_checkbox.isChecked()
                ),
                "custom_heatmap_edge_offsets": np.asarray(
                    self._heatmap_edge_offsets, dtype=float
                ).tolist(),
            }
        )
        return geometry

    def closeEvent(self, event):
        self._close_plotter()
        super().closeEvent(event)

    def done(self, result):
        self._close_plotter()
        super().done(result)

    def _close_plotter(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.plotter.interactor.removeEventFilter(self)
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
