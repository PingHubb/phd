from types import SimpleNamespace

import numpy as np
import pyvista as pv

from phd.ui import humanoid_viewer
from phd.ui.humanoid_viewer import (
    HumanoidSensorAutoWorker,
    HumanoidViewerWidget,
    _stable_usb_port_identity,
)
from phd.dependence.humanoid_sensor_registry import (
    humanoid_sensor_annotation,
    humanoid_sensor_grid_shape,
    humanoid_sensor_uses_extra_column,
)
from phd.dependence.humanoid_signal_parts import discover_signal_parts


def test_humanoid_signal_txt_maps_vertices_to_zero_based_channels(tmp_path):
    signal_obj = tmp_path / "curves_col_signal.obj"
    (tmp_path / "signal.txt").write_text(
        "1\n2\n-1\n4\n",
        encoding="utf-8",
    )

    (
        indices,
        transposed_indices,
        required,
        expected_shape,
    ) = HumanoidViewerWidget._load_signal_channel_indices(
        signal_obj,
        point_count=4,
    )

    np.testing.assert_array_equal(indices, [0, 1, -1, 3])
    np.testing.assert_array_equal(
        transposed_indices,
        [0, 1, -1, 3],
    )
    assert required == 4
    assert expected_shape == (0, 0)


def test_humanoid_part_uses_pipeline_drive_sensor_shape_not_max_id(
    tmp_path,
):
    part_dir = tmp_path / "11"
    part_dir.mkdir()
    signal_obj = part_dir / "curves_col_signal.obj"
    (part_dir / "signal.txt").write_text(
        "1\n16\n17\n18\n19\n284\n",
        encoding="utf-8",
    )
    (part_dir / "pipeline.cfg").write_text(
        "selected_weft_num = 17\n"
        "selected_warp_num = 16\n",
        encoding="utf-8",
    )

    indices, transposed_indices, required, expected_shape = (
        HumanoidViewerWidget._load_signal_channel_indices(
            signal_obj,
            point_count=6,
        )
    )

    np.testing.assert_array_equal(
        indices,
        [0, 15, 16, -1, 17, 268],
    )
    np.testing.assert_array_equal(
        transposed_indices,
        [0, 240, 256, -1, 1, 223],
    )
    assert expected_shape == (16, 17)
    assert required == 272


def test_humanoid_part_uses_packaged_layout_metadata(tmp_path):
    part_dir = tmp_path / "3"
    part_dir.mkdir()
    signal_obj = part_dir / "curves_col_signal.obj"
    signal_obj.write_text("", encoding="utf-8")
    (part_dir / "signal.txt").write_text(
        "1\n8\n10\n71\n",
        encoding="utf-8",
    )
    (tmp_path / "layouts.json").write_text(
        '{"3": {"selected_weft_num": 8, '
        '"selected_warp_num": 13}}',
        encoding="utf-8",
    )

    indices, _transposed, required, expected_shape = (
        HumanoidViewerWidget._load_signal_channel_indices(
            signal_obj,
            point_count=4,
        )
    )

    assert expected_shape == (8, 8)
    assert required == 64
    np.testing.assert_array_equal(indices, [0, 7, 8, 63])


def test_packaged_signal_manifest_replaces_generator_source_objs(
    tmp_path,
):
    part_dir = tmp_path / "1"
    part_dir.mkdir()
    signal_obj = part_dir / "curves_col_signal.obj"
    signal_obj.write_text("", encoding="utf-8")
    (tmp_path / "parts.json").write_text(
        '{"parts": [{"number": "1", '
        '"part_name": "head_link"}]}',
        encoding="utf-8",
    )
    model = SimpleNamespace(
        links={"head_link": SimpleNamespace(visuals=[])}
    )

    parts = discover_signal_parts(tmp_path, model)

    assert len(parts) == 1
    assert parts[0].number == "1"
    assert parts[0].link_name == "head_link"
    assert parts[0].signal_obj == signal_obj


def test_head_signal_mesh_resolves_to_nine_by_fourteen(tmp_path):
    part_dir = tmp_path / "1"
    part_dir.mkdir()
    signal_obj = part_dir / "curves_col_signal.obj"
    # The maximum ID 134 occupies band 8 with a stride of 15:
    # nine observed bands and fourteen interior sensor regions.
    (part_dir / "signal.txt").write_text(
        "1\n14\n16\n134\n",
        encoding="utf-8",
    )
    (part_dir / "pipeline.cfg").write_text(
        "selected_weft_num = 14\n"
        "selected_warp_num = 13\n",
        encoding="utf-8",
    )

    (
        indices,
        _transposed_indices,
        required,
        expected_shape,
    ) = HumanoidViewerWidget._load_signal_channel_indices(
        signal_obj,
        point_count=4,
    )

    assert expected_shape == (9, 14)
    assert required == 126
    np.testing.assert_array_equal(indices, [0, 13, 14, 125])


def test_fast_auto_detect_only_targets_head_torso_and_hands():
    accepted = {
        "head_link",
        "torso_link",
        "left_rubber_hand",
        "right_rubber_hand",
    }
    rejected = {
        "pelvis_contour_link",
        "left_knee_link",
        "right_shoulder_pitch_link",
    }

    assert all(
        HumanoidViewerWidget._is_fast_auto_detect_link(name)
        for name in accepted
    )
    assert not any(
        HumanoidViewerWidget._is_fast_auto_detect_link(name)
        for name in rejected
    )


def test_saved_fast_port_assignments_resolve_exact_humanoid_parts():
    harness = SimpleNamespace(
        _signal_keys=[
            "signal:1:head_link",
            "signal:3:left_rubber_hand",
            "signal:3:right_rubber_hand",
            "signal:10:torso_link",
            "signal:11:torso_link",
        ]
    )

    assert HumanoidViewerWidget._preferred_auto_port_parts(harness) == {
        "ttyacm1": "signal:1:head_link",
        "ttyacm2": "signal:10:torso_link",
        "ttyacm3": "signal:11:torso_link",
        "ttyacm5": "signal:3:right_rubber_hand",
    }


def test_humanoid_usb_identity_does_not_depend_on_ttyacm_number():
    first = SimpleNamespace(
        serial_number="48:F6:EE:22:7E:68",
        vid=0x303A,
        pid=0x1001,
    )
    renamed = SimpleNamespace(
        serial_number="48:F6:EE:22:7E:68",
        vid=0x303A,
        pid=0x1001,
    )

    assert _stable_usb_port_identity(first) == (
        "usb:303a:1001:48:f6:ee:22:7e:68"
    )
    assert _stable_usb_port_identity(renamed) == (
        _stable_usb_port_identity(first)
    )


def test_humanoid_sensor_annotation_includes_part_and_grid_shape():
    port = SimpleNamespace(
        serial_number="48:F6:EE:22:7E:68",
        vid=0x303A,
        pid=0x1001,
    )
    assignments = {
        "usb:303a:1001:48:f6:ee:22:7e:68": {
            "part": "signal:1:head_link"
        }
    }

    assert humanoid_sensor_annotation(port, assignments) == (
        "Head (9x14)"
    )
    assert humanoid_sensor_grid_shape(port, assignments) == (9, 14)


def test_end_effector_sensor_profile_includes_extra_packet_column():
    port = SimpleNamespace(
        serial_number="F4:12:FA:68:EC:20",
        vid=0x303A,
        pid=0x1001,
    )
    assignments = {
        "usb:303a:1001:f4:12:fa:68:ec:20": {
            "part": "sensor:robot_arm_end_effector"
        }
    }

    assert humanoid_sensor_annotation(port, assignments) == (
        "Robot Arm End-Effector Sensor (10x10)"
    )
    assert humanoid_sensor_grid_shape(port, assignments) == (10, 10)
    assert humanoid_sensor_uses_extra_column(port, assignments) is True


def test_testing_sensor_profile_uses_eight_by_ten_without_extra_column():
    port = SimpleNamespace(
        serial_number="DC:54:75:E9:D6:6C",
        vid=0x303A,
        pid=0x1001,
    )
    assignments = {
        "usb:303a:1001:dc:54:75:e9:d6:6c": {
            "part": "sensor:testing_8x10"
        }
    }

    assert humanoid_sensor_annotation(port, assignments) == (
        "Testing Sensor (8x10)"
    )
    assert humanoid_sensor_grid_shape(port, assignments) == (8, 10)
    assert humanoid_sensor_uses_extra_column(port, assignments) is False


def test_remembered_head_sensor_follows_a_renamed_port(monkeypatch):
    head_identity = "usb:303a:1001:48:f6:ee:22:7e:68"
    renamed_head = SimpleNamespace(
        device="/dev/ttyACM4",
        serial_number="48:F6:EE:22:7E:68",
        vid=0x303A,
        pid=0x1001,
    )
    monkeypatch.setattr(
        humanoid_viewer,
        "_ttyacm_port_details",
        lambda: [renamed_head],
    )
    harness = SimpleNamespace(
        _signal_keys=["signal:1:head_link"],
        _saved_auto_device_assignments={
            head_identity: {"part": "signal:1:head_link"}
        },
    )

    assert HumanoidViewerWidget._preferred_auto_port_parts(harness) == {
        "ttyacm4": "signal:1:head_link"
    }


def test_fast_scan_ignores_ports_without_saved_assignments():
    worker = HumanoidSensorAutoWorker(
        [],
        preferred_port_parts={
            "ttyACM1": "signal:1:head_link",
            "ttyACM3": "signal:11:torso_link",
        },
    )
    worker._candidate_ports = lambda: [
        "/dev/ttyACM0",
        "/dev/ttyACM1",
        "/dev/ttyACM2",
        "/dev/ttyACM3",
    ]

    assert worker._candidate_port_targets() == [
        ("/dev/ttyACM1", "signal:1:head_link"),
        ("/dev/ttyACM3", "signal:11:torso_link"),
    ]


def test_humanoid_live_heatmap_applies_signed_sensor_colours():
    mesh = pv.PolyData(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    mesh["RGB"] = np.full((4, 3), 128, dtype=np.uint8)

    harness = SimpleNamespace(
        _world_meshes={"signal:test": mesh},
        _signal_channel_indices={
            "signal:test": np.asarray([0, 1, 2, 3], dtype=np.int64)
        },
        _signal_static_colors={
            "signal:test": np.full((4, 3), 128, dtype=np.uint8)
        },
    )

    rendered = HumanoidViewerWidget._apply_live_heatmap_frame(
        harness,
        "signal:test",
        np.asarray([[110.0, 90.0], [100.0, 100.0]]),
        np.full((2, 2), 100.0),
        settings={
            "response_mode": "linear_relative",
            "saturation_pct": 5.0,
            "noise_floor_pct": 0.0,
            "color_gain_3d": 1.0,
            "use_absolute_signal": False,
        },
    )

    colors = np.asarray(mesh["RGB"])
    assert rendered is True
    assert colors[0, 0] > colors[0, 2]
    assert colors[2, 2] > colors[2, 0]
    np.testing.assert_array_equal(colors[1], [255, 255, 255])
    np.testing.assert_array_equal(colors[3], [255, 255, 255])


def test_auto_sensor_matching_supports_exact_and_extra_column_payloads():
    requirements = [
        ("signal:small", 80),
        ("signal:large", 100),
    ]

    assert HumanoidSensorAutoWorker.match_payload(
        80,
        requirements,
    ) == ("signal:small", 80, 0)
    assert HumanoidSensorAutoWorker.match_payload(
        110,
        requirements,
    ) == ("signal:large", 100, 10)
    assert HumanoidSensorAutoWorker.match_payload(
        99,
        requirements,
    ) is None


def test_auto_sensor_warmup_rejects_startup_sentinels_and_waits_for_stable_size():
    class _Api:
        def __init__(self):
            self.frames = [
                [999, 999, 999, 999],
                [1, 2, 3],
                [10, 20, 30, 40],
                [11, 21, 31, 41],
                [12, 22, 32, 42],
            ]

        def read_raw(self):
            return self.frames.pop(0)

    worker = HumanoidSensorAutoWorker([])
    worker._running = True

    result = worker._read_stable_frame(
        _Api(),
        expected_lengths={4},
        frame_count=3,
        attempts=8,
    )

    assert result == [12, 22, 32, 42]
    assert worker._frame_is_plausible([0, 0]) is False
    assert worker._frame_is_plausible([999, 999]) is False
    assert worker._frame_is_plausible([10, 20]) is True


def test_auto_worker_updates_live_calibration_without_reopening_sensor():
    class _Api:
        def __init__(self):
            self.update_calls = 0

        def update_cal(self):
            self.update_calls += 1
            return [101, 102, 103, 104]

        def read_cal(self):
            return [111, 112, 113, 114]

    api = _Api()
    assignment = {
        "api": api,
        "active_count": 4,
        "calibration": [1, 2, 3, 4],
    }
    worker = HumanoidSensorAutoWorker([])
    worker.CALIBRATION_SETTLE_SECONDS = 0.0
    worker._running = True

    updated, failed = worker._refresh_calibrations([assignment])

    assert updated == 1
    assert failed == 0
    assert api.update_calls == 1
    assert assignment["calibration"] == [111, 112, 113, 114]


def test_shared_reader_payload_keeps_latest_frame_per_humanoid_part():
    worker = HumanoidSensorAutoWorker([])
    worker._stream_assignments_by_port = {
        "/dev/ttyACM0": {
            "port": "/dev/ttyACM0",
            "part": "signal:head",
            "active_count": 2,
            "calibration": [10, 20],
        }
    }

    worker._on_shared_reader_payload(
        0,
        "/dev/ttyACM0",
        [1, 2],
    )
    worker._on_shared_reader_payload(
        0,
        "/dev/ttyACM0",
        [3, 4],
    )

    frame = worker._latest_frames["signal:head"]
    assert frame == (
        "/dev/ttyACM0",
        "signal:head",
        [3, 4],
        [10, 20],
    )


def test_shared_reader_ignores_short_humanoid_payload():
    worker = HumanoidSensorAutoWorker([])
    worker._stream_assignments_by_port = {
        "/dev/ttyACM0": {
            "port": "/dev/ttyACM0",
            "part": "signal:head",
            "active_count": 3,
            "calibration": [10, 20, 30],
        }
    }
    worker._on_shared_reader_payload(
        0,
        "/dev/ttyACM0",
        [1, 2],
    )

    assert worker._latest_frames == {}


def test_humanoid_update_routes_to_active_independent_worker():
    class _Toggle:
        @staticmethod
        def isChecked():
            return True

    class _Thread:
        @staticmethod
        def isRunning():
            return True

    class _Worker:
        requested = False

        def request_calibration(self):
            self.requested = True

    class _Status:
        text = ""

        def setText(self, text):
            self.text = str(text)

    worker = _Worker()
    harness = SimpleNamespace(
        auto_live_button=_Toggle(),
        _auto_sensor_worker=worker,
        _auto_sensor_thread=_Thread(),
        auto_live_status=_Status(),
    )

    accepted = HumanoidViewerWidget.request_sensor_update(harness)

    assert accepted is True
    assert worker.requested is True
    assert "queued" in harness.auto_live_status.text.lower()


def test_released_humanoid_scene_can_be_restored_without_reloading_urdf():
    calls = []

    class _Status:
        def setText(self, text):
            calls.append(("status", str(text)))

    class _Harness:
        _closed = False
        model = object()
        _scene_active = False
        status = _Status()

        @staticmethod
        def _before_scene_load():
            calls.append(("activate", None))

        def _rebuild_scene(self, reset_camera):
            calls.append(("rebuild", bool(reset_camera)))
            self._scene_active = True

    harness = _Harness()

    restored = HumanoidViewerWidget.restore_scene(
        harness,
        reset_camera=False,
    )

    assert restored is True
    assert ("activate", None) in calls
    assert ("rebuild", False) in calls


def test_suspended_humanoid_scene_restores_cached_actors():
    calls = []

    class _Status:
        current_text = "paused"
        current_tooltip = ""

        def setText(self, text):
            self.current_text = str(text)

        def text(self):
            return self.current_text

        def setToolTip(self, text):
            self.current_tooltip = str(text)

        def toolTip(self):
            return self.current_tooltip

    class _AutoButton:
        enabled = False

        @staticmethod
        def isChecked():
            return False

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    actor = object()
    auto_button = _AutoButton()
    harness = SimpleNamespace(
        _closed=False,
        model=object(),
        _scene_active=False,
        _scene_suspended=True,
        _actors={"cached": actor},
        plotter=SimpleNamespace(
            renderer=SimpleNamespace(actors={"cached": actor})
        ),
        _status_before_suspend=("Humanoid ready", "details"),
        _auto_sensor_thread=None,
        _resume_auto_live_after_restore=False,
        auto_live_button=auto_button,
        status=_Status(),
        _before_scene_load=lambda: calls.append("activate"),
        _set_scene_controls_enabled=(
            lambda enabled: calls.append(("controls", bool(enabled)))
        ),
        _refresh_live_mapping_controls=lambda: calls.append("controls-ready"),
        _apply_visibility=lambda: calls.append("visible"),
    )

    restored = HumanoidViewerWidget.restore_scene(harness)

    assert restored is True
    assert harness._scene_active is True
    assert harness._scene_suspended is False
    assert harness.status.text() == "Humanoid ready"
    assert auto_button.enabled is True
    assert "visible" in calls
    assert not any(
        isinstance(call, tuple) and call[0] == "rebuild"
        for call in calls
    )


def test_live_sensor_auto_detect_is_restarted_after_scene_restore(
    monkeypatch,
):
    class _Button:
        enabled = False
        checked = False

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def setChecked(self, checked):
            self.checked = bool(checked)

    class _Harness:
        _closed = False
        _scene_active = True
        _resume_auto_live_after_restore = True
        _auto_restart_pending = False
        _auto_sensor_thread = None
        auto_live_button = _Button()

        def _restart_auto_live_after_restore(self):
            HumanoidViewerWidget._restart_auto_live_after_restore(self)

    monkeypatch.setattr(
        humanoid_viewer.QtCore.QTimer,
        "singleShot",
        lambda _delay, callback: callback(),
    )
    harness = _Harness()

    HumanoidViewerWidget._schedule_auto_live_restore(harness)

    assert harness.auto_live_button.enabled is True
    assert harness.auto_live_button.checked is True
    assert harness._resume_auto_live_after_restore is False
    assert harness._auto_restart_pending is False


def test_connected_auto_sensor_stream_resumes_without_reconnecting():
    class _Thread:
        @staticmethod
        def isRunning():
            return True

    class _Button:
        enabled = False

        @staticmethod
        def isChecked():
            return True

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _Timer:
        started = False

        def start(self):
            self.started = True

    class _Status:
        text = ""

        def setText(self, text):
            self.text = str(text)

    button = _Button()
    timer = _Timer()
    status = _Status()
    harness = SimpleNamespace(
        _scene_active=True,
        _auto_sensor_thread=_Thread(),
        _auto_sensor_worker=object(),
        auto_live_button=button,
        _resume_auto_live_after_restore=True,
        _auto_restart_pending=True,
        _live_render_timer=timer,
        auto_live_status=status,
    )

    kept = HumanoidViewerWidget._resume_kept_auto_live_stream(
        harness
    )

    assert kept is True
    assert button.enabled is True
    assert timer.started is True
    assert "remained connected" in status.text
    assert harness._resume_auto_live_after_restore is False
    assert harness._auto_restart_pending is False


def test_batched_auto_frames_keep_only_latest_frame_per_part():
    class _Toggle:
        @staticmethod
        def isChecked():
            return True

    class _Harness:
        auto_live_button = _Toggle()
        _auto_live_parts = {"signal:head"}
        _pending_auto_frames = {}

        def _on_auto_sensor_frame(self, *args):
            HumanoidViewerWidget._on_auto_sensor_frame(self, *args)

    harness = _Harness()
    HumanoidViewerWidget._on_auto_sensor_frames(
        harness,
        [
            ("ttyACM0", "signal:head", [1, 2], [10, 20]),
            ("ttyACM0", "signal:head", [3, 4], [10, 20]),
        ],
    )

    raw, calibration = harness._pending_auto_frames["signal:head"]
    np.testing.assert_array_equal(raw, [3, 4])
    np.testing.assert_array_equal(calibration, [10, 20])


def test_drive_sensor_swap_transposes_channel_order():
    values = np.arange(6, dtype=float)

    normal = HumanoidViewerWidget._apply_drive_sensor_order(
        values,
        drive_count=2,
        sensor_count=3,
        swap_drive_sensor=False,
    )
    swapped = HumanoidViewerWidget._apply_drive_sensor_order(
        values,
        drive_count=2,
        sensor_count=3,
        swap_drive_sensor=True,
    )

    np.testing.assert_array_equal(normal, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(swapped, [0, 3, 1, 4, 2, 5])


def test_humanoid_point_mapping_can_flip_each_visual_axis():
    indices = np.arange(6, dtype=np.int64)

    horizontal = (
        HumanoidViewerWidget._transform_point_channel_indices(
            indices,
            drive_count=3,
            sensor_count=2,
            flip_horizontal=True,
        )
    )
    vertical = (
        HumanoidViewerWidget._transform_point_channel_indices(
            indices,
            drive_count=3,
            sensor_count=2,
            flip_vertical=True,
        )
    )

    np.testing.assert_array_equal(
        horizontal,
        [4, 5, 2, 3, 0, 1],
    )
    np.testing.assert_array_equal(
        vertical,
        [1, 0, 3, 2, 5, 4],
    )


def test_drive_sensor_swap_setting_is_saved_and_reloaded(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "humanoid_sensor_mappings.json"
    monkeypatch.setattr(
        humanoid_viewer,
        "HUMANOID_SENSOR_MAPPING_FILE",
        str(config_path),
    )

    class _Harness:
        _mapping_setting_key = staticmethod(
            HumanoidViewerWidget._mapping_setting_key
        )
        _save_sensor_mapping_setting = (
            HumanoidViewerWidget._save_sensor_mapping_setting
        )

        def __init__(self):
            self._saved_sensor_mappings = {}

    harness = _Harness()
    assert harness._save_sensor_mapping_setting(
        {
            "part": "signal:arm",
            "port": "/dev/ttyACM1",
            "drive_count": 8,
            "sensor_count": 10,
            "swap_drive_sensor": True,
            "transpose_point_mapping": True,
            "flip_horizontal_mapping": True,
            "flip_vertical_mapping": False,
        }
    )

    loaded = HumanoidViewerWidget._load_saved_sensor_mappings()
    assert loaded[
        "signal:arm|ttyACM1|8x10"
    ]["swap_drive_sensor"] is True
    assert loaded[
        "signal:arm|ttyACM1|8x10"
    ]["transpose_point_mapping"] is True
    assert loaded[
        "signal:arm|ttyACM1|8x10"
    ]["flip_horizontal_mapping"] is True
    assert loaded[
        "signal:arm|ttyACM1|8x10"
    ]["flip_vertical_mapping"] is False
