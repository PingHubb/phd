"""Model-driven checkpoint selection never starts or replaces a running policy."""
from types import SimpleNamespace
from unittest.mock import Mock

from phd.ui.ui_ping_direct_finger_motion import DirectFingerMotionMixin


def controls():
    ui = DirectFingerMotionMixin()
    ui.ai_direct_execution_model_path_input = Mock()
    selected = {"path": ""}
    ui.ai_direct_execution_model_path_input.setText.side_effect = lambda value: selected.update(path=value)
    ui.ai_direct_execution_model_path_input.text.side_effect = lambda: selected["path"]
    ui.ai_direct_execution_model_status = Mock()
    ui.ai_direct_execution_select_model_button = Mock()
    ui.ai_direct_execution_model_details = Mock()
    ui.ai_direct_execution_dry_run_checkbox = Mock()
    ui.ai_direct_execution_velocity_scale_spin = Mock()
    ui.ai_direct_execution_velocity_scale_spin.blockSignals.return_value = False
    return ui


def test_selection_previews_metadata_without_instantiating_a_controller(tmp_path):
    path = tmp_path / "arbitrary_model_filename.pt"
    path.touch()
    ui = controls()
    ui._get_ai_direct_finger_motion_execution_helper = Mock(side_effect=AssertionError("must not load"))
    ui._set_ai_direct_execution_model_path(str(path))
    ui.ai_direct_execution_model_status.setText.assert_called_with(f"Model: Selected: {path.name}")
    ui.ai_direct_execution_model_status.setToolTip.assert_called_with(str(path))
    ui.ai_direct_execution_model_details.set_source.assert_called_with(str(path), loaded_bundle=None, loaded_kind=None)
    ui.ai_direct_execution_dry_run_checkbox.setChecked.assert_called_with(True)
    ui.ai_direct_execution_velocity_scale_spin.setValue.assert_called_with(1.0)
    assert [c.args for c in ui.ai_direct_execution_velocity_scale_spin.blockSignals.call_args_list] == [(True,), (False,)]
    ui._get_ai_direct_finger_motion_execution_helper.assert_not_called()


def test_running_model_cannot_be_replaced_or_its_preview_mode_changed(tmp_path):
    running, selected = tmp_path / "currently_running.pt", tmp_path / "selected_for_next_run.pt"
    running.touch(); selected.touch()
    ui = controls()
    ui._set_ai_direct_execution_model_path(str(running))
    helper = SimpleNamespace(is_running=True, model_loaded=True, model_kind="cnn_gru",
                             loaded_model_checkpoint_path=str(running))
    ui._refresh_ai_direct_execution_model_status(helper)
    ui.ai_direct_execution_dry_run_checkbox.reset_mock()
    ui._set_ai_direct_execution_model_path(str(selected))
    assert ui.ai_direct_execution_model_path_input.text() == str(running)
    ui.ai_direct_execution_dry_run_checkbox.setChecked.assert_not_called()
    ui.ai_direct_execution_select_model_button.setEnabled.assert_called_with(False)
    ui.ai_direct_execution_model_status.setText.assert_called_with(
        "Model: Running: currently_running.pt | CNN–GRU")
    ui.ai_direct_execution_model_details.set_source.assert_called_with(
        str(running), loaded_bundle={}, loaded_kind="cnn_gru")
    helper.is_running = False
    ui._set_ai_direct_execution_model_path(str(selected))
    assert ui.ai_direct_execution_model_path_input.text() == str(selected)
    ui.ai_direct_execution_select_model_button.setEnabled.assert_called_with(True)


def test_active_flag_also_blocks_file_dialog_and_path_change(tmp_path):
    ui = controls()
    ui._ai_direct_finger_execution_active = True
    ui._on_select_ai_direct_execution_model()
    ui._set_ai_direct_execution_model_path(str(tmp_path / "new.pt"))
    ui.ai_direct_execution_model_path_input.setText.assert_not_called()


def test_missing_checkpoint_has_explicit_status(tmp_path):
    ui = controls()
    ui._set_ai_direct_execution_model_path(str(tmp_path / "missing.pt"))
    ui.ai_direct_execution_model_status.setText.assert_called_with(
        "Model: Selected: missing.pt\nCheckpoint file not found")


def test_running_model_details_use_actual_bundle_and_not_filename(tmp_path):
    path = tmp_path / "ambiguous_filename.pt"
    path.touch()
    ui = controls()
    ui._set_ai_direct_execution_model_path(str(path))
    bundle = {"class_names": ["swipe_up", "swipe_down", "swipe_left", "swipe_right"],
        "members": [{"config": {"architecture": "cnn_gru"}}],
        "preprocess": {"prediction_unit": "recent_touch_window", "window_duration_sec": .2}}
    helper = SimpleNamespace(is_running=True, model_loaded=True, model_kind="cnn_tcn_gesture",
        loaded_model_checkpoint_path=str(path), _touch_gesture_bundle=bundle)
    ui._refresh_ai_direct_execution_model_status(helper)
    ui.ai_direct_execution_model_status.setText.assert_called_with(
        f"Model: Running: {path.name} | CNN–GRU · 4 gestures · continuous 200 ms")
    ui.ai_direct_execution_model_details.set_source.assert_called_with(
        str(path), loaded_bundle=bundle, loaded_kind="cnn_tcn_gesture")
    helper.model_kind = "legacy_transformer"
    ui._refresh_ai_direct_execution_model_status(helper)
    ui.ai_direct_execution_model_details.set_source.assert_called_with(
        str(path), loaded_bundle={}, loaded_kind="legacy_transformer")
