"""Compact, shared characterisation settings for single/full-taxel experiments.

The protocol choice stays visible. Only settings the selected protocol uses are
shown: required ones beside it, the rest under collapsed "Advanced" and
"Record details" sections. Hiding a field never changes its value.
"""
from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QLineEdit, QFormLayout,
)
from phd.ui import components, theme

PROTOCOLS = (
    ("Loading / unloading", "standard"),
    ("Hold & recovery (fixed position)", "hold_recovery"),
    ("Speed comparison (cycle speeds)", "speed_sweep"),
    ("Low-force / force-level sweep", "force_sweep"),
    ("Drift / repeated cycling (fixed tactile baseline)", "drift"),
    ("Spatial sliding (fixed Z; supervised)", "sliding"),
    ("Proximity profile (stepwise, contact-referenced)", "proximity_profile"),
)

REQUIRED = {
    "standard": "Max contact threshold (force target), Maximum travel, Timeout and Repetitions",
    "hold_recovery": "Max contact threshold, Maximum travel, Timeout and Repetitions (hold and "
                     "recovery are preset to 30 s under Advanced)",
    "speed_sweep": "the speed list below, Max contact threshold, Maximum travel, Timeout and "
                   "Repetitions (one speed per repetition)",
    "force_sweep": "the force-level list below, Maximum travel, Timeout and Repetitions (one level "
                   "per repetition)",
    "drift": "Max contact threshold, Maximum travel, Timeout and Repetitions",
    "sliding": "the sliding distance, speed and direction below, Max contact threshold (≤200 g), "
               "Maximum travel and Timeout, then tick the preflight",
    "proximity_profile": "Maximum travel (larger than the start-to-surface distance) and Timeout "
                         "(above the estimate below)",
}


def _form_label(field):
    """The QFormLayout label beside ``field`` in its parent widget, if any."""
    parent = field.parentWidget()
    if parent is None:
        return None
    for form in parent.findChildren(QFormLayout):
        label = form.labelForField(field)
        if label is not None:
            return label
    return None


def _set_row_visible(field, visible):
    field.setVisible(visible)
    label = _form_label(field)
    if label is not None:
        label.setVisible(visible)


def build_protocol_panel(
    task, force_spin, repeat_spin, speed_spin=None, *, allow_profile=True,
    selector_layout=None,
):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_SM)
    selector = QWidget()
    selector.setObjectName("calibrationProtocolSelector")
    form = components.SettingsForm(selector)
    panel.protocol_selector = selector
    advanced = components.CollapsibleGroup("Advanced settings")
    record = components.CollapsibleGroup("Record details · IDs, probe and notes")
    advanced_boxes = []

    def section(add):
        # Qt 5 form layouts keep the spacing of hidden rows; hide whole sections instead.
        box = QWidget()
        box_form = components.SettingsForm(box)
        add(box)
        return box, box_form

    mode = QComboBox()
    for label, value in PROTOCOLS:
        if allow_profile or value != "proximity_profile":
            mode.addItem(label, value)
    mode.setObjectName("characterisationProtocol")
    mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    mode.setMinimumContentsLength(18)
    form.addRow("Protocol", mode)
    if selector_layout is None:
        layout.addWidget(selector)
    else:
        selector_layout.insertWidget(0, selector)

    def numeric(target, label, attribute, low, high, suffix, integer=False, decimals=None):
        spin = QSpinBox() if integer else QDoubleSpinBox()
        spin.setObjectName(attribute)
        if decimals is not None and not integer:
            spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setSuffix(suffix)
        spin.setValue(getattr(task, attribute))
        spin.valueChanged.connect(lambda value: setattr(task, attribute, int(value) if integer else float(value)))
        target.addRow(label, spin)
        return spin

    motion = QComboBox()
    for label, value in (
        ("Stepwise position moves (recommended)", "stepwise"),
        ("Continuous velocity (legacy)", "velocity"),
    ):
        motion.addItem(label, value)
    motion.setObjectName("motionMode")
    motion.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    motion.setMinimumContentsLength(18)
    motion.setCurrentIndex(max(0, motion.findData(task.motion_mode)))
    motion.setToolTip(
        "Stepwise: every approach, loading and unloading step is a TM line move measured at rest. "
        "TM velocity mode stalls and jumps by up to ~0.6 mm at low speed; keep it only to "
        "reproduce older recordings."
    )
    motion_box, motion_form = section(layout.addWidget)
    motion_form.addRow("Motion", motion)

    # Required inputs of particular protocols, shown beside the protocol choice.
    speeds = QLineEdit(", ".join(str(v) for v in task.speed_schedule_mm_s))
    speeds.setObjectName("speedSchedule")
    speeds.setToolTip("mm/s, 0.01–2. One speed per repetition/full scan; cycles in this order. Repetitions are total trials, not trials per speed.")

    def set_speeds(text):
        try:
            task.speed_schedule_mm_s = [float(v.strip()) for v in text.split(",") if v.strip()]
        except ValueError:
            task.speed_schedule_mm_s = []  # Start validation explains the error.

    speeds.textChanged.connect(set_speeds)
    speeds_box, speeds_form = section(layout.addWidget)
    speeds_form.addRow("Speed comparison list (mm/s)", speeds)
    forces = QLineEdit(", ".join(str(v) for v in task.force_schedule_g))
    forces.setObjectName("forceSchedule")
    forces.setToolTip("Gram-force levels, >0 to 800 g. One level per trial/full pass; repetitions are total trials/passes, not repeats per level.")

    def set_forces(text):
        try:
            task.force_schedule_g = [float(v.strip()) for v in text.split(",") if v.strip()]
        except ValueError:
            task.force_schedule_g = []

    forces.textChanged.connect(set_forces)
    forces_box, forces_form = section(layout.addWidget)
    forces_form.addRow("Force-level list (g)", forces)
    sliding_box, sliding_form = section(layout.addWidget)
    distance = numeric(sliding_form, "Sliding distance, one way", "sliding_distance_mm", 0.1, 5, " mm")
    speed = numeric(sliding_form, "Sliding speed", "sliding_speed_mm_s", 0.01, 0.5, " mm/s")
    direction = QComboBox()
    for value in ("x+", "x-", "y+", "y-"):
        direction.addItem(f"Robot base {value.upper()}", value)
    direction.currentIndexChanged.connect(lambda _: setattr(task, "sliding_direction", direction.currentData()))
    sliding_form.addRow("Sliding direction", direction)
    confirm = QCheckBox("Path checked; I will supervise")
    confirm.setToolTip("Check the entire out-and-back path. There is no shear-force measurement or guard.")
    confirm.setObjectName("slidingPathConfirmed")
    confirm.toggled.connect(lambda value: setattr(task, "sliding_path_confirmed", bool(value)))
    sliding_form.addRow("Sliding preflight", confirm)
    info = QLabel()
    info.setObjectName("protocolRequired")
    info.setWordWrap(True)
    layout.addWidget(info)
    about = components.CollapsibleGroup("About this protocol")
    details = QLabel()
    details.setObjectName("protocolDescription")
    details.setWordWrap(True)
    details.setStyleSheet(f"color: {theme.TEXT_MUTED};")
    about.add(details)
    layout.addWidget(about)

    # Advanced: every value has a working default.
    def advanced_section():
        box, box_form = section(advanced.add)
        advanced_boxes.append((box, box_form))
        return box, box_form

    step_box, step_form = advanced_section()
    step_widgets = [
        numeric(step_form, label, attribute, low, high, suffix, decimals=3)
        for label, attribute, low, high, suffix in (
            ("Approach step (far)", "step_far_mm", 0.05, 1.0, " mm"),
            ("Approach step (near)", "step_near_mm", 0.01, 0.2, " mm"),
            ("Loading / unloading step", "step_loading_mm", 0.005, 0.1, " mm"),
            ("Time at rest per step", "step_dwell_sec", 0.3, 30, " s"),
        )
    ]
    step_widgets[2].setToolTip(
        "Loading rate = step ÷ step period. Base -Z speed (or the speed list) is the nominal "
        "rate; the time at rest per step sets the fastest attainable rate."
    )
    light_box, light_form = advanced_section()
    light_contact = numeric(
        light_form, "Light-contact threshold", "light_contact_threshold_g", 0.2, 5, " g", decimals=2,
    )
    light_contact.setSingleStep(0.1)
    light_contact.setToolTip(
        "Contact is declared this far above a force zero re-taken at rest near the surface: before "
        "the near approach and before every light-contact probe. Cable drag and recovery after "
        "pressing shift the trial-start zero by up to ~1 g. Raised automatically if the force at "
        "rest does not settle."
    )
    sliding_step_box, sliding_step_form = advanced_section()
    numeric(sliding_step_form, "Sliding step", "sliding_step_mm", 0.01, 1.0, " mm", decimals=3)
    approach_box, approach_form = advanced_section()
    far_field = numeric(approach_form, "Far-field approach speed (0 = off)", "far_field_speed_mm_s", 0, 2, " mm/s")
    far_field.setSingleStep(0.1)
    far_field.setToolTip(
        "Faster base -Z speed used only above this taxel's contact height measured "
        "earlier in the same run, plus the slow zone. The first trial of each taxel "
        "runs entirely at the normal speed. Lets you start high above the surface "
        "(to measure the full proximity range) without hour-long approaches."
    )
    slow_zone = numeric(approach_form, "Slow zone above previous contact", "slow_zone_mm", 1, 50, " mm")
    slow_zone.setToolTip(
        "Distance above the previously measured contact height where the approach "
        "returns to the normal speed. Keep it well above the expected surface "
        "variation; fast approach also stops at the first 1 g force change."
    )
    hold = numeric(approach_form, "Stationary contact hold", "contact_hold_sec", 0, 300, " s")
    _common_box, common_form = advanced_section()
    recovery = numeric(common_form, "Unloaded recovery observation", "recovery_observation_sec", 0, 300, " s")
    numeric(common_form, "Stationary baseline frames", "sensor_baseline_frames", 20, 1000, " unique frames", True)
    numeric(common_form, "Minimum baseline time", "sensor_baseline_duration_sec", 1, 25, " s")
    reset = QCheckBox("Recalibrate between trials")
    reset.setObjectName("recalibrateBetweenTrials")
    reset.setChecked(task.recalibrate_between_trials)
    reset.toggled.connect(lambda checked: setattr(task, "recalibrate_between_trials", bool(checked)))
    common_form.addRow("Baseline policy", reset)
    gaps = QLineEdit(", ".join(f"{v:g}" for v in task.profile_gaps_mm))
    gaps.setObjectName("profileGaps")
    gaps.setToolTip(
        "Gaps above the light-contact surface, 0.2–100 mm. The largest is the far-field "
        "reference, revisited during every sweep to remove baseline drift."
    )

    def set_gaps(text):
        try:
            task.profile_gaps_mm = [float(v.strip()) for v in text.split(",") if v.strip()]
        except ValueError:
            task.profile_gaps_mm = []
        refresh_info()

    gaps.textChanged.connect(set_gaps)
    profile_box, profile_form = advanced_section()
    profile_form.addRow("Profile gaps (mm)", gaps)
    order = QComboBox()
    for label, value in (("Alternate down / up sweeps", "alternate"), ("Always far to near", "descending"), ("Random order per sweep", "random")):
        order.addItem(label, value)
    order.setObjectName("profileOrder")
    order.setCurrentIndex(max(0, order.findData(task.profile_order)))
    order.currentIndexChanged.connect(lambda _: setattr(task, "profile_order", order.currentData()))
    profile_form.addRow("Profile level order", order)
    for label, attribute, low, high, suffix, integer in (
        ("Profile sweeps", "profile_sweeps", 1, 20, "", True),
        ("Frames per level", "profile_frames_per_level", 10, 500, " unique frames", True),
        ("Settle before each level", "profile_settle_sec", 0, 10, " s", False),
        ("Far reference every", "profile_reference_every", 1, 20, " levels", True),
        ("Surface light-contact probes", "profile_surface_probes", 1, 10, "", True),
        ("Contact guard at levels", "profile_surface_stop_g", 1, 20, " g", False),
        ("Surface search step (far)", "profile_search_step_far_mm", 0.05, 1.0, " mm", False),
        ("Surface search step (near)", "profile_search_step_near_mm", 0.01, 0.2, " mm", False),
        ("Light-contact probe step", "profile_probe_step_mm", 0.005, 0.05, " mm", False),
        ("Profile move speed limit", "profile_move_speed_mm_s", 1, 20, " mm/s", False),
    ):
        spin = numeric(profile_form, label, attribute, low, high, suffix, integer, decimals=None if integer else 3)
        spin.valueChanged.connect(lambda _value: refresh_info())
    remeasure = QCheckBox("Re-measure the surface at the end")
    remeasure.setObjectName("profileRemeasureSurface")
    remeasure.setChecked(task.profile_remeasure_surface)
    remeasure.toggled.connect(lambda value: setattr(task, "profile_remeasure_surface", bool(value)))
    profile_form.addRow("Surface drift check", remeasure)
    layout.addWidget(advanced)

    # Record details: written to the metadata; they do not change the motion.
    _record_box, record_form = section(record.add)
    for label, attribute in (("Physical specimen ID", "specimen_id"), ("Session / day ID", "session_id"), ("Mounting / remount ID", "mounting_id")):
        field = QLineEdit(getattr(task, attribute))
        field.setObjectName(attribute)
        field.setPlaceholderText("Record explicitly for publication comparisons")
        field.textChanged.connect(lambda value, name=attribute: setattr(task, name, value.strip()))
        record_form.addRow(label, field)
    geometry = QComboBox()
    for label, value in (("Unknown / unspecified", "unspecified"), ("Flat circular tip", "flat_circular"), ("Hemispherical tip", "hemisphere"), ("Human finger", "finger"), ("Other", "other")):
        geometry.addItem(label, value)
    geometry.setObjectName("probeGeometry")
    geometry.currentIndexChanged.connect(lambda _: setattr(task, "probe_geometry", geometry.currentData()))
    record_form.addRow("Probe geometry", geometry)
    material = QComboBox()
    material.setObjectName("probeMaterial")
    for label, value in (
        ("Conductive", "conductive"),
        ("Insulative", "insulative"),
        ("Human-like", "human_like"),
        ("Other", "other"),
    ):
        material.addItem(label, value)
    material.setCurrentIndex(max(0, material.findData(task.probe_material)))
    material_note = QLineEdit(task.probe_material_note)
    material_note.setObjectName("probe_material_note")
    material_note.setMaxLength(80)
    material_note.textChanged.connect(lambda text: setattr(task, "probe_material_note", text.strip()))

    def sync_material():
        task.probe_material = material.currentData()
        if task.probe_material == "other":
            material_note.setPlaceholderText("Type the probe material. This names the folder.")
        else:
            material_note.setPlaceholderText("Optional. For example, copper foil. Added to the folder name.")
        _set_row_visible(material_note, True)

    material.currentIndexChanged.connect(lambda _: sync_material())
    record_form.addRow("Probe material", material)
    record_form.addRow("Probe material description", material_note)
    sync_material()
    numeric(record_form, "Probe diameter (0 = unknown)", "probe_diameter_mm", 0, 100, " mm")
    verified = QCheckBox("Checked zero and known loads within the rated range")
    verified.setObjectName("forceReferenceVerified")
    verified.setChecked(task.force_reference_verified)
    verified.toggled.connect(lambda value: setattr(task, "force_reference_verified", bool(value)))
    record_form.addRow("Force-reference verification", verified)
    area = QCheckBox("Known flat-tip area is fully in contact (nominal pressure only)")
    area.setObjectName("contactAreaConfirmed")
    area.setChecked(task.contact_area_confirmed)
    area.toggled.connect(lambda value: setattr(task, "contact_area_confirmed", bool(value)))
    notes = QLineEdit()
    notes.setObjectName("characterisationNotes")
    notes.setPlaceholderText("Probe diameter/material, grounding, mounting, session, temperature…")
    notes.textChanged.connect(lambda text: setattr(task, "experiment_notes", text))
    record_form.addRow("Setup / condition notes", notes)
    area_box, area_form = section(record.add)
    area_form.addRow("Contact-area declaration", area)
    layout.addWidget(record)

    def select_protocol(_index, presets=True):
        task.protocol = mode.currentData()
        profile = task.protocol == "proximity_profile"
        sliding = task.protocol == "sliding"
        stepwise = task.motion_mode == "stepwise" and not profile
        motion_box.setVisible(not profile)
        speeds_box.setVisible(task.protocol == "speed_sweep")
        speeds.setEnabled(task.protocol == "speed_sweep")
        forces_box.setVisible(task.protocol == "force_sweep")
        sliding_box.setVisible(sliding)
        for widget in (distance, speed, direction, confirm):
            widget.setEnabled(sliding)
        step_box.setVisible(stepwise)
        light_box.setVisible(stepwise or profile)
        sliding_step_box.setVisible(stepwise and sliding)
        approach_box.setVisible(not profile)
        profile_box.setVisible(profile)
        area_box.setVisible(not profile)
        count = sum(box_form.rowCount() for box, box_form in advanced_boxes if not box.isHidden())
        advanced.toggle.setText(f"Advanced settings ({count}) · defaults are fine")
        _set_row_visible(force_spin, not profile)
        if speed_spin is not None:
            _set_row_visible(speed_spin, not profile)
        force_spin.setEnabled(task.protocol not in {"force_sweep", "proximity_profile"})
        confirm.setChecked(False)
        if presets:
            hold.setValue(30 if task.protocol in {"hold_recovery", "drift"} else 1)
            recovery.setValue(30 if task.protocol in {"hold_recovery", "drift"} else 5)
            reset.setChecked(task.protocol != "drift")
            force_spin.setValue(100 if sliding else 500)
            if task.protocol == "speed_sweep":
                if task.motion_mode == "stepwise":
                    speeds.setText("0.005, 0.01, 0.02")  # Attainable with 0.02 mm steps.
                repeat_spin.setValue(max(3, len(task.speed_schedule_mm_s)))
            elif task.protocol == "force_sweep":
                repeat_spin.setValue(max(1, len(task.force_schedule_g)))
            elif profile:
                repeat_spin.setValue(1)
        refresh_info()

    def refresh_info():
        required = f"<b>To set:</b> {REQUIRED[task.protocol]}. Under Advanced the defaults are fine."
        if task.id == "full_taxel_calibration":
            required += (
                "<br><b>Recording:</b> tested taxel only, plus force, position and timing. "
                "Other taxels are not saved in each trial CSV."
            )
        if task.protocol == "proximity_profile":
            try:
                _speed, percent = task._profile_speed_mm_s()
                required += (
                    f"<br><b>Duration:</b> about {task.estimate_profile_duration_sec() / 60.0:.0f} min after "
                    f"the surface plus up to {task.estimate_profile_descent_sec() / 60.0:.0f} min of descent, "
                    f"at {percent:.0f}% TM project speed; set Timeout above it (Run re-checks it with the "
                    "speed the robot reports)."
                )
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                required += "<br><b>Duration:</b> unknown until the settings are valid."
            info.setText(required)
            details.setText(
                "Never uses velocity mode (it moves in stick-slip jumps at low speed). Steps down with "
                "position moves, checking force at rest after every step: far steps until the proximity "
                "signal is strong (10σ, at least 20 counts), then near steps to light contact, then fine "
                "probes fix the surface. The force zero follows slow drift during the far steps and is "
                "re-taken at rest before the near steps and before every probe; light contact is the "
                "threshold above that zero. It then moves to each gap, holds still and records every taxel. "
                "The largest gap is the far-field reference, revisited during each sweep so baseline drift "
                "is interpolated and removed. Reports detection distance at 3σ/5σ of single-frame noise for "
                "the target taxel and for the localized signal (target minus distant taxels, which rejects "
                "drift and arm/tool coupling shared by the whole sensor), each with its per-sweep spread, "
                "plus distance resolution and neighbour dominance. Use at least 5 sweeps for a usable range "
                "interval. Keep the start pose above the surface and Maximum travel larger than the "
                "start-to-surface distance. Use the same grounded probe for every run and note it under "
                "Record details."
            )
            return
        info.setText(required)
        motion_note = (
            "Stepwise motion: approach, loading and unloading are TM line steps measured at rest; "
            "loading halves its step above 80% of the target and the rate is step ÷ step period. "
            "The force zero is re-taken at rest before the near approach, so light contact and every "
            "later force are measured from it. "
            if task.motion_mode == "stepwise" else
            "Legacy velocity motion stalls and jumps by up to ~0.6 mm at low speed on this robot. "
        )
        details.setText(
            "Installed load cell: 1 kg rated. Target ≤800 g; independent stop at 900 g. "
            "All nodes are recorded. Pauses remain in real elapsed time and interrupted trials are flagged. "
            + motion_note
            + {
                "standard": "Each trial reports the proximity range, the drop-to-rise turning force and the force where the signal returns to baseline. Start clearly outside proximity range and use a target above the turning force.",
                "hold_recovery": "Position is fixed during the hold: force may relax. This is not constant-force creep.",
                "speed_sweep": "One nominal speed per repetition (or full scan), with the same ¼-speed slowdown above 80% target. Use repeats of each speed for uncertainty.",
                "force_sweep": "One force level per trial/full pass; retract completely between levels. Low-force detectability requires independent force-reference verification, baseline noise and repeated held-out contacts; this is not constant-force control.",
                "drift": "The tactile calibration is kept fixed. Each trial still measures its own stationary noise and force baseline. Raw signals preserve long-term drift.",
                "sliding": "Robot base XY out-and-back at fixed Z, then +Z return. Target ≤200 g. Stops on lost contact, excessive normal force or path deviation. No automatic force control or shear measurement.",
            }[task.protocol]
        )

    def select_motion(_index):
        task.motion_mode = motion.currentData()
        select_protocol(0, presets=False)

    mode.currentIndexChanged.connect(select_protocol)
    motion.currentIndexChanged.connect(select_motion)
    select_protocol(0, presets=False)

    def sync_from_task():
        """Display a restored run without applying protocol presets or emitting edits."""
        widgets = panel.findChildren(QWidget) + selector.findChildren(QWidget)
        blockers = [QSignalBlocker(widget) for widget in widgets]
        for widget in widgets:
            name = widget.objectName()
            if name and hasattr(task, name):
                value = getattr(task, name)
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
        for widget, value in ((mode, task.protocol), (motion, task.motion_mode),
                              (geometry, task.probe_geometry), (material, task.probe_material),
                              (direction, task.sliding_direction)):
            widget.setCurrentIndex(widget.findData(value))
        speeds.setText(", ".join(str(v) for v in task.speed_schedule_mm_s))
        forces.setText(", ".join(str(v) for v in task.force_schedule_g))
        notes.setText(task.experiment_notes)
        reset.setChecked(task.recalibrate_between_trials)
        verified.setChecked(task.force_reference_verified)
        area.setChecked(task.contact_area_confirmed)
        sync_material()
        select_protocol(0, presets=False)
        del blockers

    panel.sync_from_task = sync_from_task
    return panel
