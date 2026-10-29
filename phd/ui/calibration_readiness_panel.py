"""Inline loading/unloading setup checks; applying a plan never starts motion."""
from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
    QSpinBox, QComboBox, QCheckBox, QLineEdit,
)

from phd.ui import components, theme
from phd.ui.calibration_readiness import (
    loading_plan, loading_plan_error, suggested_loading_plan, travel_error,
)


class CalibrationReadinessPanel(QWidget):
    def __init__(self, task, travel_spin, timeout_spin, protocol_panel, controls, parent=None, *, host=None, auto_plan=True):
        # Refresh changes visibility, so attach before it can run; never show
        # an accidental standalone window while the form is being built.
        super().__init__(parent or protocol_panel.parentWidget())
        self.task, self.travel_spin, self.timeout_spin = task, travel_spin, timeout_spin
        self.host = host
        self.task.loading_auto_plan = bool(auto_plan)
        self._refreshing = False
        self._limit_tooltips = [spin.toolTip() for spin in (travel_spin, timeout_spin)]
        self.setObjectName("calibrationReadiness")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.setAlignment(Qt.AlignTop)
        title = QLabel("Before running · loading / unloading")
        title.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-weight: 600;")
        layout.addWidget(title)
        self.auto_plan = QCheckBox("Automatically plan travel and timeout from map")
        self.auto_plan.setObjectName("autoPlanLoadingLimits")
        self.auto_plan.setChecked(task.loading_auto_plan)
        self.auto_plan.setToolTip(
            "Loading / unloading only. Updates these two values while idle and rechecks on Run. "
            "Uncheck to edit them manually. Never starts motion or learns extra travel from failed trials."
        )
        self.auto_plan.toggled.connect(self._set_automatic)
        layout.addWidget(self.auto_plan)
        self.status = QLabel()
        self.status.setObjectName("calibrationReadinessStatus")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Calibration setup checks")
        layout.addWidget(self.status)
        self.options = components.CollapsibleGroup("Plan travel and time limits")
        form_widget = QWidget()
        form = components.SettingsForm(form_widget)
        self.extra = QDoubleSpinBox()
        self.extra.setObjectName("calibrationExtraTravel")
        self.extra.setRange(.5, 20.)
        self.extra.setDecimals(1)
        self.extra.setValue(task.loading_extra_travel_mm)
        self.extra.valueChanged.connect(lambda value: setattr(task, "loading_extra_travel_mm", float(value)))
        self.extra.setSuffix(" mm")
        self.extra.setToolTip(
            "Planning allowance below the mapped non-contact scan plane—not measured compression. "
            "Check clearance and confirm the required travel with a supervised single-taxel pilot."
        )
        form.addRow("Extra travel below mapped scan", self.extra)
        self.options.add(form_widget)
        self.proposal = QLabel()
        self.proposal.setWordWrap(True)
        self.proposal.setTextFormat(Qt.PlainText)
        self.options.add(self.proposal)
        self.apply_button = QPushButton("Apply suggested travel and timeout")
        self.apply_button.setObjectName("applyCalibrationLimits")
        self.apply_button.setToolTip("Updates these two settings only. Does not move the robot, start a run or change force/speed.")
        self.apply_button.clicked.connect(self.apply_suggestions)
        self.options.add(self.apply_button)
        layout.addWidget(self.options)
        self.extra.valueChanged.connect(self.refresh)
        watched = list(controls) + [travel_spin, timeout_spin]
        for cls in (QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QLineEdit):
            watched.extend(protocol_panel.findChildren(cls))
        # Protocol selector may be placed outside protocol_panel.
        watched.extend(protocol_panel.protocol_selector.findChildren(QComboBox))
        for widget in dict.fromkeys(watched):
            for signal in ("valueChanged", "currentIndexChanged", "toggled", "textChanged"):
                callback = getattr(widget, signal, None)
                if callback is not None:
                    callback.connect(self.refresh)
                    break
        self.refresh()

    def _plans(self):
        percent = None
        getter = getattr(self.task._robot_api(self.host), "get_project_speed_percent", None)
        if callable(getter):
            try:
                percent = getter()  # Cached robot feedback, never a motion command.
            except Exception:
                pass
        current = loading_plan(self.task, extra_mm=self.extra.value(), project_speed_percent=percent)
        proposed = suggested_loading_plan(self.task, project_speed_percent=percent)
        return current, proposed

    def _busy(self):
        active = getattr(self.host, "_sidebar_active_task", None)
        return (self.task._test_active or self.task._phase == "connecting_force_meter"
                or getattr(self.task, "resume_plan", None) is not None
                or (active is not None and active is not self.task))

    def _set_automatic(self, enabled):
        if self._busy():
            blocker = QSignalBlocker(self.auto_plan)
            self.auto_plan.setChecked(self.task.loading_auto_plan)
            return
        self.task.loading_auto_plan = bool(enabled)
        self.refresh()

    def sync_limits(self):
        """Show an atomic task update without emitting partially updated settings."""
        for spin, value in ((self.travel_spin, self.task.max_travel_m * 1000.),
                            (self.timeout_spin, self.task.timeout_sec)):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker

    def _apply_plan(self, proposed):
        self.task.max_travel_m = proposed["suggested_travel_mm"] / 1000.
        self.task.timeout_sec = proposed["suggested_timeout_sec"]
        self.sync_limits()

    def refresh(self, *_args, apply_auto=True):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._refresh(apply_auto=apply_auto)
        finally:
            self._refreshing = False

    def _refresh(self, *, apply_auto):
        self.setVisible(self.task.protocol == "standard")
        busy = self._busy()
        automatic = self.task.loading_auto_plan and self.task.protocol == "standard"
        current, proposed = self._plans()
        error = loading_plan_error(proposed)
        if automatic and apply_auto and not busy and not error:
            self._apply_plan(proposed)
            current = proposed
        self.auto_plan.setEnabled(not busy)
        for spin, tooltip in zip((self.travel_spin, self.timeout_spin), self._limit_tooltips):
            spin.setReadOnly(automatic)
            spin.setToolTip(
                "Automatically planned. Uncheck automatic planning to edit manually. " + tooltip
                if automatic else tooltip
            )
        mode = "Automatic" if automatic else "Manual"
        lines = [f"{mode} limits: {self.task.max_travel_m * 1000.:.1f} mm travel · "
                 f"{self.task.timeout_sec:.0f} s per taxel/trial" + (" · locked during run" if busy else "")]
        if automatic and error:
            lines.append("AUTO PLAN BLOCKED: " + error + " Values have not been applied.")
        if current is None:
            if not automatic:
                lines.append("Load a matching position map; for Single taxel, select the target to check travel.")
        else:
            lines.append(f"Map: Taxel {current['limiting_taxel']} scan {current['mapped_descent_mm']:.1f} mm below start; "
                         f"extra allowance {self.extra.value():g} mm (not measured compression).")
            if travel_error(self.task):
                lines.append("BLOCKED: Maximum travel cannot pass the mapped scan plane. Review the limits below.")
            budget = current["suggested_timeout_sec"]
            if self.task.timeout_sec < budget:
                lines.append(f"TIME WARNING: {self.task.timeout_sec:.0f} s is below the planning allowance of {budget:.0f} s per taxel/trial. Review the limits below.")
        lines.append("Check clearance and run a supervised low-force pilot. Target ≤800 g; emergency stop 900 g.")
        self.status.setText("\n".join(lines))
        valid = not error
        self.apply_button.setVisible(not automatic)
        self.apply_button.setEnabled(valid and not busy)
        self.extra.setEnabled(not busy)
        if proposed is None:
            self.proposal.setText("Suggestions become available after a mapped target is loaded/selected.")
        else:
            suffix = ""
            if not valid:
                suffix = "\nOutside the 100 mm / 7,200 s limits: review the anchor, steps and allowance. Values will not be silently clipped."
            self.proposal.setText(
                f"Proposed travel: {proposed['suggested_travel_mm']:.1f} mm. Timeout: {proposed['suggested_timeout_sec']:.0f} s per taxel/trial.\n"
                "Estimate assumes the final 30 mm above the scan plane use near steps and all extra travel uses slow loading/unloading. "
                f"TM project speed: {proposed['project_speed_percent']:.0f}% "
                f"({'assumed; no speed feedback' if proposed['project_speed_assumed'] else 'reported'}). "
                "Includes baseline, hold, return, recovery and 25% time margin. It is a planning model, not a guaranteed duration or measured contact height. "
                "Confirm the extra travel is physically clear before Run. Force, speed and repetitions stay unchanged. "
                "A 1,000 g target is not allowed." + suffix
                + "\nStep settling can make the actual loading rate slower than Base -Z speed; the recording includes the actual rate."
            )

    def apply_suggestions(self):
        if self._busy():
            return
        _current, proposed = self._plans()
        if loading_plan_error(proposed):
            return
        self._apply_plan(proposed)
        self.refresh()
