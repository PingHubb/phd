"""Read-only planning for loading/unloading; never a hardware safety approval.

A mapping scan height is a known non-contact position, not a surface/force
calibration. Use it only as a travel lower bound. Suggested extra travel and
time are explicit planning allowances, not promises of reaching a force.
"""
from __future__ import annotations

import math


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mapped_descent(task):
    """Deepest mapped scan below the actual trial start, in mm, and its taxel."""
    result = getattr(task, "position_map", None) or {}
    records = result.get("positions") or []
    full = task.id == "full_taxel_calibration"
    if full and task.initial_position is None:
        return None
    if not full:
        initial = task.initial_position or {}
        if initial.get("source") != "taxel_position_map":
            return None
        records = [r for r in records if r.get("taxel_index") == task.taxel_index]
    depths = []
    for record in records:
        if record.get("status") != "measured":
            continue
        start = record.get("calibration_start_position_m") if full else initial.get("tool_position_m")
        scan = record.get("tool_position_m")
        if not isinstance(start, (list, tuple)) or not isinstance(scan, (list, tuple)) or len(start) != 3 or len(scan) != 3:
            continue
        start_z, scan_z = _number(start[2]), _number(scan[2])
        if start_z is None or scan_z is None or start_z < scan_z:
            continue
        depths.append(((start_z - scan_z) * 1000., int(record["taxel_index"])))
    return max(depths) if depths else None


def travel_error(task):
    bound = mapped_descent(task)
    travel = _number(task.max_travel_m)
    if bound is None or travel is None:
        return ""
    depth, taxel = bound
    if travel * 1000. <= depth + 1e-6:
        return (
            f"Maximum travel is {travel * 1000.:.1f} mm, but Taxel {taxel}'s mapped non-contact "
            f"scan is already {depth:.1f} mm below its safe start. This cannot complete loading. "
            "Review the Before running panel and allow additional travel for contact/compression "
            "after checking the physical setup (100 mm absolute limit)."
        )
    return ""


def loading_plan(task, *, travel_mm=None, extra_mm=5., project_speed_percent=None):
    """Conservative planning budget for standard cycles, with visible assumptions.

    Budget the last 30 mm above the mapped plane as near steps and all travel
    below that plane as slow loading/unloading. Real contact depth/stiffness,
    controller latency, pauses and settling remain unknown until a pilot.
    """
    if task.protocol != "standard":
        return None
    bound = mapped_descent(task)
    if bound is None:
        return None
    depth, taxel = bound
    extra = _number(extra_mm)
    travel = _number(task.max_travel_m * 1000. if travel_mm is None else travel_mm)
    percent = _number(project_speed_percent)
    speed_assumed = percent is None or percent <= 0
    percent = min(percent, 100.) if percent is not None and percent > 0 else 25.
    if extra is None or extra <= 0 or travel is None or travel <= 0:
        return None
    far, near, step, dwell, speed = map(_number, (
        task.step_far_mm, task.step_near_mm, task.step_loading_mm,
        task.step_dwell_sec, abs(task.approach_speed_m_s) * 1000.,
    ))
    if any(value is None or value <= 0 for value in (far, near, step, speed)) or dwell is None or dwell < 0:
        return None
    suggested_travel = math.ceil((depth + extra) * 10. - 1e-8) / 10.
    remaining = max(0., travel - depth)
    if task.motion_mode == "stepwise":
        move_speed = task.STEP_MOVE_SPEED_M_S * 1000. * percent / 100.
        # Includes controller/settle time and the force window. It is a model,
        # not a guaranteed upper bound and never bypasses runtime guards.
        def period(size, nominal=False):
            return max(1.3 + size / move_speed, dwell + 0.8,
                       step / speed if nominal else 0.)
        near_depth = min(depth, 30.)
        approach = (math.ceil(max(0., depth - near_depth) / far) * period(far)
                    + math.ceil(near_depth / near) * period(near))
        small = max(0.005, step * .5)
        loaded = (math.ceil(remaining / small) * period(small, True)
                  + math.ceil(remaining / step) * period(step, True))
        retract = travel / (10. * percent / 100.)
    else:
        approach = depth / speed
        loaded = remaining / (speed * .25) + remaining / speed
        retract = depth / speed
    fixed = (max(task.sensor_baseline_duration_sec, task.force_baseline_duration_sec)
             + task.FORCE_ZERO_MAX_WAIT_SEC + task.contact_hold_sec
             + task.recovery_observation_sec + task.return_settle_sec + 10.)
    estimate = approach + loaded + retract + fixed
    suggested_timeout = math.ceil((estimate * 1.25 + 60.) / 60.) * 60.
    return {
        "mapped_descent_mm": depth, "limiting_taxel": taxel,
        "extra_mm": extra, "travel_mm": travel,
        "suggested_travel_mm": suggested_travel,
        "estimated_cycle_sec": estimate, "suggested_timeout_sec": suggested_timeout,
        "project_speed_percent": percent, "near_zone_assumption_mm": 30.,
        "project_speed_assumed": speed_assumed,
        "nominal_loading_rate_mm_s": speed,
        "step_rate_ceiling_mm_s": step / task.STEP_MIN_PERIOD_SEC,
    }


def suggested_loading_plan(task, *, project_speed_percent=None):
    """Recalculate the timeout at the proposed travel, not the old travel."""
    plan = loading_plan(task, extra_mm=task.loading_extra_travel_mm,
                        project_speed_percent=project_speed_percent)
    if plan is None:
        return None
    return loading_plan(task, travel_mm=plan["suggested_travel_mm"],
                        extra_mm=task.loading_extra_travel_mm,
                        project_speed_percent=project_speed_percent)


def loading_plan_error(plan):
    """Do not partially apply or silently clip an unachievable suggestion."""
    if plan is None:
        return "Load a matching map and select a mapped target (Single taxel) before automatic planning. Check the step and speed settings."
    if not 0. < plan["extra_mm"] <= 20.:
        return "Extra travel below the mapped scan must be above 0 and at most 20 mm."
    if not 0. < plan["suggested_travel_mm"] <= 100.:
        return (f"Planned travel {plan['suggested_travel_mm']:.1f} mm exceeds the 100 mm limit. "
                "Review the anchor and extra allowance; the limit will not be increased or clipped.")
    if not 0. < plan["suggested_timeout_sec"] <= 7200.:
        return (f"Planned timeout {plan['suggested_timeout_sec']:.0f} s exceeds the 7,200 s limit. "
                "Review travel, step and dwell settings, or switch to reviewed manual limits after a pilot.")
    return ""
