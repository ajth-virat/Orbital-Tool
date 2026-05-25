"""
adcs_timeline.py — ADCS mode timeline and attitude control calculations.

Logic:
This file generates the sequence of attitude control modes a spacecraft cycles through.
Each mode has a trigger condition, an exit condition, and an estimated power and duration.
Detumbling is always first — the spacecraft enters a chaotic spin state at separation.
Sun acquisition follows detumbling to establish a power-positive safe attitude.
Nominal nadir-pointing is the operational mode for most Earth-observation missions.
Slew maneuvers are computed geometrically from angular separation between targets.
Power budget contributions are included so the output feeds directly into power subsystem design.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# ADCS MODE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

ADCS_MODES = {
    "DETUMBLE":      {"id": 0, "label": "Detumbling (B-dot)",       "color": "#EF5350"},
    "SUN_ACQ":       {"id": 1, "label": "Sun Acquisition",          "color": "#FF8A65"},
    "SAFE":          {"id": 2, "label": "Safe Mode (Sun-pointing)",  "color": "#FFA726"},
    "NADIR":         {"id": 3, "label": "Nominal Nadir-pointing",    "color": "#66BB6A"},
    "SLEW":          {"id": 4, "label": "Slew Maneuver",            "color": "#42A5F5"},
    "INERTIAL":      {"id": 5, "label": "Inertial Hold",            "color": "#AB47BC"},
    "SUN_POINTING":  {"id": 6, "label": "Solar Charging Hold",      "color": "#26C6DA"},
}


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ADCSMode:
    """Single ADCS mode interval."""
    mode_key      : str
    label         : str
    start_s       : float    # start time from epoch [s]
    duration_s    : float    # duration in this mode [s]
    power_w       : float    # ADCS subsystem power draw [W]
    notes         : str = ""

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def energy_wh(self) -> float:
        return self.power_w * self.duration_s / 3600


@dataclass
class SlewResult:
    """Single slew maneuver computation."""
    slew_angle_deg    : float
    slew_duration_s   : float    # time to complete slew [s]
    slew_rate_deg_s   : float    # peak slew rate [deg/s]
    reaction_wheel_Nm : float    # required torque [Nm]
    energy_wh         : float    # energy consumed [Wh]
    feasible          : bool     # True if within actuator limits
    notes             : str = ""


@dataclass
class DetumbleResult:
    """B-dot detumbling analysis."""
    initial_rate_deg_s : float   # assumed initial tumble rate [deg/s]
    estimated_time_s   : float   # time to reach stable rate [s]
    estimated_time_hr  : float
    magnetorquer_Am2   : float   # required dipole moment [Am²]
    notes              : str = ""


@dataclass
class ADCSTimeline:
    """Full ADCS mode timeline output."""
    modes              : list        # list of ADCSMode in sequence
    total_duration_s   : float
    detumble           : DetumbleResult
    eclipse_mask       : np.ndarray  # bool (N,) — True = eclipse
    power_budget       : dict        # mode → avg power [W]
    mode_fractions     : dict        # mode → fraction of total time

    def summary(self) -> str:
        total_hr = self.total_duration_s / 3600
        lines = [
            f"ADCS Mode Timeline Summary",
            f"--------------------------",
            f"Total duration      : {total_hr:.2f} hr",
            f"",
            f"Detumbling estimate:",
            f"  Initial rate      : {self.detumble.initial_rate_deg_s:.1f} deg/s",
            f"  Est. duration     : {self.detumble.estimated_time_hr:.2f} hr",
            f"  Magnetorquer req  : {self.detumble.magnetorquer_Am2:.2f} Am²",
            f"",
            f"Mode breakdown:",
        ]
        for mode_key, frac in self.mode_fractions.items():
            label = ADCS_MODES[mode_key]["label"]
            pw    = self.power_budget.get(mode_key, 0.0)
            lines.append(
                f"  {label:<35}: {frac*100:5.1f} %  |  {pw:.1f} W"
            )
        lines += [
            f"",
            f"Mode sequence ({len(self.modes)} segments):",
        ]
        for m in self.modes:
            lines.append(
                f"  [{m.start_s/3600:6.2f} - {m.end_s/3600:6.2f} hr]  "
                f"{m.label:<35}  {m.power_w:.1f} W"
            )
            if m.notes:
                lines.append(f"    → {m.notes}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# DETUMBLING — B-DOT CONTROLLER ESTIMATE
# ══════════════════════════════════════════════════════════════════

def estimate_detumble(mass: float,
                      inertia: float,
                      initial_rate_deg_s: float = 10.0,
                      target_rate_deg_s: float  = 0.5,
                      dipole_Am2: float          = None,
                      altitude_km: float         = 400.0) -> DetumbleResult:
    """
    Estimate B-dot detumbling time and required magnetorquer size.

    B-dot control applies a magnetic dipole anti-parallel to the rate of change
    of the measured magnetic field. Each half-orbit provides a damping impulse.

    Parameters
    ----------
    mass               : float — Spacecraft mass [kg]
    inertia            : float — Approximate moment of inertia [kg·m²]
                                 (use mass * (size/2)^2 as rough estimate)
    initial_rate_deg_s : float — Initial tumble rate after separation [deg/s]
    target_rate_deg_s  : float — Target rate to exit detumbling [deg/s]
    dipole_Am2         : float — Magnetorquer dipole moment [Am²]
                                 (auto-sized if None)
    altitude_km        : float — Orbit altitude [km] for B-field estimate

    Returns
    -------
    DetumbleResult
    """
    # Earth magnetic field at altitude — approximate dipole model
    # B ≈ 3e-5 * (Re/r)^3 T at equator
    r_m = CONST.R_EARTH + altitude_km * 1e3
    B_field_T = 3.0e-5 * (CONST.R_EARTH / r_m)**3   # Tesla

    # Auto-size magnetorquer if not given
    # Rule of thumb: dipole ≈ inertia * initial_rate / (B * time_constant)
    time_constant_s = 3600.0   # target ~1 hour per decade of rate reduction
    if dipole_Am2 is None:
        omega_rad = np.radians(initial_rate_deg_s)
        dipole_Am2 = inertia * omega_rad / (B_field_T * time_constant_s)
        dipole_Am2 = max(dipole_Am2, 0.01)   # minimum 0.01 Am² for CubeSat

    # Damping torque per half-orbit
    torque_Nm = dipole_Am2 * B_field_T

    # Angular momentum to remove
    omega_i = np.radians(initial_rate_deg_s)
    omega_f = np.radians(target_rate_deg_s)
    H_to_remove = inertia * (omega_i - omega_f)   # kg·m²/s

    # Impulse per orbit from magnetorquer — effective over half orbit
    T_orbit_s = 2 * np.pi * np.sqrt((CONST.R_EARTH + altitude_km * 1e3)**3 / CONST.GM)
    impulse_per_orbit = torque_Nm * T_orbit_s / 2   # Nm·s per orbit

    n_orbits = max(H_to_remove / impulse_per_orbit, 1.0)
    detumble_time_s = n_orbits * T_orbit_s

    return DetumbleResult(
        initial_rate_deg_s = initial_rate_deg_s,
        estimated_time_s   = detumble_time_s,
        estimated_time_hr  = detumble_time_s / 3600,
        magnetorquer_Am2   = dipole_Am2,
        notes              = (
            f"B-field at {altitude_km:.0f} km: {B_field_T*1e6:.2f} μT | "
            f"Torque: {torque_Nm*1e6:.2f} μNm | "
            f"Est. {n_orbits:.1f} orbits to detumble"
        )
    )


# ══════════════════════════════════════════════════════════════════
# SLEW MANEUVER
# ══════════════════════════════════════════════════════════════════

def compute_slew(slew_angle_deg: float,
                 inertia: float,
                 max_torque_Nm: float,
                 max_rate_deg_s: float = 1.0,
                 rw_power_w: float     = 10.0) -> SlewResult:
    """
    Compute slew maneuver time and energy using a bang-bang torque profile.

    Bang-bang is the minimum-time control law: full torque until halfway,
    then full opposite torque to decelerate. Rate is capped by wheel limit.

    Parameters
    ----------
    slew_angle_deg : float — Angular separation to slew [deg]
    inertia        : float — Spacecraft moment of inertia [kg·m²]
    max_torque_Nm  : float — Reaction wheel peak torque [Nm]
    max_rate_deg_s : float — Maximum slew rate [deg/s] (wheel saturation limit)
    rw_power_w     : float — Reaction wheel power draw [W]

    Returns
    -------
    SlewResult
    """
    theta = np.radians(slew_angle_deg)
    tau   = max_torque_Nm
    I     = inertia

    # Angular acceleration from torque
    alpha = tau / I   # rad/s²

    # Time to peak rate using bang-bang (half slew)
    omega_max = np.radians(max_rate_deg_s)
    t_accel   = omega_max / alpha
    theta_accel = 0.5 * alpha * t_accel**2   # angle covered in acceleration phase

    if 2 * theta_accel >= theta:
        # Short slew — rate limit not reached
        t_half     = np.sqrt(theta / (2 * alpha))
        slew_time  = 2 * t_half
        peak_rate  = np.degrees(alpha * t_half)
    else:
        # Longer slew — coasts at max rate in between
        theta_coast = theta - 2 * theta_accel
        t_coast     = theta_coast / omega_max
        slew_time   = 2 * t_accel + t_coast
        peak_rate   = max_rate_deg_s

    # Energy: power × time + settling margin (10%)
    energy_wh = rw_power_w * slew_time / 3600 * 1.1

    feasible = slew_angle_deg <= 360 and slew_time < 3600

    notes = ""
    if slew_time > 600:
        notes = f"Long slew ({slew_time/60:.1f} min) — check target availability window"
    if peak_rate > max_rate_deg_s * 0.95:
        notes += " | Rate-limited slew"

    return SlewResult(
        slew_angle_deg    = slew_angle_deg,
        slew_duration_s   = slew_time,
        slew_rate_deg_s   = peak_rate,
        reaction_wheel_Nm = max_torque_Nm,
        energy_wh         = energy_wh,
        feasible          = feasible,
        notes             = notes.strip(" |")
    )


# ══════════════════════════════════════════════════════════════════
# MODE POWER DEFINITIONS
# ══════════════════════════════════════════════════════════════════

DEFAULT_MODE_POWERS = {
    "DETUMBLE":     5.0,    # magnetorquers only [W]
    "SUN_ACQ":      8.0,    # magnetorquers + coarse sun sensor [W]
    "SAFE":         6.0,    # sun-pointing hold, low power [W]
    "NADIR":        15.0,   # reaction wheels + star tracker [W]
    "SLEW":         20.0,   # reaction wheels at high torque [W]
    "INERTIAL":     12.0,   # reaction wheels momentum hold [W]
    "SUN_POINTING": 6.0,    # charging hold [W]
}


# ══════════════════════════════════════════════════════════════════
# TIMELINE GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_timeline(t_s: np.ndarray,
                      eclipse_mask: np.ndarray,
                      detumble: DetumbleResult,
                      mission_profile: str = "earth_observation",
                      slew_period_s: float = 5400.0,
                      slew_angle_deg: float = 30.0,
                      mode_powers: dict = None) -> list:
    """
    Generate ADCS mode sequence over the propagated trajectory.

    Parameters
    ----------
    t_s            : np.ndarray (N,) — Elapsed time [s]
    eclipse_mask   : np.ndarray (N,) bool — True = in eclipse
    detumble       : DetumbleResult — from estimate_detumble()
    mission_profile: str — "earth_observation", "comms", "technology_demo"
    slew_period_s  : float — How often slews happen in ops [s]
    slew_angle_deg : float — Typical slew angle [deg]
    mode_powers    : dict — Override default power values per mode

    Returns
    -------
    list of ADCSMode in time order
    """
    powers = {**DEFAULT_MODE_POWERS, **(mode_powers or {})}
    modes  = []
    t_now  = 0.0
    t_end  = float(t_s[-1])

    # ── Phase 1: Detumbling ────────────────────────────────────────
    detumble_end = min(detumble.estimated_time_s, t_end)
    modes.append(ADCSMode(
        mode_key   = "DETUMBLE",
        label      = ADCS_MODES["DETUMBLE"]["label"],
        start_s    = 0.0,
        duration_s = detumble_end,
        power_w    = powers["DETUMBLE"],
        notes      = f"B-dot control | target rate < {detumble.initial_rate_deg_s/20:.2f} deg/s"
    ))
    t_now = detumble_end

    if t_now >= t_end:
        return modes

    # ── Phase 2: Sun Acquisition ───────────────────────────────────
    sun_acq_dur = min(2 * 5568.0, t_end - t_now)   # ~2 orbits
    modes.append(ADCSMode(
        mode_key   = "SUN_ACQ",
        label      = ADCS_MODES["SUN_ACQ"]["label"],
        start_s    = t_now,
        duration_s = sun_acq_dur,
        power_w    = powers["SUN_ACQ"],
        notes      = "Coarse sun acquisition using CSS + magnetorquers"
    ))
    t_now += sun_acq_dur

    if t_now >= t_end:
        return modes

    # ── Phase 3: Operational modes ─────────────────────────────────
    # Determine primary mode from mission profile
    primary_mode = {
        "earth_observation": "NADIR",
        "comms":             "NADIR",
        "technology_demo":   "INERTIAL",
    }.get(mission_profile, "NADIR")

    # Estimate slew duration from a nominal 30-deg slew
    slew_dur_s = compute_slew(slew_angle_deg, inertia=10.0, max_torque_Nm=0.01).slew_duration_s

    while t_now < t_end:
        remaining = t_end - t_now

        if mission_profile == "earth_observation" and remaining > slew_period_s:
            # Nominal nadir hold until next slew
            nadir_dur = min(slew_period_s - slew_dur_s, remaining)
            nadir_dur = max(nadir_dur, 0)

            # Check if majority of this window is in eclipse
            idx_start = np.searchsorted(t_s, t_now)
            idx_end   = np.searchsorted(t_s, t_now + nadir_dur)
            idx_end   = min(idx_end, len(eclipse_mask) - 1)

            in_eclipse_frac = float(np.mean(eclipse_mask[idx_start:idx_end])) if idx_end > idx_start else 0.0

            if nadir_dur > 0:
                if in_eclipse_frac > 0.8:
                    # Mostly eclipse — use safe mode or sun-pointing
                    modes.append(ADCSMode(
                        mode_key   = "SUN_POINTING",
                        label      = ADCS_MODES["SUN_POINTING"]["label"],
                        start_s    = t_now,
                        duration_s = nadir_dur,
                        power_w    = powers["SUN_POINTING"],
                        notes      = f"Eclipse — battery charging attitude"
                    ))
                else:
                    modes.append(ADCSMode(
                        mode_key   = primary_mode,
                        label      = ADCS_MODES[primary_mode]["label"],
                        start_s    = t_now,
                        duration_s = nadir_dur,
                        power_w    = powers[primary_mode],
                        notes      = f"Payload operations"
                    ))
                t_now += nadir_dur

            # Slew to next target
            if t_now + slew_dur_s <= t_end:
                modes.append(ADCSMode(
                    mode_key   = "SLEW",
                    label      = ADCS_MODES["SLEW"]["label"],
                    start_s    = t_now,
                    duration_s = slew_dur_s,
                    power_w    = powers["SLEW"],
                    notes      = f"{slew_angle_deg:.1f} deg slew to next target"
                ))
                t_now += slew_dur_s

        else:
            # Fill remainder with primary mode
            modes.append(ADCSMode(
                mode_key   = primary_mode,
                label      = ADCS_MODES[primary_mode]["label"],
                start_s    = t_now,
                duration_s = remaining,
                power_w    = powers[primary_mode],
                notes      = "Nominal operations"
            ))
            t_now = t_end

    return modes


# ══════════════════════════════════════════════════════════════════
# MODE STATISTICS
# ══════════════════════════════════════════════════════════════════

def mode_statistics(modes: list, total_s: float) -> tuple:
    """
    Compute mode fraction and power budget from timeline.

    Returns
    -------
    fractions : dict — mode_key → fraction of total time
    powers    : dict — mode_key → power draw [W]
    """
    time_in_mode = {}
    power_in_mode = {}

    for m in modes:
        key = m.mode_key
        time_in_mode[key]  = time_in_mode.get(key, 0.0)  + m.duration_s
        power_in_mode[key] = m.power_w

    fractions = {k: v / total_s for k, v in time_in_mode.items()}
    return fractions, power_in_mode


# ══════════════════════════════════════════════════════════════════
# FULL ADCS PIPELINE
# ══════════════════════════════════════════════════════════════════

def compute_adcs_timeline(t_s: np.ndarray,
                           eclipse_mask: np.ndarray,
                           mass: float,
                           spacecraft_size_m: float   = 0.3,
                           altitude_km: float          = 400.0,
                           initial_rate_deg_s: float   = 10.0,
                           mission_profile: str        = "earth_observation",
                           slew_angle_deg: float       = 30.0,
                           dipole_Am2: float           = None,
                           mode_powers: dict           = None) -> ADCSTimeline:
    """
    Full ADCS timeline generation — call from app.py.

    Parameters
    ----------
    t_s                : np.ndarray (N,)  — Elapsed time [s]
    eclipse_mask       : np.ndarray (N,)  — True = in eclipse
    mass               : float            — Spacecraft mass [kg]
    spacecraft_size_m  : float            — Characteristic dimension [m]
    altitude_km        : float            — Mean orbit altitude [km]
    initial_rate_deg_s : float            — Post-separation tumble rate [deg/s]
    mission_profile    : str              — See generate_timeline()
    slew_angle_deg     : float            — Typical imaging slew angle [deg]
    dipole_Am2         : float            — Magnetorquer moment [Am²] (auto if None)
    mode_powers        : dict             — Override default powers [W]

    Returns
    -------
    ADCSTimeline
    """
    # Moment of inertia — uniform cube approximation
    inertia = mass * (spacecraft_size_m / 2)**2

    detumble = estimate_detumble(
        mass               = mass,
        inertia            = inertia,
        initial_rate_deg_s = initial_rate_deg_s,
        altitude_km        = altitude_km,
        dipole_Am2         = dipole_Am2
    )

    modes = generate_timeline(
        t_s            = t_s,
        eclipse_mask   = eclipse_mask,
        detumble       = detumble,
        mission_profile = mission_profile,
        slew_period_s  = 5400.0,
        slew_angle_deg = slew_angle_deg,
        mode_powers    = mode_powers
    )

    total_s = float(t_s[-1] - t_s[0])
    fractions, powers = mode_statistics(modes, total_s)

    return ADCSTimeline(
        modes            = modes,
        total_duration_s = total_s,
        detumble         = detumble,
        eclipse_mask     = eclipse_mask,
        power_budget     = powers,
        mode_fractions   = fractions
    )


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from converter import kep_to_cart
    from propagator import propagate
    from coverage import compute_ground_track, compute_eclipse

    print("=== ADCS Timeline Self-Test ===\n")

    a, e, i, raan, argp, nu = 6778e3, 0.0001, 51.6, 120.0, 90.0, 0.0
    r0, v0 = kep_to_cart(a, e, i, raan, argp, nu)

    res = propagate(r0, v0, duration_s=3*86400, step_s=60,
                    mass=500, cd=2.2, area=4.0)

    jd0          = 2451545.0
    eclipse_mask = compute_eclipse(res.t, res.r, jd0)

    # ── Test 1: Detumbling estimate ───────────────────────────────
    print("--- Test 1: Detumbling (500 kg, 0.6m, 400km) ---")
    det = estimate_detumble(mass=500, inertia=500*(0.3)**2,
                            initial_rate_deg_s=10.0, altitude_km=400)
    print(f"  Est. detumble time : {det.estimated_time_hr:.2f} hr")
    print(f"  Magnetorquer needed: {det.magnetorquer_Am2:.3f} Am²")
    print(f"  {det.notes}\n")

    # ── Test 2: Slew calculations ─────────────────────────────────
    print("--- Test 2: Slew maneuvers ---")
    for angle in [5, 15, 30, 45, 90]:
        slew = compute_slew(angle, inertia=45.0, max_torque_Nm=0.02,
                            max_rate_deg_s=1.0, rw_power_w=10.0)
        print(f"  {angle:3d} deg slew: {slew.slew_duration_s:6.1f} s  |  "
              f"peak rate {slew.slew_rate_deg_s:.3f} deg/s  |  "
              f"{slew.energy_wh:.4f} Wh")

    # ── Test 3: Full timeline ─────────────────────────────────────
    print("\n--- Test 3: Full ADCS Timeline (3 days, earth_observation) ---")
    timeline = compute_adcs_timeline(
        t_s                = res.t,
        eclipse_mask       = eclipse_mask,
        mass               = 500.0,
        spacecraft_size_m  = 0.6,
        altitude_km        = 400.0,
        initial_rate_deg_s = 10.0,
        mission_profile    = "earth_observation",
        slew_angle_deg     = 30.0
    )
    print(timeline.summary())

    # ── Test 4: CubeSat (3U) ─────────────────────────────────────
    print("\n--- Test 4: 3U CubeSat (6 kg, 0.1m, 550km) ---")
    cube_det = estimate_detumble(
        mass=6.0, inertia=6.0*(0.05)**2,
        initial_rate_deg_s=15.0, altitude_km=550, dipole_Am2=0.1
    )
    print(f"  Est. detumble time : {cube_det.estimated_time_hr:.2f} hr")
    print(f"  Magnetorquer       : {cube_det.magnetorquer_Am2:.3f} Am²")
    print(f"  {cube_det.notes}")

    # ── Test 5: Power budget summary ──────────────────────────────
    print("\n--- Test 5: Power Budget ---")
    total_energy_wh = sum(m.energy_wh for m in timeline.modes)
    print(f"  Total ADCS energy (3 days): {total_energy_wh:.2f} Wh")
    print(f"  Mean ADCS power           : {total_energy_wh/(3*24):.2f} W")
    for key, frac in timeline.mode_fractions.items():
        pw = timeline.power_budget.get(key, 0.0)
        print(f"  {ADCS_MODES[key]['label']:<35}: {frac*100:5.1f}%  @  {pw:.1f} W")
