"""
propagator.py — Numerical orbit propagator with perturbations.

Logic:
This is the core engine — every output module depends on what this file produces.
It integrates the equations of motion forward in time using SciPy's RK45 solver.
At each step, central gravity plus all perturbation accelerations are summed.
The output is a time-series of state vectors covering the full propagation duration.
Keplerian elements are extracted at every step so the caller gets both representations.
The propagator is perturbation-agnostic — it calls total_perturbation() and does not
need to know which forces are active, keeping it clean and reusable.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional
from constants import CONST
from perturbations import total_perturbation
from converter import cart_to_kep


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════

@dataclass
class PropagationResult:
    """
    Holds the full output of a propagation run.

    Attributes
    ----------
    t          : np.ndarray (N,)   — Elapsed time from epoch [s]
    r          : np.ndarray (N, 3) — ECI position vectors [m]
    v          : np.ndarray (N, 3) — ECI velocity vectors [m/s]
    kep        : list of dict      — Keplerian elements at each step
    altitude   : np.ndarray (N,)   — Altitude above surface [m]
    speed      : np.ndarray (N,)   — Orbital speed [m/s]
    success    : bool              — True if integration completed without error
    message    : str               — Solver status message
    """
    t        : np.ndarray = field(default_factory=lambda: np.array([]))
    r        : np.ndarray = field(default_factory=lambda: np.array([]))
    v        : np.ndarray = field(default_factory=lambda: np.array([]))
    kep      : list       = field(default_factory=list)
    altitude : np.ndarray = field(default_factory=lambda: np.array([]))
    speed    : np.ndarray = field(default_factory=lambda: np.array([]))
    success  : bool       = True
    message  : str        = ""

    def summary(self) -> str:
        if len(self.t) == 0:
            return "No data — propagation did not produce output."
        lines = [
            f"Propagation Summary",
            f"-------------------",
            f"Duration          : {self.t[-1]/3600:.3f} hr  ({self.t[-1]/86400:.3f} days)",
            f"Steps output      : {len(self.t)}",
            f"Initial altitude  : {self.altitude[0]/1e3:.2f} km",
            f"Final altitude    : {self.altitude[-1]/1e3:.2f} km",
            f"Altitude change   : {(self.altitude[-1]-self.altitude[0])/1e3:.4f} km",
            f"Initial speed     : {self.speed[0]/1e3:.4f} km/s",
            f"Final speed       : {self.speed[-1]/1e3:.4f} km/s",
            f"Initial a         : {self.kep[0]['a']/1e3:.3f} km",
            f"Final a           : {self.kep[-1]['a']/1e3:.3f} km",
            f"Initial e         : {self.kep[0]['e']:.6f}",
            f"Final e           : {self.kep[-1]['e']:.6f}",
            f"Initial i         : {self.kep[0]['i']:.4f} deg",
            f"Final i           : {self.kep[-1]['i']:.4f} deg",
            f"Success           : {self.success}",
            f"Solver message    : {self.message}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# EQUATIONS OF MOTION
# ══════════════════════════════════════════════════════════════════

def equations_of_motion(t, state, mass, cd, area, cr, jd0,
                         use_j2, use_drag, use_srp):
    """
    Equations of motion for a spacecraft under gravity and perturbations.

    State vector: [x, y, z, vx, vy, vz]
    Returns derivative: [vx, vy, vz, ax, ay, az]

    Parameters
    ----------
    t      : float       — Current elapsed time [s]
    state  : array (6,)  — [x, y, z, vx, vy, vz] in ECI [m, m/s]
    mass   : float       — Spacecraft mass [kg]
    cd     : float       — Drag coefficient
    area   : float       — Cross-sectional area [m^2]
    cr     : float       — SRP reflectivity coefficient
    jd0    : float       — Julian Date at epoch (t=0)
    use_j2, use_drag, use_srp : bool — Perturbation toggles

    Returns
    -------
    dstate : list (6,)   — [vx, vy, vz, ax, ay, az]
    """
    r_eci = state[:3]
    v_eci = state[3:]

    r_mag = np.linalg.norm(r_eci)

    if r_mag < CONST.R_EARTH:
        # Spacecraft has re-entered — stop integration
        return [0.0] * 6

    # ── Central gravity ────────────────────────────────────────────
    a_grav = -CONST.GM / r_mag**3 * r_eci

    # ── Julian Date at current time ────────────────────────────────
    jd_now = jd0 + t / CONST.SECONDS_PER_DAY

    # ── Perturbations ──────────────────────────────────────────────
    a_perturb = total_perturbation(
        r_eci, v_eci, mass, cd, area, cr, jd_now,
        use_j2=use_j2, use_drag=use_drag, use_srp=use_srp
    )

    # ── Total acceleration ─────────────────────────────────────────
    a_total = a_grav + a_perturb

    return [v_eci[0], v_eci[1], v_eci[2],
            a_total[0], a_total[1], a_total[2]]


# ══════════════════════════════════════════════════════════════════
# RE-ENTRY DETECTION EVENT
# ══════════════════════════════════════════════════════════════════

def reentry_event(t, state, *args):
    """Terminal event — stops integration when altitude hits 80 km."""
    r_mag = np.linalg.norm(state[:3])
    return r_mag - (CONST.R_EARTH + 80e3)

reentry_event.terminal  = True
reentry_event.direction = -1


# ══════════════════════════════════════════════════════════════════
# MAIN PROPAGATOR
# ══════════════════════════════════════════════════════════════════

def propagate(r0_eci: np.ndarray,
              v0_eci: np.ndarray,
              duration_s: float,
              step_s: float,
              mass: float,
              cd: float,
              area: float,
              cr: float          = 1.3,
              jd0: float         = 2451545.0,
              use_j2: bool       = True,
              use_drag: bool     = True,
              use_srp: bool      = True,
              rtol: float        = 1e-9,
              atol: float        = 1e-9) -> PropagationResult:
    """
    Propagate a spacecraft orbit forward in time.

    Parameters
    ----------
    r0_eci     : np.ndarray (3,) — Initial ECI position [m]
    v0_eci     : np.ndarray (3,) — Initial ECI velocity [m/s]
    duration_s : float           — Propagation duration [s]
    step_s     : float           — Output timestep [s]
    mass       : float           — Spacecraft mass [kg]
    cd         : float           — Drag coefficient
    area       : float           — Cross-sectional area [m^2]
    cr         : float           — SRP reflectivity (1.0=absorb, 2.0=reflect)
    jd0        : float           — Julian Date at epoch
    use_j2     : bool            — Include J2 perturbation
    use_drag   : bool            — Include atmospheric drag
    use_srp    : bool            — Include solar radiation pressure
    rtol, atol : float           — Integrator tolerances

    Returns
    -------
    PropagationResult
    """
    result = PropagationResult()

    state0 = np.concatenate([r0_eci, v0_eci])
    t_eval = np.arange(0, duration_s + step_s, step_s)

    try:
        sol = solve_ivp(
            fun      = equations_of_motion,
            t_span   = (0.0, duration_s),
            y0       = state0,
            method   = "RK45",
            t_eval   = t_eval,
            events   = reentry_event,
            rtol     = rtol,
            atol     = atol,
            args     = (mass, cd, area, cr, jd0,
                        use_j2, use_drag, use_srp)
        )

        result.success = sol.success
        result.message = sol.message
        result.t       = sol.t
        result.r       = sol.y[:3].T   # shape (N, 3)
        result.v       = sol.y[3:].T   # shape (N, 3)

        # ── Derived scalars ────────────────────────────────────────
        r_mags           = np.linalg.norm(result.r, axis=1)
        result.altitude  = r_mags - CONST.R_EARTH
        result.speed     = np.linalg.norm(result.v, axis=1)

        # ── Keplerian elements at each step ────────────────────────
        result.kep = [
            cart_to_kep(result.r[i], result.v[i], deg=True)
            for i in range(len(result.t))
        ]

        # ── Check for re-entry ─────────────────────────────────────
        if sol.t_events and len(sol.t_events[0]) > 0:
            reentry_t = sol.t_events[0][0]
            result.message = (
                f"Re-entry detected at t = {reentry_t/3600:.2f} hr "
                f"({reentry_t/86400:.2f} days). Integration stopped."
            )

    except Exception as ex:
        result.success = False
        result.message = f"Propagation failed: {str(ex)}"

    return result


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER — Keplerian input
# ══════════════════════════════════════════════════════════════════

def propagate_from_keplerian(a, e, i, raan, argp, nu,
                              duration_s, step_s,
                              mass, cd, area,
                              cr=1.3, jd0=2451545.0,
                              use_j2=True, use_drag=True, use_srp=True,
                              deg=True) -> PropagationResult:
    """
    Propagate directly from Keplerian elements.
    Converts to Cartesian internally before calling propagate().
    """
    from converter import kep_to_cart
    r0, v0 = kep_to_cart(a, e, i, raan, argp, nu, deg=deg)
    return propagate(r0, v0, duration_s, step_s, mass, cd, area,
                     cr, jd0, use_j2, use_drag, use_srp)


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Propagator Self-Test ===\n")

    # ISS-like orbit — one full day propagation
    a    = 6778e3      # ~400 km altitude [m]
    e    = 0.0001
    i    = 51.6        # deg
    raan = 120.0
    argp = 90.0
    nu   = 0.0

    mass = 500.0       # [kg] — generic LEO satellite
    cd   = 2.2
    area = 4.0         # [m^2]

    duration_s = 86400.0   # 1 day
    step_s     = 60.0      # 1-minute output steps

    print(f"Propagating ISS-like orbit for {duration_s/3600:.0f} hours...")
    print(f"  a={a/1e3:.0f} km  e={e}  i={i} deg")
    print(f"  mass={mass} kg  Cd={cd}  area={area} m^2")
    print(f"  Perturbations: J2=ON  Drag=ON  SRP=ON\n")

    result = propagate_from_keplerian(
        a, e, i, raan, argp, nu,
        duration_s, step_s,
        mass, cd, area,
        use_j2=True, use_drag=True, use_srp=True
    )

    print(result.summary())

    # ── Orbital period check ───────────────────────────────────────
    T_theory = 2 * np.pi * np.sqrt(a**3 / CONST.GM)
    print(f"\nTheoretical period : {T_theory/60:.2f} min")

    # ── RAAN drift check (J2 precession) ──────────────────────────
    raan_0 = result.kep[0]["raan"]
    raan_f = result.kep[-1]["raan"]
    dt_days = result.t[-1] / 86400

    # Unwrap RAAN if it crossed 360
    delta_raan = raan_f - raan_0
    if delta_raan > 180:
        delta_raan -= 360
    elif delta_raan < -180:
        delta_raan += 360

    raan_rate = delta_raan / dt_days

    # Theoretical J2 RAAN drift rate [deg/day]
    n = np.sqrt(CONST.GM / a**3)
    p = a * (1 - e**2)
    raan_dot_theory = -1.5 * n * CONST.J2 * (CONST.R_EARTH / p)**2 * np.cos(np.radians(i))
    raan_dot_theory_deg = np.degrees(raan_dot_theory) * 86400

    print(f"RAAN drift (simulated) : {raan_rate:.4f} deg/day")
    print(f"RAAN drift (theory)    : {raan_dot_theory_deg:.4f} deg/day")
    print(f"Agreement              : {abs(raan_rate - raan_dot_theory_deg):.4f} deg/day error")

    # ── J2-only test (no drag, no SRP) ────────────────────────────
    print("\n--- J2-only propagation (drag OFF, SRP OFF) ---")
    res_j2 = propagate_from_keplerian(
        a, e, i, raan, argp, nu,
        duration_s, step_s,
        mass, cd, area,
        use_j2=True, use_drag=False, use_srp=False
    )
    alt_change_j2 = (res_j2.altitude[-1] - res_j2.altitude[0]) / 1e3
    print(f"Altitude change (J2 only): {alt_change_j2:.6f} km  (should be ~0)")

    # ── Drag-only test ─────────────────────────────────────────────
    print("\n--- Drag-only propagation (J2 OFF, SRP OFF) ---")
    res_drag = propagate_from_keplerian(
        a, e, i, raan, argp, nu,
        duration_s, step_s,
        mass, cd, area,
        use_j2=False, use_drag=True, use_srp=False
    )
    alt_change_drag = (res_drag.altitude[-1] - res_drag.altitude[0]) / 1e3
    print(f"Altitude change (drag only, 1 day): {alt_change_drag:.4f} km")
