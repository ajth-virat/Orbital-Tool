"""
lifetime.py — Orbital lifetime estimation via drag-induced decay.

Logic:
This file estimates how long a spacecraft remains in orbit before re-entry.
It uses the King-Hele analytical method for a fast first-order lifetime estimate.
A numerical decay curve then integrates semi-major axis loss orbit by orbit.
Three solar activity scenarios (low/mean/high) bracket the uncertainty range.
The 25-year compliance check is computed automatically for every run.
All outputs are in days to keep numbers readable at mission timescales.
This module runs fast — it does not call the full propagator.
"""

import numpy as np
from dataclasses import dataclass, field
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# SOLAR ACTIVITY SCENARIOS
# ══════════════════════════════════════════════════════════════════

# F10.7 solar flux index and corresponding scale factors for density
SOLAR_SCENARIOS = {
    "low":  {"F107": 70,  "label": "Solar Minimum",   "color": "#4FC3F7"},
    "mean": {"F107": 150, "label": "Solar Mean",       "color": "#FFB74D"},
    "high": {"F107": 250, "label": "Solar Maximum",    "color": "#EF5350"},
}

# Exponential atmosphere reference table — (base_alt_km, rho_ref kg/m3, H_km)
# Scaled per solar activity via a simple F10.7-dependent multiplier
ATMO_TABLE = [
    (200,  2.541e-10,  9.473),
    (250,  6.073e-11, 11.263),
    (300,  1.916e-11, 12.636),
    (350,  5.721e-12, 13.568),
    (400,  1.585e-12, 14.142),
    (450,  5.551e-13, 14.570),
    (500,  1.970e-13, 15.228),
    (600,  2.396e-14, 16.010),
    (700,  5.408e-15, 16.457),
    (800,  1.905e-15, 16.562),
    (900,  3.396e-16, 16.657),
    (1000, 5.297e-17, 16.600),
]


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class DecayCurve:
    """Altitude vs time decay profile for one solar scenario."""
    scenario    : str           # "low", "mean", "high"
    label       : str
    t_days      : np.ndarray   # elapsed time [days]
    alt_km      : np.ndarray   # mean altitude [km]
    lifetime_days : float      # estimated days to re-entry


@dataclass
class LifetimeResult:
    """Full lifetime estimation output."""
    initial_alt_km  : float
    initial_a_km    : float
    eccentricity    : float
    ballistic_coeff : float     # B = m / (Cd * A)  [kg/m^2]
    decay_curves    : list      # list of DecayCurve (one per scenario)
    compliant_25yr  : bool      # True if all scenarios exceed 25 years
    reentry_alt_km  : float = 80.0

    def summary(self) -> str:
        lines = [
            f"Orbital Lifetime Estimate",
            f"-------------------------",
            f"Initial altitude   : {self.initial_alt_km:.1f} km",
            f"Semi-major axis    : {self.initial_a_km:.1f} km",
            f"Eccentricity       : {self.eccentricity:.6f}",
            f"Ballistic coeff B  : {self.ballistic_coeff:.2f} kg/m^2",
            f"",
        ]
        for dc in self.decay_curves:
            yr = dc.lifetime_days / 365.25
            lines.append(
                f"  {dc.label:<22}: {dc.lifetime_days:>8.1f} days  "
                f"({yr:6.2f} years)"
            )
        lines += [
            f"",
            f"25-year rule compliance: {'✓ COMPLIANT' if self.compliant_25yr else '✗ NON-COMPLIANT'}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# ATMOSPHERE MODEL — F10.7 SCALED
# ══════════════════════════════════════════════════════════════════

def density_f107(alt_km: float, F107: float) -> float:
    """
    Atmospheric density at altitude with F10.7 solar flux scaling.

    Higher solar activity heats and expands the atmosphere, increasing
    density at a given altitude and accelerating orbital decay.

    Parameters
    ----------
    alt_km : float — Altitude [km]
    F107   : float — Solar flux index (70 = min, 150 = mean, 250 = max)

    Returns
    -------
    rho : float — Atmospheric density [kg/m^3]
    """
    if alt_km < 200:
        alt_km = 200.0
    if alt_km > 1000:
        return 1e-30

    # Find layer
    base_alt, rho_ref, H = ATMO_TABLE[0]
    for row in ATMO_TABLE:
        if alt_km >= row[0]:
            base_alt, rho_ref, H = row
        else:
            break

    # Base density from exponential model
    rho_base = rho_ref * np.exp(-(alt_km - base_alt) / H)

    # F10.7 scaling — density roughly doubles from solar min to max at 400 km
    # Scale factor: 1.0 at F10.7=150, ~0.4 at F10.7=70, ~2.5 at F10.7=250
    f107_scale = (F107 / 150.0) ** 1.5
    return max(rho_base * f107_scale, 1e-30)


# ══════════════════════════════════════════════════════════════════
# KING-HELE ANALYTICAL LIFETIME ESTIMATE
# ══════════════════════════════════════════════════════════════════

def king_hele_lifetime(a_m: float, e: float,
                       B: float, F107: float = 150.0) -> float:
    """
    King-Hele analytical estimate of orbital lifetime.

    Valid for near-circular orbits (e < 0.1). For higher eccentricities
    the numerical decay integrator is more reliable.

    Parameters
    ----------
    a_m   : float — Semi-major axis [m]
    e     : float — Eccentricity
    B     : float — Ballistic coefficient m/(Cd*A) [kg/m^2]
    F107  : float — Solar flux index

    Returns
    -------
    lifetime_days : float
    """
    Re = CONST.R_EARTH
    mu = CONST.GM

    alt_m   = a_m - Re
    alt_km  = alt_m / 1e3
    rho     = density_f107(alt_km, F107)

    # Scale height at this altitude
    H_m = _scale_height(alt_km) * 1e3

    # Mean motion
    n = np.sqrt(mu / a_m**3)

    # King-Hele formula: T ≈ B * H / (rho * v * a)
    v = np.sqrt(mu / a_m)   # circular velocity

    # Eccentricity correction factor
    if e < 0.001:
        ecc_factor = 1.0
    else:
        ecc_factor = np.exp(-(a_m * e) / H_m)  # perigee density dominates

    lifetime_s = (B * H_m) / (rho * ecc_factor * v * a_m) * (1 / (3 * np.pi))

    # Sanity clamp — analytical method breaks down below 200 km
    lifetime_s = max(lifetime_s, 0.0)
    return lifetime_s / CONST.SECONDS_PER_DAY


def _scale_height(alt_km: float) -> float:
    """Return scale height [km] at given altitude from table."""
    if alt_km <= ATMO_TABLE[0][0]:
        return ATMO_TABLE[0][2]
    if alt_km >= ATMO_TABLE[-1][0]:
        return ATMO_TABLE[-1][2]
    for i in range(len(ATMO_TABLE) - 1):
        if ATMO_TABLE[i][0] <= alt_km < ATMO_TABLE[i + 1][0]:
            # Linear interpolation
            frac = (alt_km - ATMO_TABLE[i][0]) / (ATMO_TABLE[i+1][0] - ATMO_TABLE[i][0])
            return ATMO_TABLE[i][2] + frac * (ATMO_TABLE[i+1][2] - ATMO_TABLE[i][2])
    return 15.0


# ══════════════════════════════════════════════════════════════════
# NUMERICAL DECAY INTEGRATOR
# ══════════════════════════════════════════════════════════════════

def numerical_decay_curve(a0_m: float, e: float,
                           B: float, F107: float = 150.0,
                           reentry_alt_km: float = 80.0,
                           max_days: float = 36525.0,
                           dt_orbits: int = 10) -> DecayCurve:
    """
    Numerically integrate semi-major axis decay orbit by orbit.

    More accurate than King-Hele for eccentric orbits and
    long propagation durations where density varies significantly.

    Parameters
    ----------
    a0_m           : float — Initial semi-major axis [m]
    e              : float — Eccentricity (assumed constant — valid for e<0.3)
    B              : float — Ballistic coefficient [kg/m^2]
    F107           : float — Solar flux index
    reentry_alt_km : float — Re-entry altitude threshold [km]
    max_days       : float — Maximum integration time [days] (100 years)
    dt_orbits      : int   — Steps between output saves

    Returns
    -------
    DecayCurve (scenario filled by caller)
    """
    Re = CONST.R_EARTH
    mu = CONST.GM

    reentry_r = Re + reentry_alt_km * 1e3

    a     = a0_m
    t_s   = 0.0
    t_out = [0.0]
    a_out = [(a - Re) / 1e3]

    orbit_count = 0

    while a > reentry_r and t_s < max_days * CONST.SECONDS_PER_DAY:
        # Orbital period at current a
        T_orb = 2 * np.pi * np.sqrt(a**3 / mu)

        # Mean altitude for density lookup
        alt_km = (a * (1 - e**2 / 2) - Re) / 1e3   # approx mean altitude
        alt_km = max(alt_km, 200.0)

        rho = density_f107(alt_km, F107)
        v   = np.sqrt(mu / a)                        # circular velocity

        # da/dt from drag (orbit-averaged, circular approximation)
        # Energy method: da/dt = -rho * Cd * A/m * v^2 * a  (Wertz/Vallado)
        # Per orbit: da = -2*pi * rho * a^2 / B  * (2*pi*a/T) * T / (2*pi)
        # Simplified: da_per_orbit = -2 * pi * rho * a^2 / B
        da_per_orbit = -2 * np.pi * rho * a**2 / B  # [m per orbit]
        da_dt        = da_per_orbit / T_orb           # [m/s]

        # Cross-check with energy method for consistency
        # F_drag = 0.5 * rho * v^2 * (Cd*A) = 0.5 * rho * v^2 * m/B
        # da/dt  = -2 * F_drag * a^2 / (mu * m) * v  — equivalent form

        # Adaptive step — larger steps at high altitude where decay is slow
        n_orbits_step = max(1, int(1e5 / max(1, abs(da_per_orbit))))
        n_orbits_step = min(n_orbits_step, 500)

        dt_s  = T_orb * n_orbits_step
        da    = da_dt * dt_s

        # Prevent overshooting re-entry altitude
        if a + da < reentry_r:
            da = reentry_r - a - 1.0

        a   += da
        t_s += dt_s

        orbit_count += n_orbits_step

        if orbit_count % (dt_orbits * 100) == 0 or a <= reentry_r:
            t_out.append(t_s)
            a_out.append(max((a - Re) / 1e3, reentry_alt_km))

    t_out = np.array(t_out)
    a_out = np.array(a_out)

    lifetime_days = t_s / CONST.SECONDS_PER_DAY

    return DecayCurve(
        scenario      = "",
        label         = "",
        t_days        = t_out / CONST.SECONDS_PER_DAY,
        alt_km        = a_out,
        lifetime_days = lifetime_days
    )


# ══════════════════════════════════════════════════════════════════
# FULL LIFETIME ANALYSIS
# ══════════════════════════════════════════════════════════════════

def compute_lifetime(a_m: float, e: float,
                     mass: float, cd: float, area: float,
                     reentry_alt_km: float = 80.0) -> LifetimeResult:
    """
    Full orbital lifetime analysis across all three solar scenarios.

    Parameters
    ----------
    a_m            : float — Semi-major axis [m]
    e              : float — Eccentricity
    mass           : float — Spacecraft mass [kg]
    cd             : float — Drag coefficient
    area           : float — Cross-sectional area [m^2]
    reentry_alt_km : float — Re-entry threshold [km]

    Returns
    -------
    LifetimeResult
    """
    B               = mass / (cd * area)   # ballistic coefficient [kg/m^2]
    initial_alt_km  = (a_m - CONST.R_EARTH) / 1e3

    decay_curves = []

    for key, scenario in SOLAR_SCENARIOS.items():
        dc = numerical_decay_curve(
            a0_m           = a_m,
            e              = e,
            B              = B,
            F107           = scenario["F107"],
            reentry_alt_km = reentry_alt_km
        )
        dc.scenario = key
        dc.label    = scenario["label"]
        decay_curves.append(dc)

    # 25-year compliance — worst case is high solar activity (fastest decay)
    worst_lifetime = min(dc.lifetime_days for dc in decay_curves)
    compliant      = worst_lifetime >= 25 * 365.25

    return LifetimeResult(
        initial_alt_km  = initial_alt_km,
        initial_a_km    = a_m / 1e3,
        eccentricity    = e,
        ballistic_coeff = B,
        decay_curves    = decay_curves,
        compliant_25yr  = compliant,
        reentry_alt_km  = reentry_alt_km
    )


# ══════════════════════════════════════════════════════════════════
# MINIMUM ALTITUDE FOR 25-YEAR COMPLIANCE
# ══════════════════════════════════════════════════════════════════

def min_altitude_for_compliance(mass: float, cd: float, area: float,
                                 target_years: float = 25.0,
                                 alt_min_km: float   = 200.0,
                                 alt_max_km: float   = 1200.0) -> dict:
    """
    Binary search for the minimum altitude that satisfies the target lifetime
    under the worst-case (high solar activity) scenario.

    Parameters
    ----------
    mass, cd, area : spacecraft properties
    target_years   : float — Target compliance lifetime [years]
    alt_min_km     : float — Search lower bound [km]
    alt_max_km     : float — Search upper bound [km]

    Returns
    -------
    dict with "alt_km" and "lifetime_days" for each scenario at that altitude
    """
    target_days = target_years * 365.25
    B = mass / (cd * area)
    Re = CONST.R_EARTH

    lo, hi = alt_min_km, alt_max_km

    for _ in range(30):   # binary search, converges in <30 iterations
        mid = (lo + hi) / 2
        a_m = Re + mid * 1e3

        dc = numerical_decay_curve(
            a0_m  = a_m,
            e     = 0.0,
            B     = B,
            F107  = SOLAR_SCENARIOS["high"]["F107"]   # worst case
        )

        if dc.lifetime_days >= target_days:
            hi = mid
        else:
            lo = mid

        if hi - lo < 0.5:   # converged to 0.5 km
            break

    result_alt = (lo + hi) / 2
    a_result   = Re + result_alt * 1e3

    lifetimes = {}
    for key, scen in SOLAR_SCENARIOS.items():
        dc = numerical_decay_curve(a0_m=a_result, e=0.0, B=B, F107=scen["F107"])
        lifetimes[key] = dc.lifetime_days

    return {"alt_km": result_alt, "lifetimes_days": lifetimes}


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Lifetime Estimator Self-Test ===\n")

    mass = 500.0   # kg
    cd   = 2.2
    area = 4.0     # m^2
    B    = mass / (cd * area)
    print(f"Spacecraft: {mass} kg  Cd={cd}  A={area} m^2  B={B:.2f} kg/m^2\n")

    # ── Test 1: ISS altitude ~400 km ──────────────────────────────
    print("--- Test 1: LEO 400 km ---")
    result = compute_lifetime(CONST.R_EARTH + 400e3, 0.0001, mass, cd, area)
    print(result.summary())

    # ── Test 2: Higher LEO 600 km ─────────────────────────────────
    print("\n--- Test 2: LEO 600 km ---")
    result2 = compute_lifetime(CONST.R_EARTH + 600e3, 0.0, mass, cd, area)
    print(result2.summary())

    # ── Test 3: Graveyard-bound 800 km ────────────────────────────
    print("\n--- Test 3: LEO 800 km ---")
    result3 = compute_lifetime(CONST.R_EARTH + 800e3, 0.0, mass, cd, area)
    print(result3.summary())

    # ── Test 4: King-Hele spot check ──────────────────────────────
    print("\n--- Test 4: King-Hele analytical vs numerical ---")
    for alt in [300, 400, 500, 600]:
        a_m = CONST.R_EARTH + alt * 1e3
        kh  = king_hele_lifetime(a_m, 0.0, B, F107=150)
        dc  = numerical_decay_curve(a_m, 0.0, B, F107=150)
        print(f"  {alt} km: King-Hele={kh:8.1f} days  |  Numerical={dc.lifetime_days:8.1f} days")

    # ── Test 5: Minimum altitude for 25-year compliance ───────────
    print("\n--- Test 5: Min altitude for 25-year compliance ---")
    res = min_altitude_for_compliance(mass, cd, area, target_years=25.0)
    print(f"  Minimum altitude (worst case): {res['alt_km']:.1f} km")
    for key, days in res["lifetimes_days"].items():
        print(f"  {SOLAR_SCENARIOS[key]['label']}: {days:.1f} days ({days/365.25:.1f} years)")
