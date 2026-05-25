"""
environment.py — Space environment characterisation module.

Logic:
This file quantifies the radiation and particle environment along a propagated orbit.
Van Allen belt flux levels drive component selection and shielding thickness decisions.
Total Ionizing Dose accumulates over mission lifetime and sets electronics lifetime limits.
Atomic oxygen fluence degrades exposed polymer and metal surfaces in LEO below 700 km.
South Atlantic Anomaly crossings deliver elevated proton flux even at low inclinations.
Eclipse fraction from coverage.py feeds in directly rather than being recomputed here.
All outputs are in standard engineering units used in space environment specifications.
"""

import numpy as np
from dataclasses import dataclass, field
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# CONSTANTS AND BELT BOUNDARIES
# ══════════════════════════════════════════════════════════════════

# Van Allen belt boundaries in Earth radii (geocentric distance / R_Earth)
INNER_BELT = {"r_min": 1.1, "r_max": 2.5}   # proton dominated
OUTER_BELT = {"r_min": 3.0, "r_max": 7.0}   # electron dominated

# South Atlantic Anomaly geographic boundary (approximate ellipse)
SAA_LAT_CENTER  = -25.0    # deg
SAA_LON_CENTER  = -40.0    # deg
SAA_LAT_RADIUS  = 20.0     # deg
SAA_LON_RADIUS  = 40.0     # deg

# Atomic oxygen — reference flux at 400 km [atoms/m^2/s]
AO_REF_FLUX_400KM = 2.0e15

# TID conversion — rough shielding model [rad/day per unit flux]
TID_DOSE_RATE_1MM  = 10.0    # rad/day at 1 mm Al shielding in inner belt
TID_DOSE_RATE_3MM  = 1.5     # rad/day at 3 mm Al shielding
TID_DOSE_RATE_10MM = 0.05    # rad/day at 10 mm Al shielding


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class BeltExposure:
    """Time spent in each Van Allen belt region."""
    inner_belt_frac   : float   # fraction of orbit time in inner belt
    outer_belt_frac   : float   # fraction of orbit time in outer belt
    safe_zone_frac    : float   # fraction outside both belts
    inner_belt_hours  : float
    outer_belt_hours  : float
    peak_L_shell      : float   # max L-shell value reached


@dataclass
class TIDEstimate:
    """Total Ionizing Dose for common shielding thicknesses."""
    mission_days      : float
    dose_1mm_krad     : float   # TID at 1 mm Al [krad]
    dose_3mm_krad     : float   # TID at 3 mm Al [krad]
    dose_10mm_krad    : float   # TID at 10 mm Al [krad]
    dose_rate_1mm     : float   # daily dose rate at 1mm [rad/day]


@dataclass
class SAAResult:
    """South Atlantic Anomaly crossing statistics."""
    n_crossings       : int     # number of SAA passes
    total_time_s      : float   # total time inside SAA [s]
    frac_in_saa       : float   # fraction of total time in SAA
    crossing_times_s  : list    # list of crossing entry times [s from epoch]


@dataclass
class AtomicOxygenResult:
    """Atomic oxygen fluence for LEO surfaces."""
    mean_alt_km       : float
    flux_atoms_m2_s   : float   # mean AO flux [atoms/m^2/s]
    fluence_1yr       : float   # 1-year fluence [atoms/m^2]
    fluence_mission   : float   # mission fluence [atoms/m^2]
    significant       : bool    # True if AO is a design concern


@dataclass
class EnvironmentResult:
    """Full space environment output."""
    orbit_type        : str
    mean_alt_km       : float
    L_shell_mean      : float
    belt_exposure     : BeltExposure
    tid               : TIDEstimate
    saa               : SAAResult
    atomic_oxygen     : AtomicOxygenResult
    warnings          : list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Space Environment Report",
            f"------------------------",
            f"Orbit type          : {self.orbit_type}",
            f"Mean altitude       : {self.mean_alt_km:.1f} km",
            f"Mean L-shell        : {self.L_shell_mean:.3f} R_E",
            f"",
            f"Van Allen Belts:",
            f"  Inner belt        : {self.belt_exposure.inner_belt_frac*100:.1f} % of time  "
            f"({self.belt_exposure.inner_belt_hours:.2f} hr)",
            f"  Outer belt        : {self.belt_exposure.outer_belt_frac*100:.1f} % of time  "
            f"({self.belt_exposure.outer_belt_hours:.2f} hr)",
            f"  Safe zone         : {self.belt_exposure.safe_zone_frac*100:.1f} % of time",
            f"  Peak L-shell      : {self.belt_exposure.peak_L_shell:.3f} R_E",
            f"",
            f"Total Ionizing Dose ({self.tid.mission_days:.0f} days mission):",
            f"  1 mm Al shielding : {self.tid.dose_1mm_krad:.2f} krad",
            f"  3 mm Al shielding : {self.tid.dose_3mm_krad:.2f} krad",
            f"  10mm Al shielding : {self.tid.dose_10mm_krad:.4f} krad",
            f"",
            f"South Atlantic Anomaly:",
            f"  SAA crossings     : {self.saa.n_crossings}",
            f"  Time in SAA       : {self.saa.total_time_s/3600:.2f} hr  "
            f"({self.saa.frac_in_saa*100:.2f} %)",
            f"",
            f"Atomic Oxygen (LEO only):",
            f"  AO flux           : {self.atomic_oxygen.flux_atoms_m2_s:.2e} atoms/m²/s",
            f"  1-year fluence    : {self.atomic_oxygen.fluence_1yr:.2e} atoms/m²",
            f"  AO concern        : {'YES — design for AO resistance' if self.atomic_oxygen.significant else 'Negligible at this altitude'}",
        ]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  ⚠️  {w}" for w in self.warnings]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# L-SHELL CALCULATION
# ══════════════════════════════════════════════════════════════════

def l_shell(r_m: float, lat_deg: float) -> float:
    """
    McIlwain L-shell parameter — magnetic field line label.

    L ≈ r / (R_E * cos²(magnetic_latitude))
    Approximation valid for the dipole field model.

    Parameters
    ----------
    r_m     : float — Geocentric radius [m]
    lat_deg : float — Geodetic latitude [deg] (used as proxy for mag lat)

    Returns
    -------
    L : float — L-shell [Earth radii]
    """
    cos_lat = np.cos(np.radians(lat_deg))
    if abs(cos_lat) < 1e-6:
        return 99.0   # polar region — undefined
    r_Re = r_m / CONST.R_EARTH
    L    = r_Re / max(cos_lat**2, 1e-4)
    return float(L)


# ══════════════════════════════════════════════════════════════════
# VAN ALLEN BELT EXPOSURE
# ══════════════════════════════════════════════════════════════════

def compute_belt_exposure(r_eci: np.ndarray,
                           lat_deg: np.ndarray,
                           t_s: np.ndarray) -> BeltExposure:
    """
    Determine time spent in inner and outer Van Allen belts.

    Parameters
    ----------
    r_eci   : np.ndarray (N, 3) — ECI position [m]
    lat_deg : np.ndarray (N,)   — Geodetic latitude [deg]
    t_s     : np.ndarray (N,)   — Elapsed time [s]

    Returns
    -------
    BeltExposure
    """
    r_mags = np.linalg.norm(r_eci, axis=1)
    total_s = t_s[-1] - t_s[0]

    # L-shell at each step
    L_vals = np.array([
        l_shell(r_mags[i], lat_deg[i])
        for i in range(len(t_s))
    ])

    inner_mask = (L_vals >= INNER_BELT["r_min"]) & (L_vals <= INNER_BELT["r_max"])
    outer_mask = (L_vals >= OUTER_BELT["r_min"]) & (L_vals <= OUTER_BELT["r_max"])

    inner_frac = float(np.mean(inner_mask))
    outer_frac = float(np.mean(outer_mask))
    safe_frac  = 1.0 - inner_frac - outer_frac

    return BeltExposure(
        inner_belt_frac  = inner_frac,
        outer_belt_frac  = outer_frac,
        safe_zone_frac   = max(safe_frac, 0.0),
        inner_belt_hours = inner_frac * total_s / 3600,
        outer_belt_hours = outer_frac * total_s / 3600,
        peak_L_shell     = float(np.min(L_vals[L_vals < 90]))  # exclude polar
    )


# ══════════════════════════════════════════════════════════════════
# TOTAL IONIZING DOSE
# ══════════════════════════════════════════════════════════════════

def compute_tid(belt_exposure: BeltExposure,
                mean_alt_km: float,
                mission_days: float) -> TIDEstimate:
    """
    Estimate Total Ionizing Dose for standard aluminium shielding thicknesses.

    Uses a simplified trapped particle flux model based on L-shell and altitude.
    For a full analysis use SPENVIS with AP8/AE8 models.

    Parameters
    ----------
    belt_exposure : BeltExposure
    mean_alt_km  : float — Mean orbit altitude [km]
    mission_days : float — Mission duration [days]

    Returns
    -------
    TIDEstimate
    """
    # Base dose rate depends on where the spacecraft spends its time
    # Inner belt: intense proton radiation
    # Outer belt: intense electron radiation
    # LEO below belts: GCR + SAA contribution

    inner_frac = belt_exposure.inner_belt_frac
    outer_frac = belt_exposure.outer_belt_frac

    # Dose rate at 1 mm Al [rad/day]
    # Inner belt proton contribution
    inner_dose_rate = inner_frac * 500.0      # inner belt ~500 rad/day at 1mm
    # Outer belt electron contribution
    outer_dose_rate = outer_frac * 200.0      # outer belt ~200 rad/day at 1mm
    # LEO background (GCR + low-altitude trapped)
    alt_factor = np.exp(-(mean_alt_km - 400) / 300) if mean_alt_km < 700 else 0.01
    leo_dose_rate = (1 - inner_frac - outer_frac) * 10.0 * max(alt_factor, 0.01)

    dose_rate_1mm = inner_dose_rate + outer_dose_rate + leo_dose_rate

    # Scale to thicker shielding — roughly exponential reduction
    dose_rate_3mm  = dose_rate_1mm * 0.15    # ~85% reduction at 3mm
    dose_rate_10mm = dose_rate_1mm * 0.005   # ~99.5% reduction at 10mm

    return TIDEstimate(
        mission_days   = mission_days,
        dose_1mm_krad  = dose_rate_1mm  * mission_days / 1000,
        dose_3mm_krad  = dose_rate_3mm  * mission_days / 1000,
        dose_10mm_krad = dose_rate_10mm * mission_days / 1000,
        dose_rate_1mm  = dose_rate_1mm
    )


# ══════════════════════════════════════════════════════════════════
# SOUTH ATLANTIC ANOMALY
# ══════════════════════════════════════════════════════════════════

def in_saa(lat_deg: float, lon_deg: float) -> bool:
    """
    Check if a ground point is inside the South Atlantic Anomaly.
    Uses an elliptical boundary approximation.

    Parameters
    ----------
    lat_deg : float — Latitude [deg]
    lon_deg : float — Longitude [deg]

    Returns
    -------
    bool
    """
    dlat = (lat_deg - SAA_LAT_CENTER) / SAA_LAT_RADIUS
    dlon = (lon_deg - SAA_LON_CENTER) / SAA_LON_RADIUS
    return (dlat**2 + dlon**2) <= 1.0


def compute_saa(lat_deg: np.ndarray,
                lon_deg: np.ndarray,
                t_s: np.ndarray,
                alt_km: np.ndarray) -> SAAResult:
    """
    Compute SAA crossing statistics from ground track.

    Parameters
    ----------
    lat_deg : np.ndarray (N,) — Geodetic latitude [deg]
    lon_deg : np.ndarray (N,) — Longitude [deg]
    t_s     : np.ndarray (N,) — Elapsed time [s]
    alt_km  : np.ndarray (N,) — Altitude [km]

    Returns
    -------
    SAAResult
    """
    # SAA primarily affects altitudes 200-1000 km
    in_saa_mask = np.array([
        in_saa(lat_deg[i], lon_deg[i]) and alt_km[i] < 1200
        for i in range(len(t_s))
    ])

    # Count distinct crossings and total time
    crossing_times = []
    in_crossing    = False
    total_time_s   = 0.0

    for i in range(len(in_saa_mask)):
        if in_saa_mask[i] and not in_crossing:
            in_crossing = True
            crossing_times.append(float(t_s[i]))
        elif not in_saa_mask[i] and in_crossing:
            in_crossing = False
            if i > 0:
                dt = t_s[i] - t_s[i-1]
                # Add up time in this crossing
        if i > 0 and in_saa_mask[i]:
            total_time_s += float(t_s[i] - t_s[i-1])

    total_duration = float(t_s[-1] - t_s[0])
    frac = total_time_s / total_duration if total_duration > 0 else 0.0

    return SAAResult(
        n_crossings      = len(crossing_times),
        total_time_s     = total_time_s,
        frac_in_saa      = frac,
        crossing_times_s = crossing_times
    )


# ══════════════════════════════════════════════════════════════════
# ATOMIC OXYGEN
# ══════════════════════════════════════════════════════════════════

def compute_atomic_oxygen(mean_alt_km: float,
                           mission_days: float) -> AtomicOxygenResult:
    """
    Estimate atomic oxygen flux and fluence for a LEO mission.

    AO is significant below ~700 km and attacks polymers, silver, and osmium.
    Above 700 km AO density drops to negligible levels.

    Parameters
    ----------
    mean_alt_km  : float — Mean orbital altitude [km]
    mission_days : float — Mission duration [days]

    Returns
    -------
    AtomicOxygenResult
    """
    # AO flux falls exponentially with altitude
    # Reference: ~2e15 atoms/m2/s at 400 km (solar mean)
    if mean_alt_km > 800:
        flux = 1e8   # negligible
    else:
        flux = AO_REF_FLUX_400KM * np.exp(-(mean_alt_km - 400) / 100)
        flux = max(flux, 1e8)

    mission_s      = mission_days * CONST.SECONDS_PER_DAY
    fluence_1yr    = flux * 365.25 * CONST.SECONDS_PER_DAY
    fluence_mission = flux * mission_s
    significant    = mean_alt_km < 700 and flux > 1e12

    return AtomicOxygenResult(
        mean_alt_km      = mean_alt_km,
        flux_atoms_m2_s  = flux,
        fluence_1yr      = fluence_1yr,
        fluence_mission  = fluence_mission,
        significant      = significant
    )


# ══════════════════════════════════════════════════════════════════
# ORBIT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify_orbit(mean_alt_km: float, inclination_deg: float) -> str:
    """
    Classify orbit type from altitude and inclination.

    Returns a human-readable orbit type string.
    """
    if mean_alt_km < 2000:
        if abs(inclination_deg - 90) < 10:
            return "LEO — Polar"
        elif abs(inclination_deg - 98) < 5:
            return "LEO — Sun-Synchronous (SSO)"
        else:
            return f"LEO — {inclination_deg:.1f} deg inclination"
    elif mean_alt_km < 35000:
        return "MEO"
    elif 35500 < mean_alt_km < 36100:
        return "GEO"
    elif mean_alt_km > 36000:
        return "HEO / Super-GEO"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════
# FULL ENVIRONMENT PIPELINE
# ══════════════════════════════════════════════════════════════════

def compute_environment(t_s: np.ndarray,
                         r_eci: np.ndarray,
                         lat_deg: np.ndarray,
                         lon_deg: np.ndarray,
                         alt_km: np.ndarray,
                         inclination_deg: float,
                         mission_days: float) -> EnvironmentResult:
    """
    Full space environment analysis — call from app.py.

    Parameters
    ----------
    t_s             : np.ndarray (N,)   — Elapsed time [s]
    r_eci           : np.ndarray (N, 3) — ECI positions [m]
    lat_deg         : np.ndarray (N,)   — Geodetic latitude [deg]
    lon_deg         : np.ndarray (N,)   — Longitude [deg]
    alt_km          : np.ndarray (N,)   — Altitude [km]
    inclination_deg : float             — Orbit inclination [deg]
    mission_days    : float             — Total mission duration [days]

    Returns
    -------
    EnvironmentResult
    """
    mean_alt  = float(np.mean(alt_km))
    orbit_type = classify_orbit(mean_alt, inclination_deg)

    # L-shell
    r_mags    = np.linalg.norm(r_eci, axis=1)
    L_vals    = np.array([l_shell(r_mags[i], lat_deg[i]) for i in range(len(t_s))])
    L_finite  = L_vals[L_vals < 90]
    L_mean    = float(np.mean(L_finite)) if len(L_finite) > 0 else 0.0

    # Belt exposure
    belt = compute_belt_exposure(r_eci, lat_deg, t_s)

    # TID
    tid = compute_tid(belt, mean_alt, mission_days)

    # SAA
    saa = compute_saa(lat_deg, lon_deg, t_s, alt_km)

    # Atomic oxygen
    ao = compute_atomic_oxygen(mean_alt, mission_days)

    # Warnings
    warnings = []
    if belt.inner_belt_frac > 0.01:
        warnings.append(
            f"Inner belt exposure {belt.inner_belt_frac*100:.1f}% — "
            "use radiation-hardened components"
        )
    if belt.outer_belt_frac > 0.05:
        warnings.append(
            f"Outer belt exposure {belt.outer_belt_frac*100:.1f}% — "
            "electron charging risk, review grounding design"
        )
    if tid.dose_3mm_krad > 20:
        warnings.append(
            f"TID at 3mm Al: {tid.dose_3mm_krad:.1f} krad exceeds 20 krad — "
            "review component TID tolerance"
        )
    if ao.significant:
        warnings.append(
            f"AO fluence {ao.fluence_1yr:.1e} atoms/m² per year — "
            "protect polymers and silver surfaces"
        )
    if saa.n_crossings > 0 and inclination_deg > 20:
        warnings.append(
            f"{saa.n_crossings} SAA crossings detected — "
            "consider SEU mitigation in SAA"
        )

    return EnvironmentResult(
        orbit_type    = orbit_type,
        mean_alt_km   = mean_alt,
        L_shell_mean  = L_mean,
        belt_exposure = belt,
        tid           = tid,
        saa           = saa,
        atomic_oxygen = ao,
        warnings      = warnings
    )


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from converter import kep_to_cart
    from propagator import propagate
    from coverage import compute_ground_track

    print("=== Space Environment Self-Test ===\n")

    # ── Test 1: ISS-like LEO ──────────────────────────────────────
    print("--- Test 1: LEO 400 km / 51.6 deg (ISS-like) ---")
    a, e, i, raan, argp, nu = 6778e3, 0.0001, 51.6, 120.0, 90.0, 0.0
    r0, v0 = kep_to_cart(a, e, i, raan, argp, nu)

    res = propagate(r0, v0, duration_s=86400, step_s=60,
                    mass=500, cd=2.2, area=4.0)

    jd0 = 2451545.0
    gt  = compute_ground_track(res.t, res.r, jd0)

    env = compute_environment(
        t_s             = res.t,
        r_eci           = res.r,
        lat_deg         = gt.lat,
        lon_deg         = gt.lon,
        alt_km          = res.altitude / 1e3,
        inclination_deg = i,
        mission_days    = 365.0
    )
    print(env.summary())

    # ── Test 2: MEO (GPS-like) ────────────────────────────────────
    print("\n--- Test 2: MEO 20200 km / 55 deg (GPS-like) ---")
    a2 = CONST.R_EARTH + 20200e3
    r0b, v0b = kep_to_cart(a2, 0.01, 55.0, 0.0, 0.0, 0.0)
    res2 = propagate(r0b, v0b, duration_s=86400*2, step_s=300,
                     mass=1000, cd=2.2, area=5.0,
                     use_drag=False)   # negligible drag at MEO

    gt2  = compute_ground_track(res2.t, res2.r, jd0)
    env2 = compute_environment(
        t_s             = res2.t,
        r_eci           = res2.r,
        lat_deg         = gt2.lat,
        lon_deg         = gt2.lon,
        alt_km          = res2.altitude / 1e3,
        inclination_deg = 55.0,
        mission_days    = 365.0 * 10
    )
    print(env2.summary())

    # ── Test 3: SSO ───────────────────────────────────────────────
    print("\n--- Test 3: SSO 550 km / 97.6 deg ---")
    r0c, v0c = kep_to_cart(CONST.R_EARTH + 550e3, 0.0, 97.6, 0.0, 0.0, 0.0)
    res3 = propagate(r0c, v0c, duration_s=86400, step_s=60,
                     mass=6.0, cd=2.2, area=0.06)   # 3U CubeSat

    gt3  = compute_ground_track(res3.t, res3.r, jd0)
    env3 = compute_environment(
        t_s             = res3.t,
        r_eci           = res3.r,
        lat_deg         = gt3.lat,
        lon_deg         = gt3.lon,
        alt_km          = res3.altitude / 1e3,
        inclination_deg = 97.6,
        mission_days    = 365.0 * 3
    )
    print(env3.summary())

    # ── SAA spot checks ───────────────────────────────────────────
    print("\n--- SAA Boundary Spot Checks ---")
    test_pts = [
        (-30, -40, "Centre — should be IN"),
        (-30, -40 + 50, "Right edge — should be OUT"),
        (0, 0, "Equator/0lon — should be OUT"),
        (-50, -60, "South edge — should be OUT"),
    ]
    for lat, lon, desc in test_pts:
        result = in_saa(lat, lon)
        print(f"  ({lat:+.0f}, {lon:+.0f}): {desc} → {'IN' if result else 'OUT'}")
