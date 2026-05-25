"""
coverage.py — Ground track, access windows, and coverage metrics.

Logic:
This file projects the propagated ECI trajectory onto Earth's rotating surface.
ECI position converts to ECEF by rotating by Earth's sidereal angle at each timestep.
ECEF then converts to geodetic latitude and longitude for ground track plotting.
Access windows are computed by checking elevation angle above a minimum threshold.
Revisit time statistics summarise how often a target region is visible.
Eclipse fraction uses the same shadow model as perturbations.py for consistency.
All outputs feed directly into the app.py dashboard without further processing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from constants import CONST
from perturbations import sun_position_eci, in_eclipse


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class GroundTrack:
    """Latitude/longitude time series of the sub-satellite point."""
    t_s   : np.ndarray   # elapsed time [s]
    lat   : np.ndarray   # geodetic latitude [deg],  shape (N,)
    lon   : np.ndarray   # longitude [deg],           shape (N,)
    alt   : np.ndarray   # altitude [km],             shape (N,)


@dataclass
class AccessWindow:
    """Single continuous pass over a ground target."""
    start_s      : float   # pass start time [s from epoch]
    end_s        : float   # pass end time   [s from epoch]
    max_el_deg   : float   # maximum elevation angle [deg]
    max_el_t_s   : float   # time of max elevation  [s from epoch]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def report(self) -> str:
        return (
            f"  Start: {self.start_s/3600:7.3f} hr  |  "
            f"End: {self.end_s/3600:7.3f} hr  |  "
            f"Duration: {self.duration_s/60:5.1f} min  |  "
            f"Max El: {self.max_el_deg:5.1f} deg"
        )


@dataclass
class CoverageResult:
    ground_track   : GroundTrack
    access_windows : list             # list of AccessWindow
    eclipse_mask   : np.ndarray       # bool array — True = in eclipse
    eclipse_frac   : float            # fraction of time in eclipse [0-1]
    revisit_mean_s : float = 0.0      # mean revisit time [s]
    revisit_max_s  : float = 0.0      # max revisit time  [s]
    revisit_min_s  : float = 0.0      # min revisit time  [s]
    coverage_pct   : float = 0.0      # % of time target is visible

    def summary(self) -> str:
        n = len(self.access_windows)
        lines = [
            f"Coverage Summary",
            f"----------------",
            f"Total passes        : {n}",
            f"Eclipse fraction    : {self.eclipse_frac*100:.2f} %",
            f"Target coverage     : {self.coverage_pct:.2f} %",
        ]
        if n > 0:
            lines += [
                f"Mean revisit time   : {self.revisit_mean_s/60:.1f} min",
                f"Max revisit time    : {self.revisit_max_s/60:.1f} min",
                f"Min revisit time    : {self.revisit_min_s/60:.1f} min",
                f"Mean pass duration  : {np.mean([w.duration_s for w in self.access_windows])/60:.1f} min",
                f"Mean max elevation  : {np.mean([w.max_el_deg for w in self.access_windows]):.1f} deg",
            ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMS
# ══════════════════════════════════════════════════════════════════

def gmst(jd: float) -> float:
    """
    Greenwich Mean Sidereal Time in radians.
    Gives the rotation angle between ECI and ECEF frames.

    Parameters
    ----------
    jd : float — Julian Date

    Returns
    -------
    theta : float — GMST [rad]
    """
    T = (jd - CONST.J2000_JD) / 36525.0
    theta_deg = (
        280.46061837
        + 360.98564736629 * (jd - CONST.J2000_JD)
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    )
    return np.radians(theta_deg % 360.0)


def eci_to_ecef(r_eci: np.ndarray, jd: float) -> np.ndarray:
    """
    Rotate ECI position vector to ECEF frame.

    Parameters
    ----------
    r_eci : np.ndarray (3,) or (N, 3) — ECI position [m]
    jd    : float or array             — Julian Date(s)

    Returns
    -------
    r_ecef : np.ndarray same shape — ECEF position [m]
    """
    r = np.asarray(r_eci)
    jd = np.asarray(jd)

    if r.ndim == 1:
        theta = gmst(float(jd))
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, st, 0],
                      [-st, ct, 0],
                      [0,   0,  1]])
        return R @ r
    else:
        # Vectorised for array of positions
        thetas = np.array([gmst(float(j)) for j in jd])
        ct = np.cos(thetas)
        st = np.sin(thetas)
        x_ecef =  ct * r[:, 0] + st * r[:, 1]
        y_ecef = -st * r[:, 0] + ct * r[:, 1]
        z_ecef = r[:, 2]
        return np.column_stack([x_ecef, y_ecef, z_ecef])


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple:
    """
    Convert ECEF position to geodetic latitude, longitude, altitude.
    Uses Bowring's iterative method for accuracy.

    Parameters
    ----------
    r_ecef : np.ndarray (3,) or (N, 3) — ECEF position [m]

    Returns
    -------
    lat : float or array — Geodetic latitude [deg]
    lon : float or array — Longitude [deg]
    alt : float or array — Altitude above ellipsoid [m]
    """
    r = np.asarray(r_ecef, dtype=float)
    scalar = r.ndim == 1
    if scalar:
        r = r[np.newaxis, :]

    Re  = CONST.R_EARTH
    f   = CONST.F_EARTH
    e2  = 2 * f - f**2   # first eccentricity squared

    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    lon = np.degrees(np.arctan2(y, x))

    p   = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))   # initial estimate

    for _ in range(5):   # Bowring iteration
        sin_lat = np.sin(lat)
        N   = Re / np.sqrt(1 - e2 * sin_lat**2)
        lat = np.arctan2(z + e2 * N * sin_lat, p)

    sin_lat = np.sin(lat)
    N   = Re / np.sqrt(1 - e2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    lat = np.degrees(lat)

    if scalar:
        return float(lat[0]), float(lon[0]), float(alt[0])
    return lat, lon, alt


# ══════════════════════════════════════════════════════════════════
# GROUND TRACK
# ══════════════════════════════════════════════════════════════════

def compute_ground_track(t_s: np.ndarray,
                          r_eci: np.ndarray,
                          jd0: float) -> GroundTrack:
    """
    Compute sub-satellite ground track from propagated ECI trajectory.

    Parameters
    ----------
    t_s   : np.ndarray (N,)   — Elapsed time from epoch [s]
    r_eci : np.ndarray (N, 3) — ECI position vectors [m]
    jd0   : float             — Julian Date at epoch

    Returns
    -------
    GroundTrack
    """
    jd_arr  = jd0 + t_s / CONST.SECONDS_PER_DAY
    r_ecef  = eci_to_ecef(r_eci, jd_arr)
    lat, lon, alt = ecef_to_geodetic(r_ecef)

    return GroundTrack(
        t_s = t_s,
        lat = lat,
        lon = lon,
        alt = alt / 1e3   # convert to km
    )


# ══════════════════════════════════════════════════════════════════
# ELEVATION ANGLE
# ══════════════════════════════════════════════════════════════════

def elevation_angle(r_eci: np.ndarray,
                    gs_lat_deg: float,
                    gs_lon_deg: float,
                    gs_alt_m: float,
                    jd: float) -> float:
    """
    Compute elevation angle of spacecraft as seen from a ground station.

    Parameters
    ----------
    r_eci      : np.ndarray (3,) — Spacecraft ECI position [m]
    gs_lat_deg : float           — Ground station geodetic latitude [deg]
    gs_lon_deg : float           — Ground station longitude [deg]
    gs_alt_m   : float           — Ground station altitude [m]
    jd         : float           — Julian Date

    Returns
    -------
    el : float — Elevation angle [deg]  (negative = below horizon)
    """
    # Ground station ECEF position
    lat = np.radians(gs_lat_deg)
    lon = np.radians(gs_lon_deg)
    Re  = CONST.R_EARTH
    f   = CONST.F_EARTH
    e2  = 2 * f - f**2

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = Re / np.sqrt(1 - e2 * sin_lat**2)

    gs_ecef = np.array([
        (N + gs_alt_m) * cos_lat * np.cos(lon),
        (N + gs_alt_m) * cos_lat * np.sin(lon),
        (N * (1 - e2) + gs_alt_m) * sin_lat
    ])

    # Rotate spacecraft ECI to ECEF at this time
    sc_ecef = eci_to_ecef(r_eci, jd)

    # Range vector from ground station to spacecraft (ECEF)
    rho = sc_ecef - gs_ecef
    rho_mag = np.linalg.norm(rho)

    if rho_mag < 1:
        return -90.0

    # Local vertical (up direction) at ground station
    up = np.array([
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        sin_lat
    ])

    # Elevation = angle between range vector and local horizontal plane
    sin_el = np.dot(rho, up) / rho_mag
    el = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    return el


# ══════════════════════════════════════════════════════════════════
# ACCESS WINDOWS
# ══════════════════════════════════════════════════════════════════

def compute_access_windows(t_s: np.ndarray,
                            r_eci: np.ndarray,
                            gs_lat_deg: float,
                            gs_lon_deg: float,
                            gs_alt_m: float = 0.0,
                            min_el_deg: float = 5.0,
                            jd0: float = 2451545.0) -> list:
    """
    Find all access windows (passes) over a ground station.

    Parameters
    ----------
    t_s        : np.ndarray (N,) — Elapsed time [s]
    r_eci      : np.ndarray (N,3)— ECI positions [m]
    gs_lat_deg : float           — Ground station latitude [deg]
    gs_lon_deg : float           — Ground station longitude [deg]
    gs_alt_m   : float           — Ground station altitude [m]
    min_el_deg : float           — Minimum elevation for access [deg]
    jd0        : float           — Julian Date at epoch

    Returns
    -------
    list of AccessWindow
    """
    jd_arr = jd0 + t_s / CONST.SECONDS_PER_DAY

    # Compute elevation at every timestep
    elevations = np.array([
        elevation_angle(r_eci[i], gs_lat_deg, gs_lon_deg, gs_alt_m, jd_arr[i])
        for i in range(len(t_s))
    ])

    # Find access intervals — where elevation exceeds minimum
    in_access = elevations >= min_el_deg
    windows   = []

    i = 0
    while i < len(in_access):
        if in_access[i]:
            # Start of a pass
            start_idx = i
            while i < len(in_access) and in_access[i]:
                i += 1
            end_idx = i - 1

            pass_el = elevations[start_idx:end_idx + 1]
            max_idx = start_idx + int(np.argmax(pass_el))

            windows.append(AccessWindow(
                start_s    = t_s[start_idx],
                end_s      = t_s[end_idx],
                max_el_deg = float(np.max(pass_el)),
                max_el_t_s = float(t_s[max_idx])
            ))
        else:
            i += 1

    return windows


# ══════════════════════════════════════════════════════════════════
# ECLIPSE FRACTION
# ══════════════════════════════════════════════════════════════════

def compute_eclipse(t_s: np.ndarray,
                    r_eci: np.ndarray,
                    jd0: float) -> np.ndarray:
    """
    Compute eclipse mask over the propagated trajectory.

    Parameters
    ----------
    t_s   : np.ndarray (N,)   — Elapsed time [s]
    r_eci : np.ndarray (N, 3) — ECI positions [m]
    jd0   : float             — Julian Date at epoch

    Returns
    -------
    eclipse_mask : np.ndarray (N,) bool — True = in Earth shadow
    """
    jd_arr = jd0 + t_s / CONST.SECONDS_PER_DAY
    mask   = np.array([
        in_eclipse(r_eci[i], sun_position_eci(jd_arr[i]))
        for i in range(len(t_s))
    ])
    return mask


# ══════════════════════════════════════════════════════════════════
# REVISIT STATISTICS
# ══════════════════════════════════════════════════════════════════

def revisit_statistics(windows: list, total_duration_s: float) -> dict:
    """
    Compute revisit time statistics from a list of access windows.

    Revisit time = gap between end of one pass and start of the next.

    Parameters
    ----------
    windows          : list of AccessWindow
    total_duration_s : float — total propagation duration [s]

    Returns
    -------
    dict with mean, max, min revisit [s] and coverage_pct
    """
    if len(windows) == 0:
        return {"mean_s": 0, "max_s": 0, "min_s": 0, "coverage_pct": 0.0}

    # Total time in view
    total_access_s = sum(w.duration_s for w in windows)
    coverage_pct   = 100.0 * total_access_s / total_duration_s

    if len(windows) < 2:
        return {
            "mean_s":       total_duration_s - total_access_s,
            "max_s":        total_duration_s - total_access_s,
            "min_s":        0.0,
            "coverage_pct": coverage_pct
        }

    gaps = [
        windows[i + 1].start_s - windows[i].end_s
        for i in range(len(windows) - 1)
    ]

    return {
        "mean_s":       float(np.mean(gaps)),
        "max_s":        float(np.max(gaps)),
        "min_s":        float(np.min(gaps)),
        "coverage_pct": coverage_pct
    }


# ══════════════════════════════════════════════════════════════════
# FULL COVERAGE PIPELINE
# ══════════════════════════════════════════════════════════════════

def compute_coverage(t_s: np.ndarray,
                     r_eci: np.ndarray,
                     jd0: float,
                     gs_lat_deg: float  = 0.0,
                     gs_lon_deg: float  = 0.0,
                     gs_alt_m: float    = 0.0,
                     min_el_deg: float  = 5.0) -> CoverageResult:
    """
    Full coverage analysis pipeline — call this from app.py.

    Parameters
    ----------
    t_s        : np.ndarray (N,)   — Elapsed time [s]
    r_eci      : np.ndarray (N, 3) — ECI positions [m]
    jd0        : float             — Julian Date at epoch
    gs_lat_deg : float             — Ground station latitude [deg]
    gs_lon_deg : float             — Ground station longitude [deg]
    gs_alt_m   : float             — Ground station altitude [m]
    min_el_deg : float             — Minimum elevation for access [deg]

    Returns
    -------
    CoverageResult
    """
    gt      = compute_ground_track(t_s, r_eci, jd0)
    windows = compute_access_windows(
        t_s, r_eci, gs_lat_deg, gs_lon_deg, gs_alt_m, min_el_deg, jd0
    )
    eclipse_mask = compute_eclipse(t_s, r_eci, jd0)
    eclipse_frac = float(np.mean(eclipse_mask))

    stats = revisit_statistics(windows, float(t_s[-1] - t_s[0]))

    return CoverageResult(
        ground_track   = gt,
        access_windows = windows,
        eclipse_mask   = eclipse_mask,
        eclipse_frac   = eclipse_frac,
        revisit_mean_s = stats["mean_s"],
        revisit_max_s  = stats["max_s"],
        revisit_min_s  = stats["min_s"],
        coverage_pct   = stats["coverage_pct"]
    )


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from converter import kep_to_cart
    from propagator import propagate

    print("=== Coverage Self-Test (ISS-like orbit, 3 days) ===\n")

    a, e, i, raan, argp, nu = 6778e3, 0.0001, 51.6, 120.0, 90.0, 0.0
    r0, v0 = kep_to_cart(a, e, i, raan, argp, nu)

    result = propagate(r0, v0,
                       duration_s=3 * 86400,
                       step_s=60,
                       mass=500, cd=2.2, area=4.0,
                       use_j2=True, use_drag=True, use_srp=True)

    jd0 = 2451545.0

    # ── Ground track spot check ────────────────────────────────────
    gt = compute_ground_track(result.t, result.r, jd0)
    print(f"Ground track points : {len(gt.lat)}")
    print(f"Latitude  range     : {gt.lat.min():.1f} to {gt.lat.max():.1f} deg")
    print(f"Longitude range     : {gt.lon.min():.1f} to {gt.lon.max():.1f} deg")
    print(f"Altitude  range     : {gt.alt.min():.2f} to {gt.alt.max():.2f} km\n")

    # ── Access windows — Kennedy Space Center ─────────────────────
    gs_lat, gs_lon = 28.573, -80.649   # KSC, Florida
    print(f"Ground station: KSC ({gs_lat}N, {gs_lon}E)")
    print(f"Min elevation : 5 deg\n")

    windows = compute_access_windows(
        result.t, result.r,
        gs_lat, gs_lon,
        gs_alt_m=0.0,
        min_el_deg=5.0,
        jd0=jd0
    )

    print(f"Passes over KSC in 3 days: {len(windows)}")
    for i, w in enumerate(windows[:5], 1):
        print(f"  Pass {i}: {w.report()}")
    if len(windows) > 5:
        print(f"  ... and {len(windows)-5} more passes")

    # ── Eclipse ────────────────────────────────────────────────────
    eclipse_mask = compute_eclipse(result.t, result.r, jd0)
    eclipse_frac = np.mean(eclipse_mask)
    print(f"\nEclipse fraction : {eclipse_frac*100:.2f} %")
    print(f"  (typical LEO eclipse fraction: ~35-40%)")

    # ── Full pipeline ──────────────────────────────────────────────
    print("\n--- Full Coverage Pipeline ---")
    cov = compute_coverage(
        result.t, result.r, jd0,
        gs_lat_deg=gs_lat, gs_lon_deg=gs_lon,
        min_el_deg=5.0
    )
    print(cov.summary())

    # ── ECEF / geodetic round-trip check ──────────────────────────
    print("\n--- ECEF/Geodetic Round-trip Check ---")
    r_test = np.array([6378e3, 0.0, 0.0])
    lat, lon, alt = ecef_to_geodetic(r_test)
    print(f"  Input  : [{r_test[0]/1e3:.1f}, 0, 0] km  (equatorial, 0 lon)")
    print(f"  Output : lat={lat:.4f} deg  lon={lon:.4f} deg  alt={alt/1e3:.2f} km")
    print(f"  Expected: lat=0.0  lon=0.0  alt≈0")
