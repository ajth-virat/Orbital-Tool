"""
perturbations.py — Acceleration models for J2, atmospheric drag, and SRP.

Logic:
This file computes the three dominant perturbation forces acting on a spacecraft.
Each perturbation returns an acceleration vector in the ECI frame [m/s^2].
J2 is Earth's oblateness effect and dominates for all LEO and MEO orbits.
Drag decays the orbit over time and depends on spacecraft geometry and atmosphere density.
SRP pushes the spacecraft away from the Sun and matters most for high-altitude or large-area craft.
The propagator calls all three and sums them at every integration timestep.
Keeping each force in its own function makes it easy to switch any one off for comparison.
"""

import numpy as np
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# J2 PERTURBATION — Earth Oblateness
# ══════════════════════════════════════════════════════════════════

def accel_j2(r_eci: np.ndarray) -> np.ndarray:
    """
    Compute J2 gravitational perturbation acceleration in ECI frame.

    J2 accounts for Earth's equatorial bulge — the dominant orbital perturbation
    for any Earth orbit. It causes RAAN to precess and argument of perigee to drift.

    Parameters
    ----------
    r_eci : np.ndarray shape (3,) — Position vector in ECI [m]

    Returns
    -------
    a_j2 : np.ndarray shape (3,) — Acceleration vector [m/s^2]
    """
    x, y, z = r_eci
    r = np.linalg.norm(r_eci)

    if r < 1e3:
        return np.zeros(3)

    mu  = CONST.GM
    J2  = CONST.J2
    Re  = CONST.R_EARTH

    # J2 acceleration components in ECI
    factor = (3 / 2) * J2 * mu * Re**2 / r**5
    z_r2   = (z / r)**2

    ax = factor * x * (5 * z_r2 - 1)
    ay = factor * y * (5 * z_r2 - 1)
    az = factor * z * (5 * z_r2 - 3)

    return np.array([ax, ay, az])


# ══════════════════════════════════════════════════════════════════
# ATMOSPHERIC DRAG
# ══════════════════════════════════════════════════════════════════

def atmospheric_density(altitude_m: float) -> float:
    """
    Compute atmospheric density using an exponential model.

    This is a simplified but physically reasonable model for altitudes 100–1000 km.
    For high-fidelity work this would be replaced with NRLMSISE-00.

    Parameters
    ----------
    altitude_m : float — Altitude above Earth's surface [m]

    Returns
    -------
    rho : float — Atmospheric density [kg/m^3]
    """
    alt_km = altitude_m / 1e3

    # Piecewise exponential model — reference values from USSA 1976
    # Format: (base_alt_km, ref_density_kg/m3, scale_height_km)
    layers = [
        (0,    1.225,      8.44),
        (100,  5.297e-7,   5.877),
        (150,  2.076e-9,   7.263),
        (200,  2.541e-10,  9.473),
        (250,  6.073e-11,  11.263),
        (300,  1.916e-11,  12.636),
        (350,  5.721e-12,  13.568),
        (400,  1.585e-12,  14.142),
        (450,  5.551e-13,  14.570),
        (500,  1.970e-13,  15.228),
        (600,  2.396e-14,  16.010),
        (700,  5.408e-15,  16.457),
        (800,  1.905e-15,  16.562),
        (900,  3.396e-16,  16.657),
        (1000, 5.297e-17,  16.600),
    ]

    if alt_km < 0:
        return CONST.ATMO_REF_DENSITY

    # Find the appropriate layer
    base_alt, rho_ref, H = layers[0]
    for layer in layers:
        if alt_km >= layer[0]:
            base_alt, rho_ref, H = layer
        else:
            break

    rho = rho_ref * np.exp(-(alt_km - base_alt) / H)
    return max(rho, 1e-30)   # floor to avoid zero division


def accel_drag(r_eci: np.ndarray, v_eci: np.ndarray,
               mass: float, cd: float, area: float) -> np.ndarray:
    """
    Compute atmospheric drag acceleration in ECI frame.

    Drag opposes the spacecraft's velocity relative to the (rotating) atmosphere.
    It continuously removes energy, lowering the orbit over time.

    Parameters
    ----------
    r_eci : np.ndarray shape (3,) — Position vector [m]
    v_eci : np.ndarray shape (3,) — Velocity vector in ECI [m/s]
    mass  : float — Spacecraft mass [kg]
    cd    : float — Drag coefficient [dimensionless, typically 2.2]
    area  : float — Cross-sectional area [m^2]

    Returns
    -------
    a_drag : np.ndarray shape (3,) — Acceleration vector [m/s^2]
    """
    r_mag   = np.linalg.norm(r_eci)
    alt_m   = r_mag - CONST.R_EARTH

    if alt_m < 0:
        return np.zeros(3)

    rho = atmospheric_density(alt_m)

    # Velocity relative to rotating atmosphere
    omega_earth = np.array([0.0, 0.0, CONST.OMEGA_EARTH])
    v_rel = v_eci - np.cross(omega_earth, r_eci)
    v_rel_mag = np.linalg.norm(v_rel)

    if v_rel_mag < 1e-6:
        return np.zeros(3)

    # Drag force: F = -0.5 * rho * Cd * A * v_rel^2 * v_hat
    ballistic_coeff = mass / (cd * area)   # [kg/m^2]
    a_drag = -0.5 * rho * v_rel_mag**2 / ballistic_coeff * (v_rel / v_rel_mag)

    return a_drag


# ══════════════════════════════════════════════════════════════════
# SOLAR RADIATION PRESSURE (SRP)
# ══════════════════════════════════════════════════════════════════

def sun_position_eci(jd: float) -> np.ndarray:
    """
    Approximate Sun position in ECI frame using a low-precision solar model.

    Accurate to ~1 degree — sufficient for SRP perturbation calculation.

    Parameters
    ----------
    jd : float — Julian Date

    Returns
    -------
    r_sun : np.ndarray shape (3,) — Sun position vector [m] in ECI
    """
    # Days since J2000.0
    T = (jd - CONST.J2000_JD) / 36525.0   # Julian centuries

    # Mean longitude and mean anomaly of the Sun [degrees]
    L0 = 280.46646 + 36000.76983 * T
    M  = 357.52911 + 35999.05029 * T - 0.0001537 * T**2

    L0 = np.radians(L0 % 360)
    M  = np.radians(M  % 360)

    # Equation of centre
    C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M)
    C += (0.019993 - 0.000101 * T) * np.sin(2 * M)
    C += 0.000289 * np.sin(3 * M)

    sun_lon = L0 + np.radians(C)   # apparent longitude

    # Obliquity of ecliptic [rad]
    eps = np.radians(23.439291 - 0.013004 * T)

    # Sun unit vector in ECI
    r_sun_hat = np.array([
        np.cos(sun_lon),
        np.sin(sun_lon) * np.cos(eps),
        np.sin(sun_lon) * np.sin(eps)
    ])

    # Scale to AU then convert to metres
    r_sun = r_sun_hat * CONST.AU
    return r_sun


def in_eclipse(r_eci: np.ndarray, r_sun: np.ndarray) -> bool:
    """
    Determine if spacecraft is in Earth's shadow (cylindrical model).

    Parameters
    ----------
    r_eci : np.ndarray shape (3,) — Spacecraft position [m]
    r_sun : np.ndarray shape (3,) — Sun position [m]

    Returns
    -------
    bool — True if spacecraft is in eclipse
    """
    r_sun_hat = r_sun / np.linalg.norm(r_sun)

    # Project spacecraft position onto Sun direction
    proj = np.dot(r_eci, r_sun_hat)

    # If projection is positive, spacecraft is on the Sun side — no eclipse
    if proj > 0:
        return False

    # Perpendicular distance from spacecraft to Sun-Earth line
    perp = np.linalg.norm(r_eci - proj * r_sun_hat)

    return perp < CONST.R_EARTH


def accel_srp(r_eci: np.ndarray, mass: float,
              area: float, cr: float, jd: float) -> np.ndarray:
    """
    Compute Solar Radiation Pressure acceleration in ECI frame.

    SRP pushes the spacecraft away from the Sun. It is negligible in LEO
    but significant for GEO, high-altitude, or large-area spacecraft.

    Parameters
    ----------
    r_eci : np.ndarray shape (3,) — Position vector [m]
    mass  : float — Spacecraft mass [kg]
    area  : float — Sun-facing cross-sectional area [m^2]
    cr    : float — Radiation pressure coefficient (1.0 absorb, 2.0 reflect)
    jd    : float — Julian Date (for Sun position)

    Returns
    -------
    a_srp : np.ndarray shape (3,) — Acceleration vector [m/s^2]
    """
    r_sun = sun_position_eci(jd)

    # No SRP in eclipse
    if in_eclipse(r_eci, r_sun):
        return np.zeros(3)

    # Vector from Sun to spacecraft
    r_sc_from_sun     = r_eci - r_sun
    r_sc_from_sun_mag = np.linalg.norm(r_sc_from_sun)
    r_hat             = r_sc_from_sun / r_sc_from_sun_mag

    # Scale SRP by (AU / actual distance)^2
    srp_at_sc = CONST.SRP_PRESSURE * (CONST.AU / r_sc_from_sun_mag)**2

    # Acceleration: a = (P * Cr * A / m) * r_hat
    a_srp = (srp_at_sc * cr * area / mass) * r_hat

    return a_srp


# ══════════════════════════════════════════════════════════════════
# COMBINED PERTURBATION ACCELERATIONS
# ══════════════════════════════════════════════════════════════════

def total_perturbation(r_eci: np.ndarray, v_eci: np.ndarray,
                       mass: float, cd: float, area: float,
                       cr: float = 1.3, jd: float = None,
                       use_j2: bool = True,
                       use_drag: bool = True,
                       use_srp: bool = True) -> np.ndarray:
    """
    Sum all perturbation accelerations. Called by the propagator at each step.

    Parameters
    ----------
    r_eci : np.ndarray (3,) — Position [m]
    v_eci : np.ndarray (3,) — Velocity [m/s]
    mass  : float — Spacecraft mass [kg]
    cd    : float — Drag coefficient
    area  : float — Cross-sectional area [m^2]
    cr    : float — SRP reflectivity coefficient (default 1.3)
    jd    : float — Julian Date (required if use_srp=True)
    use_j2, use_drag, use_srp : bool — Toggle each perturbation on/off

    Returns
    -------
    a_total : np.ndarray (3,) — Total perturbation acceleration [m/s^2]
    """
    a_total = np.zeros(3)

    if use_j2:
        a_total += accel_j2(r_eci)

    if use_drag:
        a_total += accel_drag(r_eci, v_eci, mass, cd, area)

    if use_srp and jd is not None:
        a_total += accel_srp(r_eci, mass, area, cr, jd)

    return a_total


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Perturbations Self-Test (ISS-like orbit) ===\n")

    # ISS-like state — ECI position and velocity
    r = np.array([-181776.0, -5638771.0, 3755797.0])   # [m]
    v = np.array([5628.6,    -3012.6,    -4249.6])      # [m/s]
    alt = (np.linalg.norm(r) - CONST.R_EARTH) / 1e3

    mass = 420000.0   # ISS mass [kg]
    cd   = 2.2
    area = 2500.0     # ISS cross-section [m^2] (rough)
    cr   = 1.3
    jd   = 2451545.0  # J2000.0

    print(f"Altitude          : {alt:.1f} km")
    print(f"Atmospheric density: {atmospheric_density(alt*1e3):.3e} kg/m^3\n")

    a_j2   = accel_j2(r)
    a_drag = accel_drag(r, v, mass, cd, area)
    a_srp  = accel_srp(r, mass, area, cr, jd)
    a_tot  = total_perturbation(r, v, mass, cd, area, cr, jd)

    print(f"J2   acceleration : {np.linalg.norm(a_j2):.4e} m/s^2  |  {a_j2}")
    print(f"Drag acceleration : {np.linalg.norm(a_drag):.4e} m/s^2  |  {a_drag}")
    print(f"SRP  acceleration : {np.linalg.norm(a_srp):.4e} m/s^2  |  {a_srp}")
    print(f"Total perturbation: {np.linalg.norm(a_tot):.4e} m/s^2")

    print(f"\nEclipse check     : {in_eclipse(r, sun_position_eci(jd))}")

    # Compare magnitudes to central gravity
    r_mag  = np.linalg.norm(r)
    a_grav = CONST.GM / r_mag**2
    print(f"\nCentral gravity   : {a_grav:.4f} m/s^2")
    print(f"J2  / gravity     : {np.linalg.norm(a_j2)/a_grav:.2e}")
    print(f"Drag / gravity    : {np.linalg.norm(a_drag)/a_grav:.2e}")
    print(f"SRP  / gravity    : {np.linalg.norm(a_srp)/a_grav:.2e}")
