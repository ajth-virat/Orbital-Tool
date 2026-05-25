"""
converter.py — Keplerian ↔ Cartesian state vector conversions.

Logic:
This file bridges the two input modes the tool accepts.
Keplerian elements describe orbit shape and orientation geometrically.
Cartesian vectors describe position and velocity in 3D space numerically.
All downstream physics modules work in Cartesian, so Keplerian inputs convert first.
The reverse conversion (Cartesian → Keplerian) lets output modules report elements at any propagated point.
Edge cases (circular, equatorial, retrograde) are handled explicitly to avoid division-by-zero errors.
All angles enter in degrees and convert internally to radians for computation.
"""

import numpy as np
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# KEPLERIAN → CARTESIAN
# ══════════════════════════════════════════════════════════════════

def kep_to_cart(a, e, i, raan, argp, nu, deg=True):
    """
    Convert Keplerian orbital elements to Cartesian state vector in ECI frame.

    Parameters
    ----------
    a    : float — Semi-major axis [m]
    e    : float — Eccentricity [dimensionless]
    i    : float — Inclination [deg if deg=True, else rad]
    raan : float — Right Ascension of Ascending Node [deg or rad]
    argp : float — Argument of perigee [deg or rad]
    nu   : float — True anomaly [deg or rad]
    deg  : bool  — If True, all angles are in degrees (default: True)

    Returns
    -------
    r_eci : np.ndarray shape (3,) — Position vector [m]
    v_eci : np.ndarray shape (3,) — Velocity vector [m/s]
    """
    if deg:
        i    = np.radians(i)
        raan = np.radians(raan)
        argp = np.radians(argp)
        nu   = np.radians(nu)

    mu = CONST.GM

    # ── Step 1: Position and velocity in perifocal (PQW) frame ────
    p = a * (1 - e**2)                          # semi-latus rectum [m]
    r_mag = p / (1 + e * np.cos(nu))            # orbit radius at true anomaly

    r_pqw = np.array([
        r_mag * np.cos(nu),
        r_mag * np.sin(nu),
        0.0
    ])

    v_pqw = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
         np.sqrt(mu / p) * (e + np.cos(nu)),
        0.0
    ])

    # ── Step 2: Rotation matrix PQW → ECI ─────────────────────────
    # R = Rz(-RAAN) · Rx(-i) · Rz(-argp)
    cos_O = np.cos(raan);  sin_O = np.sin(raan)
    cos_i = np.cos(i);     sin_i = np.sin(i)
    cos_w = np.cos(argp);  sin_w = np.sin(argp)

    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i,  -cos_O*sin_w - sin_O*cos_w*cos_i,  sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i,  -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i,                        cos_w*sin_i,                       cos_i      ]
    ])

    # ── Step 3: Rotate to ECI ──────────────────────────────────────
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw

    return r_eci, v_eci


# ══════════════════════════════════════════════════════════════════
# CARTESIAN → KEPLERIAN
# ══════════════════════════════════════════════════════════════════

def cart_to_kep(r_eci, v_eci, deg=True):
    """
    Convert Cartesian ECI state vector to Keplerian orbital elements.

    Parameters
    ----------
    r_eci : array-like shape (3,) — Position vector [m]
    v_eci : array-like shape (3,) — Velocity vector [m/s]
    deg   : bool — If True, return angles in degrees (default: True)

    Returns
    -------
    dict with keys:
        a    — Semi-major axis [m]
        e    — Eccentricity
        i    — Inclination [deg or rad]
        raan — RAAN [deg or rad]
        argp — Argument of perigee [deg or rad]
        nu   — True anomaly [deg or rad]
        T    — Orbital period [s]
        h    — Specific angular momentum magnitude [m^2/s]
    """
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)
    mu = CONST.GM

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # ── Angular momentum vector ────────────────────────────────────
    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    # ── Node vector (points toward ascending node) ─────────────────
    K = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(K, h_vec)
    n_mag = np.linalg.norm(n_vec)

    # ── Eccentricity vector (points toward perigee) ────────────────
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # ── Semi-major axis ────────────────────────────────────────────
    energy = v_mag**2 / 2 - mu / r_mag
    a = -mu / (2 * energy)

    # ── Inclination ────────────────────────────────────────────────
    i = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    # ── RAAN ───────────────────────────────────────────────────────
    if n_mag < 1e-10:
        # Equatorial orbit — RAAN undefined, set to 0
        raan = 0.0
    else:
        raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan

    # ── Argument of perigee ────────────────────────────────────────
    if n_mag < 1e-10 or e < 1e-10:
        # Circular or equatorial — argp undefined, set to 0
        argp = 0.0
    else:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp

    # ── True anomaly ───────────────────────────────────────────────
    if e < 1e-10:
        # Circular — nu measured from ascending node
        nu = np.arccos(np.clip(np.dot(n_vec, r) / (n_mag * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu

    # ── Orbital period ─────────────────────────────────────────────
    T = 2 * np.pi * np.sqrt(a**3 / mu)

    if deg:
        i    = np.degrees(i)
        raan = np.degrees(raan)
        argp = np.degrees(argp)
        nu   = np.degrees(nu)

    return {
        "a":    a,
        "e":    e,
        "i":    i,
        "raan": raan,
        "argp": argp,
        "nu":   nu,
        "T":    T,
        "h":    h_mag
    }


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def state_to_dict(r_eci, v_eci):
    """Package position and velocity arrays into a labelled dict."""
    return {
        "rx": r_eci[0], "ry": r_eci[1], "rz": r_eci[2],
        "vx": v_eci[0], "vy": v_eci[1], "vz": v_eci[2]
    }


def orbital_period(a):
    """Compute orbital period from semi-major axis. [s]"""
    return 2 * np.pi * np.sqrt(a**3 / CONST.GM)


def circular_velocity(r):
    """Compute circular orbit velocity at radius r from Earth centre. [m/s]"""
    return np.sqrt(CONST.GM / r)


def escape_velocity(r):
    """Compute escape velocity at radius r from Earth centre. [m/s]"""
    return np.sqrt(2 * CONST.GM / r)


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Converter Self-Test (ISS-like orbit) ===\n")

    # ISS approximate elements
    a    = 6778e3      # ~400 km altitude
    e    = 0.0001
    i    = 51.6
    raan = 120.0
    argp = 90.0
    nu   = 45.0

    print(f"Input Keplerian elements:")
    print(f"  a={a/1e3:.1f} km  e={e}  i={i} deg")
    print(f"  RAAN={raan} deg  argp={argp} deg  nu={nu} deg\n")

    r, v = kep_to_cart(a, e, i, raan, argp, nu)
    print(f"ECI Position : [{r[0]/1e3:.3f}, {r[1]/1e3:.3f}, {r[2]/1e3:.3f}] km")
    print(f"ECI Velocity : [{v[0]/1e3:.4f}, {v[1]/1e3:.4f}, {v[2]/1e3:.4f}] km/s\n")

    kep = cart_to_kep(r, v)
    print("Recovered Keplerian elements (should match input):")
    print(f"  a    = {kep['a']/1e3:.3f} km  (input: {a/1e3:.3f} km)")
    print(f"  e    = {kep['e']:.6f}       (input: {e})")
    print(f"  i    = {kep['i']:.4f} deg   (input: {i} deg)")
    print(f"  RAAN = {kep['raan']:.4f} deg  (input: {raan} deg)")
    print(f"  argp = {kep['argp']:.4f} deg  (input: {argp} deg)")
    print(f"  nu   = {kep['nu']:.4f} deg   (input: {nu} deg)")
    print(f"  T    = {kep['T']/60:.2f} min")

    print(f"\nCircular velocity at ISS alt : {circular_velocity(a)/1e3:.4f} km/s")
    print(f"Escape velocity at ISS alt   : {escape_velocity(a)/1e3:.4f} km/s")
    print(f"Orbital period               : {orbital_period(a)/60:.2f} min")
