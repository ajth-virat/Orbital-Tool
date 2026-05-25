"""
deltav.py — Delta-V budget calculations for orbital maneuvers.

Logic:
This file computes the fuel cost of changing from one orbit to another.
Every maneuver reduces to impulses — instantaneous velocity changes at specific points.
Hohmann transfers use two burns and are the minimum-energy solution between circular orbits.
Plane changes are expensive and are always combined with altitude changes when possible.
Bi-elliptic transfers beat Hohmann when the target orbit is more than ~11.94x larger.
All inputs are in metres and metres-per-second — matching the rest of the tool.
The Tsiolkovsky equation converts delta-V into propellant mass, closing the design loop.
"""

import numpy as np
from dataclasses import dataclass, field
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ManeuverResult:
    """Single maneuver output."""
    name        : str
    delta_v     : float          # total delta-V [m/s]
    burns       : list           # list of dicts — each burn's dV and location
    tof_s       : float = 0.0   # time of flight [s]
    prop_mass   : float = 0.0   # propellant mass consumed [kg] (0 if Isp not given)
    notes       : str   = ""

    def report(self) -> str:
        lines = [
            f"Maneuver : {self.name}",
            f"Total ΔV : {self.delta_v:.4f} m/s  ({self.delta_v/1e3:.4f} km/s)",
        ]
        for i, b in enumerate(self.burns, 1):
            lines.append(f"  Burn {i} : {b['dv']:.4f} m/s  at {b['location']}")
        if self.tof_s > 0:
            lines.append(f"ToF      : {self.tof_s/3600:.3f} hr  ({self.tof_s/60:.1f} min)")
        if self.prop_mass > 0:
            lines.append(f"Prop mass: {self.prop_mass:.3f} kg")
        if self.notes:
            lines.append(f"Notes    : {self.notes}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# BASIC ORBITAL VELOCITY HELPERS
# ══════════════════════════════════════════════════════════════════

def circular_velocity(r: float) -> float:
    """Circular orbit speed at radius r from Earth centre [m/s]."""
    return np.sqrt(CONST.GM / r)


def ellipse_velocity(r: float, a: float) -> float:
    """
    Speed at radius r on an ellipse with semi-major axis a (vis-viva) [m/s].
    """
    return np.sqrt(CONST.GM * (2 / r - 1 / a))


def orbital_period(a: float) -> float:
    """Orbital period for semi-major axis a [s]."""
    return 2 * np.pi * np.sqrt(a**3 / CONST.GM)


def propellant_mass(m0: float, delta_v: float, isp: float) -> float:
    """
    Tsiolkovsky rocket equation — propellant mass consumed.

    Parameters
    ----------
    m0      : float — Wet mass (spacecraft + propellant) [kg]
    delta_v : float — Required delta-V [m/s]
    isp     : float — Specific impulse [s]

    Returns
    -------
    mp : float — Propellant mass consumed [kg]
    """
    ve = isp * 9.80665            # effective exhaust velocity [m/s]
    mp = m0 * (1 - np.exp(-delta_v / ve))
    return mp


# ══════════════════════════════════════════════════════════════════
# HOHMANN TRANSFER
# ══════════════════════════════════════════════════════════════════

def hohmann(r1: float, r2: float,
            mass: float = None, isp: float = None) -> ManeuverResult:
    """
    Two-impulse Hohmann transfer between two circular orbits.

    The most fuel-efficient transfer for orbits where r2/r1 < 11.94.
    Burn 1 raises apogee to r2. Burn 2 circularises at r2.

    Parameters
    ----------
    r1   : float — Initial orbit radius [m]  (= R_Earth + altitude1)
    r2   : float — Target orbit radius [m]   (= R_Earth + altitude2)
    mass : float — Spacecraft mass [kg] (optional — for propellant calc)
    isp  : float — Specific impulse [s] (optional — for propellant calc)

    Returns
    -------
    ManeuverResult
    """
    # Transfer ellipse semi-major axis
    a_transfer = (r1 + r2) / 2

    # Velocities
    v1          = circular_velocity(r1)
    v2          = circular_velocity(r2)
    v_trans_p   = ellipse_velocity(r1, a_transfer)   # at perigee of transfer
    v_trans_a   = ellipse_velocity(r2, a_transfer)   # at apogee of transfer

    dv1 = abs(v_trans_p - v1)
    dv2 = abs(v2 - v_trans_a)
    dv_total = dv1 + dv2

    tof = orbital_period(a_transfer) / 2   # half-period

    mp = 0.0
    if mass is not None and isp is not None:
        mp = propellant_mass(mass, dv_total, isp)

    direction = "ascending" if r2 > r1 else "descending"

    return ManeuverResult(
        name    = f"Hohmann Transfer ({direction})",
        delta_v = dv_total,
        burns   = [
            {"dv": dv1, "location": f"Perigee of transfer ellipse (r = {r1/1e3:.1f} km)"},
            {"dv": dv2, "location": f"Apogee  of transfer ellipse (r = {r2/1e3:.1f} km)"}
        ],
        tof_s       = tof,
        prop_mass   = mp,
        notes   = f"Transfer ellipse: a = {a_transfer/1e3:.1f} km, "
                  f"r2/r1 ratio = {r2/r1:.3f}"
    )


# ══════════════════════════════════════════════════════════════════
# BI-ELLIPTIC TRANSFER
# ══════════════════════════════════════════════════════════════════

def bielliptic(r1: float, r2: float, r_b: float,
               mass: float = None, isp: float = None) -> ManeuverResult:
    """
    Three-impulse bi-elliptic transfer between two circular orbits.

    More efficient than Hohmann when r2/r1 > ~11.94.
    Requires an intermediate high-apogee orbit at radius r_b.

    Parameters
    ----------
    r1  : float — Initial orbit radius [m]
    r2  : float — Target orbit radius [m]
    r_b : float — Intermediate apogee radius [m]  (r_b > max(r1, r2))
    mass : float, isp : float — optional for propellant

    Returns
    -------
    ManeuverResult
    """
    if r_b <= max(r1, r2):
        raise ValueError(
            f"Bi-elliptic intermediate radius r_b={r_b/1e3:.0f} km "
            f"must exceed both r1={r1/1e3:.0f} km and r2={r2/1e3:.0f} km."
        )

    # First transfer ellipse: r1 → r_b
    a1 = (r1 + r_b) / 2
    # Second transfer ellipse: r_b → r2
    a2 = (r_b + r2) / 2

    v1         = circular_velocity(r1)
    v2         = circular_velocity(r2)
    v_t1_peri  = ellipse_velocity(r1, a1)   # at r1 on first ellipse
    v_t1_apo   = ellipse_velocity(r_b, a1)  # at r_b on first ellipse
    v_t2_apo   = ellipse_velocity(r_b, a2)  # at r_b on second ellipse
    v_t2_peri  = ellipse_velocity(r2, a2)   # at r2 on second ellipse

    dv1 = abs(v_t1_peri - v1)
    dv2 = abs(v_t2_apo  - v_t1_apo)
    dv3 = abs(v2 - v_t2_peri)
    dv_total = dv1 + dv2 + dv3

    tof = orbital_period(a1) / 2 + orbital_period(a2) / 2

    mp = 0.0
    if mass is not None and isp is not None:
        mp = propellant_mass(mass, dv_total, isp)

    return ManeuverResult(
        name    = "Bi-elliptic Transfer",
        delta_v = dv_total,
        burns   = [
            {"dv": dv1, "location": f"r1 perigee ({r1/1e3:.1f} km)"},
            {"dv": dv2, "location": f"Intermediate apogee ({r_b/1e3:.1f} km)"},
            {"dv": dv3, "location": f"r2 target ({r2/1e3:.1f} km)"}
        ],
        tof_s     = tof,
        prop_mass = mp,
        notes     = f"r2/r1 = {r2/r1:.2f} | r_b = {r_b/1e3:.0f} km"
    )


# ══════════════════════════════════════════════════════════════════
# PURE PLANE CHANGE
# ══════════════════════════════════════════════════════════════════

def plane_change(v_orbit: float, delta_i_deg: float,
                 mass: float = None, isp: float = None) -> ManeuverResult:
    """
    Pure inclination change at constant altitude.

    This is very expensive — use combined_maneuver() whenever possible.

    Parameters
    ----------
    v_orbit    : float — Orbital speed before maneuver [m/s]
    delta_i_deg: float — Inclination change required [deg]

    Returns
    -------
    ManeuverResult
    """
    delta_i = np.radians(delta_i_deg)
    dv = 2 * v_orbit * np.sin(delta_i / 2)

    mp = 0.0
    if mass is not None and isp is not None:
        mp = propellant_mass(mass, dv, isp)

    return ManeuverResult(
        name    = f"Pure Plane Change (Δi = {delta_i_deg:.2f} deg)",
        delta_v = dv,
        burns   = [{"dv": dv, "location": "Ascending or descending node"}],
        prop_mass = mp,
        notes   = "Pure plane changes are expensive. Combine with altitude change if possible."
    )


# ══════════════════════════════════════════════════════════════════
# COMBINED PLANE CHANGE + ALTITUDE CHANGE
# ══════════════════════════════════════════════════════════════════

def combined_maneuver(r1: float, r2: float, delta_i_deg: float,
                      split: float = 0.0,
                      mass: float = None, isp: float = None) -> ManeuverResult:
    """
    Hohmann transfer with inclination change split between the two burns.

    Splitting the plane change across burns is more efficient than doing it
    separately. The optimal split minimises total delta-V.

    Parameters
    ----------
    r1          : float — Initial radius [m]
    r2          : float — Target radius [m]
    delta_i_deg : float — Total inclination change [deg]
    split       : float — Fraction of plane change at burn 1 (0.0 to 1.0).
                          If 0.0, the optimal split is computed automatically.
    mass, isp   : float — Optional for propellant calculation

    Returns
    -------
    ManeuverResult
    """
    a_t = (r1 + r2) / 2

    v1        = circular_velocity(r1)
    v2        = circular_velocity(r2)
    v_trans_p = ellipse_velocity(r1, a_t)
    v_trans_a = ellipse_velocity(r2, a_t)

    delta_i = np.radians(delta_i_deg)

    if split == 0.0:
        # Find optimal split by minimising total delta-V
        def total_dv(f):
            i1 = f * delta_i
            i2 = (1 - f) * delta_i
            d1 = np.sqrt(v1**2 + v_trans_p**2 - 2*v1*v_trans_p*np.cos(i1))
            d2 = np.sqrt(v2**2 + v_trans_a**2 - 2*v2*v_trans_a*np.cos(i2))
            return d1 + d2

        # Search over fractions 0 → 1 in fine steps
        fractions = np.linspace(0, 1, 1000)
        dvs       = [total_dv(f) for f in fractions]
        best_f    = fractions[np.argmin(dvs)]
        split     = best_f

    i1 = split * delta_i
    i2 = (1 - split) * delta_i

    dv1 = np.sqrt(v1**2 + v_trans_p**2 - 2*v1*v_trans_p*np.cos(i1))
    dv2 = np.sqrt(v2**2 + v_trans_a**2 - 2*v2*v_trans_a*np.cos(i2))
    dv_total = dv1 + dv2

    tof = orbital_period(a_t) / 2

    mp = 0.0
    if mass is not None and isp is not None:
        mp = propellant_mass(mass, dv_total, isp)

    return ManeuverResult(
        name    = f"Combined Transfer + Plane Change (Δi = {delta_i_deg:.2f} deg)",
        delta_v = dv_total,
        burns   = [
            {"dv": dv1, "location": f"Burn 1 at r1={r1/1e3:.1f} km — {np.degrees(i1):.2f} deg plane change"},
            {"dv": dv2, "location": f"Burn 2 at r2={r2/1e3:.1f} km — {np.degrees(i2):.2f} deg plane change"}
        ],
        tof_s     = tof,
        prop_mass = mp,
        notes     = f"Optimal split: {split*100:.1f}% at burn 1, {(1-split)*100:.1f}% at burn 2"
    )


# ══════════════════════════════════════════════════════════════════
# PHASING MANEUVER
# ══════════════════════════════════════════════════════════════════

def phasing(r_orbit: float, delta_angle_deg: float, n_orbits: int = 1,
            mass: float = None, isp: float = None) -> ManeuverResult:
    """
    Phasing maneuver — adjust in-track position by delta_angle_deg.

    Used for constellation slot insertion or rendezvous approach.

    Parameters
    ----------
    r_orbit        : float — Current circular orbit radius [m]
    delta_angle_deg: float — Phase angle to gain (+ve = catch up, -ve = fall back)
    n_orbits       : int   — Number of phasing orbits to complete

    Returns
    -------
    ManeuverResult
    """
    T_target = orbital_period(r_orbit)
    delta_angle = np.radians(delta_angle_deg)

    # Required phasing orbit period
    T_phase = T_target * (1 - delta_angle / (2 * np.pi * n_orbits))

    # Phasing orbit semi-major axis
    a_phase = (CONST.GM * (T_phase / (2 * np.pi))**2) ** (1/3)

    v_current = circular_velocity(r_orbit)
    v_phase_p = ellipse_velocity(r_orbit, a_phase)

    dv = abs(v_phase_p - v_current)
    dv_total = 2 * dv   # burn in + burn out

    tof = T_phase * n_orbits

    mp = 0.0
    if mass is not None and isp is not None:
        mp = propellant_mass(mass, dv_total, isp)

    return ManeuverResult(
        name    = f"Phasing Maneuver (Δθ = {delta_angle_deg:.1f} deg in {n_orbits} orbit(s))",
        delta_v = dv_total,
        burns   = [
            {"dv": dv, "location": "Entry burn — enter phasing orbit"},
            {"dv": dv, "location": "Exit burn  — return to target orbit"}
        ],
        tof_s     = tof,
        prop_mass = mp,
        notes     = f"Phasing orbit: a = {a_phase/1e3:.2f} km | T = {T_phase/60:.2f} min"
    )


# ══════════════════════════════════════════════════════════════════
# RECOMMEND BEST TRANSFER
# ══════════════════════════════════════════════════════════════════

def recommend_transfer(r1: float, r2: float,
                       delta_i_deg: float = 0.0,
                       r_b: float = None,
                       mass: float = None,
                       isp: float = None) -> dict:
    """
    Compute all applicable transfers and recommend the lowest delta-V option.

    Parameters
    ----------
    r1, r2       : float — Initial and target orbit radii [m]
    delta_i_deg  : float — Required inclination change [deg] (0 = no plane change)
    r_b          : float — Bi-elliptic intermediate radius [m] (auto-set if None)
    mass, isp    : optional for propellant

    Returns
    -------
    dict with keys "recommended", "hohmann", "bielliptic", "combined"
    """
    results = {}

    if delta_i_deg == 0.0:
        h = hohmann(r1, r2, mass, isp)
        results["hohmann"] = h

        if r_b is None:
            r_b = max(r1, r2) * 5
        try:
            be = bielliptic(r1, r2, r_b, mass, isp)
            results["bielliptic"] = be
        except ValueError:
            pass

        best = min(results.values(), key=lambda x: x.delta_v)

    else:
        combined = combined_maneuver(r1, r2, delta_i_deg, mass=mass, isp=isp)
        results["combined"] = combined
        best = combined

    results["recommended"] = best
    return results


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Delta-V Calculator Self-Test ===\n")

    Re = CONST.R_EARTH

    # ── Test 1: LEO → GEO Hohmann ─────────────────────────────────
    r_leo = Re + 400e3
    r_geo = Re + 35786e3
    mass  = 1000.0
    isp   = 311.0    # Hydrazine thruster

    print("--- Test 1: LEO (400 km) → GEO Hohmann ---")
    result = hohmann(r_leo, r_geo, mass, isp)
    print(result.report())

    # ── Test 2: Bi-elliptic LEO → GEO ─────────────────────────────
    print("\n--- Test 2: LEO → GEO Bi-elliptic (r_b = 3.5 R_GEO) ---")
    r_b = r_geo * 3.5
    result = bielliptic(r_leo, r_geo, r_b, mass, isp)
    print(result.report())

    # ── Test 3: Pure plane change at GEO ──────────────────────────
    print("\n--- Test 3: 28.5 deg plane change at GEO ---")
    v_geo = circular_velocity(r_geo)
    result = plane_change(v_geo, 28.5, mass, isp)
    print(result.report())

    # ── Test 4: Combined transfer + plane change ───────────────────
    print("\n--- Test 4: LEO 400km → GTO + 28.5 deg plane change ---")
    r_gto_apo = r_geo
    result = combined_maneuver(r_leo, r_gto_apo, 28.5, mass=mass, isp=isp)
    print(result.report())

    # ── Test 5: Phasing maneuver ───────────────────────────────────
    print("\n--- Test 5: 90 deg phasing at LEO (1 orbit) ---")
    result = phasing(r_leo, 90.0, n_orbits=1, mass=mass, isp=isp)
    print(result.report())

    # ── Test 6: Recommendation engine ─────────────────────────────
    print("\n--- Test 6: Recommend best transfer LEO → GEO ---")
    rec = recommend_transfer(r_leo, r_geo, mass=mass, isp=isp)
    print(f"Hohmann   ΔV : {rec['hohmann'].delta_v:.2f} m/s")
    print(f"Bi-ellip  ΔV : {rec['bielliptic'].delta_v:.2f} m/s")
    print(f"Recommended  : {rec['recommended'].name}")

    # ── Known-answer check: LEO→GEO Hohmann ≈ 3930 m/s ───────────
    h_dv = rec["hohmann"].delta_v
    print(f"\nLEO→GEO Hohmann ΔV : {h_dv:.2f} m/s  (expected ~3930 m/s)")
    assert 3800 < h_dv < 4000, f"Hohmann ΔV {h_dv:.2f} outside expected range!"
    print("Known-answer check: PASSED")
