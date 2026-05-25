"""
validator.py — Input validation for all orbital parameters.

Logic:
This file runs before any calculation touches the inputs.
Every parameter is checked against physical bounds — not just data types.
Keplerian and Cartesian inputs are validated through separate functions.
Warnings flag suspicious but legal values; errors stop execution immediately.
A clean ValidationResult object tells the caller exactly what passed or failed.
This prevents silent wrong answers, which are worse than loud crashes.
All units are SI — metres and metres-per-second — matching constants.py.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from constants import CONST


# ══════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(f"[ERROR] {msg}")
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(f"[WARNING] {msg}")

    def report(self) -> str:
        lines = []
        lines += self.errors
        lines += self.warnings
        if self.valid and not self.warnings:
            lines.append("[OK] All inputs validated successfully.")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# KEPLERIAN VALIDATOR
# ══════════════════════════════════════════════════════════════════

def validate_keplerian(a, e, i, raan, argp, nu, deg=True) -> ValidationResult:
    """
    Validate Keplerian orbital elements.

    Parameters
    ----------
    a    : float — Semi-major axis [m]
    e    : float — Eccentricity [dimensionless]
    i    : float — Inclination [deg if deg=True, else rad]
    raan : float — RAAN [deg or rad]
    argp : float — Argument of perigee [deg or rad]
    nu   : float — True anomaly [deg or rad]
    deg  : bool  — True if angles supplied in degrees

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    angle_max = 360.0 if deg else 2 * np.pi
    incl_max  = 180.0 if deg else np.pi
    label     = "deg" if deg else "rad"

    # ── Semi-major axis ────────────────────────────────────────────
    if not isinstance(a, (int, float)):
        result.add_error("Semi-major axis (a) must be a number.")
    elif a <= CONST.R_EARTH:
        result.add_error(
            f"Semi-major axis {a/1e3:.1f} km is below Earth's surface "
            f"(R_Earth = {CONST.R_EARTH/1e3:.1f} km). Orbit would intersect Earth."
        )
    elif a < CONST.R_EARTH + 150e3:
        result.add_warning(
            f"Altitude ~{(a - CONST.R_EARTH)/1e3:.1f} km is very low. "
            "Rapid orbital decay expected within days."
        )
    elif a > 5e8:
        result.add_warning(
            f"Semi-major axis {a/1e9:.2f} Gm exceeds reasonable Earth orbit range. "
            "Verify units are metres."
        )

    # ── Eccentricity ───────────────────────────────────────────────
    if not isinstance(e, (int, float)):
        result.add_error("Eccentricity (e) must be a number.")
    elif e < 0:
        result.add_error(f"Eccentricity {e} is negative. Must be >= 0.")
    elif e >= 1.0:
        result.add_error(
            f"Eccentricity {e} >= 1. Tool handles elliptical orbits only (e < 1)."
        )
    elif e > 0.9:
        result.add_warning(
            f"Eccentricity {e:.3f} is very high. "
            "Perigee may be inside Earth's atmosphere."
        )

    # ── Perigee altitude check (when both a and e are valid) ───────
    if isinstance(a, (int, float)) and isinstance(e, (int, float)):
        if 0 <= e < 1 and a > 0:
            r_perigee = a * (1 - e)
            if r_perigee < CONST.R_EARTH:
                result.add_error(
                    f"Perigee radius {r_perigee/1e3:.1f} km is below Earth's surface. "
                    f"Perigee altitude = {(r_perigee - CONST.R_EARTH)/1e3:.1f} km."
                )
            elif r_perigee < CONST.R_EARTH + 80e3:
                result.add_warning(
                    f"Perigee altitude {(r_perigee - CONST.R_EARTH)/1e3:.1f} km is "
                    "below the Karman line (100 km). Extreme drag expected."
                )

    # ── Inclination ────────────────────────────────────────────────
    if not isinstance(i, (int, float)):
        result.add_error("Inclination (i) must be a number.")
    elif not (0 <= i <= incl_max):
        result.add_error(
            f"Inclination {i} {label} is out of range [0, {incl_max}] {label}."
        )

    # ── RAAN ───────────────────────────────────────────────────────
    if not isinstance(raan, (int, float)):
        result.add_error("RAAN must be a number.")
    elif not (0 <= raan < angle_max):
        result.add_warning(
            f"RAAN {raan} {label} outside [0, {angle_max}) {label}. "
            "Will be normalised automatically."
        )

    # ── Argument of perigee ────────────────────────────────────────
    if not isinstance(argp, (int, float)):
        result.add_error("Argument of perigee must be a number.")
    elif not (0 <= argp < angle_max):
        result.add_warning(
            f"Argument of perigee {argp} {label} outside [0, {angle_max}) {label}. "
            "Will be normalised automatically."
        )

    # ── True anomaly ───────────────────────────────────────────────
    if not isinstance(nu, (int, float)):
        result.add_error("True anomaly must be a number.")
    elif not (0 <= nu < angle_max):
        result.add_warning(
            f"True anomaly {nu} {label} outside [0, {angle_max}) {label}. "
            "Will be normalised automatically."
        )

    return result


# ══════════════════════════════════════════════════════════════════
# CARTESIAN VALIDATOR
# ══════════════════════════════════════════════════════════════════

def validate_cartesian(r_eci, v_eci) -> ValidationResult:
    """
    Validate Cartesian ECI state vector.

    Parameters
    ----------
    r_eci : array-like shape (3,) — Position vector [m]
    v_eci : array-like shape (3,) — Velocity vector [m/s]

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # ── Shape checks ───────────────────────────────────────────────
    try:
        r = np.asarray(r_eci, dtype=float).flatten()
        v = np.asarray(v_eci, dtype=float).flatten()
    except Exception:
        result.add_error("Position and velocity must be numeric arrays.")
        return result

    if r.shape != (3,):
        result.add_error(f"Position vector must have 3 components. Got shape {r.shape}.")
    if v.shape != (3,):
        result.add_error(f"Velocity vector must have 3 components. Got shape {v.shape}.")

    if not result.valid:
        return result

    if np.any(np.isnan(r)) or np.any(np.isinf(r)):
        result.add_error("Position vector contains NaN or Inf.")
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        result.add_error("Velocity vector contains NaN or Inf.")

    if not result.valid:
        return result

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # ── Position magnitude ─────────────────────────────────────────
    if r_mag < CONST.R_EARTH:
        result.add_error(
            f"Position magnitude {r_mag/1e3:.1f} km is inside Earth "
            f"(R_Earth = {CONST.R_EARTH/1e3:.1f} km)."
        )
    elif r_mag < CONST.R_EARTH + 150e3:
        result.add_warning(
            f"Altitude ~{(r_mag - CONST.R_EARTH)/1e3:.1f} km is very low. "
            "Rapid decay expected."
        )
    elif r_mag > 5e8:
        result.add_warning(
            f"Position magnitude {r_mag/1e9:.3f} Gm is very large. "
            "Verify units are metres."
        )

    # ── Velocity magnitude ─────────────────────────────────────────
    v_circular = np.sqrt(CONST.GM / r_mag)
    v_escape   = np.sqrt(2 * CONST.GM / r_mag)

    if v_mag < 10:
        result.add_warning(
            f"Velocity magnitude {v_mag:.2f} m/s is extremely low. "
            "Verify units are metres per second, not km/s."
        )
    elif v_mag > v_escape:
        result.add_error(
            f"Velocity {v_mag/1e3:.3f} km/s exceeds escape velocity "
            f"{v_escape/1e3:.3f} km/s at this altitude. Orbit is hyperbolic."
        )
    elif v_mag > 0.95 * v_escape:
        result.add_warning(
            f"Velocity {v_mag/1e3:.3f} km/s is close to escape velocity "
            f"({v_escape/1e3:.3f} km/s). Orbit is highly eccentric."
        )

    # ── Energy check — must be negative for bound orbit ───────────
    energy = v_mag**2 / 2 - CONST.GM / r_mag
    if energy >= 0:
        result.add_error(
            f"Orbital energy {energy:.3e} J/kg >= 0. "
            "Spacecraft is not in a bound orbit."
        )

    return result


# ══════════════════════════════════════════════════════════════════
# SPACECRAFT PROPERTIES VALIDATOR
# ══════════════════════════════════════════════════════════════════

def validate_spacecraft(mass, cd, area) -> ValidationResult:
    """
    Validate spacecraft physical properties used in perturbation models.

    Parameters
    ----------
    mass : float — Spacecraft mass [kg]
    cd   : float — Drag coefficient [dimensionless]
    area : float — Cross-sectional area [m^2]

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    if not isinstance(mass, (int, float)) or mass <= 0:
        result.add_error(f"Mass must be a positive number. Got: {mass}")
    elif mass < 0.1:
        result.add_warning(f"Mass {mass} kg is very small. Is this a CubeSat?")
    elif mass > 100000:
        result.add_warning(f"Mass {mass} kg is very large. Verify units are kg.")

    if not isinstance(cd, (int, float)) or cd <= 0:
        result.add_error(f"Drag coefficient must be a positive number. Got: {cd}")
    elif not (1.5 <= cd <= 3.5):
        result.add_warning(
            f"Cd = {cd} is outside typical range [1.5, 3.5]. "
            "Standard assumption for LEO spacecraft is 2.2."
        )

    if not isinstance(area, (int, float)) or area <= 0:
        result.add_error(f"Cross-sectional area must be positive. Got: {area}")
    elif area > 1000:
        result.add_warning(f"Area {area} m^2 is very large. Verify units are m^2.")

    return result


# ══════════════════════════════════════════════════════════════════
# PROPAGATION SETTINGS VALIDATOR
# ══════════════════════════════════════════════════════════════════

def validate_propagation(duration_s, step_s) -> ValidationResult:
    """
    Validate propagation duration and timestep.

    Parameters
    ----------
    duration_s : float — Total propagation duration [s]
    step_s     : float — Output timestep [s]

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    if not isinstance(duration_s, (int, float)) or duration_s <= 0:
        result.add_error(f"Propagation duration must be positive. Got: {duration_s} s")
    elif duration_s > 365 * CONST.SECONDS_PER_DAY:
        result.add_warning(
            f"Propagation duration {duration_s/CONST.SECONDS_PER_DAY:.1f} days "
            "exceeds 1 year. This may take significant compute time."
        )

    if not isinstance(step_s, (int, float)) or step_s <= 0:
        result.add_error(f"Timestep must be positive. Got: {step_s} s")
    elif step_s > 3600:
        result.add_warning(
            f"Timestep {step_s} s ({step_s/3600:.1f} hr) is coarse. "
            "Orbital features may be missed."
        )
    elif step_s < 1:
        result.add_warning(
            f"Timestep {step_s} s is very fine. Output will be large."
        )

    if isinstance(duration_s, (int, float)) and isinstance(step_s, (int, float)):
        if step_s > 0 and duration_s > 0 and step_s >= duration_s:
            result.add_error(
                "Timestep must be smaller than propagation duration."
            )

    return result


# ══════════════════════════════════════════════════════════════════
# COMBINED FULL VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_all_keplerian(a, e, i, raan, argp, nu,
                            mass, cd, area,
                            duration_s, step_s,
                            deg=True) -> ValidationResult:
    """Run all validators for a Keplerian input session."""
    combined = ValidationResult()

    for result in [
        validate_keplerian(a, e, i, raan, argp, nu, deg),
        validate_spacecraft(mass, cd, area),
        validate_propagation(duration_s, step_s)
    ]:
        combined.errors   += result.errors
        combined.warnings += result.warnings
        if not result.valid:
            combined.valid = False

    return combined


def validate_all_cartesian(r_eci, v_eci,
                            mass, cd, area,
                            duration_s, step_s) -> ValidationResult:
    """Run all validators for a Cartesian input session."""
    combined = ValidationResult()

    for result in [
        validate_cartesian(r_eci, v_eci),
        validate_spacecraft(mass, cd, area),
        validate_propagation(duration_s, step_s)
    ]:
        combined.errors   += result.errors
        combined.warnings += result.warnings
        if not result.valid:
            combined.valid = False

    return combined


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Validator Self-Test ===\n")

    print("--- Test 1: Valid ISS-like Keplerian orbit ---")
    r = validate_keplerian(6778e3, 0.0001, 51.6, 120.0, 90.0, 45.0)
    print(r.report())

    print("\n--- Test 2: Orbit inside Earth (bad a) ---")
    r = validate_keplerian(5000e3, 0.0001, 51.6, 120.0, 90.0, 45.0)
    print(r.report())

    print("\n--- Test 3: Hyperbolic eccentricity ---")
    r = validate_keplerian(6778e3, 1.2, 51.6, 120.0, 90.0, 45.0)
    print(r.report())

    print("\n--- Test 4: Valid Cartesian state vector ---")
    r_eci = np.array([-181776, -5638771, 3755797], dtype=float)
    v_eci = np.array([5628.6, -3012.6, -4249.6], dtype=float)
    r = validate_cartesian(r_eci, v_eci)
    print(r.report())

    print("\n--- Test 5: Velocity too high (escape) ---")
    r = validate_cartesian(r_eci, v_eci * 3.0)
    print(r.report())

    print("\n--- Test 6: Spacecraft properties ---")
    r = validate_spacecraft(500, 2.2, 4.0)
    print(r.report())

    print("\n--- Test 7: Bad Cd ---")
    r = validate_spacecraft(500, 8.0, 4.0)
    print(r.report())

    print("\n--- Test 8: Full combined validation ---")
    r = validate_all_keplerian(6778e3, 0.0001, 51.6, 120.0, 90.0, 45.0,
                                500, 2.2, 4.0, 86400, 60)
    print(r.report())
