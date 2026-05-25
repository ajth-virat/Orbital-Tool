"""
constants.py — Physical and orbital constants for the GNC Tool.

Logic:
This file stores every fixed physical value the tool needs.
All values are in SI units (meters, seconds, kilograms).
The dataclass groups related constants so imports stay clean.
No calculations happen here — only definitions.
Other modules import from here, so changing a value once updates the entire tool.
Units are noted in comments to prevent silent errors downstream.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    # ── Gravitational ──────────────────────────────────────────────
    GM: float = 3.986004418e14        # Earth gravitational parameter [m^3/s^2]
    G: float = 6.67430e-11            # Universal gravitational constant [m^3/kg/s^2]
    M_EARTH: float = 5.972168e24      # Earth mass [kg]

    # ── Earth Geometry ─────────────────────────────────────────────
    R_EARTH: float = 6378136.6        # Earth mean equatorial radius [m]
    R_EARTH_POLAR: float = 6356751.9  # Earth polar radius [m]
    F_EARTH: float = 1 / 298.257223563  # Earth flattening (WGS-84)

    # ── Earth Rotation ─────────────────────────────────────────────
    OMEGA_EARTH: float = 7.2921150e-5  # Earth rotation rate [rad/s]

    # ── Gravitational Harmonics ────────────────────────────────────
    J2: float = 1.08262668e-3         # Second zonal harmonic (oblateness) [dimensionless]
    J3: float = -2.53265649e-6        # Third zonal harmonic [dimensionless]
    J4: float = -1.61962159e-6        # Fourth zonal harmonic [dimensionless]

    # ── Atmospheric ────────────────────────────────────────────────
    ATMO_REF_DENSITY: float = 1.225   # Sea-level air density [kg/m^3]
    SCALE_HEIGHT: float = 8500.0      # Atmospheric scale height — simple exponential model [m]

    # ── Solar Radiation Pressure ───────────────────────────────────
    SOLAR_FLUX: float = 1361.0        # Solar irradiance at 1 AU [W/m^2]
    C_LIGHT: float = 2.99792458e8     # Speed of light [m/s]
    AU: float = 1.495978707e11        # 1 Astronomical Unit [m]

    # ── Time ───────────────────────────────────────────────────────
    J2000_JD: float = 2451545.0       # Julian Date of J2000.0 epoch
    SECONDS_PER_DAY: float = 86400.0  # Seconds in one day [s]

    # ── Derived (computed from above) ──────────────────────────────
    @property
    def SRP_PRESSURE(self) -> float:
        """Solar radiation pressure at 1 AU [N/m^2]."""
        return self.SOLAR_FLUX / self.C_LIGHT

    @property
    def MU(self) -> float:
        """Alias for GM — used interchangeably in astrodynamics literature."""
        return self.GM


# ── Module-level singleton — import and use directly ───────────────
CONST = Constants()


if __name__ == "__main__":
    print("=== GNC Tool Constants ===")
    print(f"GM            : {CONST.GM:.6e} m^3/s^2")
    print(f"R_EARTH       : {CONST.R_EARTH:.3f} m")
    print(f"J2            : {CONST.J2:.8e}")
    print(f"OMEGA_EARTH   : {CONST.OMEGA_EARTH:.7e} rad/s")
    print(f"SRP Pressure  : {CONST.SRP_PRESSURE:.6e} N/m^2")
    print(f"AU            : {CONST.AU:.6e} m")
