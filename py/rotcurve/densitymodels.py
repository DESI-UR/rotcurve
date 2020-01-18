# coding: utf-8
"""Spherically symmetric density models.

For examples of models, see David Merritt, Alister W. Graham, Ben Moore, Juerg
Diemand, and Balsa Terzic, "Empirical Models for Dark Matter Halos. I.
Nonparametric Construction of Density Profiles and Comparison with Parametric
Models," Astron J. 132:2685–2700, 2006, astro-ph/0509417.
"""

import numpy as np
from scipy.integrate import quad
from abc import ABC

class DensityProfile(ABC):
    """Abstract base class for spherically symmetric density profiles. Includes
    functions to estimate density as a function of radius, mass enclosed within
    radius r, and velocity a location r."""

    def __init__(self, name):
        """Initialize a mass profile.

        Parameters
        ----------
        name : str
            Name of the profile model.
        """
        self.name = name

    def density(self, r):
        """Density at galactocentric radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].

        Returns
        -------
        density : float or ndarray
            Density at r [solar mass / kpc^3]
        """
        pass

    def _r2density(self, r):
        """r^2 x density at galactocentric radius r."""
        return r**2 * self.density(r)

    def mass(self, r):
        """Mass contained inside radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].

        Returns
        -------
        M : float or ndarray
            Mass at r'<r [solar mass].
        """
        if np.isscalar(r):
            M, dM = quad(self._r2density, np.finfo(dtype=float).eps, r)
            return M

        M = []
        for _r in r:
            _M, _dM = quad(self._r2density, np.finfo(dtype=float).eps, _r)
            M.append(_M)
        return np.asarray(M)

    def velocity(self, r, return_mass=False):
        """Velocity at radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].
        return_mass : bool
            If true, return both mass and velocity.

        Returns
        -------
        v : float or ndarray
            Velocity at r [km/s].
        """
        G = 4.3009173e-6 # kpc km^2 / Mpc / s^2
        M = self.mass(r)
        v = np.sqrt(G * M / r)
        if return_mass:
            return M, v
        return v


class ConstDensityProfile(DensityProfile):

    def __init__(self, name, rho0):
        """Initialize a constant mass profile.

        Parameters
        ----------
        name : str
            Name of the profile model.
        rho0 : float
            Constant density [solar mass/kpc^3].
        """
        super().__init__(name)
        self.rho0 = rho0

    def density(self, r):
        """Density at galactocentric radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].

        Returns
        -------
        density : float or ndarray
            Density at r [solar mass / kpc^3]
        """
        if np.isscalar(r):
            return self.rho0
        return np.full_like(r, self.rho0)


class PowerlawDensityProfile(DensityProfile):

    def __init__(self, name, rho0, gamma):
        """Initialize a constant mass profile.

        Parameters
        ----------
        name : str
            Name of the profile model.
        rho0 : float
            Density normalization [solar mass/kpc^3].
        gamma : float
            Power law index.
        """
        super().__init__(name)
        self.rho0 = rho0
        self.gamma = gamma

    def density(self, r):
        """Density at galactocentric radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].

        Returns
        -------
        density : float or ndarray
            Density at r [solar mass / kpc^3]
        """
        return r**self.gamma * self.rho0


class NFWDensityProfile(DensityProfile):

    def __init__(self, name, rho0, Rs):
        """Initialize a constant mass profile.

        Parameters
        ----------
        name : str
            Name of the profile model.
        rho0 : float
            Density normalization [solar mass/kpc^3].
        Rs : float
            Scale radius [kpc].
        """
        super().__init__(name)
        self.rho0 = rho0
        self.Rs = Rs

    def density(self, r):
        """Density at galactocentric radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].

        Returns
        -------
        density : float or ndarray
            Density at r [solar mass / kpc^3]
        """
        return self.rho0 / (r/self.Rs * (1+r/self.Rs)**2)


class EinastoDensityProfile(DensityProfile):

    def __init__(self, name, rho_e, r_e, n):
        """Initialize a constant mass profile.

        Parameters
        ----------
        name : str
            Name of the profile model.
        rho_e : float
            Density at radius r_e [solar mass/kpc^3].
        r_e : float
            Radius where volume contains 1/2 of total mass [kpc].
        n : float
            Einasto index (>0.5).
        """
        super().__init__(name)
        self.rho_e = rho_e
        self.r_e = r_e
        self.n = n

        if n >= 0.5:
            self.dn = 3*n - 1/3 + 0.0079/n
        else:
            raise ValueError('Use n < {:g}'.format(n))

    def density(self, r):
        """Density at galactocentric radius r.

        Parameters
        ----------
        r : float or ndarray
            Galactocentric radius [kpc].

        Returns
        -------
        density : float or ndarray
            Density at r [solar mass / kpc^3]
        """
        return self.rho_e * np.exp(-self.dn * ((r/self.r_e)**(1/self.n) - 1))
