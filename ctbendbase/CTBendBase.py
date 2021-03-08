import numpy as np
from math import pi
from scipy.optimize import minimize
from abc import ABC, abstractmethod


def radians(deg):
    return 2. * pi / 360. * deg


class CTBendBase(ABC):
    """Base class from which all ctbend models are to be derived.
    """

    def __init__(self, parameters):
        # type: (dict) -> None

        """Constructor.

        Args:
            parameters: Dictionary with parameters of the bending model.
        """

        self.parameters = parameters
        self._math = np
        self.deg2arcsec = 3600.

    @abstractmethod
    def azimuth_model_terms(self, az_rad, el_rad):
        pass

    @abstractmethod
    def azimuth_derivative_phi(self, az_rad, el_rad):
        pass

    @abstractmethod
    def azimuth_derivative_theta(self, az_rad, el_rad):
        pass

    @abstractmethod
    def elevation_model_terms(self, az_rad, el_rad):
        pass

    @abstractmethod
    def elevation_derivative_phi(self, az_rad, el_rad):
        pass

    @abstractmethod
    def elevation_derivative_theta(self, az_rad, el_rad):
        pass

    @classmethod
    def modelname(cls):
        return cls.__name__

    def _pointing_correction(self, az, el, altaz):
        # type: (float, float, str) -> float

        """Pointing correction.

        Args:
            az: Requested azimuth in degrees.
            el: Requested elevation in degrees.
            altaz: Requested axis; Either 'azimuth' or 'elevation'.

        Returns:
            Pointing correction for the requested axis in degrees.
        """

        if altaz not in ["azimuth", "elevation"]:
            info = "altaz argument must be either azimuth or"
            info += " elevation"
            raise RuntimeError(info)

        p = self.model_parameters
        az_rad = radians(az)
        el_rad = radians(el)

        term_function = {}
        term_function["azimuth"] = self.azimuth_model_terms
        term_function["elevation"] = self.elevation_model_terms

        term_dict = term_function[altaz](az_rad, el_rad)
        delta = 0.
        for term in term_dict.keys():
            delta += p[term] * term_dict[term]

        return delta

    def delta_azimuth_derivative_phi(self, az, el):

        p = self.model_parameters
        az_rad = radians(az)
        el_rad = radians(el)

        term_dict = self.azimuth_derivative_phi(az_rad, el_rad)
        delta = 0.
        for term in term_dict.keys():
            delta += p[term] * term_dict[term]

        delta = radians(delta)
        return delta

    def delta_elevation_derivative_phi(self, az, el):

        p = self.model_parameters
        az_rad = radians(az)
        el_rad = radians(el)

        term_dict = self.elevation_derivative_phi(az_rad, el_rad)
        delta = 0.
        for term in term_dict.keys():
            delta += p[term] * term_dict[term]

        return radians(delta)

    def delta_azimuth_derivative_theta(self, az, el):

        p = self.model_parameters
        az_rad = radians(az)
        el_rad = radians(el)

        term_dict = self.azimuth_derivative_theta(az_rad, el_rad)
        delta = 0.
        for term in term_dict.keys():
            delta += p[term] * term_dict[term]

        return radians(delta)

    def delta_elevation_derivative_theta(self, az, el):

        p = self.model_parameters
        az_rad = radians(az)
        el_rad = radians(el)

        term_dict = self.elevation_derivative_theta(az_rad, el_rad)
        delta = 0.
        for term in term_dict.keys():
            delta += p[term] * term_dict[term]

        return radians(delta)

    def delta_azimuth(self, az, el):
        # type: (float, float) -> float

        """Pointing correction in azimuth.

        Args:
            az: Requested azimuth in degrees.
            el: Requested elevation in degrees.

        Returns:
            Pointing correction in azimuth in degrees.
        """

        return self._pointing_correction(az, el, altaz="azimuth")

    def delta_elevation(self, az, el):
        # type: (float, float) -> float

        """Pointing correction in elevation.

        Args:
            az: Requested azimuth in degrees.
            el: Requested elevation in degrees.

        Returns:
            Pointing correction in elevation in degrees.
        """

        return self._pointing_correction(az, el, altaz="elevation")

    @property
    def model_parameter_names(self):

        name_list = list(self.azimuth_model_terms(0, 0).keys())
        name_list += list(self.elevation_model_terms(0, 0).keys())

        return np.unique(name_list)

    def serialize(self):
        return {"model_name": self.modelname(), "parameters": self.parameters}

    @property
    def model_parameters(self):
        if hasattr(self, "parameters_are_distributions"):
            parameter_dict = {}
            for par in self.model_parameter_names:
                parameter_dict[par] = self.parameters["priors"][par]
            return parameter_dict

        return self.parameters

    def invert_bending_model(self,
                             azimuth,
                             elevation,
                             verbose=False,
                             tolerance=1.e-8):
        """Invert the current bending model, i.e. given input (azimuth,
           elevation), find the altaz coordinates
           to which the telescope would be pointing without correction.

           Technically, values for az0 and el0 are searched such that 
           the difference between the prediction BendingModel(az0, el0)
           and the input altaz is minimized.

           Args:
           azimuth (List[float]): List of corrected azimuth coordinates
                                  in degrees.
           elevation (List[float]): List of corrected elevation coordinates
                                    in degrees.
           verbose (bool): Flag to control the verbosity of the minimization
                           algorithm.
           tolerance (float): Tolerance parameter of the minimization
                              algorithm.

           Returns:
           altaz (List[float], List[float]): Uncorrected altaz coordinates in
                                             degrees.
        """
        uncorrected_azimuth = []
        uncorrected_elevation = []

        def _telescope_pointing_inverter_loss_function(x, az, el):
            az0 = x[0]
            el0 = x[1]
            a = np.abs(az - az0 - self.delta_azimuth(az0, el0))
            b = np.abs(el - el0 - self.delta_elevation(az0, el0))

            loss = a + b

            return loss

        for az, el in zip(azimuth, elevation):
            range_error = False
            x0 = (az, el)
            res = minimize(_telescope_pointing_inverter_loss_function,
                           x0, args=(az, el), method="nelder-mead",
                           options={"xtol": tolerance, "disp": verbose})

            if range_error or res.success is False:
                res = minimize(self._telescope_pointing_inverter_loss_function,
                               x0, args=(az, el), method="L-BFGS-B",
                               options={"disp": verbose})

            uncorrected_azimuth.append(res.x[0])
            uncorrected_elevation.append(res.x[1])

        return np.array(uncorrected_azimuth), np.array(uncorrected_elevation)
