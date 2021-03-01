import numpy as np
import math

try:
    import pymc3 as pm

except ImportError:
    info = "Warning: No model training possible in this environment"
    print(info)


def radians(deg):
    return deg / 360. * 2. * math.pi


class XYZVector(object):

    """
    Unit vector in the horizon/AltAz system

    Constructor:
    az: Azimuth in degrees
    el: Altitude/elevation in degrees
    """

    def __init__(self, az, el, math=np):

        az_rad = radians(az)
        el_rad = radians(el)

        sin = math.sin
        cos = math.cos

        self.x = cos(el_rad) * cos(az_rad)
        self.y = -cos(el_rad) * sin(az_rad)
        self.z = sin(el_rad)

    @property
    def az(self):
        return np.degrees(np.arctan2(-self.y, self.x))

    @property
    def alt(self):
        return np.degrees(np.arcsin(self.z))

    def __mul__(self, other):

        if isinstance(other, XYZVector):
            product = self.x * other.x
            product += self.y * other.y
            product += self.z * other.z

            return product

        x = self.x * other
        y = self.y * other
        z = self.z * other
        res = XYZVector(0, 0)
        res.x = x
        res.y = y
        res.z = z

        return res

    def __rmul__(self, other):
        return self * other

    def distance(self, other):
        scalar_product = self * other
        scalar_product = scalar_product.eval()
        scalar_product[np.abs(scalar_product > 1.)] = 1.
        dist_rad = np.arccos(scalar_product)
        deg2arcsec = 3600
        dist_arcsec = np.degrees(dist_rad) * deg2arcsec

        return dist_arcsec

    def __sub__(self, other):

        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z

        diff = XYZVector(0, 0)
        diff.x = x
        diff.y = y
        diff.z = z

        return diff

    def __add__(self, other):

        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z

        sumv = XYZVector(0, 0)
        sumv.x = x
        sumv.y = y
        sumv.z = z

        return sumv

    def __str__(self):

        return str((self.x.eval(), self.y.eval(), self.z.eval()))


class e_phi(XYZVector):

    def __init__(self, az_tel, el_tel, delta_az_derivative_phi,
                 delta_el_derivative_phi, math=np):

        sin = math.sin
        cos = math.cos

        az_rad = radians(az_tel)
        el_rad = radians(el_tel)

        #delta_az_derivative_phi = 0
        #delta_el_derivative_phi = 0

        x = -sin(az_rad) * cos(el_rad) * (1. - delta_az_derivative_phi) + sin(el_rad) * cos(az_rad) * delta_el_derivative_phi
        y = -cos(az_rad) * cos(el_rad) * (1. - delta_az_derivative_phi) - sin(el_rad) * sin(az_rad) * delta_el_derivative_phi
        z = -cos(el_rad) * delta_el_derivative_phi

        norm = math.sqrt(x * x + y * y + z * z)

        self.x = x / norm
        self.y = y / norm
        self.z = z / norm


class e_theta(XYZVector):

    def __init__(self, az_tel, el_tel, delta_az_derivative_theta,
                 delta_el_derivative_theta, math=np):

        sin = math.sin
        cos = math.cos

        az_rad = radians(az_tel)
        el_rad = radians(el_tel)

        #delta_az_derivative_theta = 0
        #delta_el_derivative_theta = 0

        x = -sin(el_rad) * cos(az_rad) * (1. - delta_el_derivative_theta) + cos(el_rad) * sin(az_rad) * delta_az_derivative_theta
        y = sin(el_rad) * sin(az_rad) * (1. - delta_el_derivative_theta) + cos(el_rad) * cos(az_rad) * delta_az_derivative_theta
        z = cos(el_rad) * (1. - delta_el_derivative_theta)

        norm = math.sqrt(x * x + y * y + z * z)

        self.x = x / norm
        self.y = y / norm
        self.z = z / norm


