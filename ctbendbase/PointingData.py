import numpy as np
import pickle
from ctbend.ctbendbase import CTBend as CTBend


class UVCoordinate(object):
    """Dataclass to hold uv-plane coordinates.

       Attributes:
        u (float): u
        v (float): v
    """
    def __init__(self, u, v):
        # type (float, float) -> None
        self.u = u
        self.v = v


class CCDCoordinate(object):
    """Dataclass to hold CCD image coordinates.

       Attributes:
        x (float): x-position in units of CCD pixels.
        y (float): y-position in units of CCD pixels.
    """

    def __init__(self, x, y):
        # type (float, float) -> None
        self.x = x
        self.y = y

    def __str__(self):
        info = "(x, y)=(" + str(self.x) + ", " + str(self.y) + ") px"
        return info

    def __add__(self, other):
        xnew = self.x + other.x
        ynew = self.y + other.y
        return CCDCoordinate(xnew, ynew)

    def __sub__(self, other):
        xnew = self.x - other.x
        ynew = self.y - other.y
        return CCDCoordinate(xnew, ynew)

    def rotate(self, alpha_deg):
        # type (float) -> CCDCoordinate

        """
        Rotate the CCD coordinate by the angle alpha_deg.
        Args:
            alpha_deg: Rotation angle in degrees, e.g. CCD camera rotation
                       angle.
        """

        alpha_rad = np.radians(alpha_deg)

        x_tilde = self.x * np.cos(alpha_rad) + self.y * np.sin(alpha_rad)
        y_tilde = -self.x * np.sin(alpha_rad) + self.y * np.cos(alpha_rad)

        return CCDCoordinate(x_tilde, y_tilde)

    def project2uv(self, pixelscale):
        # type (float) -> UVCoordinate

        """Project the CCD coordinate to the tangential plane of the unit ...

            Args:
                pixelscale (float): Pixelscale of the CCD camera in arcsec
        """

        deg2arcsec = 3600
        pixel2rad = np.radians(pixelscale * 1. / deg2arcsec)
        u = self.x * pixel2rad
        v = self.y * pixel2rad

        return UVCoordinate(u, v)


class DriveCoordinate(object):
    """Dataclass to hold telescope drive coordinates.

       Attributes:
        azimuth: Azimuth position in degrees.
        elevation: Elevation position in degrees.
    """

    def __init__(self, azimuth, elevation):
        # type (float, float) -> None
        self.azimuth = azimuth
        self.elevation = elevation

    def __str__(self):
        info = "(az, el)=(" + str(self.azimuth) + ", "
        info += str(self.elevation) + ") deg"
        return info


class PointingData(object):
    """Dataclass to hold one pointing datum.

       Attributes:
        star (CCDCoordinate): Position of the star on the CCD image.
        telescope (CCDCoordinate): Position of the center of the LED pattern on
                                   the CCD image, i.e. the telescope pointing
                                   direction.
        drive_position (DriveCoordinate): Position of the telescope
                                          drive system.
    """

    def __init__(self, star, telescope, drive_position):
        # type (CCDCoordiante, CCDCoordinate, DriveCoordinate) -> None

        self.star = star
        self.telescope = telescope
        self.drive_position = drive_position

    def __str__(self):
        info = "PointingData--------------------------------:" + "\n"
        info += "Telescope drive: " + str(self.drive_position) + "\n"
        info += "Star on CCD: " + str(self.star) + "\n"
        info += "Telescope pointing on CCD: " + str(self.telescope)
        return info


class PointingDataset(object):
    """Collection of PointingData.

    Attributes:
        pointing_data (list[PointingData]): List of PointingData.
        pixelscale (float): "Pixel scale" of the CCD camera in arcsec.
        bending_model (dict): Description of the pointing model
                              used during data-taking.
    """

    def __init__(self, pixelscale, pointing_model):
        # type (float, CTBendBase) -> None

        self.pointing_data_list = np.array([])
        self.pixelscale = pixelscale
        self._pointing_model = pointing_model

    def pointing_model(self):
        # type (None) -> CTBendBase

        """Returns the pointing model applied while taking the
           PointingDataset.
        """

        parameters = self._pointing_model["parameters"]
        model_name = self._pointing_model["model_name"]

        return getattr(CTBend, model_name)(parameters)

    def __str__(self):
        info = "pixel scale: " + str(self.pixelscale) + " arcsec"

        for pointing_data in self.pointing_data_list:
            info += "\n"
            info += str(pointing_data)
        return info

    def append(self, pointing_data):
        # type (PointingData) -> None

        """
        Append a PointingData point.

        Args:
            pointing_data (PointingData): Data to be appended.
        """
        self.pointing_data_list = np.append(self.pointing_data_list,
                                            pointing_data)

    @property
    def elevation(self):
        # type (None) -> List(float)
        el = []
        for drive_position in self.drive_position():
            elevation = drive_position.elevation
            el.append(elevation)

        return np.array(el)

    @property
    def azimuth(self):
        # type (None) -> List(float)

        az = []
        for drive_position in self.drive_position():
            azimuth = drive_position.azimuth
            az.append(azimuth)

        return np.array(az)

    def old_bending_correction(self, bending_model):
        # type (CTBendBase) -> float, float

        """Returns the bending correction that was applied while taking data.

           Args:
                bending_model (CTBendBase): Bending model applied while taking
                                            data.
        """
        inverter_func = bending_model.invert_bending_model
        azimuth = self.azimuth
        elevation = self.elevation
        azimuth0, elevation0 = inverter_func(azimuth, elevation)

        daz0 = azimuth - azimuth0
        del0 = elevation - elevation0

        return daz0, del0

    def drive_position(self):
        for pointing_data in self.pointing_data_list:
            yield pointing_data.drive_position

    def uv(self, alpha_deg=0):
        # type (float) -> UVCoordinate

        """Returns bending data points in the UV-plane.

        Args:
            alpha_deg (float): CCD rotation angle in degrees.

        Returns:
            ctbend.UVCoordinate
        """

        for pointing_data in self.pointing_data_list:

            delta_ccd = pointing_data.star - pointing_data.telescope
            delta_ccd = delta_ccd.rotate(alpha_deg)
            uv = delta_ccd.project2uv(self.pixelscale)
            yield uv

    def __len__(self):
        # type (None) -> int
        return len(self.pointing_data_list)

    def train_test_split(self, train_fraction):
        # type (float) -> PointingDataset, PointingDataset

        train_length = int(len(self) * train_fraction)

        indices = np.random.permutation(len(self))
        train_indices = indices[:train_length]
        test_indices = indices[train_length:]

        train = PointingDataset(pixelscale=self.pixelscale,
                                pointing_model=self._pointing_model)
        train.pointing_data_list = self.pointing_data_list[train_indices]

        test = PointingDataset(pixelscale=self.pixelscale,
                               pointing_model=self._pointing_model)

        test.pointing_data_list = self.pointing_data_list[test_indices]

        return train, test

    def save(self, filename):
        # type (str) -> None
        """Saves the dataset as a pickle file.

        Args:
            filename: Filename
        """

        with open(filename, "wb") as fout:
            pickle.dump(self, fout)
