import numpy as np


class UVCoordinate(object):
    """Dataclass to hold uv-plane coordinates.

       Attributes:
        u (float): u
        v (float): v
    """
    def __init__(self, u, v):
        self.u = u
        self.v = v


class CCDCoordinate(object):
    """Dataclass to hold CCD image coordinates.

       Attributes:
        x (float): x-position in units of CCD pixels.
        y (float): y-position in units of CCD pixels.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        xnew = self.x + other.x
        ynew = self.y + other.y
        return CCDCoordinate(xnew, ynew)

    def __sub__(self, other):
        xnew = self.x - other.x
        ynew = self.y - other.y
        return CCDCoordinate(xnew, ynew)

    def rotate(self, alpha_deg):
        """
        QQQQQQQQQQQ
        Args:
            alpha_deg: CCD camera rotation angle in degrees
        """

        alpha_rad = np.radians(alpha_deg)

        x_tilde = self.x * np.cos(alpha_rad) + self.y * np.sin(alpha_rad)
        y_tilde = -self.x * np.sin(alpha_rad) + self.y * np.cos(alpha_rad)

        return CCDCoordinate(x_tilde, y_tilde)

    def project2uv(self, pixelscale):
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
        self.azimuth = azimuth
        self.elevation = elevation


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

    def __init__(self, star, telescope, drive_position, timestamp):
        self.star = star
        self.telescope = telescope
        self.drive_position = drive_position


class PointingDataset(object):
    """Collection of PointingData.

    Attributes:
        pointing_data (list[PointingData]): List of PointingData.
        pixelscale (float): "Pixel scale" of the CCD camera.
    """

    def __init__(self, pixelscale):

        self.pointing_data_list = []
        self.pixelscale = pixelscale

    def append(self, pointing_data):
        """
        Append a PointingData point.

        Args:
            pointing_data (PointingData): Data to be appended.
        """
        self.pointing_data_list.append(pointing_data)

    def old_bending_correction(self, bending_model):

        inverter_func = bending_model.invert_bending_model
        azimuth0, elevation0 = inverter_func(self.azimuth,
                                             self.elevation)

        daz0 = self.azimuth - azimuth0
        del0 = self.elevation - elevation0

        return daz0, del0

    def drive_position(self):
        for pointing_data in self.pointing_data_list:
            yield pointing_data.drive_position

    def uv(self, alpha_deg=0):
        """aa
        Args:
            alpha_deg: CCD rotation angle in degrees.

        Returns:
            np.array ...
        """

        for pointing_data in self.pointig_data_list:

            delta_ccd = pointing_data.star - pointing_data.telescope
            delta_ccd = delta_ccd.rotate(alpha_deg)
            uv = delta_ccd.project2uv(self.pixelscale)

            yield uv

    def __len__(self):
        return len(self.x1_star)

    def train_test_split(self, train_fraction=0.8):

        train_length = int(len(self) * train_fraction)

        indices = np.random.permutation(len(self))
        train_indices = indices[:train_length]
        test_indices = indices[train_length:]

        train = PointingDataset(pixelscale=self.pixelscale)
        train.pointing_data_list = self.pointing_data_list[train_indices]

        test = PointingDataset(pixelscale=self.pixelscale)
        test.pointing_data_list = self.pointing_Data_list[test_indices]

        return train, test
