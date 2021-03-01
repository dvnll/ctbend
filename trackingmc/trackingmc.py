import argparse
import logging
import json
from pprint import pprint
import numpy as np
import os

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz

from datetime import datetime

import ctbend.ctbendbase.CTBend as CTBend
from ctbend.ctbendtrainer.CTBendGeometry import XYZVector, e_phi, e_theta
from ctbend.ctbendbase.PointingData import CCDCoordinate, DriveCoordinate,\
     PointingData, PointingDataset

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class MCSetup(object):
    def __init__(self, config_dict):

        self.config_dict = config_dict

    @property
    def star(self):
        return SkyCoord(ra=self.config_dict["star_ra"],
                        dec=self.config_dict["star_dec"])

    @property
    def random_star(self):

        ra_h = np.random.uniform(0, 24)
        ra_m = np.random.uniform(0, 60)
        ra_s = np.random.uniform(0, 60)

        star_ra = str(int(ra_h)) + "h"
        star_ra += str(int(ra_m)) + "m"
        star_ra += str(int(ra_s)) + "s"

        dec_d = np.random.uniform(0, 90)
        dec_m = np.random.uniform(0, 60)
        dec_s = np.random.uniform(0, 60)

        star_dec = str(int(dec_d)) + "d"
        star_dec += str(int(dec_m)) + "m"
        star_dec += str(int(dec_s)) + "s"

        return SkyCoord(ra=star_ra, dec=star_dec)
	
    @property
    def location(self):
        lat = u.Quantity(self.config_dict["location_lat"])
        lon = u.Quantity(self.config_dict["location_lon"])
        height = u.Quantity(self.config_dict["location_height"])

        return EarthLocation(lat=lat,
                             lon=lon,
                             height=height)

    @property
    def telescope_focal_length(self):
        return u.Quantity(self.config_dict["telescope_focal_length"])

    @property
    def ccd_focal_length(self):
        return u.Quantity(self.config_dict["ccd_focal_length"])

    @property
    def ccd_pixel_size(self):
        return u.Quantity(self.config_dict["ccd_pixel_size"])

    @property
    def delta_t(self):
        return u.Quantity(self.config_dict["delta_t"])

    @property
    def n_tracking(self):
        return int(self.config_dict["n_tracking"])

    @property
    def start_timestamp(self):
        return self.config_dict["start_timestamp"]

    def tracking_timestamps(self):

        delta_t = u.Quantity(self.config_dict["delta_t"])
        start_timestamp = self.config_dict["start_timestamp"]

        for i in range(int(self.config_dict["n_tracking"])):
            timestamp = start_timestamp + i * delta_t.to(u.s).value
            yield timestamp

    @property
    def pixel_scale(self):

        return self.ccd_pixel_size.to(u.m).value / self.ccd_focal_length.to(u.m).value * u.rad

    @property
    def bending(self):
        parameters = self.config_dict["bending"]["parameters"]
        parameters0 = {"mean": parameters}
        return getattr(CTBend, self.config_dict["bending"]["model"])(parameters0)

    def measured_x1x2(self, true_x1, true_x2):

        sigma = self.config_dict["sigma"]["size"]
        sigma2 = np.power(sigma, 2)
        (dx1, dx2) = np.random.multivariate_normal([0, 0], cov=[[sigma2, 0],[0, sigma2]])
        
        offset_x1 = float(self.config_dict["sigma"]["offset_x1"])
        offset_x2 = float(self.config_dict["sigma"]["offset_x2"])

        return (true_x1 + dx1 + offset_x1, true_x2 + dx2 + offset_x2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type=str,
                        help="MC config json file",
                        dest="CONFIG",
                        required=True)

    parser.add_argument("--outfile",
                        type=str,
                        help="MC tracking data outfile",
                        dest="OUTFILE",
                        required=True)

    options = parser.parse_args()

    if os.path.isfile(options.OUTFILE):
        raise RuntimeError(options.OUTFILE + " already exists")

    with open(options.CONFIG) as fin:
        config = json.load(fin)

    pprint(config)
    setup = MCSetup(config)
    info = "Pixelscale: " + str(setup.pixel_scale.to(u.arcsec).value)
    info += " arcsec"
    logger.info(info)

    pointing_model_for_data_taking = CTBend.ConstantOffsetModel(
                                    parameters={"model": {"mean": {"azimuth_offset_deg": 0.,
                                                                   "elevation_offset_deg": 0.}}})

    pointing_dataset = PointingDataset(
                pixelscale=setup.pixel_scale.to(u.arcsec).value,
                pointing_model=pointing_model_for_data_taking.serialize())

    for timestamp in setup.tracking_timestamps():

        def random_star_altaz(minimal_altitude_deg : float = 5):
            """Returns random altaz coordinates for a test star.

                Args:
                minimal_altitude_deg: Minimal altitude of the star in degrees.
            """ 
            def get_random_star():
                time = datetime.fromtimestamp(timestamp)
                observing_time = Time(time)

                aa = AltAz(location=setup.location,
                           obstime=observing_time)

                return setup.random_star.transform_to(aa)
        
            star_altaz = get_random_star()
            while star_altaz.alt.to(u.deg).value < minimal_altitude_deg:
                star_altaz = get_random_star()
            return star_altaz

        star_altaz = random_star_altaz()

        #star_altaz = setup.star.transform_to(aa)

        bending = setup.bending
        az_star_deg = star_altaz.az.to(u.deg).value
        el_star_deg = star_altaz.alt.to(u.deg).value

        delta_az_deg = bending.delta_azimuth(az=az_star_deg,
                                             el=el_star_deg)
        delta_el_deg = bending.delta_elevation(az=az_star_deg,
                                               el=el_star_deg)

        telescope_az = (az_star_deg - delta_az_deg) * u.deg
        telescope_el = (el_star_deg - delta_el_deg) * u.deg

        telescope = XYZVector(az=telescope_az.to(u.deg).value,
                              el=telescope_el.to(u.deg).value)

        def get_e_phi():     
            delta_az_derivative_phi = bending.delta_azimuth_derivative_phi(
                                        az=telescope_az.to(u.deg).value,
                                        el=telescope_el.to(u.deg).value)

            delta_el_derivative_phi = bending.delta_elevation_derivative_phi(
                                        az=telescope_az.to(u.deg).value,
                                        el=telescope_el.to(u.deg).value)

            _e_phi = e_phi(az_tel=telescope.az,
                           el_tel=telescope.alt,
                           delta_az_derivative_phi=delta_az_derivative_phi,
                           delta_el_derivative_phi=delta_el_derivative_phi)
            return _e_phi

        def get_e_theta(): 
            delta_az_derivative_theta = bending.delta_azimuth_derivative_theta(
                                            az=telescope_az.to(u.deg).value,
                                            el=telescope_el.to(u.deg).value)

            delta_el_derivative_theta = \
                    bending.delta_elevation_derivative_theta(
                                            az=telescope_az.to(u.deg).value,
                                            el=telescope_el.to(u.deg).value)

            _e_theta = e_theta(
                        az_tel=telescope.az,
                        el_tel=telescope.alt,
                        delta_az_derivative_theta=delta_az_derivative_theta,
                        delta_el_derivative_theta=delta_el_derivative_theta)
            return _e_theta

        def telescope_ccd_position():
            """The pointing direction of the telescope in CCD coordinates
               is the center of the LED pattern. By definition, this is the 
               zero-vector in CCD coordinates in this MC.
            """
            x1_tel = 0
            x2_tel = 0

            return CCDCoordinate(x1_tel, x2_tel)

        def star_ccd_position():
            star_vector = XYZVector(star_altaz.az.to(u.deg).value,
                                    star_altaz.alt.to(u.deg).value)


            focal_length_mm = setup.ccd_focal_length.to(u.mm).value
            image_star_length = focal_length_mm / (telescope * star_vector)

            image = telescope * (star_vector * telescope) * 2. - star_vector

            image_star = image * image_star_length

            
            fp_u = image_star * get_e_phi()
            fp_v = image_star * get_e_theta()

            x1_star = fp_u / setup.ccd_pixel_size.to(u.mm).value
            x2_star = fp_v / setup.ccd_pixel_size.to(u.mm).value

            measured_x1_star, measured_x2_star = setup.measured_x1x2(x1_star,
                                                                     x2_star)

            return CCDCoordinate(measured_x1_star, measured_x2_star)

        drive_position = DriveCoordinate(telescope.az, telescope.alt)

        pointing_data = PointingData(telescope=telescope_ccd_position(),
                                     star=star_ccd_position(),
                                     drive_position=drive_position)

        pointing_dataset.append(pointing_data)
        #print(pointing_data)
    
    pointing_dataset.save(options.OUTFILE)
