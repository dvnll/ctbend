import ctbend.ctbendbase as ctbend
import pymc3 as pm
import theano.tensor as tensor
import numpy as np
from math import sqrt, log
from scipy.special import erfinv
import datetime
import json
import logging
from astropy.units import Quantity
from astropy import units as units
import ctbend.ctbendtrainer.CTBendGeometry as CTBendGeometry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class ModelTrainer(object):

    def __init__(self,
                 training_dataset: ctbend.PointingDataset,
                 bending_model_dict: dict,
                 n_cpu_cores: int):

        self.n_cpu_cores = n_cpu_cores
        self.bending_model_dict = bending_model_dict
        try:
            model_name = bending_model_dict["model_name"]
            self.model = getattr(ctbend.CTBend, model_name)(
                                 parameters={"priors": {}})
            self.model.parameters_are_distributions = True

        except AttributeError as e:
            raise RuntimeError(e)
        except Exception as e:
            info = "Unknown exception while loading CTBend model: "
            info += str(e)
            raise RuntimeError(e)

        self.training_dataset = training_dataset

        self.logger = logging.getLogger(__name__)
        self.logger.info("Training size: " + str(len(training_dataset)))

    def train(self,
              n_samples: int = 4000,
              burn: int = 1000,
              tuning_steps: int = 1000,
              progressbar: bool = False):

        parameters = self.model.model_parameter_names

        azimuth = []
        elevation = []
        for drive_position in self.training_dataset.drive_position():
            azimuth.append(drive_position.azimuth)
            elevation.append(drive_position.elevation)

        bending_prior_dict = self.bending_model_dict["bending_priors"]
        with pm.Model() as model:

            for parameter_name in parameters:

                distribution = getattr(
                            pm,
                            bending_prior_dict[parameter_name]["distribution"])
                prior_parameter_values = {"name": parameter_name}
                for key in bending_prior_dict[parameter_name].keys():
                    if key == "distribution":
                        continue
                    parameter_value = Quantity(
                                    bending_prior_dict[parameter_name][key])

                    pp_value = parameter_value.to(units.deg).value
                    prior_parameter_values[key] = pp_value

                prior_dict = self.model.parameters["priors"]
                prior_dict[parameter_name] = distribution(
                                                **prior_parameter_values)

            nuisance_priors = self.bending_model_dict["nuisance_priors"]

            def get_sigma_prior():
                pixel_sigma_median = nuisance_priors["sigma"]["median"]
                pixel_sigma_90quantile = nuisance_priors["sigma"]["q90"]

                def pixel_sigma_mu(pixel_sigma_median):
                    return log(pixel_sigma_median)

                def pixel_sigma_sd(pixel_sigma_90quantile, pixel_sigma_median):
                    pixel_sigma_sd = log(pixel_sigma_90quantile)
                    pixel_sigma_sd -= pixel_sigma_mu(pixel_sigma_median)
                    pixel_sigma_sd /= sqrt(2.) * erfinv(0.8)
                    return pixel_sigma_sd

                sigma = pm.Lognormal("sigma",
                                     mu=pixel_sigma_mu(pixel_sigma_median),
                                     sd=pixel_sigma_sd(pixel_sigma_90quantile,
                                                       pixel_sigma_median))

                pixelscale = self.training_dataset.pixelscale
                arcsec2deg = 3600
                arcsec2rad = np.radians(arcsec2deg)
                sigma = sigma * pixelscale * arcsec2rad

                return sigma

            def get_nu_prior():
                nu_parameter = float(nuisance_priors["nu"]["lam"])
                nu = pm.Exponential("nu", lam=nu_parameter)
                return nu

            def get_alpha_prior():

                if nuisance_priors["alpha"]["distribution"] == "fixed":
                    alpha = Quantity(nuisance_priors["alpha"]["value"])
                    return alpha.to(units.deg).value

                distribution = getattr(
                                    pm,
                                    nuisance_priors["alpha"]["distribution"])
                prior_parameter_values = {"name": "alpha"}
                for key in nuisance_priors["alpha"].keys():
                    if key == "distribution":
                        continue
                    parameter_value = Quantity(nuisance_priors["alpha"][key])

                    pp_value = parameter_value.to(units.deg).value
                    prior_parameter_values[key] = pp_value

                alpha = distribution(**prior_parameter_values)

                return alpha

            def _model_u(model, az_drive, el_drive, daz0, del0):

                az_star = az_drive - daz0
                el_star = el_drive - del0

                daz0_tensor = tensor.as_tensor_variable(daz0)
                del0_tensor = tensor.as_tensor_variable(del0)

                mis_el = model.delta_elevation(az_star, el_star)
                mis_az = model.delta_azimuth(az_star, el_star)

                el_star_tensor = tensor.as_tensor_variable(el_star)
                az_star_tensor = tensor.as_tensor_variable(az_star)

                delta_az = daz0_tensor - mis_az
                delta_el = del0_tensor - mis_el

                az_tel = az_star_tensor + delta_az
                el_tel = el_star_tensor + delta_el

                telescope = CTBendGeometry.XYZVector(az_tel, el_tel)

                star = CTBendGeometry.XYZVector(az_star, el_star)
                image = telescope * (star * telescope) * 2. - star

                e_phi = CTBendGeometry.e_phi(
                    az_tel,
                    el_tel,
                    model.delta_azimuth_derivative_phi(az_tel, el_tel),
                    model.delta_elevation_derivative_phi(az_tel, el_tel))

                length = 1. / (image * telescope)

                image_uv = image * length

                return image_uv * e_phi

            def _model_v(model, az_drive, el_drive, daz0, del0):

                az_star = az_drive - daz0
                el_star = el_drive - del0

                daz0_tensor = tensor.as_tensor_variable(daz0)
                del0_tensor = tensor.as_tensor_variable(del0)

                mis_el = model.delta_elevation(az_star, el_star)
                mis_az = model.delta_azimuth(az_star, el_star)

                el_star_tensor = tensor.as_tensor_variable(el_star)
                az_star_tensor = tensor.as_tensor_variable(az_star)

                delta_az = daz0_tensor - mis_az
                delta_el = del0_tensor - mis_el

                az_tel = az_star_tensor + delta_az
                el_tel = el_star_tensor + delta_el

                telescope = CTBendGeometry.XYZVector(az_tel, el_tel)

                star = CTBendGeometry.XYZVector(az_star, el_star)
                image = telescope * (star * telescope) * 2. - star

                e_theta = CTBendGeometry.e_theta(
                    az_tel,
                    el_tel,
                    model.delta_azimuth_derivative_theta(az_tel, el_tel),
                    model.delta_elevation_derivative_theta(az_tel, el_tel))

                length = 1. / (image * telescope)

                image_uv = image * length

                return image_uv * e_theta

            sigma = get_sigma_prior()
            nu = get_nu_prior()
            alpha_deg = get_alpha_prior()

            u = []
            v = []
            for uv in self.training_dataset.uv(alpha_deg=alpha_deg):
                u.append(uv.u)
                v.append(uv.v)

            b = self.training_dataset.pointing_model()

            daz0, del0 = self.training_dataset.old_bending_correction(b)
            model_u = _model_u(self.model, azimuth, elevation, daz0, del0)
            model_v = _model_v(self.model, azimuth, elevation, daz0, del0)

            pm.StudentT("u",
                        mu=model_u,
                        sd=sigma,
                        nu=nu,
                        observed=u)

            pm.StudentT("v",
                        mu=model_v,
                        sd=sigma,
                        nu=nu,
                        observed=v)

            self.trace = pm.sample(n_samples,
                                   burn=burn,
                                   cores=self.n_cpu_cores,
                                   progressbar=progressbar,
                                   tune=tuning_steps)

            try:
                alpha_mean = np.mean(self.trace["alpha"])
                info = "Alpha degrees in trainer: " + str(alpha_mean)
                self.logger.debug(info)

            except KeyError:
                """Alpha was not sampled -> ignore.
                """

                pass

            self.waic = pm.stats.waic(self.trace)

    def posterior_parameter_info(self):

        parameter_list = self.model.model_parameter_names

        def parameter_info(parameter_name):
            try:
                parameter_trace = self.trace[parameter_name]
            except KeyError:
                self.logger.debug(parameter_name + " not sampled")
                return

            median = np.percentile(parameter_trace, q=50)

            width = 68
            plus = np.percentile(parameter_trace, q=50 + width / 2)
            minus = np.percentile(parameter_trace, q=50 - width / 2)

            info = parameter + ": Median " + str(round(median, 2))
            info += ", 68% CL [" + str(round(minus, 2))
            info += ", " + str(round(plus, 2)) + "]"
            self.logger.info(info)

        for parameter in parameter_list:

            parameter_info(parameter)

        nuisance_parameter_list = ["alpha", "nu", "sigma"]

        for parameter in nuisance_parameter_list:
            parameter_info(parameter)

    @property
    def model_dictionary(self):

        model_json = json.loads(pm.stats.summary(self.trace).to_json())

        model_dict = {"created": datetime.datetime.now(),
                      "model": model_json,
                      "name": self.model.name,
                      "gelman_rubin": pm.diagnostics.gelman_rubin(self.trace),
                      "effective_n": pm.diagnostics.effective_n(self.trace),
                      "trace": self.trace,
                      "tpoints": self.tpoints,
                      "waic": self.waic}

        return model_dict
