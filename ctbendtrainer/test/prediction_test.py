import ctbend.ctbendbase as ctbendbase
import argparse
import pickle
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--model_file",
                     type=str,
                     help="Pointing model file (pickle format)",
                     dest="INFILE",
                     required=True)

parser_options = parser.parse_args()

with open(parser_options.INFILE, "rb") as fin:
    model_json = pickle.load(fin)

    print(model_json)


model = ctbendbase.bending_factory(model_json)

for i in range(10):
    az = np.random.uniform(0, 360)
    el = np.random.uniform(0, 90)

    az = round(az, 2)
    el = round(el, 2)

    delta_az = round(model.delta_azimuth(az, el), 2)
    delta_el = round(model.delta_elevation(az, el), 2)
    
    info = "Pointing (az, el)=(" + str(az) + ", " + str(el) + ") deg"
    info += " -> Correction: (" + str(delta_az) + ", " + str(delta_el) + ") deg"
    print(info)
