#import os
from ctbend.ctbendtrainer.ModelTrainer import ModelTrainer
import argparse
import pickle
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pointing_dataset_file",
                        type=str,
                        help="Pointing dataset input file (pickle format)",
                        dest="INFILE",
                        required=True)

    parser.add_argument("--bending_model",
                        type=str,
                        help="Fit ctbend model json definition",
                        dest="CTBEND",
                        required=True)

    #parser.add_argument(n_cpu)
    #parser.add_argument(train_fraction)
    #parser.add_argument(progressbar)

    parser_options = parser.parse_args()

    with open(parser_options.INFILE, "rb") as fin:
        pointing_ds = pickle.load(fin)

    with open(parser_options.CTBEND) as fin:
        bending_model_dict = json.load(fin)

    print(pointing_ds)

    training_dataset, test_dataset = pointing_ds.train_test_split(
                                                    train_fraction=0.8)
    print("----------")
    print(training_dataset)

    trainer = ModelTrainer(training_dataset=training_dataset,
                           bending_model_dict=bending_model_dict,
                           n_cpu_cores=2)
    trainer.train(progressbar=True)
    trainer.posterior_parameter_info()
    """
    if os.path.isfile(p_options.OUTFILE):
        error = "Model output file " + p_options.OUTFILE + " already exists"
        raise RuntimeError(error)
    """
