import os
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

    parser.add_argument("--n_cpu",
                        type=int,
                        help="Number of CPU cores to use for sampling",
                        dest="NCPU",
                        default=2)

    parser.add_argument("--training_fraction",
                        type=float,
                        help="Fraction of pointing data used for training",
                        dest="FTRAIN",
                        default=0.8)

    parser.add_argument("--progress_bar",
                        type=bool,
                        help="Show a progress bar",
                        dest="PROGRESS",
                        default=True)

    parser.add_argument("--output_file",
                        type=str,
                        help="Output file",
                        dest="OUTFILE",
                        default="NONE")

    parser_options = parser.parse_args()

    with open(parser_options.INFILE, "rb") as fin:
        pointing_ds = pickle.load(fin)

    with open(parser_options.CTBEND) as fin:
        bending_model_dict = json.load(fin)

    print(pointing_ds)

    training_dataset, test_dataset = pointing_ds.train_test_split(
                                        train_fraction=parser_options.FTRAIN)
    print("----------")
    print(training_dataset)

    trainer = ModelTrainer(training_dataset=training_dataset,
                           bending_model_dict=bending_model_dict,
                           n_cpu_cores=parser_options.NCPU)
    trainer.train(progressbar=parser_options.PROGRESS)
    trainer.posterior_parameter_info()
    
    if not parser_options.OUTFILE == "NONE" and os.path.isfile(parser_options.OUTFILE):
        error = "Model output file " + parser_options.OUTFILE + " already exists"
        raise RuntimeError(error)

    if not parser_options.OUTFILE == "NONE":
        with open(parser_options.OUTFILE, "wb") as fout:
            pickle.dump(trainer.model_dictionary, fout)
