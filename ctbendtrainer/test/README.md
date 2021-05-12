A consistency check for the ctbend package.

First, make sure that the ctbend directory is in the PYTHONPATH and that the ctbendtrainer conda environment is loaded.

The following commands are relative to the ctbend/ctbendtrainer/test directory.

1) Create tracking Monte-Carlo data:

```
python3 trackingmc.py --config test_input_model_basic4.json --outfile test.pickle
```

Tracking data with a bending configuration which is defined in test_input_model_basic4.json is simulated and written to test.pickle.

2) Analyze the Monte-Carlo pointing dataset:

```
python3 analyze_datafile.py --pointing_dataset_file test.pickle --bending_model test_fit_model_basic4.json --n_cpu 5
```

The Monte-Carlo tracking data (test.pickle) is analyzed with the model defined in test_fit_model_basic4.json. 
Make sure that the posterior estimation is compatible with the input bending configuration.
