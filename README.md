# Predicting-and-overcoming-inter-participant-variability-in-c-VEP-based-BCI
Repository of the code and files to create results for the paper "Predicting and overcoming inter-participant variability in c-VEP-based BCI"

First you need to create an environment with python 3.11.7 and with pip.
Then you can use the command below to install all the necessary libraries:

```
pip install -r requirements.txt
```

## Benchmark of the decoding pipelines

To perform the benchmark of the decoding pipelines and get the following metrics : balanced epoch-level accuravy, trial-level accuracy, training time, prediction time, epoch-level recall score, trial-level f1 score, you will have to use the file STL.py

You can use the command :
```
python ./scr/STL.py --participants *list_of_participants_name* --timewise time_sample --clf_name *name of the decoding model in the available ones* --ws 0.35 --nb_epoch 20 --path *path to the participant data* --fpath *path to the folder containing the results and preprecessed data* --method *DA or SiSu*
```

SiSu correspond to the WP procedure.
You can create the preprocessed data for the PT based decoding models with the **create_data.py** file.

## Make the correlation results
You can just run the **get_neuro_predictors.py** file.

## Plot the results
You can run the **plt_results.py** file.

## Complementary files

A few complementray files are available in complements folder and helps to runs the differents files. (wavelets parameters, results...).




