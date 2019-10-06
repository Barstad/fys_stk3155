Project 1 of subject fys_stk3155

Group members:
* Johannes Barstad
* Olve Heitmann


HOWTO:
* Create a folder called data/ and add SRTM_data_Norway_1.tif to that file
* Run GenerateData.py
* To choose the different data sets, modify the Config file by setting the EXPERIMENT paramater to either 'FRANKE', 'NOISY_FRANKE' or 'TERRAIN'
* Run the different experiment files to get the results for the different model types
  - OLSExperiemnt.py for using regular OLS regression
  - RidgeExperiment.py for using ridge regression
  - LassoExperiment.py for using lasso regression
* The results folder already contains the results from our runs of the experiment.
