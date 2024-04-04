# Explanation-method4MTSC_captum

## Overview 
This repo contain the code used in "Improving the Evaluation and Actionability of Explanation Methods for Multivariate Time Series Classification" (temporary repo to be updated after accept/reject notification)

## Dataset
To be updated after accept/reject notification. Please create a directory named "datasets"

## Trained models
To be updated after accept/reject notification. Please create a directory named "saved_models"

## Computed explanation
To be updated after accept/reject notification. Please create a directory named "explanations"

## Interpret Time outputs
To be updated after accept/reject notification. Please create a directory named "IntepretTime_results"

## Code usage
Code run using python 3.9.18, using the library listed in requirements_py3.9.18.txt file. Executable files are:

### train_models.py: 
Script to train the models. Mandatory arguments are dataset(possible choices are CounterMovementJump, Military press along with synthetic and ECG data used in InterpretTime publication)
and classifier (possible choices are ResNet, dResNet, Rocket, MiniRocket and ConvTran)
. Optional field is concat i.e. whether to concatenate all channels to get a univariate time series

### compare_explanations.py
Script used to run the ground truth methodology described in the paper on synthetic data.
No argument is required nut some assumptions are made as
saliency maps are in "explanations" folder" datasets are in "datasets" folder, etc.

### compute_attributions.py
Script to be used for computing the saliency maps. 
At the top of the script a dictionary named captum_methods is defined to list the methods to be used (edit it if you want to remove/add methods)
Mandatory fields are datasets (same as before), classifiers (same as before) and ns_chunks i.e. which number of chunks (definition in the paper) to try. Use -1 to compute point-wise i.e. no chunking

### interpretTime.py
Script to be used to run interpret Time method. Some modifications were made for instance adding other masks, as specified in the paper, etc.
Only mandatory argument is datasets (same as before)