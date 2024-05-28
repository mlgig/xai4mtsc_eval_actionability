# Explanation-method4MTSC_captum

## Overview 
This repo contain the code used in "Improving the Evaluation and Actionability of Explanation Methods for Multivariate Time Series Classification" (temporary repo to be updated after accept/reject notification)

## Dataset
To create a dir named datasets and place the content found here  https://drive.google.com/drive/folders/18tbVOkbac8Bvr8-8VZLf3fpzzJNc1vss?usp=drive_link

## Trained models
To create a dir named 'saved_models' and place the content found here https://drive.google.com/drive/folders/1_Ld_6JFriAWLq18xRGzp-uxHOfrM5V5f?usp=drive_link

## Computed explanation
(optional) Saliency maps produced in our experiments, originally placed in a  dir named 'explanations'. Content can be found here https://drive.google.com/drive/folders/1B6NmIuekDVwyJxBZQX6uPOulc-_uGvPU?usp=drive_link

## Interpret Time outputs
(optional) Outputs produced in our experiments by interpret time, originally placed in a  dir named 'IntepretTime_results'. Content can be found here https://drive.google.com/drive/folders/1S4y9_R1S7ba5XTUBEpmqDT4E9eNHuP8b?usp=drive_link

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