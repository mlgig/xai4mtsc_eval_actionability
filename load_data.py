import os.path

import numpy as np
from sktime.datasets  import  load_from_tsfile_to_dataframe
from os.path import join
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.datasets import load_from_arff_to_dataframe

def load_data(data_name, concat=False , explanation_gt=False):

    if data_name.startswith("synth"):

        path="./datasets/synthetics/"
        name = "synth_2lines.npy"

        data = np.load(os.path.join(path,name),allow_pickle=True).item()

        train_X  = data['train']['X']
        train_y  = data['train']['y']
        test_X  = data['test']['X']
        test_y = data['test']['y']

    elif data_name=="CMJ":

        data = np.load("./datasets/CounterMovementJump/CMJ_resampled.npy", allow_pickle=True).item()

        train_X  = data['train']['X']
        train_y  = data['train']['y']
        test_X  = data['test']['X']
        test_y = data['test']['y']

    elif data_name=="CMJ_orig":
        # Counter Movement Jump raw data i.e. varying length, including more data before and after jump (NOT USED in any experiment)
        train_X, train_y = load_from_arff_to_dataframe("./datasets/CounterMovementJump/CounterMovementJump_TRAIN.arff",replace_missing_vals_with='0')
        test_X, test_y = load_from_arff_to_dataframe("./datasets/CounterMovementJump/CounterMovementJump_TEST.arff",replace_missing_vals_with='0')
        train_X, test_X = from_nested_to_3d_numpy(train_X), from_nested_to_3d_numpy(test_X)
        a =3

    elif data_name.startswith('MP_centered'):
        # Military press centered i.e. subtracting mean from each channel
        data = np.load('./datasets/MilitaryPress/MP_centered.npy', allow_pickle=True).item()
        train_X  = data['train']['X']
        train_y  = data['train']['y']
        test_X  = data['test']['X']
        test_y = data['test']['y']

    elif data_name.startswith('MP'):
        # Military Press raw data
        base_path = "./datasets/MilitaryPress/"
        train_X, train_y = load_from_tsfile_to_dataframe(join(base_path,"TRAIN_full_X.ts"))
        test_X, test_y = load_from_tsfile_to_dataframe(join(base_path,"TEST_full_X.ts"))

        column_names = ('Nose_X', 'Neck_X', 'RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'MidHip_X',
                        'RHip_X', 'RKnee_X', 'RAnkle_X', 'LHip_X', 'LKnee_X', 'LAnkle_X', 'REye_X', 'LEye_X', 'REar_X', 'LEar_X', 'LBigToe_X',
                        'LSmallToe_X', 'LHeel_X', 'RBigToe_X', 'RSmallToe_X', 'RHeel_X', 'Nose_Y', 'Neck_Y', 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y',
                        'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'MidHip_Y', 'RHip_Y', 'RKnee_Y', 'RAnkle_Y', 'LHip_Y', 'LKnee_Y', 'LAnkle_Y',
                        'REye_Y', 'LEye_Y', 'REar_Y', 'LEar_Y', 'LBigToe_Y', 'LSmallToe_Y', 'LHeel_Y', 'RBigToe_Y', 'RSmallToe_Y', 'RHeel_Y')

        train_X.columns = column_names
        test_X.columns = column_names

        if data_name=="MP":
            # using the domain expert channels selection
            columns_subset = [ 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y','RHip_Y', 'LHip_Y']
            train_X = from_nested_to_3d_numpy( train_X[columns_subset])
            test_X = from_nested_to_3d_numpy( test_X[columns_subset])
        else:
            # or using the full 50 channels
            train_X = from_nested_to_3d_numpy( train_X)
            test_X = from_nested_to_3d_numpy( test_X)

    elif data_name=="ECG":
        train = np.load("./datasets/ECG/ecg_train.npy",allow_pickle=True).item()
        val = np.load("./datasets/ECG/ecg_val.npy",allow_pickle=True).item()
        test = np.load("./datasets/ECG/ecg_test.npy",allow_pickle=True).item()

        train_X = np.concatenate( (train["X"], val["X"]) ,axis=0)
        train_y = np.concatenate( ( np.argmax( train["y"],axis=-1) ,np.argmax( val["y"],axis=-1) ), axis=0)
        test_X = test["X"]
        test_y = np.argmax( test["y"],axis=-1)

    else:
        raise Exception("data_name must be either synth_data, 'CMJ', 'MP50', 'MP', 'MP_centred' or 'ECG' ")

    if concat:
        # concat if required
        train_X = np.squeeze( train_X.reshape(train_X.shape[0],1,-1) )
        test_X = np.squeeze( test_X.reshape(test_X.shape[0],1,-1) )

    # get info about the selected dataset
    seq_len = train_X.shape[-1]
    n_channels = train_X.shape[-2]
    n_classes = len( np.unique(train_y) )

    # return the ground truth data if required
    if explanation_gt and data_name.startswith("synth"):
        return  test_X, test_y, data["test"]['ground_truth'] , seq_len, n_channels, n_classes
    else:
        return  train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes

