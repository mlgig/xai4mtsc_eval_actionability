import numpy as np


def load_data(data_name, concat=False):

    if data_name.startswith("synth"):
        # synthetic datasets
        path="./datasets/synthetics/"
        if data_name=="synth_1line":
            name = "nSamples_7500_diffLines_1_nTotFeatures_8.npy"
        elif data_name=="synth_2lines":
            name = "nSamples_7500_diffLines_2_nTotFeatures_8.npy"

        data = np.load(path+name,allow_pickle=True).item()

        n = 7500
        train_X  = data['train']['X'][:n]
        train_y  = data['train']['y'][:n]
        test_X  = data['test']['X'][:n]
        test_y = data['test']['y'][:n]

    elif data_name=="CMJ":
        # Counter Movement Jump
        data = np.load("./datasets/CounterMovementJump/resampled.npy", allow_pickle=True).item()
        train_X  = data['X_train']
        train_y  = data['y_train']
        test_X  = data['X_test']
        test_y = data['y_test']

    elif data_name=="MP":
        a = 2
    else:
        raise Exception("data_name must 'synth_1line', 'synth_2lines' , 'CMJ' or 'MP' ")

    # transform (in case) and get info
    if concat:
        train_X = train_X.reshape(train_X.shape[0],-1)
        test_X = test_X.reshape(test_X.shape[0],-1)

    seq_len = train_X.shape[-1]
    n_channels = train_X.shape[-2]
    n_classes = len( np.unique(train_y) )


    return  train_X, train_y.astype(int), test_X, test_y.astype(int), seq_len, n_channels, n_classes
