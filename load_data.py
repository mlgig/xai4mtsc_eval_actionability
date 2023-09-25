import numpy as np


def load_data(data_name):
    # TODO just temp" ; to specify number of channels
    path="/home/davide/Desktop/gianmarco_datasets/new_ones/"
    if data_name=="tmp_synth_1line":
        name = "nSamples_7500_diffLines_1_nTotFeatures_8.npy"
    elif data_name=="tmp_synth_2lines":
        name = "nSamples_7500_diffLines_2_nTotFeatures_8.npy"
    else:
        raise ("data_name either 'tmp_synth_1line' or 'tmp_synth_2lines'")

    data = np.load(path+name,allow_pickle=True).item()
    return data, 500, 8, 2
