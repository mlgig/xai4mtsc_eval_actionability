from dCAM.src.models.CNN_models import dResNetBaseline
from models.MyMiniRocket import MyMiniRocket
from load_data import load_data
from utils import MyDataset
import torch
import numpy as np
import timeit

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #X_train,y_train,X_test,y_test,n_classes, n_channels,seq_len = load_data(device,"my_synth")
    data, seq_len, n_channels, n_classes = load_data("tmp_synth_2lines")
    # TODO remove the following!


    n = 5000
    data['train']['X']  = data['train']['X'][:n]
    data['train']['y']  = data['train']['y'][:n]
    data['test']['X']  = data['test']['X'][:n]
    data['test']['y']  = data['test']['y'][:n]


    miniRocket = MyMiniRocket("MiniRocket",n_channels,seq_len,n_classes)
    X_train = torch.tensor(data['train']['X'] ).to(device)
    X_test = torch.tensor(data['test']['X'] ).to(device)

    start = timeit.default_timer()
    X_train, X_test = miniRocket.transform_dataset(X_train=X_train,X_test=X_test, chunksize=512)
    y_train = torch.Tensor(data['train']['y'].astype(int)).type(torch.uint8).to(device)
    y_test = torch.Tensor(data['test']['y'].astype(int)).type(torch.uint8).to(device)
    torch.cuda.empty_cache()
    print("trans in ", timeit.default_timer() - start)

    start = timeit.default_timer()
    Cs = np.array( [10,100] )
    print(Cs)
    miniRocket.train_regression( X_train, y_train, X_test, y_test,
                                Cs =Cs,k=2,batch_size=128 )
    exit()
    #probs = miniRocket( torch.Tensor( data['test']['X'] ). to("cuda")  )
    print("classifier in ", timeit.default_timer() - start)
    torch.save(miniRocket,"RocketTrialCs10.pt")
    a = 2

if __name__ == "__main__" :
    main()