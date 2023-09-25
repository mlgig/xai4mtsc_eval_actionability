from dCAM.src.models.CNN_models import dResNetBaseline
from models.MyMiniRocket import MyMiniRocket
from load_data import load_data
from utils import MyDataset
import torch
import numpy as np

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #X_train,y_train,X_test,y_test,n_classes, n_channels,seq_len = load_data(device,"my_synth")
    data, seq_len, n_channels, n_classes = load_data("tmp_synth_2lines")
    # TODO remove the following!
    """
    data['train']['X']  = data['train']['X'][:100]
    data['train']['y']  = data['train']['y'][:100]
    data['test']['X']  = data['test']['X'][:100]
    data['test']['y']  = data['test']['y'][:100]
    """
    miniRocket = MyMiniRocket(n_channels,seq_len,n_classes)

    X_train, X_test = miniRocket.transform_dataset(X_train=data['train']['X'],X_test=data['test']['X'], chunksize=512)
    y_train = torch.Tensor(data['train']['y'].astype(int)).type(torch.uint8).to(device)
    y_test = torch.Tensor(data['test']['y'].astype(int)).type(torch.uint8).to(device)
    torch.cuda.empty_cache()

    miniRocket.train_regression( X_train, y_train, X_test, y_test,
                                Cs =np.logspace(2, 4, 3),k=5,batch_size=64 )

    """
    #instanciate rocket
    #rocket_transform = ROCKET(n_channels,seq_len,device=device)
    miniRocket_transfrom = MiniRocketFeatures(n_channels,seq_len,device=device)
    miniRocket_transfrom.fit(X_train)
    print("begin rocket trans")

    # transform dataset
    starttime = timeit.default_timer()
    #X_train_trans = rocket_transform(X_train)
    #X_test_trans = rocket_transform(X_test)
    X_train_trans = get_minirocket_features(X_train, miniRocket_transfrom, chunksize=512, to_np=False)
    torch.cuda.empty_cache()
    X_test_trans = get_minirocket_features(X_test, miniRocket_transfrom, chunksize=512, to_np=False)
    torch.cuda.empty_cache()
    #X_train_trans =  miniRocket_transfrom(X_train)
    #X_test_trans =  miniRocket_transfrom(X_test)
    print("rocket transformation", (timeit.default_timer() - starttime), " seconds" )

    X_train_norm,X_test_norm = pre_fature_normalization(X_train_trans,X_test_trans)

    del X_train_trans,X_test_trans
    torch.cuda.empty_cache()


    Cs = np.logspace(2, 4, 3)
    """


if __name__ == "__main__" :
    main()