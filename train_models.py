from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN
from pytorch_utils import transform_data4ResNet
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
from utils import MyDataset
import torch
import numpy as np
import timeit
from sklearn.linear_model import RidgeClassifierCV

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_X, train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data("CMJ")

    """
    miniRocket = MyMiniRocket("MiniRocket",n_channels,seq_len,n_classes)
    X_train = torch.tensor( train_X ).type(torch.float).to(device)
    X_test = torch.tensor( test_X ).type(torch.float).to(device)

    start = timeit.default_timer()
    X_train, X_test = miniRocket.transform_dataset(X_train=X_train,X_test=X_test, chunk_size=512,normalise=True)
    y_train = torch.Tensor(train_y).type(torch.uint8).to(device)
    y_test = torch.Tensor( test_y).type(torch.uint8).to(device)
    torch.cuda.empty_cache()
    print("trans in ", timeit.default_timer() - start)

    start = timeit.default_timer()
    Cs = np.logspace(-4,4,10)
    miniRocket.train_regression( X_train, y_train, X_test, y_test,
                                Cs =Cs,k=5,batch_size=128 )
    #probs = miniRocket( torch.Tensor( data['test']['X'] ). to("cuda")  )
    print("classifier in ", timeit.default_timer() - start)
    #torch.save(miniRocket,"saved_models/Rocket_1line_CS10.pt")
    """




    train_loader, test_loader,n_channels,n_classes, device, enc= (
        transform_data4ResNet(train_X,train_y,test_X,test_y,device=device))
    resNet = dResNetBaseline(n_channels, mid_channels=128,num_pred_classes= n_classes).to(device)
    model = ModelCNN(model=resNet, n_epochs_stop=50,device=device)
    acc = model.train(num_epochs=300,train_loader=train_loader,test_loader=test_loader)
    print("resNet accuracy was ",acc)
    exit()
    #torch.save(model,"saved_models/resNet128_1line.pt")

    train_X, train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data("tmp_synth_2lines", concat=True)
    print(train_X.shape,test_X.shape)
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),cv=5)
    ridge.fit(train_X,train_y)
    print("ridge accuracy was ",ridge.score(test_X,test_y))


if __name__ == "__main__" :
    main()