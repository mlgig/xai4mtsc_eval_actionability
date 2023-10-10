from sklearn.preprocessing import LabelEncoder
from torch.utils import data


def gen_cube(instance):
    result = []
    for i in range(len(instance)):
        result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
    return result

def one_hot_encoding(train_labels,test_labels):
    enc = LabelEncoder()
    y_train = enc.fit_transform(train_labels)
    y_test = enc.transform(test_labels)

    return y_train,y_test,enc



def pre_fature_normalization(X_train,X_test):
    eps = 1e-6
    f_mean = X_train.mean(axis=0, keepdims=True)
    f_std = X_train.std(axis=0, keepdims=True) + eps  # epsilon to avoid dividing by 0
    X_train_tfm2 = (X_train - f_mean) / f_std
    X_test_tfm2 = (X_test - f_mean) / f_std
    return  X_train_tfm2,X_test_tfm2


class MyDataset(data.Dataset):

    def __init__(self,X,y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]