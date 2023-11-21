import os.path

import torch
from pytorch_utils import transform2tensors, transform_data4ResNet, transform4ConvTran
import models.ConvTran.hyper_parameters  as conTrans_param
from models.dCAM.src.models.CNN_models import ModelCNN, ResNetBaseline, dResNetBaseline
from load_data import load_data
from sklearn.metrics import accuracy_score
import timeit
from models.ConvTran.Models.model import ConvTran

import models.ConvTran.hyper_parameters  as conTran_param
from models.ConvTran.utils import Initialization

from models.ConvTran.Models.utils import load_model as load_convTran
from sktime.transformations.panel.rocket import (
	MiniRocket,
	MiniRocketMultivariate,
	MiniRocketMultivariateVariable,
)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from  sklearn.preprocessing import StandardScaler
import os

"""
# TODO subsitute every occurency of synth_2lines with a "global variable" dataset_name
for dataset_name in ['synth_2lines']:
	train_X,train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data(dataset_name)
	print("test shape",test_y.shape , test_X.shape)
	device="cuda"
	for file_name in os.listdir("saved_models/"+dataset_name) : #["dResNet.pt","ResNet.pt","MiniRocket.pt",]:
		if file_name.endswith(".pt"):
			arch = torch.load( os.path.join( "./saved_models/",dataset_name,file_name), map_location=device )
			model = ModelCNN(arch,1,device=device )
			if file_name.startswith("dResNet") or file_name.startswith("dInception"):
				train_loader, test_loader, enc = transform_data4ResNet(X_train=train_X,y_train= train_y,
																   X_test=test_X,y_test=test_y,device=device,batch_s=(64,16))
			else:
				train_loader, test_loader, enc = transform2tensors(X_train=train_X,y_train= train_y,
															X_test=test_X,y_test=test_y,device=device,batch_size=(64,16))
			#print(test_loader.dataset.X.shape, test_loader.dataset.X.shape)
			start = timeit.default_timer()
			y_pred = model.predict(test_loader)
			end = timeit.default_timer()
			print(	dataset_name, file_name, accuracy_score(enc.transform (test_y),y_pred),  "\t time was",end-start)
"""

for dataset_name in  ['CMJ','MP','synth_2lines','synth_1line']:
	train_X,train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data(dataset_name)
	device="cuda"

	files =  os.listdir( "/".join( ("saved_models",dataset_name) ) )
	for file in files:

		if file.startswith("ConvTrans"):
			parser = file.split("_")
			norm = bool([parser[3]])
			seed = int(parser[5])
			conTrans_param.params['seed'] = seed

			Initialization(conTrans_param.params)
			train_loader,test_loader, conTran_params = transform4ConvTran( train_X, train_y,test_X, test_y,
			                                                               conTran_param.params,n_classes)
			model = ConvTran(conTran_params, num_classes=n_classes).to(device)
			conTran_params['Norm'] = norm
			file_path= "/".join( ("saved_models",dataset_name,file) )
			model = load_convTran(model,file_path)


			X_test_set = torch.tensor(test_loader.dataset.feature).type(torch.float).to(device)
			predictions = torch.argmax( model(X_test_set) ,dim=-1).to("cpu").detach().numpy()
			print(dataset_name,file, "has accuracy", accuracy_score(test_loader.dataset.labels,predictions))
			b=2
	exit()

"""
cls = [MiniRocket(n_jobs=-1) ,StandardScaler(),
					LogisticRegressionCV(cv = 5, random_state=0, n_jobs = -1,max_iter=1000)]

train_X = train_X[:100] ; train_y = train_y[:100]
X_train_transformed = cls[0].fit_transform(train_X)
X_train_scaled = cls[1].fit_transform(X_train_transformed)
y_traiin = cls[2].fit(X_train_scaled,train_y)
print("trained")

start1 = timeit.default_timer()
X_test_transformed = cls[0].transform (test_X)
end1= timeit.default_timer()
start2 = timeit.default_timer()
X_test_sclaed = cls[1].transform (X_test_transformed)
end2 = timeit.default_timer()
start3 = timeit.default_timer()
score = cls[2].score(X_test_sclaed, test_y)
end3 = timeit.default_timer()
print(end1-start1, end2-start2, end3-start3)
print(score)
"""