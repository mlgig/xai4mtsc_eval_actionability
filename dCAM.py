from models.dCAM.src.explanation.DCAM import DCAM
import timeit
from tqdm import trange
import numpy as np

def run_dCAM(model,device,attributions, dataset_name,X_test, y_test):
	last_conv_layer = model._modules['0'].layers[2]
	fc_layer_name = model._modules['0'].final
	dCAM = DCAM(model,device,last_conv_layer,fc_layer_name)

	nb_permutation = 6 if dataset_name=="CMJ" else 200
	generate_all = True if dataset_name=="CMJ" else False

	outupts = []
	nb_err = 0

	start = timeit.default_timer()
	for i in trange(X_test.shape[0]):
		instance = X_test[i][0].cpu().numpy()
		label = y_test[i]
		try:
			dcam,permutation_success = dCAM.run(
				instance=instance, nb_permutation=nb_permutation, label_instance=label,generate_all=generate_all)
			outupts.append(dcam)
		except IndexError:
			outupts.append( np.zeros(shape=instance.shape))
			nb_err+=1

	end = timeit.default_timer()
	attributions['dCAM'] = np.stack(outupts)
	print("dCAM took", (end-start), "seconds and couldn't find an explanation for ",nb_err,"instance")