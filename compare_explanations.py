import os
import numpy as np
from sklearn.metrics import ( roc_auc_score, average_precision_score)
from utils import minMax_normalization
from load_data import load_data


def add_random_explanation(data):
	for model in data:
		first_method = list(data[model]["explanations"].keys())[0]
		shape = data[model]["explanations"][first_method].shape
		random = np.random.normal(0, 1, size=shape)
		data[model]["explanations"]["random"] = random

	return data


def compute_metrics(gts, exps, model, exp_name):
	exps_normalized = []
	gts_toAnlyzed = []
	for i in range(gts.shape[0]):
		exps_normalized.append(minMax_normalization(exps[i]).flatten())
		gts_toAnlyzed.append(gts[i].flatten())
	exps_normalized = np.array(exps_normalized)
	gts_toAnlyzed = np.array(gts_toAnlyzed)

	roc = roc_auc_score(gts_toAnlyzed, exps_normalized, average="samples")
	ap = average_precision_score(gts_toAnlyzed, exps_normalized, average="samples")
	print(model, "\t", exp_name, "\t", ap, "\t", roc)


def main():
	exp_base_path = "./explanations/"
	datasets = os.listdir(exp_base_path)
	for dataset_name in ["synth_2lines"]:  # ["synth_1line", "synth_2lines"]:

		# load dataset
		# TODO what to do with the following line
		data_base_path = "./datasets/synthetics/"
		test_X, test_y, exp_ground_truths, seq_len, n_channels, n_classes = load_data(dataset_name, explanation_gt=True)

		# TODO delete
		exp_ground_truths = np.concatenate( ( exp_ground_truths[:50], exp_ground_truths[-50:] ))

		# load all the explanations for a single model
		models = os.listdir(os.path.join(exp_base_path, dataset_name))
		data = {}

		for model in models:
			# load all the explanations for the current model
			data[model] = np.load(os.path.join(exp_base_path, dataset_name, model), allow_pickle=True).item()

		data = add_random_explanation(data)

		for model in data:
			for current_exps_name in data[model]["explanations"].keys():
				if current_exps_name.lower() != "lime":
					current_exps = data[model]["explanations"][current_exps_name]
					compute_metrics(exp_ground_truths, current_exps, model, current_exps_name)
			print("\n\n\n")


if __name__ == "__main__":
	main()
