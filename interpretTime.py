import os
import numpy as np
import torch.cuda
import timeit
import pandas as pd
import argparse
from InterpretTime.src.postprocessing_pytorch.manipulation_results import ScoreComputation
from load_data import  load_data
from InterpretTime.src.shared_utils.utils_visualization import plot_DeltaS_results, plot_additional_results

all_qfeatures = [0.05, 0.15, 0.25, 0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.0]

def main(args):

    for dataset_name in args.datasets:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name)

        # create a dictionary storing the dataset and metadata as name , local/global mean/std for
        test_set = {'name':dataset_name,  "X": test_X ,  "y": test_y, "train_y" : train_y,
                    "seq_len" : seq_len, "n_channels":n_channels, "n_classes":n_classes,
                    "local_mean" : pd.DataFrame(np.mean(train_X,axis=0)),
                    "local_std" : pd.DataFrame(np.std(train_X,axis=0)),
                    "global_mean" : np.mean(train_X), "global_std": np.std(train_X)
                    }

        # load explanations
        explanations_dir = os.path.join( "explanations",dataset_name )
        explanations = os.listdir( explanations_dir )
        explanations.sort()

        # for each explanation and for each mask
        for explanation in explanations:
            for nt in  ["normal_distribution","zeros","global_mean","local_mean","global_gaussian","local_gaussian"]:
                print("assessing ", explanation)

                # load model and explanations to access
                model_name = explanation.split("_")[0], explanation[:-4].split("_")[-1]
                diction = np.load( os.path.join( explanations_dir, explanation ), allow_pickle=True).item()
                model_path = os.path.join("saved_models", dataset_name, model_name[0])
                dResNet = True if explanation.lower().count("dresnet") > 0 else False

                # create ScoreComputation object and run interpretTime using it
                manipulation_results = ScoreComputation(model_path=(model_path,model_name[1]), noise_type=nt,
                            names=None,dataset=test_set ,device=device,encoder=diction['classes_idx'])

                start = timeit.default_timer()
                explanations_dict = diction['explanations']
                for method in explanations_dict:
                    attrib = explanations_dict[method]
                    _ = manipulation_results.compute_scores_wrapper( all_qfeatures, method, attrib)
                    manipulation_results.create_summary(method)

                manipulation_results.summarise_results()

                # save results and plot additional info
                save_results_path = manipulation_results.save_results
                print(save_results_path, explanations_dict.keys())
                plot_DeltaS_results(save_results_path)
                plot_additional_results(save_results_path)
                end =  timeit.default_timer()
                print("it took", (end-start))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str,nargs='+', help="which dataset to be used")
    args = parser.parse_args()
    main(args)
