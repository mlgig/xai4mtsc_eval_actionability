import os
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix,average_precision_score, roc_auc_score,
                             multilabel_confusion_matrix,roc_auc_score,average_precision_score)
from utils import plot

threshold = 0.5

def select_first_row(explanations):
    return explanations[:,0,:,:]


def minMax_normalization(x):
    x = (x - x.min() ) / (x.max() - x.min())
    return x.flatten()

def compute_metrics(gts,exps, model, exp_name, toAnalyze,X):
    # TODO delete limit!
    limit = 100

    i = 0
    precision=0; recall = 0 ; f1 = 0
    roc = 0 ; ap= 0
    j=0
    while i < limit:
        # preprocessing gts and preds

        current_idx = toAnalyze[i]
        current_gt = gts[current_idx].flatten()

        current_exp = minMax_normalization(exps[current_idx])

        importants = (current_exp>=threshold).astype(int)


        #### compute precisioon, recall and f1 score
        curr_precision, curr_recall, curr_f1, support = precision_recall_fscore_support(current_gt,importants
                ,average="binary", labels=[0,1])
        if (curr_precision>0.5 and False):
            print( i, curr_precision,curr_recall,curr_f1)
            plot( instance=X[current_idx],dcam=exps[current_idx],nb_dim=8, idx=j)
            plot( instance=X[current_idx],dcam=gts[current_idx],nb_dim=8, idx=j+1000)
            plot( instance=X[current_idx],dcam=importants.reshape(8,-1),nb_dim=8, idx=j+2000)
            j+=1
        precision += curr_precision; recall+=curr_recall ; f1+=curr_f1

        ## compute curves
        roc += roc_auc_score(current_gt,current_exp,average="samples")
        ap += average_precision_score(current_gt,current_exp,average="samples")

        i+=1

    roc /= limit ; ap/=limit
    precision /= limit ; recall /= limit ; f1 /= limit
    print(model+ " \t"+exp_name+"\t {:.3f} \t {:.3f} \t{:.3f} \t{:.3f} \t{:.3f}".format(precision,
                                                                                        recall,f1,ap,roc) )
    return 2

    """
        # micro precision, recall and f1
        confusion_matrices = multilabel_confusion_matrix(gts,importants)
        tot_tn, tot_fp, tot_fn, tot_tp = np.sum(confusion_matrices, axis=0).ravel()
        micro_precision = tot_tp / (tot_tp + tot_fp)
        micro_recall = tot_tp / (tot_tp + tot_fn)
        micro_f1 = 2*(micro_precision*micro_recall) / (micro_precision+micro_recall)

        # samples  precision, recall and f1
        confusion_matrices = multilabel_confusion_matrix(gts,importants,samplewise=True)
        sample_precision = 0 ; sample_recall = 0 ; sample_f1 = 0
        for i in range(limit):
            tn, fp, fn, tp = confusion_matrices[i].ravel()
            sample_precision += tp / (tp+fp)
            sample_recall += tp / (tp+fn)
        sample_precision /= limit ; sample_recall /= limit
        sample_f1 = (2* sample_precision * sample_recall) / (sample_precision+sample_recall)

        i+=1

    roc_auc_micro = roc_auc_score(gts,exps,average="micro")
    roc_auc_samples = roc_auc_score(gts,exps,average="samples")


    ap_auc_micro = average_precision_score(gts,exps,average="micro")
    ap_auc_samples = average_precision_score(gts,exps,average="samples")


    # TODO return results when the metrics to be used are selected
   #print("micro precision, recall and f1 for",model, " are: ",micro_precision,micro_recall,micro_f1,"\n",
   #       "sample precision, recall and f1 for",model, " are: ",sample_precision,sample_recall,sample_f1,"\n",
    #   "roc auc, samples/micro average are",roc_auc_samples,roc_auc_micro,
    #      "ap auc, samples/micro average are", ap_auc_samples,ap_auc_micro,"\n")
    #print(model, exp_name, sample_precision, sample_recall, sample_f1,roc_auc_samples, ap_auc_samples )
    print(model+ " \t"+exp_name+"\t {:.3f} \t {:.3f} \t{:.3f} \t{:.3f} \t{:.3f}".format(0,roc,
                ap,ap_auc_samples,roc_auc_samples))
    #tmp = multilabel_confusion_matrix(gts,importants,labels=[1])
    #print(np.sum(confusion_matrices,axis=0),tmp, "\n\n")
    return 2
    """

def main():
    exp_base_path = "./explanations/"
    datasets = os.listdir(exp_base_path)
    for dataset_name in ["synth_2lines"]: #["synth_1line", "synth_2lines"]:

        # load dataset
        data_base_path = "./datasets/synthetics/"
        dataset = np.load( os.path.join( data_base_path,dataset_name+".npy" ) ,allow_pickle=True ).item()
        exp_ground_truths = dataset["test"]["ground_truth"]
        X =  dataset["test"]["X"]

        # load all the explanations for a single model
        models = os.listdir( os.path.join( exp_base_path,dataset_name ))
        #TODO delete!
        models = ["ResNet150.npy","dResNet150.npy"]#  ["explanation200ResNet_labels.npy", "explanation200dResNet_labels.npy"]

        for model in models:
            # load all the explanations for the current model
            data = np.load( os.path.join( exp_base_path, dataset_name, model ) , allow_pickle=True ).item()
            labels = data["ground_truth_labels"]
            preds = data["predicted_labels"]
            toAnalyze = np.where(labels==preds)[0]

            for current_exps_name in data["explanations"].keys():
                if current_exps_name=="Lime":
                    continue
                if model.count("dRes") > 0:
                    current_exps = select_first_row(data["explanations"][current_exps_name])
                else:
                    current_exps = data["explanations"][current_exps_name]
                compute_metrics(exp_ground_truths,current_exps,model ,current_exps_name, toAnalyze,X)
            print("\n\n\n")

if __name__ == "__main__" :
    main()



"""
    for i in range(limit):
        current_gt = gts[i,:,:]
        current_exp = minMax_normalization( exps[i,:,: ])
        importants = (current_exp>=treshold).astype(np.int32)


        #current_gt = current_gt.flatten() ; importants = importants.flatten()
        precision, recall, f1,support = precision_recall_fscore_support(
            current_gt,importants,average="binary", pos_label=1)
        a = 2
"""