import os
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support,
        confusion_matrix,average_precision_score, roc_auc_score,
        multilabel_confusion_matrix,roc_auc_score,average_precision_score)

threshold = 0.5

def select_first_row(explanations):
    return explanations[:,0,:,:]


def minMax_normalization(X):
    n_samples = X.shape[0]
    normalized = []
    for i in range(n_samples):
        x = X[i,:,:]
        x = x.flatten()
        x = (x - x.min() ) / (x.max() - x.min())
        normalized.append(x)
    normalized = np.stack(normalized)
    return normalized

def compute_metrics(gts,exps, model, exp_name, gt_labels, preds):
    # TODO delete limit!
    limit = 100

    # preprocessing gts and preds
    exps = minMax_normalization(exps[:limit])
    importants = (exps>=threshold).astype(int)
    gts = gts [:limit].reshape(limit,-1)

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
    print(model+ " \t"+exp_name+"\t {:.3f} \t {:.3f} \t{:.3f} \t{:.3f} \t{:.3f}".format(sample_precision,sample_recall,
                sample_f1,ap_auc_samples,roc_auc_samples))
    #tmp = multilabel_confusion_matrix(gts,importants,labels=[1])
    #print(np.sum(confusion_matrices,axis=0),tmp, "\n\n")
    return 2

def main():
    exp_base_path = "./explanations/"
    datasets = os.listdir(exp_base_path)
    for dataset_name in ["synth_2lines"]: #["synth_1line", "synth_2lines"]:

        # load dataset
        data_base_path = "./datasets/synthetics/"
        dataset = np.load( os.path.join( data_base_path,dataset_name+".npy" ) ,allow_pickle=True ).item()
        exp_ground_truths = dataset["test"]["ground_truth"]

        # load all the explanations for a single model
        models = os.listdir( os.path.join( exp_base_path,dataset_name ))
        # TODO substitute with a for
        for model in ["explanation200ResNet_labels.npy", "explanation200dResNet_labels.npy"]: #models[2:]:
            # load all the explanations for the current model
            data = np.load( os.path.join( exp_base_path, dataset_name, model ) , allow_pickle=True ).item()
            labels = data["ground_truth_labels"]
            preds = data["predicted_labels"]

            for current_exps_name in data["explanations"].keys():
                if model.count("dRes") > 0:
                    current_exps = select_first_row(data["explanations"][current_exps_name])
                else:
                    current_exps = data["explanations"][current_exps_name]
                compute_metrics(exp_ground_truths,current_exps,model ,current_exps_name, labels, preds)
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