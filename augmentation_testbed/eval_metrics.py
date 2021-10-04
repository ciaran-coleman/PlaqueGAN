from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import os

################################### AUC #######################################
# for each of the individual classes
def auc_roc(preds, label, type = 'macro'):
    auc_roc = []
    if type == 'macro':
        for i in range(preds.shape[1]):
            fpr, tpr, _ = roc_curve(label[:,i].ravel(), preds[:,i].ravel())
            auc_roc.append(auc(fpr, tpr))
    elif type == 'micro':
        fpr, tpr, _ = roc_curve(label.ravel(), preds.ravel())
        auc_roc.append(auc(fpr, tpr))
    return auc_roc

def auc_prc(preds, label, type = 'macro'):
    auc_prc = []
    if type == 'macro':
        for i in range(preds.shape[1]):
            precision, recall, _ = precision_recall_curve(label[:,i].ravel(), preds[:,i].ravel())
            auc_prc.append(auc(recall, precision))
    elif type == 'micro':
        precision, recall, _ = precision_recall_curve(label.ravel(), preds.ravel())
        auc_prc.append(auc(recall, precision))
    return auc_prc

def macro_avg_auc(preds, label, type = 'roc'):
    if type=='roc':
        auc_list = auc_roc(preds, label, type = 'macro')
    elif type=='prc':
        auc_list = auc_prc(preds, label, type = 'macro')
    return np.mean(auc_list), auc_list

def micro_avg_auc(preds, label, type = 'roc'):
    if type == 'roc':
        auc_list = auc_roc(preds, label, type = 'micro')
    elif type == 'prc':
        auc_list = auc_prc(preds, label, type = 'micro')

    return np.mean(auc_list)

def init_metrics_dict():
    headers = ['epoch',
           'tr_loss',
           'tr_acc_cored',
           'tr_acc_diffuse',
           'tr_acc_caa',
           'tr_auc_micro',
           'tr_auc_macro',
           'tr_auc_cored',
           'tr_auc_diffuse',
           'tr_auc_caa',
           'dev_loss',
           'dev_acc_cored',
           'dev_acc_diffuse',
           'dev_acc_caa',
           'dev_auc_micro',
           'dev_auc_macro',
           'dev_auc_cored',
           'dev_auc_diffuse',
           'dev_auc_caa'
          ]
    metrics_dict = {k: [] for k in headers}

    return metrics_dict

def update_metrics_dict(metrics_dict, phase, epoch, epoch_loss, epoch_acc, pr_auc_micro, pr_auc_macro, pr_auc_macro_list):
    if phase=='train':
        metrics_dict['epoch'].append(epoch)
        front = 'tr_'
    elif phase=='dev':
        front = 'dev_'

    metrics_dict[front + 'loss'].append(epoch_loss)
    metrics_dict[front + 'acc_cored'].append(epoch_acc[0])
    metrics_dict[front + 'acc_diffuse'].append(epoch_acc[1])
    metrics_dict[front + 'acc_caa'].append(epoch_acc[2])
    metrics_dict[front + 'auc_micro'].append(pr_auc_micro)
    metrics_dict[front + 'auc_macro'].append(pr_auc_macro)
    metrics_dict[front + 'auc_cored'].append(pr_auc_macro_list[0])
    metrics_dict[front + 'auc_diffuse'].append(pr_auc_macro_list[1])
    metrics_dict[front + 'auc_caa'].append(pr_auc_macro_list[2])

    return metrics_dict

def save_metrics_dict(results_dir, metrics_dict, repeat):
    import pandas as pd
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.to_csv(os.path.join(results_dir, f'metrics_{repeat}.csv'), index=False, float_format='%.3f')
    # metrics_df.to_csv('metrics_csv.csv', index=False)
