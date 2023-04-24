from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from imblearn.metrics import specificity_score, sensitivity_score
import numpy as np
def calculate_metrics(gt,pre,auc_pre,split):
    AUROCs = []
    Accuracy = []
    sens = []
    spec = []
    for i in range(14):
        Accuracy.append(accuracy_score(gt,pre))
        if 1 in gt[:,i]:
            AUROCs.append(roc_auc_score(gt[:,i],auc_pre[:,i]))
            sens.append(sensitivity_score(y_true=gt[:,i],y_pred=pre[:,i]))
            spec.append(specificity_score(y_true=gt[:, i], y_pred=pre[:, i]))

    AUROCs = np.array(AUROCs)
    AUROCs_avg = AUROCs.mean()

    result={
        split + '_samples/sensitivity_score': sum(sens)/len(sens),
        split + '_samples/specificity_score': sum(spec) / len(spec),
        split + '_samples/f1': f1_score(y_true=gt, y_pred=pre, average='weighted'),
        split + '_accuracy_score': accuracy_score(y_true=gt, y_pred=pre),
        split + '_AUROC_avg': AUROCs_avg,
        split + '_AUROC_each': AUROCs.tolist()
    }

    return result