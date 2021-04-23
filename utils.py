import os
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def seed_everything(seed=444):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def print_score(label, pred, prob_thres=0.5):
    print(f"Precision: {precision_score(label, pred>prob_thres):.5f}")
    print(f"Recall: {recall_score(label, pred>prob_thres):.5f}")
    print(f"F1 Score: {f1_score(label, pred>prob_thres):.5f}")
    print(f"ROC AUC Score: {roc_auc_score(label, pred):.5f}") 