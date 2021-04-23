import warnings
warnings.filterwarnings('ignore')
import os, gc

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer

from pytorch_tabnet.tab_model import TabNetClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from utils import seed_everything
from features import feature_engineering

# Seed / Required directory paths
SEED = 444
data_dir = '/opt/ml/code/input'
output_dir = '/opt/ml/code/output' 

# For hyperparameter searching(GridSearchCV, catboost)
PARAM_CANDIDATES = {
    'iterations': [300, 500, 700],
    'learning_rate': [0.1, 0.01, 0.001],
    'depth': [3, 4, 5, 6],
    'loss_function': ['CrossEntropy'],
    'l2_leaf_reg': np.logspace(-20, -19, 3),
    'leaf_estimation_iterations': [10],
    'eval_metric': ['AUC'],
    'task_type': ['GPU'],
}


def grid_search_cv(clf_type, X_train, X_test, y_train, model_params=PARAM_CANDIDATES, folds=5):
    """ Apply GridSearchCV

    Args:
        clf_type (str): Classifier type
        X_train (pd.DataFrame): train data
        X_test (pd.DataFrame): test data
        y_train (pd.Series): labels of train data
        model_params (dict): param_grid for GridSearchCV

    Returns:
        optimal parameters(in model_params) of the clf chosen
    """

    train_X = X_train.to_numpy()
    test_X = X_test.to_numpy()
    train_y = y_train.to_numpy()

    # -- Classifier setting
    clf = None
    if clf_type == 'cat':
        clf = CatBoostClassifier()
    elif clf_type == 'xgb':
        clf = XGBClassifier()
    elif clf_type == 'lgbm':
        clf = LGBMClassifier()
    elif clf_type == 'tabnet':
        clf = TabNetClassifier()
    else:
        print('Please write appropriate clf_type.')
        assert 0
    
    # -- Grid search based on AUC score
    scorer = make_scorer(roc_auc_score)
    clf_grid = GridSearchCV(estimator=clf, 
                            param_grid=model_params, 
                            scoring=scorer, 
                            cv=5,
                            verbose=20)
    clf_grid.fit(
        train_X, train_y,
        early_stopping_rounds=50, 
        verbose=10
    )       
    best_param = clf_grid.best_params_

    return best_param


def inference(train_X, test_X, y, model_params=None, folds=10):
    """ Inference with stratified kFold

    Args:
        train_X (pd.DataFrame): train data
        test_X (pd.DataFrame): test data
        y (pd.Series): labels of train data
        model_params (dict): model params
        folds (int): number of folds in stratified kFold

    Returns:
        Out of fold prediction, KFold prediction
    """

    y_oof = np.zeros(train_X.shape[0]) # oof pred
    test_preds = np.zeros(test_X.shape[0]) # kfold pred
    score = 0 # average of kfold(AUC score)
    
    # -- Stratified KFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=444)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_X, y)):
        X_train, X_valid = train_X.loc[train_idx, :], train_X.loc[valid_idx, :]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        X_train = X_train.to_numpy()
        X_valid = X_valid.to_numpy()
        y_train = y_train.to_numpy()
        y_valid = y_valid.to_numpy()

        X_test = test_X.to_numpy()

        print(f'fold: {fold+1}, X_train.shape: {X_train.shape}, X_valid.shape: {X_valid.shape}')

        # -- Catboost, train
        clf = CatBoostClassifier(**model_params) 
        clf.fit(
            X_train, y_train, 
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=50, 
            verbose=20
        )

        # -- Prediction/Validation/Scoring
        valid_preds = clf.predict_proba(X_valid)[:, 1]
        y_oof[valid_idx] = valid_preds

        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_valid, valid_preds)}")
        print('-'*80)

        score += roc_auc_score(y_valid, valid_preds) / folds
        test_preds += clf.predict_proba(X_test)[:, 1] / folds
        
        del X_train, X_valid, y_train, y_valid
        gc.collect()
        
    print(f"\nMean AUC = {score}") # validation score
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # oof validation
        
    return y_oof, test_preds


if __name__ == '__main__':
    # -- Fix seed
    seed_everything(SEED)
    
    # -- Read data file(train/submission)
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    submission = pd.read_csv(data_dir + '/sample_submission.csv')
    submission['probability'] = 0

    # -- Target date
    year_month = '2011-12'

    # -- Model parameter (CatBoost)
    status_params = {
        'depth': 3, 
        'eval_metric': 'AUC', 
        'iterations': 500, 
        'l2_leaf_reg': 3.162277660168379e-20, 
        'leaf_estimation_iterations': 10, 
        'learning_rate': 0.1, 
        'loss_function': 'CrossEntropy', 
        'task_type': 'GPU'
    }

    # -- Feature engineering
    print("-----Start feature engineering-----")
    train, test, y_status, y_total = feature_engineering(data, year_month)
    print("Feature preprocessing completed.")

    # -- GridSearchCV
    # print("-----Start parameter searching-----")
    # status_params = grid_search_cv('cat', train, test, y_status)
    # print("Parameter searching completed.")

    # -- Infererence
    print("-----Start inference-----")
    y_oof_status, status_preds = inference(train, test, y_status, model_params=status_params)
    submission['probability'] = np.array(status_preds) * np.array(y_total)
    print("Inference completed.")

    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(os.path.join(output_dir , 'final_submission.csv'), index=False)
    print("Success to make output file.")

    # -- Print optimal hyperparameter(GridSearchCV)
    # print(status_params)