import os
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import roc_auc_score

# Thresholds 
TOTAL_THRES = 270
MIDDLE_THRES = 210
STATUS_THRES = 20

# Required directory paths
data_dir = '/opt/ml/code/input/train.csv'
model_dir = '/opt/ml/model'


def generate_label_status(df_original, target_date, status_thres=STATUS_THRES):
    """ Labeling based on whether customer purchase at target date(month)
        This label dosen't care about whether he will buy more than $300(TOTAL_THRES)

    Args:
        df_original (pd.DataFrame): monthly frame
        target_date (str): date to apply prediction
        status_thres (int): threshold to consider as 'purchased'
    
    Returns:
        purchase status label (0/1) on target date
    """

    df = df_original.copy()

    label_values = df[target_date]
    label = (label_values > status_thres).astype(int)
    label = label.sort_index().to_frame(name='status_label').reset_index()

    return label


def generate_label_total(df_original, middle_thres=MIDDLE_THRES, total_thres=TOTAL_THRES, status_thres=STATUS_THRES):
    """ Calculate consumption expectations of each customer based on average consumption history

    https://stats.stackexchange.com/questions/135061/best-method-for-short-time-series
    Averaging is one of the strongest method. ㅇ _ㅇ

    Args:
        df_original (pd.DataFrame): monthly frame
        middle_thres (int): if 'middle_thres < expectation < total_thres' is true, apply 0.5
        total_thres (int): if 'expectation > total_thres' is true, apply 1.0
        status_thres (int): Averaging only above status_thres
    
    Returns:
        purchase total label (0/0.5/1) on target date
    """

    df = df_original.copy()
    label = df[df > status_thres].mean(axis=1)
    label[label < middle_thres] = 0
    label[(label > middle_thres) & (label < total_thres)] = 0.5
    label[label >= total_thres] = 1
    label = label.sort_index().to_frame(name='total_label').reset_index()

    return label


def generate_monthly_frame(df_original, categories):
    """ Generate monthly purchase data calculated for each customer

    Args:
        df_original (pd.DataFrame): DataFrame dropped with unnecessary columns
        categories (list[str]): numerical categories to be aggregated

    Returns:
        monthly frame
        
        ex)
        ym            2009-12  2010-01  ...   2011-10    2011-11 
        customer_id                                                              
        12346        187.2750  -22.275  ...     0.000     0.0000 
        12349        -39.8475    0.000  ...  1763.058   330.0000 
        ...               ...      ...  ...       ...        ...  
        18286        763.8675    0.000  ...     0.000     0.0000  
        18287         -8.4150    0.000  ...     0.000  1768.1565  

    """

    df = df_original.copy()

    # -- groupby / pivot_table
    df = df.groupby(['customer_id', 'ym'])[categories].sum().reset_index()
    monthly_frame = pd.pivot_table(data=df,
                            values=categories,
                            index='customer_id',
                            columns='ym',
                            fill_value=0)

    return monthly_frame


def time_series_processing(df_original, categories, train=True):
    """ Generate features that reflect the characteristics of time series data.
        In this function, SUM and SKEW will be applied to each specified time period.

        - Seasonality(Continuity): The person who bought recently will buy again
            1. Last 10 months | 2. Last 7 months | 3. Last 4 months 

        - Cyclicity(Periodicity): People buy things regularly
            1. two-months interval | 2. three-months interval | 3. annually(1 year interval)

        - Weak-Cyclicity: The person who bought last year will buy again at near month
            - Around last year's target month

    Args:
        df_original (pd.DataFrame): monthly frame
        categories (list[str]): aggregated categories
        train (bool): True if df_original is train data

    Returns:
        Specialized time series data
    """
    
    df = df_original.copy()
    
    # -- Declare period list that reflect each attribute
    if train:
        target_date = '2011-11'

        # -- seasonality
        seasons1 = ['2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10']
        seasons2 = ['2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10']
        seasons3 = ['2011-07', '2011-08', '2011-09', '2011-10']

        # -- cyclicity
        cycle1 = ['2011-01', '2011-03', '2011-05', '2011-07', '2011-09']
        cycle2 = ['2010-08', '2010-11', '2011-02', '2011-05', '2011-08']
        cycle3 = ['2010-11']

        # -- weak-periodicity
        weak_cycle = ['2009-12', '2010-10', '2010-11', '2010-12']

    else: 
        target_date = '2011-12'
        
        # -- seasonality
        seasons1 = ['2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11']
        seasons2 = ['2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11']
        seasons3 = ['2011-09', '2011-10', '2011-11']

        # -- cyclicity
        cycle1 = ['2011-02', '2011-04', '2011-06', '2011-08', '2011-10']
        cycle2 = ['2010-09', '2010-12', '2011-03', '2011-06', '2011-09']
        cycle3 = ['2010-12']

        # -- weak-periodicity
        weak_cycle = ['2010-01', '2010-11', '2010-12', '2011-01']

    df_ret = pd.DataFrame()
    time_list_bundle = [seasons1, seasons2, seasons3, cycle1, cycle2, cycle3, weak_cycle]
    attribute_names = ['seasonality1', 'seasonality2', 'seasonality3', 'cyclicity1', 'cyclicity2', 'cyclicity3', 'weak_cyclicity']

    # -- For each category, apply agg independently
    for category in categories:
        categoric_df = pd.DataFrame()

        for time_list, attribute_name in zip(time_list_bundle, attribute_names):
            now_df = df[category].loc[:, time_list]

            # -- Aggregating (sum, skew)
            attribute_sum = now_df.sum(axis=1)
            attribute_skew = now_df.skew(axis=1)

            categoric_df[f"{category}_{attribute_name}_sum"] = attribute_sum
            categoric_df[f"{category}_{attribute_name}_skew"] = attribute_skew

        df_ret = pd.concat([df_ret, categoric_df], axis=1)

    return df_ret


def calculate_date_diff(df_original, start_date, target_date):
    """ Generate the gap between first purchase date and last purchase date as feature.
        Refund data is NOT considered as purchase.

    Args:
        df_original (pd.DataFrame): monthly frame
        start_date (str): start date(2009-12 or 2010-01)
        target_date (str): target date

    Returns:
        date diff
        default: (last purchase date) - (first purchase date)
        If there is only one purchase date, (target date) - (first purchase date)
    """

    df = df_original.copy()

    # -- Convert each data to pd.Timestamp
    start_date = pd.to_datetime(start_date)
    target_date = pd.to_datetime(target_date)
    
    # -- Calculate date diff for each row(customer)
    dt_diff = []
    for customer_id, datas in df.iterrows():
        start = start_date
        end = start_date
        
        for date, value in datas.items():
            if value > 0 and start == start_date:
                start = pd.to_datetime(date)
            if value > 0:
                end = pd.to_datetime(date)
        
        # -- When only one purchase data exist
        if start == end:
            end = pd.to_datetime(target_date)    
        dt_diff.append(int((end - start).total_seconds()))

    dt_diff = np.array(dt_diff).reshape(-1, 1)

    # -- Normalize
    scaler = StandardScaler()
    dt_diff = scaler.fit_transform(dt_diff)

    return dt_diff


def apply_agg_to_feature(df_original, categories, start_date=None, target_date=None):
    """ Apply aggregate function to monthly data.
        1. Generate monthly cumsum columns
        2. Apply aggregation(skew) to original monthly data and of cumsum data.  
        3. (optional) Call 'calculate_date_diff' method

    Args:
        df_original (pd.DataFrame): monthly frame
        categories: aggregated categories

        (optional, when want to add 'date_diff' feature)
        start_date (str): start date(2009-12 or 2010-01)
        target_date (str): target date

    Returns:
        aggregated features
    """

    df = df_original.copy()

    # -- Apply cumsum/skew
    df_ret = df.copy()
    for category in categories:
        df_skew = df[category].skew(axis=1).rename(f'{category}_skew')
        df_cumsum = df[category].cumsum(axis=1)
        df_cumsum.columns = [f"cum_{category}_{x}" for x in df_cumsum.columns]
        cumsum_skew = df_cumsum.skew(axis=1).rename(f'{category}_cumsum_skew')

        df_ret = pd.concat([df_ret, df_skew, df_cumsum, cumsum_skew], axis=1)
        df_ret = df_ret.rename(columns={'skew': f'{category}_skew',
                                        'cumsum': f'{category}_cumsum',
                                        'cumsum_skew': f'{category}_cumsum_skew'})

    # -- If start_date and target_date exists, generate date_diff feature additionally
    if start_date and target_date:
        date_diff = calculate_date_diff(df['total'], start_date=start_date, target_date=target_date)
        df_ret['date_diff'] = date_diff

    return df_ret


def convert_multi_index_to_single(df_original):
    """ Convert multi-index columns to single index.

    Args:
        df_original (pd.DataFrame): monthly frame

    Returns:
        monthly frame with single-index columns.
        ex) ('total', '2011-10') => 'total_2011-10'
    """

    df = df_original.copy()

    new_columns = []
    for column in df.columns:
        new_column = column

        # -- If multi-index
        if isinstance(column, tuple):
            new_column = f"{column[0]}_{column[1]}"
        new_columns.append(new_column)
    df.columns = new_columns

    return df


def feature_engineering(df_original, target_date):
    """ Date feature engineering
        1. Drop unnecessary columns(features)
        2. Make monthly frame
        3. Feature extracting
        4. Imputing
    
    Args:
        df_original (pd.DataFrame): raw frame
        target_date (str): target date (i.e., '2011-12')

    Returns:
        preprocessed data(Split as train and test)
        train label(status, total)
    """

    df = df_original.copy()

    # -- Basic preprocessing
    df.order_date = pd.to_datetime(df.order_date)
    df['ym'] = pd.to_datetime(df['order_date']).dt.strftime('%Y-%m')
    df.drop(['order_id', 'product_id', 'description', 'price', 'country'], axis=1, inplace=True)

    # -- Calculate period of train and test and apply it.
    d = datetime.datetime.strptime(target_date, "%Y-%m")
    prev_date = (d - relativedelta(months=1)).strftime("%Y-%m")
    init_date = df.order_date.min().strftime("%Y-%m")

    train = df[df['ym'] < prev_date]
    test = df[(df['ym'] < target_date) & (df['ym'] > init_date)]

    # -- Generate monthly frame and train label
    categories = ['total', 'quantity']
    monthly_frame = generate_monthly_frame(df, categories)['total']
    train_data = generate_monthly_frame(train, categories)
    test_data = generate_monthly_frame(test, categories)

    status_label = generate_label_status(monthly_frame, prev_date)
    total_label = generate_label_total(monthly_frame)['total_label']

    # -- Denoising
    train_data[train_data < STATUS_THRES] = 0
    test_data[test_data < STATUS_THRES] = 0

    # -- Feature extracting
    train_ts = time_series_processing(train_data, categories, train=True)
    test_ts = time_series_processing(test_data, categories, train=False)
    
    train_agg = apply_agg_to_feature(train_data, categories, start_date='2009-12', target_date='2011-11')
    test_agg = apply_agg_to_feature(test_data, categories, start_date='2010-01', target_date='2011-12')

    X_train = pd.merge(train_ts, train_agg, on=['customer_id'], how='left')
    X_train = pd.merge(X_train, status_label, on=['customer_id'], how='left')
    X_test = pd.merge(test_ts, test_agg, on=['customer_id'], how='left')
    
    # -- For convenience(ignoreable)
    X_test['customer_id'] = X_test.index
    X_test = X_test[[X_test.columns.values[-1]] + list(X_test.columns.values[:-1])]
    X_test.reset_index(drop=True).sort_values(by='customer_id')

    # -- Imputing(for test data)
    checker = X_train['customer_id'].isin(X_test.index)
    imputed = X_train[~checker].drop(columns=['status_label'])
    test_cols = {x: y for x, y in zip(X_train.columns, X_test.columns)}
    X_test = X_test.append(imputed.rename(columns=test_cols)).sort_values(by='customer_id')

    # -- Detect multi-index and convert them to single
    X_train = convert_multi_index_to_single(X_train)
    X_test = convert_multi_index_to_single(X_test)

    return X_train.drop(columns=['customer_id', 'status_label']), \
           X_test.drop(columns=['customer_id']), \
           X_train['status_label'], \
           total_label


if __name__ == '__main__':
    print('data_dir', data_dir)
