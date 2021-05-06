# P Stage 2 - Tabular data classification <!-- omit in toc -->

- [Background](#background)
- [Input/Output](#inputoutput)
- [Approaches](#approaches)
- [Requirements](#requirements)
  - [Dependencies](#dependencies)
  - [Install Requirements](#install-requirements)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Model](#model)
  - [Hyperparameter](#hyperparameter)
- [Features](#features)
- [Others(EDA)](#otherseda)

## Background
최근 온라인 거래를 이용하는 고객들이 많이 늘어나고 있어 고객들의 log 데이터가 많이 늘어나고 있습니다. 온라인 거래 고객 log 데이터를 이용하여 고객들의 미래 소비를 예측 분석프로젝트를 진행하려 합니다.

고객들의 월별 총 구매 금액을 확인했을 때 연말에 소비가 많이 이루어지고 있는 것으로 확인이 되었습니다. 그리하여 12월을 대상으로 고객들에게 프로모션을 통해 성공적인 마케팅을 하기 위해 모델을 만들려고 합니다.

온라인 거래 log 데이터는 2009년 12월부터 2011년 11월까지의 온라인 상점의 거래 데이터가 주어집니다. 2011년 11월 까지 데이터를 이용하여 2011년 12월의 고객 구매액 300초과 여부를 예측해야 합니다.

## Input/Output
Input: Purchasing data of 5914 customers from 2009.12 to 2011.11

Output: Probability that each customer's total purchases in 2011.12 will exceed $300  

## Approaches
Due to lack of data, it is difficult to accurately predict the label.  
I separated this problem into two problems.
1) Will the customer purchase in 2011.12? (regardless of purchase amount)
2) Assuming the customer makes a purchase, will the customer make a purchase of above $300?
  
Assuming that customers who purchase more than $300 on average will continue to purchase more than $300, I predicted '2)' simply by the **average of the purchase.**  
  
For '1)', CatBoost was used with the features that reflected the characteristics of time series data(seasonality, periodicity, etc.) and several other aggregated features.

## Requirements  
### Dependencies
- Python >= 3.7.7
- numpy == 1.19.5
- pandas == 1.1.5
- matplotlib == 3.1.3
- seaborn == 0.11.1
- scikit-learn == 0.23.2
- scipy == 1.5.4
- xgboost == 1.3.3
- lightgbm == 3.1.1
- catboost == 0.24.4
- pytorch-tabnet == 3.1.1
   
### Install Requirements
- `pip install -r requirements.txt`
  
## Folder Structure
  ```
  code/
  ├── eda/ - my EDA works (not refined)
  │
  ├── utils.py - utility methods
  ├── features.py - feature engineering/preprocessing
  └── inference.py - inference
  ```
  
## Usage
Train & Inference  
`python inference.py`
    
## Model
CatBoost
  
### Hyperparameter    
```javascript
model_params = {
  'depth': 3, 
  'eval_metric': 'AUC', 
  'iterations': 500, 
  'l2_leaf_reg': 3.162277660168379e-20, 
  'leaf_estimation_iterations': 10, 
  'learning_rate': 0.1, 
  'loss_function': 'CrossEntropy', 
  'task_type': 'GPU'
}
```

## Features
I extracted 125 features and used them to train&inference model.  
  
125 features
- 24 | monthly data(quantity) and skew of it
- 24 | monthly data(total) and skew of it
- 24 | cumulative sum of monthly data(quantity) and skew of it
- 24 | cumulative sum of monthly data(total) and skew of it
- 14 | sum and skew of time series features (seasonality, cyclicity, ...) - quantity 
- 14 | sum and skew of time series features (seasonality, cyclicity, ...) - total
- 1 | date diff, (last purchase date) - (first purchase date)

## Others(EDA)
There are two iPython files that related to EDA.  

These iPython files are not refined.  

**For more explanation, please refer to my [Wrap-up report](./wrapup.pdf). :)**