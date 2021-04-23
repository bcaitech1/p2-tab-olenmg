# P Stage 2 - Tabular data classification <!-- omit in toc -->

- [Requirements](#requirements)
  - [Dependencies](#dependencies)
  - [Install Requirements](#install-requirements)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Model](#model)
  - [Hyperparameter](#hyperparameter)
- [Features](#features)

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
Inference  
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
125 features
- 24 | monthly data(quantity) and skew of it
- 24 | monthly data(total) and skew of it
- 24 | cumulative sum of monthly data(quantity) and skew of it
- 24 | cumulative sum of monthly data(total) and skew of it
- 14 | sum and skew of time series features (seasonality, cyclicity, ...) - quantity 
- 14 | sum and skew of time series features (seasonality, cyclicity, ...) - total
- 1 | date diff, (last purchase date) - (first purchase date)