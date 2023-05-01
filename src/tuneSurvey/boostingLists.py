"""Boosting functions tuning - following sklearn models mostly

    XGBoost (version from DMLC) pip install xgboost
    lightGBM 
    CatBoost

"""

modelList_boosting = []

from xgboost import XGBRegressor, XGBClassifier



boostingc_grid = []
boostingr_grid = []

xgbc_grid = {"modelInit":XGBClassifier,
            "par": {"n_estimators" : [10,100, 500, 1000],
                    "max_depth" : [3,5,7],
                    "eta" : [.01,.03,.1],
                    "colsample_by_tree" = [.7,.8,.9]},
            "from": "tabular"}}

xgbr_grid = {"modelInit":XGBCRegressor,
            "par": {"n_estimators" : [10,100, 500, 1000],
                    "max_depth" : [3,5,7],
                    "eta" : [.01,.03,.1],
                    "colsample_by_tree" = [.7,.8,.9]},
            "from": "tabular"}}




from catboost import CatBoostClassifier, CatBoostRegressor
catboostc_grid = {"modelInit" : CatBoostClassifier,
                  "par" : { 'iterations': 500,
                            'learning_rate': 0.1,
                            'eval_metric': metrics.Accuracy(),
                            'random_seed': 42,
                            'logging_level': 'Silent',
                            'use_best_model': False
                        },
                  "from":"tabular"}
catboostr_grid = {"modelInit" : CatBoostRegressor,
                  "par" : { 'iterations': 500,
                            'learning_rate': 0.1,
                            'eval_metric': metrics.Accuracy(),
                            'random_seed': 42,
                            'logging_level': 'Silent',
                            'use_best_model': False
                        },
                  "from":"tabular"}

