"""Boosting functions tuning - following sklearn models mostly
"""

modelList_boosting = []

from xgb import XGBRegressor, XGBClassifier



boosting_classification = {{"modelInit":XGBClassifier,
                            }}
