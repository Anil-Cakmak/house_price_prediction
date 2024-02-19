from helper import *
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, GridSearchCV

df = pd.read_csv("house_price_prediction/df.csv")
train_set = df[df.SalePrice.notnull()].drop("Id", axis=1)
y = np.log1p(train_set["SalePrice"])
X = train_set.drop(["SalePrice"], axis=1)


def grid_optimization(models, x_, y_, cv_=5, scoring="neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    for name, model, params in models:
        print(f"########## {name} ##########")

        gs_best = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

        best_model = model.set_params(**gs_best.best_params_)

        cv_results = cross_validate(best_model, x_, y_, cv=cv_, scoring=scoring)

        print(f"{scoring[4:]}: {round(-cv_results['test_score'].mean(), 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")


# ########## xg ##########
# root_mean_squared_error: 0.1188
# xg best params: {'subsample': 1.0, 'n_estimators': 3400, 'max_depth': 3, 'learning_rate': 0.02,
# 'colsample_bytree': 0.1}

xg_params = {'subsample': [1.0, 0.9],
             'n_estimators': [3400, 3600, 3800, 4000],
             'max_depth': [3, 4],
             'learning_rate': [0.01, 0.02, 0.03],
             'colsample_bytree': [0.1, 0.2]}

# ########## lgbm ##########
# root_mean_squared_error: 0.1206 (lgbm)
# lgbm best params: {'subsample': 1.0, 'num_leaves': 17, 'n_estimators': 3400, 'max_depth': 3, 'learning_rate': 0.02,
# 'colsample_bytree': 0.1}

lgbm_params = {'subsample': [1.0, 0.9],
               'num_leaves': [7, 10, 20],
               'n_estimators': [3400, 4000, 4500, 5000],
               'max_depth': [3, 4],
               'learning_rate': [0.01, 0.02, 0.03],
               'colsample_bytree': [0.1, 0.2]}

# ########## gbm ##########
# root_mean_squared_error: 0.1205 (gbm)
# gbm best params: {'subsample': 0.4, 'n_estimators': 5000, 'max_features': 0.7000000000000001, 'max_depth': 5,
# 'learning_rate': 0.01}

gbm_params = {'subsample': [0.3, 0.4, 0.5],
              'n_estimators': [4800, 5000, 5200],
              'max_features': [0.6, 0.7, 0.8],
              'max_depth': [4, 5, 6],
              'learning_rate': [0.01, 0.02]}

# ########## cat ##########
# root_mean_squared_error: 0.1175 (cat)
# cat best params: {'subsample': 0.2, 'min_data_in_leaf': 18, 'learning_rate': 0.02, 'iterations': 3210, 'depth': 5,
# 'colsample_bylevel': 0.9}

cat_params = {"iterations": [3000, 3210, 3500],
              "learning_rate": [0.01, 0.02, 0.03],
              "depth": [4, 5, 6],
              "subsample": [0.2, 0.3],
              "colsample_bylevel": [0.9],
              "min_data_in_leaf": [15, 18, 21]}


gs_models = [("xg", XGBRegressor(use_label_encoder=False, objective="reg:squarederror"), xg_params),
             ("lgbm", LGBMRegressor(verbose=-1, force_col_wise=True), lgbm_params),
             ("gbm", GradientBoostingRegressor(), gbm_params),
             ("cat", CatBoostRegressor(verbose=False), cat_params)]


grid_optimization(gs_models, X, y)


# Hyperparameter Optimization....

# ########## xg ##########
# Fitting 5 folds for each of 96 candidates, totalling 480 fits
# root_mean_squared_error: 0.1178 (xg)
# xg best params: {'colsample_bytree': 0.1, 'learning_rate': 0.02, 'max_depth': 3, 'n_estimators': 4000,
# 'subsample': 0.9}

# ########## lgbm ##########
# Fitting 5 folds for each of 288 candidates, totalling 1440 fits
# root_mean_squared_error: 0.1198 (lgbm)
# lgbm best params: {'colsample_bytree': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 4500,
# 'num_leaves': 10, 'subsample': 1.0}

# ########## gbm ##########
# Fitting 5 folds for each of 162 candidates, totalling 810 fits
# root_mean_squared_error: 0.119 (gbm)
# gbm best params: {'learning_rate': 0.01, 'max_depth': 4, 'max_features': 0.6, 'n_estimators': 4800, 'subsample': 0.5}

# ########## cat ##########
# Fitting 5 folds for each of 162 candidates, totalling 810 fits
# root_mean_squared_error: 0.1175 (cat)
# cat best params: {'colsample_bylevel': 0.9, 'depth': 5, 'iterations': 3500, 'learning_rate': 0.02,
# 'min_data_in_leaf': 15, 'subsample': 0.3}


