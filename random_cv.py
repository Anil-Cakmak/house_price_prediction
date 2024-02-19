from helper import *
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, RandomizedSearchCV

df = pd.read_csv("house_price_prediction/df.csv")
train_set = df[df.SalePrice.notnull()].drop("Id", axis=1)
y = np.log1p(train_set["SalePrice"])
X = train_set.drop(["SalePrice"], axis=1)

regressors = [("RF", RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ('XGBoost', XGBRegressor(use_label_encoder=False, objective="reg:squarederror")),
              ('LightGBM', LGBMRegressor(verbose=0, force_col_wise=True)),
              ("CatBoost", CatBoostRegressor(verbose=False))]

base_models(regressors, X, y)


def hyperparameter_optimization(models, x_, y_, cv_=5, scoring="neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    for name, model, params in models:
        print(f"########## {name} ##########")

        rs_best = RandomizedSearchCV(estimator=model,
                                     param_distributions=params,
                                     n_iter=150,  # denenecek parametre sayısı
                                     cv=cv_,
                                     verbose=True,
                                     random_state=42,
                                     n_jobs=-1)

        rs_best.fit(X, y)

        best_model = model.set_params(**rs_best.best_params_)
        cv_results = cross_validate(best_model, x_, y_, cv=cv_, scoring=scoring)
        print(f"{scoring[4:]}: {round(-cv_results['test_score'].mean(), 4)} ({name}) ")
        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")


rf_params = {"max_depth": np.arange(3, 15, 1),
             "max_features": np.arange(0.1, 1.01, 0.1),
             "min_samples_split": np.random.randint(2, 50, 20),
             "n_estimators": [int(x) for x in np.linspace(start=200, stop=5000, num=10)],
             "min_samples_leaf": np.random.randint(2, 25, 10),
             "max_samples": np.arange(0.01, 1.01, 0.01)}

lgbm_params = {"learning_rate": np.arange(0.01, 0.21, 0.01),
               "num_leaves": np.random.randint(5, 50, 1),
               "n_estimators": [int(x) for x in np.linspace(start=200, stop=5000, num=10)],
               "colsample_bytree": np.arange(0.1, 1.05, 0.1),
               "subsample": np.arange(0.1, 1.05, 0.1),
               "max_depth": np.arange(3, 10, 1)}

xg_params = {"learning_rate": np.arange(0.01, 0.2, 0.01),
             "colsample_bytree": np.arange(0.1, 1.05, 0.1),
             "n_estimators": [int(x) for x in np.linspace(start=200, stop=5000, num=10)],
             "subsample": np.arange(0.1, 1.05, 0.1),
             "max_depth": np.arange(3, 10, 1)}

gbm_params = {"learning_rate": np.arange(0.01, 0.2, 0.01),
              "max_features": np.arange(0.1, 1.05, 0.1),
              "n_estimators": [int(x) for x in np.linspace(start=200, stop=5000, num=10)],
              "subsample": np.arange(0.1, 1.05, 0.1),
              "max_depth": np.arange(3, 10, 1)}

cat_params = {"iterations": np.arange(300, 5001, 10),
              "learning_rate": np.arange(0.01, 0.21, 0.01),
              "depth": np.arange(3, 10, 1),
              "subsample": np.arange(0.1, 1.01, 0.1),
              "colsample_bylevel": np.arange(0.1, 1.01, 0.1),
              "min_data_in_leaf": np.arange(2, 50, 1)}

rs_models = [("rf", RandomForestRegressor(), rf_params),
             ("xg", XGBRegressor(use_label_encoder=False, objective="reg:squarederror"), xg_params),
             ("lgbm", LGBMRegressor(verbose=-1, force_col_wise=True), lgbm_params),
             ("gbm", GradientBoostingRegressor(), gbm_params),
             ("cat", CatBoostRegressor(verbose=False), cat_params)]

hyperparameter_optimization(rs_models, X, y)

# Base Models....
# root_mean_squared_error: 0.1401 (RF)
# root_mean_squared_error: 0.1273 (GBM)
# root_mean_squared_error: 0.1439 (XGBoost)
# root_mean_squared_error: 0.1324 (LightGBM)
# root_mean_squared_error: 0.1229 (CatBoost)
# Hyperparameter Optimization....

# ########## rf ##########
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
# root_mean_squared_error: 0.1361 (rf)
# rf best params: {'n_estimators': 200, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_samples': 0.64,
# 'max_features': 0.7000000000000001, 'max_depth': 14}

# ########## xg ##########
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
# root_mean_squared_error: 0.1188 (xg)
# xg best params: {'subsample': 1.0, 'n_estimators': 3400, 'max_depth': 3, 'learning_rate': 0.02,
# 'colsample_bytree': 0.1}

# ########## lgbm ##########
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
# root_mean_squared_error: 0.1206 (lgbm)
# lgbm best params: {'subsample': 1.0, 'num_leaves': 17, 'n_estimators': 3400, 'max_depth': 3, 'learning_rate': 0.02,
# 'colsample_bytree': 0.1}

# ########## gbm ##########
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
# root_mean_squared_error: 0.1205 (gbm)
# gbm best params: {'subsample': 0.4, 'n_estimators': 5000, 'max_features': 0.7000000000000001, 'max_depth': 5,
# 'learning_rate': 0.01}

# ########## cat ##########
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
# root_mean_squared_error: 0.1175 (cat)
# cat best params: {'subsample': 0.2, 'min_data_in_leaf': 18, 'learning_rate': 0.02, 'iterations': 3210, 'depth': 5,
# 'colsample_bylevel': 0.9}
