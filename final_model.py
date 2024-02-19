from helper import *
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from itertools import combinations
import joblib
from pathlib import Path

df = pd.read_csv("house_price_prediction/df.csv")
train_set = df[df.SalePrice.notnull()].drop("Id", axis=1)
test_set = df[df.SalePrice.isnull()].drop("SalePrice", axis=1)
pred_set = test_set.drop("Id", axis=1)

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


lgbm_params = {'colsample_bytree': [0.1],
               'learning_rate': [0.01],
               'max_depth': [3, 4],
               'n_estimators': [4500],
               'num_leaves': [7, 8, 10],
               'subsample': [1.0]}

grid_optimization([("lgbm", LGBMRegressor(verbose=-1, force_col_wise=True), lgbm_params)], X, y)

# Stacking & Ensemble Learning
lgbm_best_params = {'colsample_bytree': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 4500,
                    'num_leaves': 8, 'subsample': 1.0}

xg_best_params = {'colsample_bytree': 0.1, 'learning_rate': 0.02, 'max_depth': 3, 'n_estimators': 4000,
                  'subsample': 0.9}

gbm_best_params = {'learning_rate': 0.01, 'max_depth': 4, 'max_features': 0.6, 'n_estimators': 4800, 'subsample': 0.5}

cat_best_params = {'colsample_bylevel': 0.9, 'depth': 5, 'iterations': 3500, 'learning_rate': 0.02,
                   'min_data_in_leaf': 15, 'subsample': 0.3}

voting_models = [("xg", XGBRegressor(**xg_best_params, use_label_encoder=False, objective="reg:squarederror")),
                 ("lgbm", LGBMRegressor(**lgbm_best_params, verbose=-1, force_col_wise=True)),
                 ("gbm", GradientBoostingRegressor(**gbm_best_params)),
                 ("cat", CatBoostRegressor(**cat_best_params, verbose=False))]

for model_set in list(combinations(voting_models, 3)):
    print(f"{model_set[0][0], model_set[1][0], model_set[2][0]}")
    rmse = cross_validate(VotingRegressor(estimators=list(model_set)), X, y, cv=5,
                          scoring="neg_root_mean_squared_error")
    print(f"rmse: {-rmse['test_score'].mean()}")

# ('xg', 'lgbm', 'gbm')
# rmse: 0.11643139318910253
# ('xg', 'lgbm', 'cat')
# rmse: 0.11642111460702328
# ('xg', 'gbm', 'cat')
# rmse: 0.11602910193256044
# ('lgbm', 'gbm', 'cat')
# rmse: 0.11644479056310005


final_voting_models = [("xg", XGBRegressor(**xg_best_params, use_label_encoder=False, objective="reg:squarederror")),
                       ("gbm", GradientBoostingRegressor(**gbm_best_params)),
                       ("cat", CatBoostRegressor(**cat_best_params, verbose=False))]

final_model = voting_regressor(final_voting_models, X, y)
# 0.11624151960368154

y_pred = final_model.predict(pred_set)
predictions = np.expm1(y_pred)
dictionary = {"Id": test_set.Id.values, "SalePrice": predictions}
dfSubmission = pd.DataFrame(dictionary)
pred_path = Path("house_price_prediction/house_price_predictions.csv")
pred_path.parent.mkdir(parents=True, exist_ok=True)
dfSubmission.to_csv(pred_path, index=False)

feat_imp = voting_importance(final_model, X, [1 / 3] * 3).sort_values(by="Value", ascending=False)
feat_imp.head(10)

# 2          OverallQual  0.964864
# 46      new_total_qual  0.914798
# 47      new_totalflrsF  0.393715
# 20           GrLivArea  0.335449
# 157  new_baths_above_1  0.280829
# 33          GarageCars  0.245657
# 5            ExterQual  0.234254
# 27         KitchenQual  0.211044
# 111   Foundation_PConc  0.207645
# 15         TotalBsmtSF  0.202891

importance_path = Path("house_price_prediction/house_price_importance")
importance_path.parent.mkdir(parents=True, exist_ok=True)
feat_imp.to_csv(importance_path, index=False)

plot_voting_importance(final_model, X, [1 / 3] * 3, 20)

model_path = Path("house_price_prediction/final_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, model_path)
