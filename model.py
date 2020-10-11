import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics


def preprocess(data):
    data['is_adr'] = data['is_adr'].astype('int')
    data['is_adr'] = data['is_adr'].astype('category')

    data['distance'] = np.sqrt(np.power(data['destination_longitude'] - data['origin_longitude'], 2) +
                               np.power(data['destination_latitude'] - data['origin_latitude'], 2))

    data['shipping_date'] = pd.to_datetime(data['shipping_date'])
    data['shipping_date_month'] = pd.DatetimeIndex(data['shipping_date']).month
    data['shipping_date_day'] = pd.DatetimeIndex(data['shipping_date']).day

    data['shipping_date_month'] = data['shipping_date_month'].astype('category')
    data['shipping_date_day'] = data['shipping_date_day'].astype('category')

    # Excluding redundant columns
    cols_exclude = ["origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude", "is_adr",
                    "shipping_date"]
    cols_relevant = list(set(list(data.columns.values)).difference(cols_exclude))

    data = pd.DataFrame(data.loc[:, cols_relevant])

    # Create dummy variables
    cat_features = list(data.select_dtypes(include=['category']).columns)
    data = pd.get_dummies(data, columns=cat_features, dtype=int)

    return data


def grid_search(data):
    # Initializing model params
    eta_range = np.arange(0.2, 0.45, 0.05)
    max_depth_range = [5, 6, 7, 8]
    model_obj = 'reg:squarederror'
    eval_metric = 'rmse'
    df_perf = pd.DataFrame(columns=['eta', 'max_depth', 'rmse', 'num_rounds'])

    # Preparing data for XGBoost model
    features = list(set(list(data.columns.values)).difference(["cost"]))
    dtrain = xgb.DMatrix(data=pd.DataFrame(data.loc[:, features]), label=pd.DataFrame(np.log(data['cost'] * 100)))

    for eta in eta_range:
        for max_depth in max_depth_range:
            param = {'objective': model_obj,
                     'eta': eta,
                     'max_depth': max_depth,
                     'eval_metric': eval_metric,
                     'seed': 3137}
            eval_result = xgb.cv(params=param, dtrain=dtrain, num_boost_round=200, early_stopping_rounds=15,
                                 nfold=5, verbose_eval=True)
            print("eta: {}  max_depth: {}   RMSE:{}".format(eta, max_depth,
                                                            round(eval_result['test-rmse-mean'].min(), 7)))
            df_perf = df_perf.append({'eta': eta, 'max_depth': max_depth,
                                      'rmse': round(eval_result['test-rmse-mean'].min(), 7),
                                      'num_rounds': len(eval_result)}, ignore_index=True)

    best_eta = round(df_perf.eta[df_perf['rmse'] == df_perf['rmse'].min()].min(), 2)
    best_max_depth = int(df_perf.max_depth[df_perf['rmse'] == df_perf['rmse'].min()].min())
    num_rounds = int(df_perf.num_rounds[df_perf['rmse'] == df_perf['rmse'].min()].min())
    best_params = {'eta': best_eta,
                   'max_depth': best_max_depth,
                   'objective': model_obj,
                   'eval_metric': eval_metric,
                   'seed': 3137}
    print("Optimum params:\n {}".format(best_params))

    return best_params


def xgb_model(params, data):
    # Preparing data
    features = list(set(list(data.columns.values)).difference(["cost"]))
    dtrain = xgb.DMatrix(data=pd.DataFrame(data.loc[:, features]), label=pd.DataFrame(np.log(data['cost'] * 100)))

    watchlist = [(dtrain, 'train')]
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=200, evals=watchlist, early_stopping_rounds=15)
    return model


def xgb_predict(model, data):
    dtest = xgb.DMatrix(data=data)
    predictions = np.exp(model.predict(dtest)) / 100

    return predictions


if __name__ == '__main__':
    df = pd.read_csv(filepath_or_buffer="train_data.csv", delimiter=";")
    df_train = preprocess(df.copy(deep=True))
    print(df_train.dtypes)

    opt_params = grid_search(df_train.copy(deep=True))
    best_model = xgb_model(opt_params, df_train)

    test_data = pd.read_csv(filepath_or_buffer="test_data.csv", delimiter=";")
    df_test = preprocess(test_data.copy(deep=True))
    print(df_test.dtypes)
    predicted_costs = xgb_predict(best_model, df_test)

    ## print(metrics.r2_score(df_test['cost'], predicted_costs))



