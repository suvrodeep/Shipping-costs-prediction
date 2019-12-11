import pandas as pd
import numpy as np
import math as mt
import datetime


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

    return data

 
if __name__ == '__main__':
    df = pd.read_csv(filepath_or_buffer="train_data.csv", delimiter=";")
    df_train = preprocess(df.copy(deep=True))
    print(df_train.dtypes)
