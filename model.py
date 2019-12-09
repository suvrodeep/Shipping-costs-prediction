import pandas as pd
import numpy as np
import math as mt
import datetime
import swifter


def preprocess(data):
    data['is_adr'] = data['is_adr'].astype('int')
    data['is_adr'] = data['is_adr'].astype('category')

    data['distance'] = data.swifter.apply(lambda row: mt.sqrt(mt.pow(row['destination_longitude'] -
                                                                     row['origin_longitude'], 2) +
                                                              mt.pow(row['destination_latitude'] -
                                                                     row['origin_latitude'], 2)), axis=1)

    data['shipping_date'] = data.swifter.apply(lambda row: datetime.datetime.strptime(row['shipping_date'],
                                                                                      "%Y-%m-%d"), axis=1)
    data['shipping_date_month'] = data.swifter.apply(lambda row: row['shipping_date'].month, axis=1)
    data['shipping_date_month'] = data['shipping_date_month'].astype('int')
    data['shipping_date_month'] = data['shipping_date_month'].astype('category')
    data['shipping_date_day'] = data.swifter.apply(lambda row: row['shipping_date'].day, axis=1)
    data['shipping_date_day'] = data['shipping_date_day'].astype('int')
    data['shipping_date_day'] = data['shipping_date_day'].astype('category')

    return data

 
if __name__ == '__main__':
    df = pd.read_csv(filepath_or_buffer="train_data.csv", delimiter=";")
    df_train = preprocess(df.copy(deep=True))
    print(df_train.dtypes)
