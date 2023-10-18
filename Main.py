import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import Clean


plt.figure(figsize=(12, 10))
pd.set_option('display.max_columns', None)

models = [RandomForestRegressor(), XGBRegressor()]
realtor_data = 'Resources/core_housing_data.csv'
income_by_zip_csv = 'Resources/census_income.csv'
income_by_zip_df = pd.read_csv(income_by_zip_csv)

column_type_converters = {'postal_code': str, 'month_date_yyyymm': str}
realtor_data_reader = pd.read_csv(realtor_data, chunksize=100000, encoding='ISO-8859-1', converters=column_type_converters)
column_filters = [
    lambda df: df['month_date_yyyymm'].str.contains('2021'),
    lambda df: (100 < df['median_square_feet']),
    lambda df: (df['median_square_feet'] < 10000),
    lambda df: (30 <= df['active_listing_count'])
]

realtor_data_2021 = Clean.reader_to_filtered_df(realtor_data_reader, column_filters)
realtor_data_2021 = Clean.include_income_by_zip(realtor_data_2021, income_by_zip_df)

# plt.hist(realtor_data_2021['median_listing_price'], bins=100)
columns_of_interest = ['median', 'median_listing_price', 'median_days_on_market',
                       'new_listing_count', 'median_square_feet', 'price_reduced_count',
                       'total_listing_count']
realtor_data_2021 = realtor_data_2021[columns_of_interest]
realtor_data_2021 = realtor_data_2021.dropna()
# sns.heatmap(realtor_data_2021.corr(numeric_only=True),
#             cmap='Reds',
#             fmt='.2f',
#             linewidths=2,
#             annot=True)
# plt.show()
target = realtor_data_2021['median_listing_price'].values
features = realtor_data_2021.drop('median_listing_price', axis=1)
X_train, X_val, \
    Y_train, Y_val = train_test_split(features, target, test_size=.5, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

map_models_to_val = {}
map_models_to_pred = {}
for model in models:
    model.fit(X_train, Y_train)
    print(f'{model} : ')
    train_preds = model.predict(X_train)
    print('Training Error: ', mean_squared_error(Y_train, train_preds))

    val_preds = model.predict(X_val)
    print('Validation Error: ', mean_squared_error(Y_val, val_preds))
    map_models_to_val[model] = Y_val
    map_models_to_pred[model] = val_preds

    print(f'R2: {r2_score(Y_val, val_preds)}')
    print()

sns.residplot(x=map_models_to_val[models[1]], y=map_models_to_pred[models[1]])
plt.show()