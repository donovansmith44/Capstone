import matplotlib.pyplot as plt
import pandas as pd
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
    lambda df: (100 <= df['median_square_feet']),
    lambda df: (df['median_square_feet'] <= 10000),
    lambda df: (30 <= df['active_listing_count']),
    lambda df: (10000 <= df['median_listing_price']),
    lambda df: (df['median_listing_price'] <= 1000000),
    lambda df: (df['quality_flag'] == 1)
]

realtor_data_2021 = Clean.reader_to_filtered_df(realtor_data_reader, column_filters)
realtor_data_2021 = Clean.include_income_by_zip(realtor_data_2021, income_by_zip_df)
to_drop = ['month_date_yyyymm', 'postal_code', 'zip_name',
           'median_listing_price_mm', 'median_listing_price_yy',
           'active_listing_count_mm', 'median_listing_price_per_square_foot',
           'median_listing_price_per_square_foot_mm',
           'median_listing_price_per_square_foot_yy',
           'median_square_feet_mm', 'median_square_feet_yy',
           'average_listing_price', 'average_listing_price_mm',
           'average_listing_price_yy', 'pending_ratio', 'pending_ratio_mm',
           'pending_ratio_yy', 'price_reduced_count_mm',
           'price_reduced_count_yy', 'pending_listing_count',
           'pending_listing_count_mm', 'pending_listing_count_yy',
           'median_days_on_market_mm', 'median_days_on_market_yy',
           'quality_flag', 'new_listing_count_mm',
           'new_listing_count_yy', 'price_increased_count',
           'price_increased_count_mm', 'price_increased_count_yy',
           'total_listing_count_mm', 'total_listing_count_yy',
           'price_reduced_count', 'new_listing_count',
           'active_listing_count_yy'
           ]
realtor_data_2021 = realtor_data_2021.drop(to_drop, axis=1)
realtor_data_2021 = realtor_data_2021.dropna()
# sns.heatmap(realtor_data_2021.corr(numeric_only=True),
#             cmap='Reds',
#             fmt='.2f',
#             linewidths=2,
#             annot=True)
# plt.show()
sns.pairplot(realtor_data_2021)
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

# sns.residplot(x=map_models_to_val[models[1]], y=map_models_to_pred[models[1]])
plt.show()
