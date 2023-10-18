import pandas as pd
from pandas.io.parsers import TextFileReader
from functools import reduce


def include_income_by_zip(realtor_df, income_by_zip):
    cleaned_income_by_zip = clean_income_by_zip(income_by_zip)
    realtor_df_with_income_by_zip = realtor_df.merge(cleaned_income_by_zip, how='right', on='postal_code')
    return realtor_df_with_income_by_zip


def clean_income_by_zip(income_by_zip):
    income_by_zip = income_by_zip.rename(columns={'Estimate!!Households!!Median income ('
                                                  'dollars)': 'median', 'zip_code': 'postal_code'})
    income_by_zip = income_by_zip[pd.to_numeric(income_by_zip['median'], errors='coerce').notnull()]
    income_by_zip = income_by_zip[['postal_code', 'median']]
    income_by_zip['median'] = income_by_zip['median'].astype(int)
    income_by_zip['postal_code'] = income_by_zip['postal_code'].astype(str)
    income_by_zip = income_by_zip.drop_duplicates(subset='postal_code', keep='first')

    return income_by_zip


def combine_column_filters(column_filters, df):
    return reduce(lambda x, y: x | y, (column_filter(df) for column_filter in column_filters))


def reader_to_filtered_df(reader: TextFileReader, column_filters):
    filtered_df = pd.DataFrame()
    for chunk in reader:
        combined_filter = combine_column_filters(column_filters, chunk)
        filtered_chunk = chunk[combined_filter]
        filtered_df = pd.concat([filtered_df, filtered_chunk], ignore_index=True)
    return filtered_df
