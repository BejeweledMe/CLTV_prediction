from sklearn.preprocessing import OneHotEncoder
from typing import Dict
import pandas as pd
import numpy as np


def ohe_preprocess(df: pd.DataFrame,
                   ohe_info: Dict[str, list],
                   copy: bool = True,
                   delete_cols: bool = True) -> pd.DataFrame:
    """
    Apply one hot encoding to columns of dataframe specified in parameter *ohe_info*.
    Here is ohe_cols_info example:
    ohe_cols_info = {'State': [['Washington'], ['Arizona'], ['Nevada'], ['California'], ['Oregon']]}

    Create new columns for column's categories, where new column name is
    'column name : category of column'.

    Delete old columns if parameter *delete_cols*  set to True.

    :param df: pandas.Dataframe to preprocess
    :param ohe_info: dictionary  of column names to preprocess paired to lists of categories to fit to OneHotEncoder
    :param copy: boolean parameter to make a copy of a dataframe or apply changes to original
    :param delete_cols: boolean parameter to delete old columns with applied transformations
    :return: dataframe with new one hot features and deleted old columns if delete_cols
    """

    if copy:
        df = df.copy()

    for col in ohe_info.keys():
        # use handle_unknown='ignore' to avoid errors with unknown values if it will come in future
        ohe = OneHotEncoder(handle_unknown='ignore')

        ohe.fit(np.array(ohe_info[col]).reshape(-1, 1))
        categories = ohe.categories_[0]

        ohe_vector = ohe.transform(df[col].values.reshape(-1, 1)).toarray()

        for i in range(len(categories)):
            df[f'{col} : {categories[i]}'] = ohe_vector[:, i]

        if delete_cols:
            del df[col]

    return df


def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True,
                     copy: bool = True) -> pd.DataFrame:
    """
    Check dataframe values by columns and transformed it to memory efficient data types.

    :param df: pandas.Dataframe to preprocess
    :param verbose: boolean parameter to print memory changes
    :param copy: boolean parameter to make a copy of a dataframe or apply changes to original
    :return: dataframe with memory efficient data types
    """

    if copy:
        df = df.copy()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print(
        'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )

    return df


def normalize_df(df: pd.DataFrame,
                 copy: bool = True) -> pd.DataFrame:
    """
    Apply normalizing to dataframe.
    It means that new mean and std values of columns will be close to 0 and 1 respectively.

    :param df: pandas.Dataframe to preprocess
    :param copy: boolean parameter to make a copy of a dataframe or apply changes to original
    :return: normalized dataframe
    """

    if copy:
        df = df.copy()

    df = (df - df.mean()) / df.std()
    return df


def apply_all_transforms(df: pd.DataFrame,
                         copy: bool = True) -> pd.DataFrame:
    """
    Apply transformations to IBM dataframe for CLTV prediction.

    :param df: pandas.Dataframe to preprocess
    :param copy: boolean parameter to make a copy of a dataframe or apply changes to original
    :return: transformed dataframe
    """

    if copy:
        df = df.copy()

    # Gender preprocessing
    male_df = df.query('Gender == "M"')
    female_df = df.query('Gender == "F"')

    df['gender_log_mean_target'] = 0
    df['gender_log_mean_target'].iloc[male_df.index] = np.log(male_df['Customer Lifetime Value'].mean())
    df['gender_log_mean_target'].iloc[female_df.index] = np.log(female_df['Customer Lifetime Value'].mean())

    # Policy level
    df['policy_l_level'] = df['Policy'].apply(lambda x: int(x[-1]))

    # Vehicle size
    vehicle_size_labels = {
        'Small': 0,
        'Medsize': 1,
        'Large': 2
    }

    df['vehicle_size_labeled'] = df['Vehicle Size'].apply(lambda x: vehicle_size_labels[x])

    # Coverage
    coverage_labels = {
        'Basic': 0,
        'Extended': 1,
        'Premium': 2
    }

    df['coverage_labeled'] = df['Coverage'].apply(lambda x: coverage_labels[x])

    del df['Policy'], df['Vehicle Size'], df['Gender'], df['Coverage']

    # One hot preprocessing
    ohe_cols_info = {
        'State':
            [
                ['Washington'],
                ['Arizona'],
                ['Nevada'],
                ['California'],
                ['Oregon']
            ],
        'Response':
            [
                ['No'],
                ['Yes']
            ],
        'Education':
            [
                ['Bachelor'],
                ['College'],
                ['Master'],
                ['High School or Below'],
                ['Doctor']
            ],
        'EmploymentStatus':
            [
                ['Employed'],
                ['Unemployed'],
                ['Medical Leave'],
                ['Disabled'],
                ['Retired']
            ],
        'Location Code':
            [
                ['Suburban'],
                ['Rural'],
                ['Urban']
            ],
        'Marital Status':
            [
                ['Married'],
                ['Single'],
                ['Divorced']
            ],
        'Policy Type':
            [
                ['Corporate Auto'],
                ['Personal Auto'],
                ['Special Auto']
            ],
        'Renew Offer Type':
            [
                ['Offer1'],
                ['Offer2'],
                ['Offer3'],
                ['Offer4']
            ],
        'Sales Channel':
            [
                ['Agent'],
                ['Call Center'],
                ['Web'],
                ['Branch']
            ],
        'Vehicle Class':
            [
                ['Two-Door Car'],
                ['Four-Door Car'],
                ['SUV'],
                ['Luxury SUV'],
                ['Sports Car'],
                ['Luxury Car']
            ]
    }

    df = ohe_preprocess(df=df, ohe_info=ohe_cols_info, copy=copy, delete_cols=True)

    return df
