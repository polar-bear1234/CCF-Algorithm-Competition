# -*- coding: utf-8 -*-

# Created on 2021/11/25
# Author: 雅俗共赏 <2542174006@qq.com>

'''
新特征加入到原始数据中，对完整数据集分别进行预处理和特征工程
'''

import pandas as pd
from feature1 import policy_code_feature, sub_class_feature
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
import sys
from os.path import dirname, abspath

warnings.filterwarnings('ignore')


# ############################################feature2
def read_data():
    # 读取数据
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path + "\\raw_data")
    train_data = pd.read_csv(path + '\\raw_data\\train_public.csv')
    test_data = pd.read_csv(path + '\\raw_data\\test_public.csv')
    train_init_data = pd.read_csv(path + '\\raw_data\\train_internet.csv')

    return train_data, test_data, train_init_data


def modify_data():
    # 修改数据集里的policy_code和sub_class
    train_data, test_data = read_data()[:2]

    train, test = policy_code_feature()
    data_train, data_test = sub_class_feature()

    train_data['policy_code'] = train
    test_data['policy_code'] = test

    train_data['sub_class'] = data_train
    test_data['sub_class'] = data_test

    return train_data, test_data


def findDate(val):
    # earlies_credit_mon日期转换
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val

    return val + '-01'


def preprocessing():
    train_data, test_data = modify_data()
    train_init_data = read_data()[2]

    # 采用均值填充f系
    train_data['work_year'] = train_data['work_year'].fillna(-1)
    test_data['work_year'] = test_data['work_year'].fillna(-1)
    train_init_data['work_year'] = train_init_data['work_year'].fillna(-1)

    col_fill = ['f0', 'f1', 'f2', 'f3', 'f4']
    imp = SimpleImputer(strategy='median')
    for c in col_fill:
        train_init_data[c] = imp.fit_transform(train_init_data[c].values.reshape(-1, 1))

    # 处理特征work_year  class
    work_year_dict = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
                      '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    train_data['work_year'] = train_data['work_year'].map(work_year_dict)
    test_data['work_year'] = test_data['work_year'].map(work_year_dict)
    train_init_data['work_year'] = train_init_data['work_year'].map(work_year_dict)
    class_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    train_data['class'] = train_data['class'].map(class_dict)
    test_data['class'] = test_data['class'].map(class_dict)
    train_init_data['class'] = train_init_data['class'].map(class_dict)

    # 日期转换
    train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
    test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
    train_init_data['issue_date'] = pd.to_datetime(train_init_data['issue_date'])
    train_data['issue_date_year'] = train_data['issue_date'].dt.year
    test_data['issue_date_year'] = test_data['issue_date'].dt.year
    train_init_data['issue_date_year'] = train_init_data['issue_date'].dt.year
    train_data['issue_date_month'] = train_data['issue_date'].dt.month
    test_data['issue_date_month'] = test_data['issue_date'].dt.month
    train_init_data['issue_date_month'] = train_init_data['issue_date'].dt.month
    train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
    test_data['issue_date_dayofweek'] = test_data['issue_date'].dt.dayofweek
    train_init_data['issue_date_dayofweek'] = train_init_data['issue_date'].dt.dayofweek
    train_data.drop('issue_date', axis=1, inplace=True)
    test_data.drop('issue_date', axis=1, inplace=True)
    train_init_data.drop('issue_date', axis=1, inplace=True)
    train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDate))
    test_data['earlies_credit_mon'] = pd.to_datetime(test_data['earlies_credit_mon'].map(findDate))
    train_init_data['earlies_credit_mon'] = pd.to_datetime(train_init_data['earlies_credit_mon'].map(findDate))
    train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
    test_data['earliesCreditMon'] = test_data['earlies_credit_mon'].dt.month
    train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
    test_data['earliesCreditYear'] = test_data['earlies_credit_mon'].dt.year
    train_init_data['earliesCreditMon'] = train_init_data['earlies_credit_mon'].dt.month
    train_data.drop('earlies_credit_mon', axis=1, inplace=True)
    test_data.drop('earlies_credit_mon', axis=1, inplace=True)
    train_init_data.drop('earlies_credit_mon', axis=1, inplace=True)

    # 特征编码
    cat_cols = ['employer_type', 'industry', 'sub_class']
    for col in cat_cols:
        lab = LabelEncoder().fit(train_data[col])
        train_data[col] = lab.transform(train_data[col])
        test_data[col] = lab.transform(test_data[col])
        train_init_data[col] = lab.transform(train_init_data[col])

    # 异常值处理
    train_init_data = train_init_data[train_init_data['total_loan'] <= 38000]
    train_init_data = train_init_data[train_init_data['debt_loan_ratio'] <= 43.34]
    train_init_data = train_init_data[train_init_data['house_exist'] <= 2]
    train_init_data.reset_index()
    train_data.drop('user_id', axis=1, inplace=True)
    test_data.drop('user_id', axis=1, inplace=True)
    train_init_data.drop('user_id', axis=1, inplace=True)

    return train_data, test_data, train_init_data


def gen_target_encoding_feats(train, test, train_init, features, target_feature, target_feature1, n_fold=10):
    # 目标编码
    # -----------------------------------train
    tg_feats = np.zeros((train.shape[0], len(features)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[features], train[target_feature])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, feat in enumerate(features):
            target_mean_dict = df_train.groupby(feat)[target_feature].mean()
            df_val[f'{feat}_mean_target'] = df_val[feat].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{feat}_mean_target'].values

    for idx, feature in enumerate(features):
        train[f'{feature}_mean_target'] = tg_feats[:, idx]
    # ---------------------------------train_init
    tg_feats = np.zeros((train_init.shape[0], len(features)))
    for _, (train_index, val_index) in enumerate(kfold.split(train_init[features], train_init[target_feature1])):
        df_train, df_val = train_init.iloc[train_index], train_init.iloc[val_index]
        for idx, feat in enumerate(features):
            target_mean_dict = df_train.groupby(feat)[target_feature1].mean()
            df_val[f'{feat}_mean_target'] = df_val[feat].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{feat}_mean_target'].values

    for idx, feature in enumerate(features):
        train_init[f'{feature}_mean_target'] = tg_feats[:, idx]
    # -------------------------------------test
    for feat in features:
        target_mean_dict = train.groupby(feat)[target_feature].mean()
        test[f'{feat}_mean_target'] = test[feat].map(target_mean_dict)

    return train, test, train_init


def features():
    train_data, test_data, train_init_data = preprocessing()
    # 1.根据industry对f系特征构造
    for df in [train_data, test_data, train_init_data]:
        for item in ['f0', 'f1', 'f2', 'f3', 'f4']:
            df['industry_to_mean_' + item] = df.groupby(['industry'])[item].transform('mean')

    # 2.目标编码
    features = ['house_exist', 'debt_loan_ratio', 'industry', 'title']
    train_data, test_data, train_init_data = \
        gen_target_encoding_feats(train_data, test_data, train_init_data, features, 'isDefault', 'is_default', n_fold=10)

    # 3.构造交叉特征
    train_data['post_code_to_mean_interst'] = train_data.groupby(['post_code'])['interest'].transform('mean')
    test_data['post_code_to_mean_interst'] = test_data.groupby(['post_code'])['interest'].transform('mean')
    train_init_data['post_code_to_mean_interst'] = train_init_data.groupby(['post_code'])['interest'].transform('mean')
    train_data['industry_mean_interest'] = train_data.groupby(['industry'])['interest'].transform('mean')
    test_data['industry_mean_interest'] = test_data.groupby(['industry'])['interest'].transform('mean')
    train_init_data['industry_mean_interest'] = train_init_data.groupby(['industry'])['interest'].transform('mean')
    train_data['employer_type_mean_interest'] = train_data.groupby(['employer_type'])['interest'].transform('mean')
    test_data['employer_type_mean_interest'] = test_data.groupby(['employer_type'])['interest'].transform('mean')
    train_init_data['employer_type_mean_interest'] = \
        train_init_data.groupby(['employer_type'])['interest'].transform('mean')
    train_data['recircle_u_std_recircle_b'] = train_data.groupby(['recircle_u'])['recircle_b'].transform('std')
    test_data['recircle_u_std_recircle_b'] = test_data.groupby(['recircle_u'])['recircle_b'].transform('std')
    train_init_data['recircle_u_std_recircle_b'] = \
        train_init_data.groupby(['recircle_u'])['recircle_b'].transform('std')
    train_data['early_return_remove_early_return_amount'] = train_data['early_return_amount'] / train_data[
        'early_return']
    test_data['early_return_remove_early_return_amount'] = \
        test_data['early_return_amount'] / test_data['early_return']
    train_init_data['early_return_remove_early_return_amount'] = \
        train_init_data['early_return_amount'] / train_init_data['early_return']

    # 将比值后存在inf的值设为 0
    inf1 = np.isinf(train_data['early_return_remove_early_return_amount'])
    train_data['early_return_remove_early_return_amount'][inf1] = 0
    inf2 = np.isinf(test_data['early_return_remove_early_return_amount'])
    test_data['early_return_remove_early_return_amount'][inf2] = 0
    inf3 = np.isinf(train_init_data['early_return_remove_early_return_amount'])
    train_init_data['early_return_remove_early_return_amount'][inf3] = 0

    return train_data, test_data, train_init_data


if __name__ == "__main__":
    print(features()[0].shape)
    print(features()[1].shape)
    print(features()[2].shape)