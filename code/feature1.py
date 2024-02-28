# -*- coding: utf-8 -*-

# Created on 2021/11/25
# Author: 雅俗共赏 <2542174006@qq.com>

'''
对原始数据进行预处理以及生成新特征
'''

import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
from os.path import dirname, abspath

warnings.filterwarnings('ignore')


# ############################################feature1
def read_data():
    # 读取数据
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path + "\\raw_data")
    train_data = pd.read_csv(path + '\\raw_data\\train_public.csv')
    test_data = pd.read_csv(path + '\\raw_data\\test_public.csv')
    train_init_data = pd.read_csv(path + '\\raw_data\\train_internet.csv')

    return train_data, test_data, train_init_data


def fill_value():
    # 填补缺失值f系
    train_data, test_data, train_init_data = read_data()
    col = ['f0', 'f1', 'f2', 'f3', 'f4']
    x_col = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    imp = SimpleImputer(strategy = 'median')
    for c in col:
        train_data[c] = imp.fit_transform(train_data[c].values.reshape(-1,1))
        test_data[c] = imp.fit_transform(test_data[c].values.reshape(-1,1))
    for x_ in x_col:
        train_init_data[x_] = imp.fit_transform(train_init_data[x_].values.reshape(-1,1))

    return train_data, test_data, train_init_data


def findDate(val):
    # 处理earlies_credit_mon特征
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val

    return val + '-01'


def feature():
    # 1.处理特征work_year、class、work_type
    train_data, test_data, train_init_data = fill_value()
    work_year_dict = {'< 1 year': 0,'1 year': 1,'2 years': 2,'3 years': 3,'4 years': 4,'5 years': 5,
                      '6 years': 6,'7 years': 7,'8 years': 8,'9 years': 9,'10+ years': 10}
    train_data['work_year'] = train_data['work_year'].map(work_year_dict)
    test_data['work_year'] = test_data['work_year'].map(work_year_dict)
    train_init_data['work_year'] = train_init_data['work_year'].map(work_year_dict)
    train_data['work_year'] = train_data['work_year'].fillna(1)
    test_data['work_year'] = test_data['work_year'].fillna(1)
    train_init_data['work_year'] = train_init_data['work_year'].fillna(1)

    class_dict = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    train_data['class'] = train_data['class'].map(class_dict)
    test_data['class'] = test_data['class'].map(class_dict)
    train_init_data['class'] = train_init_data['class'].map(class_dict)

    work_type_dict = {'公务员':0,'其他':1,'工人':2,'工程师':3,'职员':4}
    train_init_data['work_type'] = train_init_data['work_type'].map(work_type_dict)

    # 2.issue_date提取时间特征
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

    train_data.drop('issue_date', axis = 1, inplace=True)
    test_data.drop('issue_date', axis = 1, inplace=True)
    train_init_data.drop('issue_date', axis = 1, inplace=True)

    # 3.处理earlies_credit_mon特征
    train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDate))
    test_data['earlies_credit_mon'] = pd.to_datetime(test_data['earlies_credit_mon'].map(findDate))
    train_init_data['earlies_credit_mon'] = pd.to_datetime(train_init_data['earlies_credit_mon'].map(findDate))

    train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
    test_data['earliesCreditMon'] = test_data['earlies_credit_mon'].dt.month
    train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
    test_data['earliesCreditYear'] = test_data['earlies_credit_mon'].dt.year
    train_init_data['earliesCreditMon'] = train_init_data['earlies_credit_mon'].dt.month

    train_data.drop('earlies_credit_mon', axis = 1, inplace=True)
    test_data.drop('earlies_credit_mon', axis = 1, inplace=True)
    train_init_data.drop('earlies_credit_mon', axis = 1, inplace=True)

    # 4.处理特征  employer_type、industry
    cat_cols = ['employer_type', 'industry']
    for col in cat_cols:
        lab = LabelEncoder().fit(train_data[col])
        train_data[col] = lab.transform(train_data[col])
        test_data[col] = lab.transform(test_data[col])
        train_init_data[col] = lab.transform(train_init_data[col])

    # 5.对数据集缺失进行填充
    train_data['pub_dero_bankrup'] = train_data['pub_dero_bankrup'].fillna(method = 'ffill')
    test_data['pub_dero_bankrup'] = test_data['pub_dero_bankrup'].fillna(method = 'ffill')
    train_init_data['pub_dero_bankrup'] = train_init_data['pub_dero_bankrup'].fillna(method = 'ffill')
    train_init_data = train_init_data.dropna(subset = ['post_code', 'debt_loan_ratio', 'title'])
    train_init_data['recircle_u'] = train_init_data['recircle_u'].fillna(train_init_data['recircle_u'].median())

    return train_data, test_data, train_init_data


def policy_code_feature():
    # 构造policy_code
    train_data, test_data, train_init_data = feature()

    # 1.构建train_data的特征
    train_data.drop('policy_code', axis = 1, inplace=True)
    scaler = MinMaxScaler()
    scaler_data = scaler.fit_transform(train_data)
    clf = KMeans(n_clusters=2, random_state=10)
    pre = clf.fit(scaler_data)
    train = pre.labels_
    train = pd.DataFrame(train)

    # 2.构建test_data的特征
    test_data.drop('policy_code', axis = 1, inplace=True)
    scaler = MinMaxScaler()
    scaler_data = scaler.fit_transform(test_data)
    clf = KMeans(n_clusters=2, random_state=10)
    pre = clf.fit(scaler_data)
    test = pre.labels_
    test = pd.DataFrame(test)

    return train, test


def Kmeans(data, num):
    # kmean处理sub_class特征
    scaler = MinMaxScaler()
    scaler_data = scaler.fit_transform(data.loc[data['class'] == num])
    clf = KMeans(n_clusters = 5, random_state=546789)
    pre = clf.fit(scaler_data)
    test = pre.labels_

    return test


def sub_class_feature():
    # 通过class的A~G特征分为七类，对此七类分别预测
    # 1.构建train集的sub_class特征
    train_data, test_data, train_init_data = feature()
    for i in range(1,8):
        data = Kmeans(train_data, i)
        if i == 1:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 1].index)
            s1 = data.map({0:'A1', 1:'A2',2:'A3',3:'A4',4:'A5'})
        if i == 2:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 2].index)
            s2 = data.map({0:'B1',1:'B2',2:'B3',3:'B4',4:'B5'})
        if i == 3:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 3].index)
            s3 = data.map({0:'C1',1:'C2',2:'C3',3:'C4',4:'C5'})
        if i == 4:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 4].index)
            s4 = data.map({0:'D1',1:'D2',2:'D3',3:'D4',4:'D5'})
        if i == 5:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 5].index)
            s5 = data.map({0:'E1',1:'E2',2:'E3',3:'E4',4:'E5'})
        if i == 6:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 6].index)
            s6 = data.map({0:'F1',1:'F2',2:'F3',3:'F4',4:'F5'})
        if i == 7:
            data = pd.Series(data, index= train_data.loc[train_data['class'] == 7].index)
            s7 = data.map({0:'G1',1:'G2',2:'G3',3:'G4',4:'G5'})
    # 合并
    data_train = pd.concat([s1, s2, s3, s4, s5, s6, s7]).reset_index(drop=True)

    # 2.构建test集的sub_class特征
    for i in range(1,8):
        data = Kmeans(test_data, i)
        if i == 1:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 1].index)
            t1 = data.map({0:'A1', 1:'A2',2:'A3',3:'A4',4:'A5'})
        if i == 2:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 2].index)
            t2 = data.map({0:'B1',1:'B2',2:'B3',3:'B4',4:'B5'})
        if i == 3:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 3].index)
            t3 = data.map({0:'C1',1:'C2',2:'C3',3:'C4',4:'C5'})
        if i == 4:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 4].index)
            t4 = data.map({0:'D1',1:'D2',2:'D3',3:'D4',4:'D5'})
        if i == 5:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 5].index)
            t5 = data.map({0:'E1',1:'E2',2:'E3',3:'E4',4:'E5'})
        if i == 6:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 6].index)
            t6 = data.map({0:'F1',1:'F2',2:'F3',3:'F4',4:'F5'})
        if i == 7:
            data = pd.Series(data, index= test_data.loc[test_data['class'] == 7].index)
            t7 = data.map({0:'G1',1:'G2',2:'G3',3:'G4',4:'G5'})
    # 合并
    data_test = pd.concat([t1, t2, t3, t4, t5, t6, t7]).reset_index(drop=True)

    return data_train, data_test


if __name__ == "__main__":
    print(policy_code_feature())
    print(sub_class_feature())