# -*- coding: utf-8 -*-

# Created on 2021/11/25
# Author: 雅俗共赏 <2542174006@qq.com>

'''
过滤train_internet中的数据，将过滤后的精确数据和原始数据集合并
'''

import pandas as pd
from feature2 import features
import numpy as np
import gc
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def filter_data():
    train_data, test_data, train_init_data = features()

    # #########################################样本扩充
    column1 = set(train_data.columns)
    column2 = set(test_data.columns)
    column3 = set(train_init_data.columns)
    same_col = list(column1.intersection(column3))
    nosasme_col = list(column1.difference(column3))
    train_init_name_data = train_init_data[same_col].copy()
    for col in nosasme_col:
        train_init_name_data[col] = np.nan

    # ########################################catboost过滤
    y = train_data['isDefault']
    oof_preds = np.zeros(train_data.shape[0])
    sub_preds = np.zeros(train_init_name_data.shape[0])
    feats = [f for f in train_data.columns if f not in ['loan_id', 'isDefault']]
    fold = KFold(n_splits=10, shuffle=True, random_state=546789)
    for n_fold, (trn_idx, val_idx) in enumerate(fold.split(train_data)):
        trn_x, trn_y = train_data[feats].iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train_data[feats].iloc[val_idx], y.iloc[val_idx]
        clf = CatBoostClassifier(class_weights=[1, 1.15], depth=6, learning_rate=0.08, iterations=4000,
                                 bootstrap_type='Bernoulli', subsample=0.9, random_seed=546789, verbose=0,
                                 allow_writing_files=False)
        clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], verbose=100, early_stopping_rounds=40)
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
        sub_preds += clf.predict_proba(train_init_name_data[feats])[:, 1] / fold.n_splits
        print('第%d次过滤auc分数：%.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    train_init_name_data['isDefault'] = sub_preds
    IntePre = train_init_name_data[['loan_id', 'isDefault']]
    InteId = IntePre.loc[IntePre.isDefault < 0.07, 'loan_id'].tolist()
    train = train_data
    test = test_data
    train_init = train_init_name_data
    train_init['isDefault'] = train_init_data['is_default']
    use_te = train_init[train_init.loan_id.isin(InteId)].copy()
    data = pd.concat([train, test, use_te]).reset_index(drop=True)
    print('=' * 30 + '\n过滤分数:%.6f\n过滤后的数据量:' % (roc_auc_score(y, oof_preds)), data.shape)

    return data


if __name__ == "__main__":
    print(filter_data().shape)