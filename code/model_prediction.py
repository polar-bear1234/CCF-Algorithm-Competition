# -*- coding: utf-8 -*-

# Created on 2021/11/25
# Author: 雅俗共赏 <2542174006@qq.com>

'''训练模型'''

from model_filter import filter_data
import numpy as np
import gc
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# #############################xgboost训练
def train_xgb_model():
    data = filter_data()
    # 重新定义新的数据集
    train = data[data['isDefault'].notna()]
    test = data[data['isDefault'].isna()]
    y = train['isDefault']
    # xgboost单模
    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feats = [f for f in train.columns if f not in ['loan_id', 'isDefault']]
    fold = KFold(n_splits=10, shuffle=True, random_state=1122)
    for n_fold, (trn_idx, val_idx) in enumerate(fold.split(train, y)):
        trn_x, trn_y = train[feats].iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train[feats].iloc[val_idx], y.iloc[val_idx]
        clf = XGBClassifier(eval_metric='auc', max_depth=5, alpha=0.3, reg_lambda=0.3, subsample=0.8,
                            colsample_bylevel=0.867, objective='binary:logistic', use_label_encoder=False,
                            learning_rate=0.08, n_estimators=4000, min_child_weight=2, tree_method='hist',
                            n_jobs=-1)
        clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric='auc', verbose=100,
                early_stopping_rounds=40)
        oof_preds[val_idx] = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
        sub_preds += clf.predict_proba(test[feats], ntree_limit=clf.best_ntree_limit)[:, 1] / fold.n_splits
        print('第%d次auc分数：%.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    test['isDefault'] = sub_preds
    print('=' * 30 + '\nXGB单模线下分:%.6f' % roc_auc_score(y, oof_preds))

    return test

