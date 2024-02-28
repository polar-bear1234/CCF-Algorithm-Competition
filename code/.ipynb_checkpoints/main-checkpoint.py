# -*- coding: utf-8 -*-

# Created on 2021/11/25
# Author: 雅俗共赏 <2542174006@qq.com>

'''
 程序启动项
 执行该文件可以启动预测程序并输出预测结果进行保存
'''

from model_prediction import train_xgb_model
import sys
from os.path import dirname, abspath

# ###########训练&预测
test = train_xgb_model()

# ###########设置输出文件路径
path = dirname(dirname(abspath(__file__)))
sys.path.append(path + "\\prediction_result")


# #############导出结果
pre = test[['loan_id', 'isDefault']]
pre.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv(path + '\\prediction_result\\result.csv', index=False)