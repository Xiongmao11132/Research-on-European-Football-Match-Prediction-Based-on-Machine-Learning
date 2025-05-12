# -*- coding:utf-8 -*-
# coding:unicode_escape
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
import random
import csv
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import StackingClassifier
import os
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

processed_data_path = "D:/desk/Oddsprediction/Datasets/Processed_merged_output_2024.csv"  # 替换为你的文件路径
merged_df = pd.read_csv(processed_data_path)
# 假设 'FTR' 列是目标变量（比赛最终结果）
# columns_to_drop = ['FTR']  # 全部数据
# columns_to_drop = ['HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTR']#只有赔率
columns_to_drop = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'SBH', 'SBD', 'SBA', 'WHH', 'WHD', 'WHA', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'GBAHH', 'GBAHA', 'GBAH', 'LBAHH', 'LBAHA', 'LBAH', 'B365AHH', 'B365AHA', 'B365AH', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BSH', 'BSD', 'BSA', 'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'BFH', 'BFD', 'BFA', 'BFEH', 'BFED', 'BFEA', 'BFE>2.5', 'BFE<2.5', 'BFEAHH', 'BFEAHA', 'BFCH', 'BFCD', 'BFCA', '1XBCH', '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5', 'BFECAHH', 'BFECAHA', 'FTR']#只有比赛数据
X = merged_df.drop(columns_to_drop, axis=1)  # 特征
y = merged_df['FTR']  # 目标变量
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# 模型选择
model = RandomForestClassifier(random_state=1)
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
param_dist = {
    'n_estimators': [400],
    'max_features': ['sqrt'],
    'max_depth': [27,],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=1,
                                   n_jobs=-1)
# 模型训练
random_search.fit(X_train, y_train)
# model.fit(X_train, y_train)
print("最佳参数:", random_search.best_params_)
best_model = random_search.best_estimator_
# 进行预测
y_pred = best_model.predict(X_test)
# y_pred = model.predict(X_test)
# 保存模型
joblib.dump(best_model, 'Random.pkl')
# 输出特征重要性
feature_importances = best_model.feature_importances_
# 创建一个 DataFrame 存储特征名称及其重要性
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
# 按重要性排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# 输出前20个最重要的特征
print('前20个最重要的特征:')
print(importance_df.head(20))
# 输出模型评估指标
print('准确率:', best_model.score(X_test, y_test))
print('召回率:', recall_score(y_test, y_pred, average='macro'))
print('F1 得分:', f1_score(y_test, y_pred, average='macro'))
# 保存训练时的特征列
train_columns = X_train.columns  # 训练时的特征名称
joblib.dump(train_columns, 'train_columns_RF.pkl')  # 保存特征名称
# 读取新的数据集进行验证
processed_data = "D:/desk/Oddsprediction/Datasets/Processed_merged_SMOTE_output_2024.csv"  # 替换为你的文件路径
merged_df = pd.read_csv(processed_data)
# 假设 'FTR' 列是目标变量（比赛最终结果）
X_new = merged_df.drop('FTR', axis=1)  # 特征
y_true = merged_df['FTR']  # 真实的目标变量
# 加载训练好的模型
model = joblib.load('Random.pkl')
# 加载训练时的特征列
train_columns = joblib.load('train_columns_RF.pkl')
# # 重新排列新数据的列，使其与训练时的列对齐
X_new = X_new.reindex(columns=train_columns, fill_value=0)
# 对新的数据集进行预测
y_pred = model.predict(X_new)
# 输出模型评估指标
print('准确率:', model.score(X_new, y_true))
print('召回率:', recall_score(y_true, y_pred, average='macro'))
print('F1 得分:', f1_score(y_true, y_pred, average='macro'))

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.show()















# -*- coding:utf-8 -*-
# coding:unicode_escape
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import recall_score, f1_score
#
# # 加载数据
# processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"  # 替换为你的文件路径
# merged_df = pd.read_csv(processed_data_path)
# X = merged_df.drop('FTR', axis=1)  # 特征
# y = merged_df['FTR']  # 目标变量
#
# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
#
# # 模型选择并设置参数
# model = RandomForestClassifier(random_state=1)
#
# # 模型训练
# model.fit(X_train, y_train)
#
# # 进行预测
# y_pred = model.predict(X_test)
#
# # 保存模型
# joblib.dump(model, 'Random.pkl')
#
# # 输出特征重要性
# feature_importances = model.feature_importances_
# # 创建一个 DataFrame 存储特征名称及其重要性
# importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
# # 按重要性排序
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
#
# # 输出前20个最重要的特征
# print('前20个最重要的特征:')
# print(importance_df.head(20))
#
# # 输出模型评估指标
# print('准确率:', model.score(X_test, y_test))
# print('召回率:', recall_score(y_test, y_pred, average='macro'))
# print('F1 得分:', f1_score(y_test, y_pred, average='macro'))
#
# # 保存训练时的特征列
# train_columns = X_train.columns  # 训练时的特征名称
# joblib.dump(train_columns, 'train_columns_RF.pkl')  # 保存特征名称
#
# # 读取新的数据集进行验证
# processed_data = "/data/coding/Processed_merged_SMOTE_output_2024.csv"  # 替换为你的文件路径
# merged_df = pd.read_csv(processed_data)
# X_new = merged_df.drop('FTR', axis=1)  # 特征
# y_true = merged_df['FTR']  # 真实的目标变量
#
# # 加载训练好的模型
# model = joblib.load('Random.pkl')
#
# # 加载训练时的特征列
# train_columns = joblib.load('train_columns_RF.pkl')
# # 重新排列新数据的列，使其与训练时的列对齐
# X_new = X_new.reindex(columns=train_columns, fill_value=0)
#
# # 对新的数据集进行预测
# y_pred = model.predict(X_new)
#
# # 输出模型评估指标
# print('准确率:', model.score(X_new, y_true))
# print('召回率:', recall_score(y_true, y_pred, average='macro'))
# print('F1 得分:', f1_score(y_true, y_pred, average='macro'))
#
# # 计算混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
# # 绘制热图
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.ylabel('真实标签')
# plt.xlabel('预测标签')
# plt.title('混淆矩阵')
# plt.show()
