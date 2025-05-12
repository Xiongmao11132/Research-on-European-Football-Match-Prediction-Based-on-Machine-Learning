# # -*- coding:utf-8 -*-
# # coding:unicode_escape
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.ensemble import RandomForestClassifier
# from IPython.display import Image
# from sklearn import tree
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import pandas as pd
# import xgboost as xgb
# import numpy as np
# import warnings
# import random
# import csv
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.metrics import classification_report
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVC
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# import joblib
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.ensemble import StackingClassifier
# import os
# from imblearn.over_sampling import SMOTE
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, recall_score, f1_score
# import joblib
# import pandas as pd
# from imblearn.over_sampling import SMOTE

# # 加载和处理数据
# processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
# merged_df = pd.read_csv(processed_data_path)
# if any(char in col for col in merged_df.columns for char in ['[', ']', '<']):
#     merged_df.columns = merged_df.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')

# X = merged_df.drop('FTR', axis=1)  # 特征
# y = merged_df['FTR']  # 目标变量

# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# # 定义基础模型和堆叠模型
# base_models = [
#     ('knn', KNeighborsClassifier()),
#     ('log_reg', LogisticRegression(random_state=1)),
#     ('svm', SVC()),
#     ('decision_tree', DecisionTreeClassifier(random_state=1)),
#     ('random_forest', RandomForestClassifier(random_state=1)),
#     ('xgb', XGBClassifier()),
#     ('GBDT', GradientBoostingClassifier())
# ]

# meta_model = XGBClassifier(random_state=1)

# stack_model = StackingClassifier(
#     estimators=base_models,
#     final_estimator=meta_model,
#     cv=KFold(n_splits=3)
# )

# # 训练堆叠模型
# stack_model.fit(X_train, y_train)
# y_pred_stack = stack_model.predict(X_test)
# joblib.dump(stack_model, 'stack_model_MLP.pkl')

# # 输出堆叠模型评估指标
# print('堆叠模型准确率:', stack_model.score(X_test, y_test))
# print('堆叠模型召回率:', recall_score(y_test, y_pred_stack, average='macro'))
# print('堆叠模型F1得分:', f1_score(y_test, y_pred_stack, average='macro'))
# print('堆叠模型分类报告:')
# print(classification_report(y_test, y_pred_stack))

# # 保存特征列
# train_columns = X_train.columns
# joblib.dump(train_columns, 'train_columns.pkl')

# # 新数据集验证
# new_data_path = "/data/coding/Processed_merged_SMOTE_output_2024.csv"
# merged_df_new = pd.read_csv(new_data_path)
# if any(char in col for col in merged_df_new.columns for char in ['[', ']', '<']):
#     merged_df_new.columns = merged_df_new.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')
# X_new = merged_df_new.drop('FTR', axis=1)
# y_true = merged_df_new['FTR']

# X_new = X_new.reindex(columns=train_columns, fill_value=0)

# # 使用堆叠模型预测新数据
# stack_model = joblib.load('stack_model_MLP.pkl')
# y_pred_new = stack_model.predict(X_new)
# print('新数据准确率:', stack_model.score(X_new, y_true))
# print('新数据召回率:', recall_score(y_true, y_pred_new, average='macro'))
# print('新数据F1得分:', f1_score(y_true, y_pred_new, average='macro'))

# # 混淆矩阵绘制
# # cm = confusion_matrix(y_true, y_pred_new)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stack_model.classes_, yticklabels=stack_model.classes_)
# # plt.ylabel('真实标签')
# # plt.xlabel('预测标签')
# # plt.title('新数据混淆矩阵')
# # plt.show()


import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# 设置并行数
n_jobs = -1  # 使用所有可用的CPU核心

# 加载和处理数据
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    if any(char in col for col in df.columns for char in ['[', ']', '<']):
        df.columns = df.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')
    return df

processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
merged_df = load_and_process_data(processed_data_path)
X = merged_df.drop('FTR', axis=1)  # 特征
y = merged_df['FTR']  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# 定义基础模型和堆叠模型
base_models = [
    ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=20, weights='distance', n_jobs=n_jobs)),
    ('log_reg', LogisticRegression(C=100, penalty='l2', solver='newton-cg', max_iter=100000, random_state=1, n_jobs=n_jobs)),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True)),
    ('decision_tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=9,
                                             max_features=None, min_samples_leaf=12,
                                             min_samples_split=90, splitter='best', random_state=1)),
    ('random_forest', RandomForestClassifier(n_estimators=700, min_samples_split=2,
                                             min_samples_leaf=1, max_features='sqrt',
                                             max_depth=46, bootstrap=True, random_state=1, n_jobs=n_jobs)),
    ('xgb', XGBClassifier(subsample=1.0, n_estimators=250, min_child_weight=1, max_depth=20,
                          learning_rate=0.1, colsample_bytree=0.5, n_jobs=n_jobs)),
    ('GBDT', GradientBoostingClassifier(learning_rate=0.1, max_depth=30,
                                         n_estimators=440, subsample=0.8))
]

meta_model = XGBClassifier(eval_metric='logloss', random_state=1, n_jobs=n_jobs)
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=3),
    n_jobs=n_jobs
)

# 训练堆叠模型
stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)
joblib.dump(stack_model, 'stack_model_XGB.pkl')

# 输出堆叠模型评估指标
print('堆叠模型准确率:', stack_model.score(X_test, y_test))
print('堆叠模型召回率:', recall_score(y_test, y_pred_stack, average='macro'))
print('堆叠模型F1得分:', f1_score(y_test, y_pred_stack, average='macro'))
print('堆叠模型分类报告:')
print(classification_report(y_test, y_pred_stack))

train_columns = X_train.columns
joblib.dump(train_columns, 'train_columns_XGB.pkl')

# 新数据集验证
new_data_path = "/data/coding/Processed_merged_SMOTE_output_2024.csv"
merged_df_new = load_and_process_data(new_data_path)
X_new = merged_df_new.drop('FTR', axis=1)
y_true = merged_df_new['FTR']
train_columns = joblib.load('train_columns_XGB.pkl')
X_new = X_new.reindex(columns=train_columns, fill_value=0)

# 使用堆叠模型预测新数据
stack_model = joblib.load('stack_model_XGB.pkl')
y_pred_new = stack_model.predict(X_new)
print('新数据准确率:', stack_model.score(X_new, y_true))
print('新数据召回率:', recall_score(y_true, y_pred_new, average='macro'))
print('新数据F1得分:', f1_score(y_true, y_pred_new, average='macro'))

# 混淆矩阵绘制
cm = confusion_matrix(y_true, y_pred_new)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stack_model.classes_, yticklabels=stack_model.classes_)
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('新数据混淆矩阵')
plt.show()
