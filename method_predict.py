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



'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
'''
# datasets = pd.read_csv('./all/sub_1.csv', engine='python', error_bad_lines=False)
# Train_X = datasets.iloc[0:49032, [1, 3, 5]].values
datasets = pd.read_csv('D:\桌面\Datasets\data.csv', encoding='gbk',engine='python', error_bad_lines=False)
Train_X = datasets.iloc[0:47266, 17:22].values
Train_y = datasets.iloc[0:47266, 23:24].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train_X, Train_y, test_size=0.4, random_state=1)
'''
'''
d1 = open('G:/EDA_emotion_recognition/data_try/case_dataset-master/1.txt', 'r')
aa = d1.readlines()
LINE = []
for ij in range(len(aa)):
    aaa = (aa[ij].split("\t")[0], aa[ij].split("\t")[1], aa[ij].split("\t")[2].split("\n")[0])
    LINE.append(aaa)
df = pd.DataFrame(LINE)
df.to_csv('G:/EDA_emotion_recognition/data_try/case_dataset-master/c1.csv', index=False)
'''
'''
#datasets1 = pd.read_csv('G:/EDA_emotion_recognition/data_try/case_dataset-master/exp.csv', engine='python', error_bad_lines=False)
#X_Test = datasets1.iloc[0:4900, 0:1].values
#Y_Test = datasets1.iloc[0:4900, 2:3].values
# TEST_SIZE = [0.2, 0.3]
Tree_C_1 = []
Tree_C_2 = []
KNN_1 = []

'''

'''
 # X_train, X_test, Y_train, Y_test = train_test_split(Train_X, Train_y, test_size=TEST_SIZE[ii], random_state=0)
datasets = pd.read_csv( 'D:\desk\Odds prediction\Datasets\data_normalized.csv', encoding='gbk',engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:22].values
Train_y = datasets.iloc[0:47266, 23:24].values
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
# clf_1 = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=0)
# clf_1.fit(Train_X, Train_y)
loaded_model = joblib.load('GDBTModel_optimized.pkl')
y_predict = loaded_model.predict(X_Test) 
# y_predict = clf_1.predict(X_Test)
print('GDBT精确率: ', precision_score(Y_Test, y_predict, average='macro'))
print('GDBT召回率: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1: ', f1_score(Y_Test, y_predict, average='macro'))
# joblib.dump(clf_1.predict, 'GDBTModel.pkl')
'''
'''
datasets = pd.read_csv("D:\\desk\\Odds prediction\\Datasets\\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:22].values
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 修正目标变量的提取
x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.4, random_state=1)
loaded_model = joblib.load('SVMmodel')# 加载保存好的模型
y_predict = loaded_model.predict(x_test)
print('准确率:', accuracy_score(y_test, y_predict))
print('召回率:', recall_score(y_test, y_predict, average='macro'))
print('F1 分数:', f1_score(y_test, y_predict, average='macro'))
'''


'''
# GBDT网格搜索优化超参
datasets = pd.read_csv('D:/desk/Odds prediction/Datasets/data_normalized.csv', encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:17].values
Train_y = datasets.iloc[0:47266, 23:24].values
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
param_grid = {
    'n_estimators': [200],
    'max_depth': [3],
    'learning_rate': [0.1, 1],
    'subsample': [0.8]
}
clf_1 = GradientBoostingClassifier(random_state=0)
grid_search = GridSearchCV(estimator=clf_1, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=3)
grid_search.fit(Train_X, Train_y.ravel())
print("最佳参数: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_Test)
print('GDBT精确率: ', precision_score(Y_Test, y_predict, average='macro'))
print('GDBT召回率: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1: ', f1_score(Y_Test, y_predict, average='macro'))
# joblib.dump(best_model, 'GDBTModel_optimized.pkl')
'''

'''
datasets = pd.read_csv( "D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk',engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 17:22].values
Train_y = datasets.iloc[0:47266, 23:24].values
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
XgBoost = xgb.XGBClassifier(n_estimators=10, max_depth=15, random_state=0)
XgBoost.fit(Train_X, Train_y)
y_predict = XgBoost.predict(X_Test)
print('XgBoost精确率: ', precision_score(Y_Test, y_predict, average='macro'))
print('XgBoost召回率: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1: ', f1_score(Y_Test, y_predict, average='macro'))
'''

'''
# XGB网格搜索超参数
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 17:22].values
Train_y = datasets.iloc[0:47266, 23:24].values
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
xgboost_model = xgb.XGBClassifier(random_state=0)
param_grid = {
    'n_estimators': [200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1],
    'subsample': [ 0.8, 1.0],
    'colsample_bytree': [0.5]
}
grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid, scoring='f1_macro', cv=3, verbose=3, n_jobs=-1)
grid_search.fit(Train_X, Train_y)
print("最优参数: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_Test)
print('XgBoost精确率: ', precision_score(Y_Test, y_predict, average='macro'))
print('XgBoost召回率: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1: ', f1_score(Y_Test, y_predict, average='macro'))
# joblib.dump(best_model, 'XGBoostModel.pkl')
'''

'''
# 独热编码KNN预测模型
encoded_data = pd.read_csv("D:\\desk\\Odds prediction\\Datasets\\encoded_data_stander.csv", encoding='gbk',engine='python', on_bad_lines='skip')
result_column = 'FTR'
X = encoded_data.drop(columns=[result_column]).values
y = encoded_data[result_column].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.1, random_state=1)
knc = KNN(n_neighbors=90)
knc.fit(X_Train, Y_Train)
y_predict = knc.predict(X_Test)
print('KNN准确率:', accuracy_score(Y_Test, y_predict))
print('KNN召回率:', recall_score(Y_Test, y_predict, average='macro'))
print('F1 分数:', f1_score(Y_Test, y_predict, average='macro'))
'''



'''
datasets = pd.read_csv( "D:\desk\Odds prediction\Datasets\data.csv", encoding='gbk',engine='python', on_bad_lines='skip')
X_Train = datasets.iloc[0:47263, 2:22].values
Y_Train = datasets.iloc[0:47263, 23:24].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train, Y_Train, test_size=0.1, random_state=1)

knc = KNN(n_neighbors=15)
knc.fit(X_Train, Y_Train)
y_predict = knc.predict(X_Test)
print('KNN准确率', knc.score(X_Test, Y_Test))
print('KNN召回率', recall_score(Y_Test, y_predict, average='macro'))
print('F1', f1_score(Y_Test, y_predict, average='macro'))
# print('KNN精确率', precision_score(Y_Test, y_predict, average='macro'))
'''
'''
 # KNN网格搜索预测优化超参数
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
X_Train = datasets.iloc[0:47263, 2:22].values
Y_Train = datasets.iloc[0:47263, 23:24].values.ravel()
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train, Y_Train, test_size=0.1, random_state=1)
from sklearn.neighbors import KNeighborsClassifier as KNN
knc = KNN()
param_grid = {'n_neighbors': [280, 290, 300, 310],
              'weights': ['uniform','distance'],  # 加入 weights 参数
              'metric': ['euclidean', 'manhattan', 'minkowski']  # 加入 metric 参数
              }
grid_search = GridSearchCV(knc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_Train, Y_Train)
best_params = grid_search.best_params_
print("最佳超参数:", best_params)
best_knc = grid_search.best_estimator_
y_predict = best_knc.predict(X_Test)
print('KNN准确率:', best_knc.score(X_Test, Y_Test))
print('KNN召回率:', recall_score(Y_Test, y_predict, average='macro'))
print('F1分数:', f1_score(Y_Test, y_predict, average='macro'))
# joblib.dump(best_knc, 'KNN_model.pkl')
'''

'''
# 随机森林1
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk',engine='python', on_bad_lines='skip')
X_Train = datasets.iloc[0:47263, 2:22].values
Y_Train = datasets.iloc[0:47263, 23:24].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train, Y_Train, test_size=0.1, random_state=1)
clf_tree_1 = RandomForestClassifier(
    bootstrap=True,               # 启用自助采样
    max_depth=15,                 # 树的最大深度
    max_features=None,          # 每次分裂时最多考虑的特征数量
    min_samples_leaf=4,           # 每个叶节点的最小样本数
    min_samples_split=2,          # 分裂内部节点所需的最小样本数
    n_estimators=300,             # 随机森林中的树木数量
    random_state=0                # 设置随机种子以获得可重复的结果
)
clf_tree_1.fit(X_Train, Y_Train)
test_predictions = clf_tree_1.predict(X_Test)
print('准确率',clf_tree_1.score(X_Test, Y_Test))
print('recall',recall_score(Y_Test, test_predictions, average='macro'))
print('F1 score',f1_score(Y_Test, test_predictions, average='macro'))
'''

'''
# 随机森林网格搜索超参
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
X = datasets.iloc[:, 17:22].values  # 替换为实际特征列号
y = datasets.iloc[:, 23].values    # 替标换为实际目变量列号
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.1, random_state=1)
param_grid = {
    'bootstrap': [True],  # 尝试使用或不使用自举抽样
    'n_estimators': [300],  # 增加不同数量的树
    'max_depth': [10, 15],  # 包含较大的深度和不限制深度的情况
    'min_samples_split': [2, 3],  # 更广泛的分裂节点数
    'min_samples_leaf': [4, 5],  # 尝试较小的叶节点样本数
    'max_features': ['sqrt', 'log2', None]  # 不同特征选择策略
}
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
rf = RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_Train, Y_Train)
print("最佳参数组合:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
test_predictions = best_rf.predict(X_Test)
# joblib.dump(best_rf, 'Random.pkl')
print('准确率:', accuracy_score(Y_Test, test_predictions))
print('recall:', recall_score(Y_Test, test_predictions, average='macro'))
print('F1 score:', f1_score(Y_Test, test_predictions, average='macro'))
'''
'''
print(Tree_1)

# 随机森林2
clf_tree_2 = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=0, class_weight="balanced")
clf_tree_2.fit(X_Train, Y_Train)
test_predictions = clf_tree_2.predict(X_Test)
Tree_2 = classification_report(Y_Test, test_predictions)
print(Tree_2)
'''

'''
#多分类逻辑回归

model = LogisticRegression()
datasets = pd.read_csv( "D:\desk\Odds prediction\Datasets\data_SMOTE_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[:, 2:22].values
Train_y = datasets.iloc[:, 23:24].values
model.fit(Train_X, Train_y)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
y_predict = model.predict(X_Test)
LogisticRegression(C=3.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=None, solver='saga', tol=0.0001,
                   verbose=0,warm_start=False)
print('准确率',model.score(X_Test, Y_Test))
print('recall',recall_score(Y_Test, y_predict, average='macro'))
print('F1 score',f1_score(Y_Test, y_predict, average='macro'))
'''

'''
#多分类逻辑回归网格搜索法优化超参数
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_SMOTE_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47263, 2:22].values
Train_y = datasets.iloc[0:47263, 23:24].values.ravel()
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
# smote = SMOTE(random_state=1)
# x_resampled, y_resampled = smote.fit_resample(Train_X, Train_y)
# 数据标准化
# scaler = StandardScaler()
# Train_X = scaler.fit_transform(Train_X)
# X_Test = scaler.transform(X_Test)
model = LogisticRegression()
param_grid = {
    'C': [3],  # 正则化强度
    'penalty': ['l1'],  # 正则化方式
    'solver': ['liblinear', 'saga'],  # 优化算法
    'max_iter': [300],  # 最大迭代次数
    'tol': [1e-4],  # 收敛阈值
    'class_weight': [None]  # 类别权重
    # 'C': [3, 4, 5,],  # 正则化强度
    # 'penalty': ['l1'],  # 正则化方式
    # 'solver': ['saga']  # 适用于l1和l2的优化算法
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)
grid_search.fit(X_Train, Y_Train)
best_params = grid_search.best_params_
print("最佳超参数:", best_params)
best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_Test)
print('准确率:', best_model.score(X_Test, Y_Test))
print('召回率:', recall_score(Y_Test, y_predict, average='macro'))
print('F1 分数:', f1_score(Y_Test, y_predict, average='macro'))
# # 保存最佳模型
# joblib.dump(best_model, 'best_model.pkl')
# loaded_model = joblib.load('best_model.pkl')
# y_predict = loaded_model.predict(X_Test)
'''

'''
# SVM
datasets = pd.read_csv("D:\\desk\\Odds prediction\\Datasets\\data.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:22].values
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 修正目标变量的提取
x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
svc = SVC(kernel='rbf', C=1, gamma='scale')  # 选择默认参数或你想要的参数
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
print('准确率:', accuracy_score(y_test, y_predict))
print('召回率:', recall_score(y_test, y_predict, average='macro'))
print('F1 分数:', f1_score(y_test, y_predict, average='macro'))
# joblib.dump(y_predict, 'SVMmodel')
'''

'''
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 17:22].values
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 修正目标变量的提取
x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
# smote = SMOTE(random_state=1)
# x_resampled, y_resampled = smote.fit_resample(Train_X, Train_y)
import os
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
svc = SVC()
param_grid = {
    'kernel': ['rbf'],  # 减少核函数的选择
    'C': [1],  # 减少 C 值的选择
    'gamma': ['scale'],  # 固定 gamma 值
    'max_iter': [1000000]  # 减少迭代次数
}
# 随机搜索
# random_search = RandomizedSearchCV(estimator=svc, param_distributions=param_grid, n_iter=5, cv=3, verbose=1, n_jobs=-1, random_state=1)
# random_search.fit(x_train, y_train)
# print('最好的得分是: %f' % random_search.best_score_)
# print('最好的参数是:', random_search.best_params_)
# 网格搜索
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1, return_train_score=True)
grid_search.fit(x_train, y_train)
print('最好的得分是: %f' % grid_search.best_score_)
print('最好的参数是:')
for key in grid_search.best_params_.keys():
    print('%s=%s' % (key, grid_search.best_params_[key]))
best_model = grid_search.best_estimator_
y_predict = best_model.predict(x_test)
print('准确率:', accuracy_score(y_test, y_predict))
print('召回率:', recall_score(y_test, y_predict, average='macro'))
print('F1 分数:', f1_score(y_test, y_predict, average='macro'))
'''

# joblib.dump(best_model, 'SVMmodel')
# loaded_model = joblib.load('SVMmodel.pkl')
# y_predict = loaded_model.predict(X_Test)


'''
# 加载模型预测
datasets = pd.read_csv("D:\\desk\\Odds prediction\\Datasets\\data.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:22].values
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 修正目标变量的提取
x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
loaded_model = joblib.load('SVMmodel')# 加载保存好的模型
y_predict = loaded_model.predict(x_test)
print('准确率:', accuracy_score(y_test, y_predict))
print('召回率:', recall_score(y_test, y_predict, average='macro'))
print('F1 分数:', f1_score(y_test, y_predict, average='macro'))
'''

'''
# 决策树网格搜索优化超参
datasets = pd.read_csv( "D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 17:22].values
Train_y = datasets.iloc[0:47266, 23].values
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
smote = SMOTE(random_state=1)
x_resampled, y_resampled = smote.fit_resample(Train_X, Train_y)
model = clf_tree = DecisionTreeClassifier()
clf_tree.fit(Train_X, Train_y)
param_grid = {   'max_depth': [7,8,9],
    'min_samples_split': [65],
    'min_samples_leaf': [16,17],  # 修改为合理范围
    'criterion': ['entropy'],
    'max_features': [None],
    'splitter': ['best']
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)
grid_search.fit(Train_X, Train_y)
best_params = grid_search.best_params_
print("最佳超参数:", best_params)
best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_Test)
print('准确率:', best_model.score(X_Test, Y_Test))
print('召回率:', recall_score(Y_Test, y_predict, average='macro'))
print('F1 分数:', f1_score(Y_Test, y_predict, average='macro'))
# joblib.dump(best_model, 'DT_model')
# y_predict = clf_tree.predict(X_Test)
# print('准确率',clf_tree.score(X_Test, Y_Test))
# print('recall',recall_score(Y_Test, y_predict, average='macro'))
# print('F1 score',f1_score(Y_Test, y_predict, average='macro'))

# print(model.predict([[0, 0]]))
# model.predict(Train_y)
# print(model.predict_proba([[0, 0]]))
'''

'''
# 逻辑回归堆叠模型训练
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:22].values
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 修正目标变量的提取
x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
base_models = [
    ('DT_model', joblib.load('DT_model.pkl')),
    ('GDBTModel', joblib.load('GDBTModel_optimized.pkl')),
    ('SVMmodel', joblib.load('SVMmodel.pkl')),
    ('XGBoostModel', joblib.load('XGBoostModel.pkl')),
    ('Random', joblib.load('Random.pkl')),
    ('KNN_model', joblib.load('KNN_model.pkl')),
    ('log_model', joblib.load('log_model.pkl'))
]
meta_model = LogisticRegression(max_iter=100000, random_state=1)  # 添加迭代次数和随机状态
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=5),
    verbose=3
)
stack_model.fit(x_train, y_train)
y_pred = stack_model.predict(x_test)
joblib.dump(stack_model, 'stack_model.pkl')
print('准确率:', stack_model.score(x_test, y_test))
print('召回率:', recall_score(y_test, y_pred, average='macro'))
print('F1 得分:', f1_score(y_test, y_pred, average='macro'))
print('分类报告:')
print(classification_report(y_test, y_pred))
'''

'''
# MLP神经网络堆叠模型训练

datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')
Train_X = datasets.iloc[0:47266, 2:22].values
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 确保目标变量是一维的
# 使用SMOTE处理不平衡数据
smote = SMOTE(random_state=1)
x_resampled, y_resampled = smote.fit_resample(Train_X, Train_y)

x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
base_models = [
    ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=300, weights='distance')),
    ('log_reg', LogisticRegression(C=3, penalty='l1', solver='saga', max_iter=1000, random_state=1)),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True)),
    ('decision_tree', DecisionTreeClassifier(criterion='entropy', max_depth=8,
                                             max_features=None, min_samples_leaf=10,
                                             min_samples_split=60, splitter='best', random_state=1)),
    ('random_forest', RandomForestClassifier(bootstrap=True, max_depth=15, max_features='sqrt',
                                             min_samples_leaf=4, min_samples_split=2,
                                             n_estimators=300, random_state=1)),
    ('xgb', XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=5,
                          n_estimators=200, subsample=1.0, use_label_encoder=False,
                          eval_metric='logloss')),
    ('GBDT', GradientBoostingClassifier(learning_rate=0.1, max_depth=3,
                                         n_estimators=200, subsample=0.8))
]

meta_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)  # 修改为多层感知器
# meta_model = LogisticRegression(max_iter=1000, random_state=1)
# meta_model = SVC(probability=True, kernel='linear', random_state=1)
# meta_model = RandomForestClassifier(random_state=1)
# meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=5)
)
stack_model.fit(x_train, y_train)
y_pred = stack_model.predict(x_test)
# joblib.dump(stack_model, 'stack_model_MPL.pkl')
print('准确率:', stack_model.score(x_test, y_test))
print('召回率:', recall_score(y_test, y_pred, average='macro'))
print('F1 得分:', f1_score(y_test, y_pred, average='macro'))

print('分类报告:')
print(classification_report(y_test, y_pred))
'''

'''
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# 读取数据
datasets = pd.read_csv("D:\\desk\\Odds prediction\\Datasets\\data.csv", encoding='gbk', engine='python', on_bad_lines='skip')
# 打印所有列名
print(datasets.columns)
# 假设'主场胜利=0 客场胜利=1 平局=2'列名为 'Result'
result_column = 'FTR'  # 请用实际列名替换此处
# 打印目标变量的分布
print(datasets[result_column].value_counts())
# 对需要独热编码的列进行编码
categorical_columns = ['HomeTeam', 'AwayTeam']
datasets_encoded = pd.get_dummies(datasets, columns=categorical_columns)
# 分离特征和目标变量
Train_X = datasets_encoded.drop(columns=[result_column])
Train_y = datasets_encoded[result_column]
# 归一化，标准化
scaler = MinMaxScaler()
# scaler = StandardScaler()
Train_X_scaled = scaler.fit_transform(Train_X)
# 将标准化后的数据转换回 DataFrame
Train_X_scaled_df = pd.DataFrame(Train_X_scaled, columns=Train_X.columns)
Train_y_df = pd.DataFrame(Train_y, columns=[result_column])
# 合并特征和目标变量
final_dataset = pd.concat([Train_X_scaled_df, Train_y_df], axis=1)
# 保存为新的 CSV 文件
final_dataset.to_csv("D:\\desk\\Odds prediction\\Datasets\\encode_data_normalized.csv", index=False)
print("独热编码后的数据已保存到新的 CSV 文件中。")
'''

'''
# SMOTE上采样
import pandas as pd
from imblearn.over_sampling import SMOTE
datasets = pd.read_csv("D:\\desk\\Odds prediction\\Datasets\\data_normalized.csv", encoding='gbk')
# 提取涉及对战队伍的信息
team_info = datasets.iloc[0:47266, [0, 1]]  # 假设 'HomeTeam' 在第0列, 'AwayTeam' 在第1列
# 分离特征和目标变量
Train_X = datasets.iloc[0:47266, 2:22].values  # 假设特征从第2列到第21列
Train_y = datasets.iloc[0:47266, 23:24].values.ravel()  # 目标变量在第23列
# 使用 SMOTE 进行上采样平衡类别
smote = SMOTE(random_state=1)
Train_X_resampled, Train_y_resampled = smote.fit_resample(Train_X, Train_y)
# 将上采样后的数据转换为 DataFrame，手动指定列名
Train_X_resampled_df = pd.DataFrame(Train_X_resampled, columns=datasets.columns[2:22])
Train_y_resampled_df = pd.DataFrame(Train_y_resampled, columns=[datasets.columns[23]])
# 将对战队伍信息重复以匹配上采样后的数量
team_info_resampled = pd.concat([team_info] * (len(Train_X_resampled) // len(Train_X)), ignore_index=True)
# 合并特征、目标变量和对战队伍信息
final_dataset_resampled = pd.concat([team_info_resampled, Train_X_resampled_df, Train_y_resampled_df], axis=1)
# 保存上采样后的数据集到 CSV 文件
final_dataset_resampled.to_csv("D:\\desk\\Odds prediction\\Datasets\\data_resampled_normalized.csv", index=False)
print("上采样后的数据已保存到新的 CSV 文件中。")
'''

'''
# 合并数据代码
import os
import pandas as pd
# 定义主文件夹路径
folder_path = r'D:\desk\Odds prediction\Datasets\Latest Datasets'  # 替换为你的主文件夹路径
# 创建一个空的列表用于存储每个CSV的DataFrame
csv_list = []
# 使用 os.walk 遍历主文件夹及其子文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.csv'):  # 确保只处理CSV文件
            file_path = os.path.join(root, filename)  # 获取完整的文件路径
            # 读取CSV文件并追加到列表中
            df = pd.read_csv(file_path)
            csv_list.append(df)
# 将所有CSV文件的DataFrame合并为一个
merged_df = pd.concat(csv_list, ignore_index=True)
# 保存合并后的数据为新的CSV文件
merged_df.to_csv('Latest_merged_output.csv', index=False)
print('合并完成，已保存为 Latest_merged_output.csv')
'''

'''
# 合并数据代码，并进行预处理，包括数据清洗，填充缺失值，独热编码以及上采样。
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
# 定义主文件夹路径
folder_path = r'D:\desk\Odds prediction\Datasets\Latest Datasets'  # 替换为你的主文件夹路径
# 创建一个空的列表用于存储每个CSV的DataFrame
csv_list = []
# 使用 os.walk 遍历主文件夹及其子文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.csv'):  # 确保只处理CSV文件
            file_path = os.path.join(root, filename)  # 获取完整的文件路径
            # 读取CSV文件并追加到列表中
            df = pd.read_csv(file_path)
            csv_list.append(df)
# 将所有CSV文件的DataFrame合并为一个
merged_df = pd.concat(csv_list, ignore_index=True)
# 删除不需要的列：'Div', 'Date', 'Time', 'Referee'
columns_to_drop = ['Div', 'Date', 'Time', 'Referee']
merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
# 将 'FTR'（全场比赛结果） 和 'HTR'（半场比赛结果） 映射为数值型数据
# 'H' -> 0 (主队胜利), 'A' -> 1 (客队胜利), 'D' -> 2 (平局)
result_mapping = {'H': 0, 'A': 1, 'D': 2}
merged_df['FTR'] = merged_df['FTR'].map(result_mapping)  # 比赛结果 (目标)
merged_df['HTR'] = merged_df['HTR'].map(result_mapping)  # 半场结果 (特征)
# 对比赛队伍进行独热编码 (One-Hot Encoding)
teams_encoded = pd.get_dummies(merged_df[['HomeTeam', 'AwayTeam']], drop_first=True).astype(int)
merged_df = pd.concat([merged_df, teams_encoded], axis=1)
# 删除编码后的原始 'HomeTeam' 和 'AwayTeam' 列
merged_df.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
# 对缺失值的行进行填充，使用一个无意义的值，例如 '-999' 或 'Unknown'
for col in merged_df.columns:
    if merged_df[col].dtype == 'object':  # 如果是类别型数据
        merged_df[col].fillna('Unknown', inplace=True)
    else:  # 如果是数值型数据
        merged_df[col].fillna(-999, inplace=True)
# 进行数据归一化
# 选择需要归一化的特征，排除 'FTR' 列和 'HTR' 列
scaler = MinMaxScaler()
# 获取独热编码生成的列名
encoded_columns = teams_encoded.columns.tolist()
# 选择数值型特征，排除 FTR, HTR 和独热编码的列
columns_to_scale = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
columns_to_scale = [col for col in columns_to_scale if col not in ['FTR', 'HTR'] + encoded_columns]  # 移除不需要归一化的列
# 应用归一化
merged_df[columns_to_scale] = scaler.fit_transform(merged_df[columns_to_scale])
# 进行 SMOTE 上采样以平衡不平衡数据
# 假设 'FTR' 列是目标变量（比赛最终结果）
X = merged_df.drop('FTR', axis=1)  # 特征
y = merged_df['FTR']  # 目标变量
# SMOTE 上采样处理不平衡问题
smote = SMOTE(random_state=1)
X_resampled, y_resampled = smote.fit_resample(X, y)
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=1)
# 模型选择
model = RandomForestClassifier(random_state=1)
# 模型训练
model.fit(X_train, y_train)
# 进行预测
y_pred = model.predict(X_test)
print('准确率:', model.score(X_test, y_test))
print('召回率:', recall_score(y_test, y_pred, average='macro'))
print('F1 得分:', f1_score(y_test, y_pred, average='macro'))
# 合并上采样后的数据
resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['FTR'])], axis=1)
# 保存处理后的数据为新的CSV文件
resampled_df.to_csv('Processed_merged,_SMOTE_output.csv' index=False)
print('数据预处理完成并进行了 SMOTE 处理，已保存为 Processed_merged_SMOTE_output.csv')
'''


# 加载处理后的数据并进行训练
processed_data_path = "D:\desk\Odds prediction\Datasets\Processed_merged_SMOTE_output.csv"  # 替换为你的文件路径
merged_df = pd.read_csv(processed_data_path)
# 假设 'FTR' 列是目标变量（比赛最终结果）
X = merged_df.drop('FTR', axis=1)  # 特征
y = merged_df['FTR']  # 目标变量
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# 模型选择
model = RandomForestClassifier(random_state=1)
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
param_dist = {
    'n_estimators': [350, 400],
    'max_features': ['sqrt'],
    'max_depth': [24, 27],
    'min_samples_split': [3],
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
joblib.dump(train_columns, 'train_columns.pkl')  # 保存特征名称
# 读取新的数据集进行验证
processed_data = "D:\desk\Odds prediction\Datasets\Processed_merged_SMOTE_output_2024.csv"  # 替换为你的文件路径
merged_df = pd.read_csv(processed_data)
# 假设 'FTR' 列是目标变量（比赛最终结果）
X_new = merged_df.drop('FTR', axis=1)  # 特征
y_true = merged_df['FTR']  # 真实的目标变量
# 加载训练好的模型
model = joblib.load('Random.pkl')
# 加载训练时的特征列
train_columns = joblib.load('train_columns.pkl')
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
# 绘制热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.show()






'''
datasets = pd.read_csv("D:\desk\Odds prediction\Datasets\data_normalized.csv", encoding='gbk', engine='python', on_bad_lines='skip')

Train_X = datasets.iloc[:, 2:22].values
Train_y = datasets.iloc[:, 23:24].values.ravel()
# x_train, x_test, y_train, y_test = train_test_split(Train_X, Train_y, test_size=0.1, random_state=1)
smote = SMOTE(random_state=1)
x_resampled, y_resampled = smote.fit_resample(Train_X, Train_y)
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.1, random_state=1)

base_models = [
    ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=300, weights='distance')),
    ('log_reg', LogisticRegression(C=3, penalty='l1', solver='saga', max_iter=1000, random_state=1)),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True)),
    ('decision_tree', DecisionTreeClassifier(criterion='entropy', max_depth=8,
                                             max_features=None, min_samples_leaf=10,
                                             min_samples_split=60, splitter='best', random_state=1)),
    ('random_forest', RandomForestClassifier(bootstrap=True, max_depth=15, max_features='sqrt',
                                             min_samples_leaf=4, min_samples_split=2,
                                             n_estimators=300, random_state=1)),
    ('xgb', XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=5,
                          n_estimators=200, subsample=1.0, use_label_encoder=False,
                          eval_metric='logloss')),
    ('GBDT', GradientBoostingClassifier(learning_rate=0.1, max_depth=3,
                                         n_estimators=200, subsample=0.8))
]
# 定义元模型（可以替换）
# meta_model = LogisticRegression(max_iter=1000, random_state=1)
# meta_model = SVC(probability=True, kernel='linear', random_state)
meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)
# meta_model = KNeighborsClassifier()
# meta_model = RandomForestClassifier(random_state=1)
# meta_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)
# meta_model = GradientBoostingClassifier()
# 定义堆叠模型
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=5)
)
stack_model.fit(x_train, y_train)
# 预测
y_pred = stack_model.predict(x_test)
# 保存模型
# joblib.dump(stack_model, 'stack_model_SVC.pkl')
print('准确率:', stack_model.score(x_test, y_test))
print('召回率:', recall_score(y_test, y_pred, average='macro'))
print('F1 得分:', f1_score(y_test, y_pred, average='macro'))
# 打印分类报告
print('分类报告:')
print(classification_report(y_test, y_pred))
'''







