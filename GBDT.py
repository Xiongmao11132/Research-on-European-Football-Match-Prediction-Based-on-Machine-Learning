import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载和处理数据
processed_data_path = r"D:\desk\Oddsprediction\Datasets\Processed_merged_output_2024.csv"
merged_df = pd.read_csv(processed_data_path)
# columns_to_drop = ['FTR']  # 全部数据
# columns_to_drop = ['HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTR']#只有赔率
columns_to_drop = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'SBH', 'SBD', 'SBA', 'WHH', 'WHD', 'WHA', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'GBAHH', 'GBAHA', 'GBAH', 'LBAHH', 'LBAHA', 'LBAH', 'B365AHH', 'B365AHA', 'B365AH', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BSH', 'BSD', 'BSA', 'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'BFH', 'BFD', 'BFA', 'BFEH', 'BFED', 'BFEA', 'BFE>2.5', 'BFE<2.5', 'BFEAHH', 'BFEAHA', 'BFCH', 'BFCD', 'BFCA', '1XBCH', '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5', 'BFECAHH', 'BFECAHA', 'FTR']#只有比赛数据
X = merged_df.drop(columns_to_drop, axis=1)  # 特征
y = merged_df['FTR']  # 目标变量

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# 模型选择，使用 GradientBoostingClassifier
model = GradientBoostingClassifier()
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
# 参数分布
param_grid = {
    'learning_rate': [0.1],
    'max_depth': [25],
    'n_estimators': [450],
    'subsample': [0.8]
}

# 网格搜索
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=3,
                           verbose=2,
                           n_jobs=-1)

# 模型训练
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 进行预测
y_pred = best_model.predict(X_test)

# 保存模型
joblib.dump(best_model, 'GBDT_model.pkl')

# 输出模型评估指标
print('准确率:', best_model.score(X_test, y_test))
print('召回率:', recall_score(y_test, y_pred, average='macro'))
print('F1 得分:', f1_score(y_test, y_pred, average='macro'))

# 保存训练时的特征列
train_columns = X_train.columns
joblib.dump(train_columns, 'train_columns_GBDT.pkl')

# 读取新的数据集进行验证
processed_data = r"D:\desk\Oddsprediction\Datasets\Processed_merged_output_2024.csv"
# 同样明确指定编码格式为UTF-8进行文件读取
merged_df = pd.read_csv(processed_data, encoding='UTF-8')
X_new = merged_df.drop('FTR', axis=1)
y_true = merged_df['FTR']

# 加载训练好的模型
model = joblib.load('GBDT_model.pkl')
train_columns = joblib.load('train_columns_GBDT.pkl')
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








# import os
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 加载和处理数据
# processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
# merged_df = pd.read_csv(processed_data_path)
# X = merged_df.drop('FTR', axis=1)  # 特征
# y = merged_df['FTR']  # 目标变量

# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# # 模型选择，使用 GradientBoostingClassifier
# model = GradientBoostingClassifier()

# # 模型训练
# model.fit(X_train, y_train)

# # 进行预测
# y_pred = model.predict(X_test)

# # 保存模型
# joblib.dump(model, 'GBDT_model.pkl')

# # 输出模型评估指标
# print('准确率:', model.score(X_test, y_test))
# print('召回率:', recall_score(y_test, y_pred, average='macro'))
# print('F1 得分:', f1_score(y_test, y_pred, average='macro'))

# # 保存训练时的特征列
# train_columns = X_train.columns
# joblib.dump(train_columns, 'train_columns_GBDT.pkl')

# # 读取新的数据集进行验证
# processed_data = "/data/coding/Processed_merged_SMOTE_output_2024.csv"
# merged_df = pd.read_csv(processed_data)
# X_new = merged_df.drop('FTR', axis=1)
# y_true = merged_df['FTR']

# # 加载训练好的模型
# model = joblib.load('GBDT_model.pkl')
# train_columns = joblib.load('train_columns_GBDT.pkl')
# X_new = X_new.reindex(columns=train_columns, fill_value=0)

# # 对新的数据集进行预测
# y_pred = model.predict(X_new)

# # 输出模型评估指标
# print('准确率:', model.score(X_new, y_true))
# print('召回率:', recall_score(y_true, y_pred, average='macro'))
# print('F1 得分:', f1_score(y_true, y_pred, average='macro'))

# # 计算混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.ylabel('真实标签')
# plt.xlabel('预测标签')
# plt.title('混淆矩阵')
# plt.show()
