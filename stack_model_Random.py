import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

# 加载和处理数据
processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
merged_df = pd.read_csv(processed_data_path)
if any(char in col for col in merged_df.columns for char in ['[', ']', '<']):
    merged_df.columns = merged_df.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')

X = merged_df.drop('FTR', axis=1)  # 特征
y = merged_df['FTR']  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# 设置 joblib 临时文件夹
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'

# 定义基础模型和堆叠模型
base_models = [
    ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=20, weights='distance', n_jobs=-1)),
    ('log_reg', LogisticRegression(C=100, penalty='l2', solver='newton-cg', max_iter=1000000, random_state=1, n_jobs=-1)),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True)),  # SVC 不支持 n_jobs 参数
    ('decision_tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=9,
                                             max_features=None, min_samples_leaf=12,
                                             min_samples_split=90, splitter='best', random_state=1)),
    ('random_forest', RandomForestClassifier(n_estimators=700, min_samples_split=2,
                                             min_samples_leaf=1, max_features='sqrt',
                                             max_depth=46, bootstrap=True, random_state=1, n_jobs=-1)),
    ('xgb', XGBClassifier(subsample=1.0, n_estimators=250, min_child_weight=1, max_depth=20,
                          learning_rate=0.1, colsample_bytree=0.5, n_jobs=-1)),
    ('GBDT', GradientBoostingClassifier(learning_rate=0.1, max_depth=30,
                                         n_estimators=440, subsample=0.8)),
    ('lightgbm', LGBMClassifier(learning_rate=0.2, n_estimators=200, num_leaves=50, n_jobs=-1))
]

meta_model = RandomForestClassifier(random_state=1, n_jobs=-1)

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=5),
    n_jobs=-1  # 启用堆叠模型的并行性
)

# 训练堆叠模型
stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)
joblib.dump(stack_model, 'stack_model_stack_Random.pkl')

# 输出堆叠模型评估指标
print('堆叠模型准确率:', stack_model.score(X_test, y_test))
print('堆叠模型召回率:', recall_score(y_test, y_pred_stack, average='macro'))
print('堆叠模型F1得分:', f1_score(y_test, y_pred_stack, average='macro'))
print('堆叠模型分类报告:')
print(classification_report(y_test, y_pred_stack))

# 保存特征列
train_columns = X_train.columns
joblib.dump(train_columns, 'train_columns_stack_Random.pkl')

# 新数据集验证
new_data_path = "/data/coding/Processed_merged_SMOTE_output_2024.csv"
merged_df_new = pd.read_csv(new_data_path)
if any(char in col for col in merged_df_new.columns for char in ['[', ']', '<']):
    merged_df_new.columns = merged_df_new.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')

X_new = merged_df_new.drop('FTR', axis=1)
y_true = merged_df_new['FTR']
train_columns = joblib.load('train_columns_stack_Random.pkl')
X_new = X_new.reindex(columns=train_columns, fill_value=0)

# 使用堆叠模型预测新数据
stack_model = joblib.load('stack_model_stack_Random.pkl')
y_pred_new = stack_model.predict(X_new)
print('新数据准确率:', stack_model.score(X_new, y_true))
print('新数据召回率:', recall_score(y_true, y_pred_new, average='macro'))
print('新数据F1得分:', f1_score(y_true, y_pred_new, average='macro'))

# 混淆矩阵绘制
cm = confusion_matrix(y_true, y_pred_new)
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
plt.rcParams['font.sans-serif'] = [font_path]  # 指定中文字体路径
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stack_model.classes_, yticklabels=stack_model.classes_)
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.show()
plt.savefig("Stack_model_Random.png")
