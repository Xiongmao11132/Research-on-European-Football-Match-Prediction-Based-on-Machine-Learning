import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import joblib

# 设置并行数
n_jobs = -1  # 使用所有可用的CPU核心

# 加载和处理数据
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    if any(char in col for col in df.columns for char in ['[', ']', '<']):
        df.columns = df.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')
    return df

# 定义和训练基础模型
def train_base_model(model, X, y):
    model.fit(X, y)
    return model

# 主要处理流程
def main():
    processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
    merged_df = load_and_process_data(processed_data_path)
    X = merged_df.drop('FTR', axis=1)  # 特征
    y = merged_df['FTR']  # 目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

    base_models = [
        ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=20, weights='distance', n_jobs=n_jobs)),
        ('log_reg', SGDClassifier(loss='log_loss', penalty='l2', alpha=1/100, max_iter=1000, random_state=1, n_jobs=n_jobs)),
        ('svm', SVC(kernel='linear', C=1, probability=True, random_state=1)),
        ('decision_tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=9,
                                                 max_features=None, min_samples_leaf=12,
                                                 min_samples_split=90, splitter='best', random_state=1)),
        ('random_forest', RandomForestClassifier(n_estimators=700, min_samples_split=2,
                                                 min_samples_leaf=1, max_features='sqrt',
                                                 max_depth=46, bootstrap=True, random_state=1, n_jobs=n_jobs, warm_start=True)),
        ('xgb', XGBClassifier(subsample=1.0, n_estimators=250, min_child_weight=1, max_depth=20,
                              learning_rate=0.1, colsample_bytree=0.5, n_jobs=n_jobs,
                              tree_method='hist')),
        ('GBDT', GradientBoostingClassifier(learning_rate=0.1, max_depth=30,
                                             n_estimators=440, subsample=0.8, warm_start=True))
    ]

    # 并行训练基础模型
    trained_models = Parallel(n_jobs=n_jobs)(delayed(train_base_model)(model, X_train, y_train) for _, model in base_models)

    # 更新base_models列表
    base_models = [(name, model) for (name, _), model in zip(base_models, trained_models)]

    # 定义和训练堆叠模型
    meta_model = XGBClassifier(eval_metric='logloss', random_state=1, n_jobs=n_jobs, tree_method='hist')
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5),
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
    X_new = X_new.reindex(columns=train_columns, fill_value=0)

    # 使用堆叠模型预测新数据
    y_pred_new = stack_model.predict(X_new)
    print('新数据准确率:', stack_model.score(X_new, y_true))
    print('新数据召回率:', recall_score(y_true, y_pred_new, average='macro'))
    print('新数据F1得分:', f1_score(y_true, y_pred_new, average='macro'))

    # 混淆矩阵绘制
    cm = confusion_matrix(y_true, y_pred_new)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stack_model.classes_, yticklabels=stack_model.classes_)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('新数据混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()
