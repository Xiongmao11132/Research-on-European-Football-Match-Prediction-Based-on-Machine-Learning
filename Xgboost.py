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
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots



from imblearn.over_sampling import SMOTE


# 加载和处理数据
# processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
# merged_df = pd.read_csv(processed_data_path)
#
# # 确保特征名称没有不合法字符
# merged_df.columns = merged_df.columns.str.replace('<', '')
# columns_to_drop = ['FTR']  # 全部数据
# # columns_to_drop = ['HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTR']#只有赔率
# # columns_to_drop = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'SBH', 'SBD', 'SBA', 'WHH', 'WHD', 'WHA', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'GBAHH', 'GBAHA', 'GBAH', 'LBAHH', 'LBAHA', 'LBAH', 'B365AHH', 'B365AHA', 'B365AH', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BSH', 'BSD', 'BSA', 'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'BFH', 'BFD', 'BFA', 'BFEH', 'BFED', 'BFEA', 'BFE>2.5', 'BFE<2.5', 'BFEAHH', 'BFEAHA', 'BFCH', 'BFCD', 'BFCA', '1XBCH', '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5', 'BFECAHH', 'BFECAHA', 'FTR']#只有比赛数据
# X = merged_df.drop(columns_to_drop, axis=1)  # 特征
# y = merged_df['FTR']  # 目标变量
#
# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
#
# # 模型选择
# model = XGBClassifier(random_state=1)
#
# # 参数分布
# param_dist = {
#     'max_depth': [20,30,40],                  # 控制树的最大深度，直接影响模型复杂度
#     'learning_rate': [0.1],       # 学习率，对收敛速度和准确率影响较大
#     'n_estimators': [240,250,260],           # 基学习器数量，与学习率配合影响模型表现
#     'min_child_weight': [1],            # 控制叶子节点的最小权重和，平衡复杂度和准确率
#     'subsample': [1.0],                  # 样本采样比例，适当调整提升泛化能力
#     'colsample_bytree': [0.5],            # 特征采样比例，控制每棵树使用的特征数量
#
# }
#
#
# random_search = RandomizedSearchCV(estimator=model,
#                                    param_distributions=param_dist,
#                                    cv=3,
#                                    verbose=2,
#                                    random_state=1,
#                                    n_jobs=-1)
#
# # 模型训练
# random_search.fit(X_train, y_train)
# print("最佳参数:", random_search.best_params_)
# best_model = random_search.best_estimator_
#
# # 进行预测
# y_pred = best_model.predict(X_test)
#
# # 保存模型
# joblib.dump(best_model, 'XGBoost_model.pkl')
#
# # 输出特征重要性
# feature_importances = best_model.feature_importances_
# importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
#
# top_features = importance_df.head(20)
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
# plt.title('Top 20 Feature Importances', fontsize=16)
# plt.xlabel('Importance', fontsize=14)
# plt.ylabel('Features', fontsize=14)
# plt.tight_layout()
# plt.savefig("Xgboost_Features_importance")
# plt.show()
#
# print('前20个最重要的特征:')
# print(importance_df.head(20))
#
# # 输出模型评估指标
# print('准确率:', best_model.score(X_test, y_test))
# print('召回率:', recall_score(y_test, y_pred, average='macro'))
# print('F1 得分:', f1_score(y_test, y_pred, average='macro'))
#
# # 保存训练时的特征列
# train_columns = X_train.columns
# joblib.dump(train_columns, 'train_columns_Xgboost.pkl')

# 读取新的数据集进行验证
processed_data = r"D:\desk\Oddsprediction\Datasets\Processed_merged_output_2024.csv"
merged_df = pd.read_csv(processed_data)
merged_df.columns = merged_df.columns.astype(str).str.replace('<', '')  # 确保特征名称是字符串
X_new = merged_df.drop('FTR', axis=1)
y_true = merged_df['FTR']

# 加载训练好的模型
model = joblib.load('XGBoost_model.pkl')
train_columns = joblib.load('train_columns_Xgboost.pkl')
X_new = X_new.reindex(columns=train_columns, fill_value=0)

# # 假设我们要预测新数据集中的第7场比赛（索引为2，因为索引从0开始），你可以根据实际需求修改这里的索引值
# specific_game_index = 6
# specific_game_data = X_new.iloc[specific_game_index].values.reshape(1, -1)
# # 对特定场比赛数据进行预测
# specific_game_prediction = model.predict(specific_game_data)
# specific_game_prediction_proba = model.predict_proba(specific_game_data)
# # 输出特定场比赛的预测结果
# print(f"对第 {specific_game_index + 1} 场比赛的预测结果: {specific_game_prediction[0]}")
# print(f"对第 {specific_game_index + 1} 场比赛的预测结果概率分布: {specific_game_prediction_proba[0]}")
# 对新的数据集进行预测
y_pred = model.predict(X_new)
# 输出每一场比赛的预测概率
# y_pred_proba = model.predict_proba(X_new)
# result_dict = {0: "主队胜", 1: "客队胜", 2: "平局"}
# for i, game_proba in enumerate(y_pred_proba):
#     print(f"对第 {i + 1} 场比赛的预测结果概率分布:")
#     for j, p in enumerate(game_proba):
#         print(f"    {result_dict[j]}: {p}")
# 模拟 400+ 场比赛的预测概率数据

y_pred_proba = model.predict_proba(X_new)
result_dict = {0: "主队胜", 1: "客队胜", 2: "平局"}
game_ids = [f"比赛 {i + 1}" for i in range(len(y_pred_proba))]
proba_df = pd.DataFrame(y_pred_proba, columns=[result_dict[i] for i in range(len(result_dict))])
proba_df['比赛'] = game_ids
proba_df['比赛编号'] = range(1, len(proba_df) + 1)

# 创建 Dash 应用
app = Dash(__name__)

# Layout
app.layout = html.Div([
    # 动态背景
    html.Div(className="background-effects"),

    # 标题部分
    html.Div([
        html.H1("比赛预测概率分布",
                style={
                    'textAlign': 'center',
                    'fontFamily': 'Orbitron, sans-serif',
                    'color': '#FFFFFF',
                    'textShadow': '0 0 30px rgba(255, 255, 255, 0.8)',
                    'fontSize': '40px',
                    'padding': '20px 0'}),
    ], style={
        'backgroundColor': 'rgba(0, 0, 0, 0.7)',  # 更深的透明背景
        'borderRadius': '15px',
        'marginBottom': '20px',
        'boxShadow': '0 0 20px rgba(255, 255, 255, 0.3)',
        'border': '1px solid rgba(255, 255, 255, 0.3)',
        'width': '80%',
        'margin': '0 auto',
    }),

    # 图表部分
    html.Div([
        dcc.Graph(id='multi-bar-chart', config={'displayModeBar': True, 'scrollZoom': True}),
    ], style={
        'width': '90%',
        'margin': '0 auto',
        'padding': '20px',
        'backgroundColor': 'rgba(255, 255, 255, 0.1)',
        'borderRadius': '20px',
        'boxShadow': '0 0 15px rgba(0, 229, 255, 0.3)',
        'backdropFilter': 'blur(10px)',
        'color': '#fff',
        'border': '1px solid rgba(255, 255, 255, 0.2)'
    }),

    # 滑块部分
    html.Div([
        dcc.Slider(
            id='game-slider',
            min=1,
            max=len(proba_df),
            value=1,
            marks={i: f"比赛 {i}" for i in range(1, len(proba_df) + 1, 50)},
            step=1,
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag',
            included=False,
        ),
    ], style={
        'marginTop': '30px',
        'width': '90%',
        'margin': '0 auto 40px',
        'border': '1px solid rgba(255, 255, 255, 0.3)',
        'borderRadius': '20px',
        'boxShadow': '0 0 10px rgba(0, 229, 255, 0.3)',
        'color': '#00E5FF',
        'fontSize': '16px'
    }),

    # 数据表部分
    dash_table.DataTable(
        id='prediction-table',
        columns=[
            {"name": "比赛", "id": "比赛"},
            {"name": "主队胜", "id": "主队胜"},
            {"name": "客队胜", "id": "客队胜"},
            {"name": "平局", "id": "平局"},
        ],
        style_table={
            'height': '300px',
            'overflowY': 'auto',
            'backgroundColor': 'rgba(255, 255, 255, 0.1)',
            'color': '#fff',
            'width': '90%',
            'margin': '0 auto',
            'boxShadow': '0 0 10px rgba(0, 229, 255, 0.3)'
        },
        style_header={
            'backgroundColor': 'rgba(0, 0, 0, 0.7)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': 'rgba(0, 0, 0, 0.5)',
            'color': '#fff',
            'textAlign': 'center'
        }
    )
], style={
    'background': 'linear-gradient(to bottom, #1A237E, #0D47A1)',
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif',
    'position': 'relative'
}, className="main-layout")

# 动态背景 CSS 样式
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        .background-effects {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(26, 35, 126, 0.8), rgba(13, 71, 161, 0.8));
            z-index: -1;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# 回调函数
@app.callback(
    [Output('multi-bar-chart', 'figure'),
     Output('prediction-table', 'data')],
    [Input('game-slider', 'value')]
)
def update_chart(game_id):
    # 确定范围
    start_game = max(0, game_id - 1)
    end_game = min(start_game + 5, len(proba_df))  # 每次最多显示5场比赛

    # 更新图表
    fig = make_subplots(
        rows=1, cols=end_game - start_game,
        subplot_titles=[f"比赛 {i + 1}" for i in range(start_game, end_game)],
        shared_yaxes=True
    )

    for i, game in enumerate(range(start_game, end_game)):
        game_proba = proba_df.iloc[game]
        categories = list(result_dict.values())
        probabilities = game_proba.iloc[:len(categories)]  # 明确使用 iloc

        fig.add_trace(
            go.Bar(
                x=categories,
                y=probabilities,
                marker_color = ['#FF9500', '#18AFFF', '#39FF14'],
                name=f"比赛 {game + 1}",
                hoverinfo='x+y',
                hoverlabel=dict(bgcolor="rgba(0, 229, 255, 0.8)", font_size=16, font_family="Arial")
            ),
            row=1, col=i + 1
        )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(255, 255, 255, 0.1)',
        paper_bgcolor='rgba(255, 255, 255, 0.1)',
        font=dict(family='Arial, sans-serif', size=14, color='#fff'),
        margin=dict(t=40, b=40, l=40, r=40),
        height=500,
        hovermode='closest',
        title_x=0.5,
        title_y=0.95
    )

    # 表格数据
    table_data = []
    for game in range(start_game, end_game):
        game_proba = proba_df.iloc[game]
        table_data.append({
            "比赛": f"比赛 {game + 1}",
            "主队胜": game_proba.iloc[0],
            "客队胜": game_proba.iloc[1],
            "平局": game_proba.iloc[2]
        })

    return fig, table_data




# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)

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
