import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
# # 设置并行参数
# n_jobs = -1
#
# # 加载和处理数据
# processed_data_path = "/data/coding/Processed_merged_SMOTE_output.csv"
# merged_df = pd.read_csv(processed_data_path)
# if any(char in col for col in merged_df.columns for char in ['[', ']', '<']):
#     merged_df.columns = merged_df.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')
# columns_to_drop = ['FTR']  # 全部数据
# columns_to_drop = ['HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTR']#只有赔率
# columns_to_drop = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'SBH', 'SBD', 'SBA', 'WHH', 'WHD', 'WHA', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'GBAHH', 'GBAHA', 'GBAH', 'LBAHH', 'LBAHA', 'LBAH', 'B365AHH', 'B365AHA', 'B365AH', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BSH', 'BSD', 'BSA', 'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'BFH', 'BFD', 'BFA', 'BFEH', 'BFED', 'BFEA', 'BFE>2.5', 'BFE<2.5', 'BFEAHH', 'BFEAHA', 'BFCH', 'BFCD', 'BFCA', '1XBCH', '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5', 'BFECAHH', 'BFECAHA', 'FTR']#只有比赛数据
# X = merged_df.drop(columns_to_drop, axis=1)  # 特征
# y = merged_df['FTR']  # 目标变量
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/desk/temp_joblib'
# # 定义基础模型和堆叠模型
# base_models = [
#     ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=20, weights='distance', n_jobs=n_jobs)),
#     ('log_reg', LogisticRegression(C=100, penalty='l2', solver='newton-cg', max_iter=1000000, random_state=1, n_jobs=n_jobs)),
#     ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True)),  # SVC不支持n_jobs参数
#     ('decision_tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=9,
#                                              max_features=None, min_samples_leaf=12,
#                                              min_samples_split=90, splitter='best', random_state=1)),
#     ('random_forest', RandomForestClassifier(n_estimators=700, min_samples_split=2,
#                                              min_samples_leaf=1, max_features='sqrt',
#                                              max_depth=46, bootstrap=True, random_state=1, n_jobs=n_jobs)),
#     ('xgb', XGBClassifier(subsample=1.0, n_estimators=250, min_child_weight=1, max_depth=20,
#                           learning_rate=0.1, colsample_bytree=0.5, n_jobs=n_jobs)),
#     ('GBDT', GradientBoostingClassifier(learning_rate=0.1, max_depth=30,
#                                          n_estimators=440, subsample=0.8)),
#     ('lightgbm', LGBMClassifier(learning_rate=0.2, n_estimators=200, num_leaves=50, n_jobs=n_jobs))
# ]
#
# meta_model = GradientBoostingClassifier()
# stack_model = StackingClassifier(
#     estimators=base_models,
#     final_estimator=meta_model,
#     cv=KFold(n_splits=5),
#     n_jobs=n_jobs  # 并行化堆叠模型的训练
# )
#
# # 训练堆叠模型
# stack_model.fit(X_train, y_train)
# y_pred_stack = stack_model.predict(X_test)
# joblib.dump(stack_model, 'stack_model_stack_GBDT.pkl')
#
# # 输出堆叠模型评估指标
# print('堆叠模型准确率:', stack_model.score(X_test, y_test))
# print('堆叠模型召回率:', recall_score(y_test, y_pred_stack, average='macro'))
# print('堆叠模型F1得分:', f1_score(y_test, y_pred_stack, average='macro'))
# print('堆叠模型分类报告:')
# print(classification_report(y_test, y_pred_stack))
#
# train_columns = X_train.columns
# joblib.dump(train_columns, 'train_columns_stack_GBDT.pkl')

# 新数据集验证
new_data_path = r"D:\desk\Oddsprediction\Datasets\Processed_merged_output_Bundesliga1.csv"
merged_df_new = pd.read_csv(new_data_path)
if any(char in col for col in merged_df_new.columns for char in ['[', ']', '<']):
    merged_df_new.columns = merged_df_new.columns.astype(str).str.replace('[', '').str.replace(']', '').str.replace('<', '')

X_new = merged_df_new.drop('FTR', axis=1)
y_true = merged_df_new['FTR']
train_columns = joblib.load('train_columns_stack_GBDT_best.pkl')
X_new = X_new.reindex(columns=train_columns, fill_value=0)

# 使用堆叠模型预测新数据
stack_model = joblib.load('stack_model_stack_GBDT_best.pkl')
y_pred_new = stack_model.predict(X_new)

y_pred_proba = stack_model.predict_proba(X_new)
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
plt.title('混淆矩阵')
plt.savefig("Stack_model_GBDT_Bundesliga1.png")
plt.show()

