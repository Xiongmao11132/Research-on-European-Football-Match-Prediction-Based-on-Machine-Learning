import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import chardet
import numpy as np
import joblib

folder_path = r'D:/desk/Oddsprediction/Datasets/2024106_2025316_Serie A'
# 遍历文件夹，读取所有 CSV 文件
new_data_list = []
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(root, filename)
            with open(file_path, 'rb') as f:
                rawdata = f.read()
                encoding = chardet.detect(rawdata)['encoding']
                print(f"检测到文件: {file_path} 使用编码: {encoding}")
            try:
                new_df = pd.read_csv(file_path, encoding=encoding, engine='python', on_bad_lines='skip')
                new_data_list.append(new_df)
                print(f"成功读取新数据文件: {file_path}")
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                print(f"无法读取新数据文件: {file_path}，错误: {e}")

if new_data_list:
    # 合并新数据
    new_data = pd.concat(new_data_list, ignore_index=True)
    # 删除无关列
    columns_to_drop = ['Div', 'Date', 'Time', 'Referee', 'BbAH', 'BbAHh',
                       '1xBet', '1XBH', '1XBD', '1XBA', 'FTHG', 'FTAG']
    new_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    # 删除 Unnamed 列
    new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]
    # 映射 FTR 和 HTR
    result_mapping = {'H': 0, 'A': 1, 'D': 2}
    new_data['FTR'] = new_data['FTR'].map(result_mapping)
    new_data['HTR'] = new_data['HTR'].map(result_mapping)
    # 对队伍进行独热编码
    new_teams_encoded = pd.get_dummies(new_data[['HomeTeam', 'AwayTeam']], drop_first=False).astype(int)
    # 加载保存的 One-Hot 编码列名
    onehot_columns = joblib.load('onehot_columns.pkl')
    # 确保独热编码的列名与 onehot_columns 对应
    new_teams_encoded = new_teams_encoded.reindex(columns=onehot_columns, fill_value=0)
    # 替换缺失值并填充
    new_data.replace('-', np.nan, inplace=True)
    new_data.dropna(subset=['FTR'], inplace=True)  # 确保 FTR 不为空
    new_data.fillna(-999, inplace=True)
    # 加载 MinMaxScaler 和需要归一化的列名
    scaler = joblib.load('scaler.pkl')
    scaler_columns = joblib.load('scaler_columns.pkl')
    # 从新数据集中排除 FTR 和 HTR 列，确保它们不会被归一化
    numeric_columns = [col for col in scaler_columns if col not in ['FTR', 'HTR']]
    # 补全缺失的数值列，并确保列顺序一致
    for col in numeric_columns:
        if col not in new_data.columns:
            new_data[col] = -999  # 用 -999 填充缺失列
    # 将需要归一化的数值列进行归一化
    new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])
    # 合并独热编码后的数据，并删除原始队伍列，最后将独热编码的列放在最后
    new_data = pd.concat([new_data.drop(['HomeTeam', 'AwayTeam'], axis=1), new_teams_encoded], axis=1)
    # 保存预处理后的文件
    new_data.to_csv('Processed_merged_output_Serie A.csv', index=False, encoding='utf-8')
    print('新数据集预处理完成并进行归一化处理，已保存为 Processed_merged_output_Serie A.csv')
    # 使用 SMOTE 进行上采样
    X_new = new_data.drop(['FTR'], axis=1)
    y_new = new_data['FTR']
    smote = SMOTE(random_state=40)
    X_new_resampled, y_new_resampled = smote.fit_resample(X_new, y_new)
    # 合并上采样后的数据
    new_data_resampled = pd.concat([pd.DataFrame(X_new_resampled, columns=X_new.columns),
                                    pd.DataFrame(y_new_resampled, columns=['FTR'])], axis=1)

    # 保存上采样后的数据
    new_data_resampled.to_csv('Processed_merged_SMOTE_output_Serie A.csv', index=False, encoding='utf-8')
    print('新数据集预处理完成并进行 SMOTE 处理，已保存为 Processed_merged_SMOTE_Serie A.csv')

else:
    print("没有成功读取任何新数据文件。")