import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import chardet
import numpy as np
import joblib

# 检测并读取所有CSV文件
folder_path = r'D:\desk\Oddsprediction\Datasets\New Data'
csv_list = []
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(root, filename)
            with open(file_path, 'rb') as f:
                rawdata = f.read()
                encoding = chardet.detect(rawdata)['encoding']
                print(f"检测到文件: {file_path} 使用编码: {encoding}")
            try:
                df = pd.read_csv(file_path, encoding=encoding, engine='python', on_bad_lines='skip')
                csv_list.append(df)
                print(f"成功读取文件: {file_path}")
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                print(f"无法读取文件: {file_path}，错误: {e}")
if csv_list:
    merged_df = pd.concat(csv_list, ignore_index=True)
    # 删除无关列
    columns_to_drop = ['Div', 'Date', 'Time', 'Referee', 'BbAH', 'BbAHh',
                       '1xBet', '1XBH', '1XBD', '1XBA', 'FTHG', 'FTAG']
    merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    # 删除所有 Unnamed 列
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
    # 映射比赛结果（FTR）和半场结果（HTR）
    result_mapping = {'H': 0, 'A': 1, 'D': 2}
    merged_df['FTR'] = merged_df['FTR'].map(result_mapping)
    merged_df['HTR'] = merged_df['HTR'].map(result_mapping)
    # 对主客队进行独热编码并保存编码列名
    teams_encoded = pd.get_dummies(merged_df[['HomeTeam', 'AwayTeam']], drop_first=False).astype(int)
    joblib.dump(teams_encoded.columns, 'onehot_columns.pkl')  # 保存编码列名
    print('已保存 One-Hot 编码的列名：onehot_columns.pkl')
    # 合并编码数据并删除原始列
    merged_df = pd.concat([merged_df, teams_encoded], axis=1)
    merged_df.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
    # 替换缺失值为 -999
    merged_df.replace(['-', '', 'None', 'NaN', np.nan], np.nan, inplace=True)
    merged_df.dropna(subset=['FTR', 'HTR'], inplace=True)  # 删除关键列缺失的数据
    merged_df.fillna(-999, inplace=True)
    # 保存预处理后的文件
    merged_df.to_csv('Processed_output.csv', index=False, encoding='utf-8')
    print('数据预处理完成并保存为 Processed_output.csv')
    # 数据归一化并保存 MinMaxScaler 模型
    scaler = MinMaxScaler()
    # 获取需要归一化的列，排除独热编码的列
    numeric_columns = [
        col for col in merged_df.select_dtypes(include=['float64', 'int64']).columns
        if col not in ['FTR', 'HTR'] + list(teams_encoded.columns)
    ]
    joblib.dump(numeric_columns, 'scaler_columns.pkl')  # 保存归一化用的列名
    print('已保存用于归一化的列名：scaler_columns.pkl')
    # 在归一化之前将 -999 替换为 NaN
    # merged_df[numeric_columns] = merged_df[numeric_columns].replace(-999, np.nan)
    # 进行归一化
    merged_df[numeric_columns] = scaler.fit_transform(merged_df[numeric_columns])
    # 归一化后将 NaN 恢复为 -999
    # merged_df.fillna(-999, inplace=True)
    # 保存归一化模型
    joblib.dump(scaler, 'scaler.pkl')
    print('已保存 MinMaxScaler 模型：scaler.pkl')
    # 保存归一化后的数据为 CSV 文件
    merged_df.to_csv('Processed_merged_output.csv', index=False, encoding='utf-8')
    print('数据预处理完成并保存为 Processed_merged_output.csv')
    # 使用 SMOTE 进行上采样并保存处理后的数据
    X = merged_df.drop('FTR', axis=1)
    y = merged_df['FTR']
    smote = SMOTE(random_state=40)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # 保存 SMOTE 处理后的数据为 CSV 文件
    resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.DataFrame(y_resampled, columns=['FTR'])], axis=1)
    resampled_df.to_csv('Processed_merged_SMOTE_output.csv', index=False, encoding='utf-8')
    print('SMOTE 处理完成并保存为 Processed_merged_SMOTE_output.csv')
else:
    print("没有成功读取任何CSV文件。")




# 新数据集处理，使用一样的独热编码和归一化处理器
folder_path = r'D:/desk/Oddsprediction/Datasets/2024106_2025316_Bundesliga1'
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
    new_data.to_csv('Processed_merged_output_Bundesliga1.csv', index=False, encoding='utf-8')
    print('新数据集预处理完成并进行归一化处理，已保存为 Processed_merged_output_Bundesliga1.csv')
    # 使用 SMOTE 进行上采样
    X_new = new_data.drop(['FTR'], axis=1)
    y_new = new_data['FTR']
    smote = SMOTE(random_state=40)
    X_new_resampled, y_new_resampled = smote.fit_resample(X_new, y_new)
    # 合并上采样后的数据
    new_data_resampled = pd.concat([pd.DataFrame(X_new_resampled, columns=X_new.columns),
                                    pd.DataFrame(y_new_resampled, columns=['FTR'])], axis=1)

    # 保存上采样后的数据
    new_data_resampled.to_csv('Processed_merged_SMOTE_output_Bundesliga1.csv', index=False, encoding='utf-8')
    print('新数据集预处理完成并进行 SMOTE 处理，已保存为 Processed_merged_SMOTE_output_Bundesliga1.csv')

else:
    print("没有成功读取任何新数据文件。")


# # 无上采样
# import os
# import pandas as pd
# import chardet
# from sklearn.preprocessing import MinMaxScaler
#
# folder_path = r'D:\desk\Odds prediction\Datasets\New Data'  # 替换为你的主文件夹路径
#  创建一个空的列表用于存储每个CSV的DataFrame
# csv_list = []
# # 使用 os.walk 遍历主文件夹及其子文件夹中的所有文件
# for root, dirs, files in os.walk(folder_path):
#     for filename in files:
#         if filename.endswith('.csv'):  # 确保只处理CSV文件
#             file_path = os.path.join(root, filename)  # 获取完整的文件路径
#             # 首先尝试检测文件编码
#             with open(file_path, 'rb') as f:
#                 rawdata = f.read()
#                 result = chardet.detect(rawdata)
#                 encoding = result['encoding']  # 获取检测到的编码
#                 print(f"检测到文件: {file_path} 使用编码: {encoding}")
#             try:
#                 # 读取CSV文件并追加到列表中
#                 df = pd.read_csv(file_path, encoding=encoding, engine='python', on_bad_lines='skip')
#                 csv_list.append(df)
#                 print(f"成功读取文件: {file_path}")
#             except (UnicodeDecodeError, pd.errors.ParserError) as e:
#                 print(f"无法读取文件: {file_path}，错误: {e}")
# # 将所有CSV文件的DataFrame合并为一个
# if csv_list:  # 确保列表不为空
#     merged_df = pd.concat(csv_list, ignore_index=True)
#     # 删除不需要的列：'Div', 'Date', 'Time', 'Referee'
#     columns_to_drop = ['Div', 'Date', 'Time', 'Referee']
#     merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
#     # 删除所有Unnamed列
#     merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
#     # 将 'FTR'（全场比赛结果） 和 'HTR'（半场比赛结果） 映射为数值型数据
#     result_mapping = {'H': 0, 'A': 1, 'D': 2}
#     merged_df['FTR'] = merged_df['FTR'].map(result_mapping)  # 比赛结果 (目标)
#     merged_df['HTR'] = merged_df['HTR'].map(result_mapping)  # 半场结果 (特征)
#     print("FTR 列缺失值数量:", merged_df['FTR'].isnull().sum())
#     # 对缺失值的行进行填充
#     for col in merged_df.columns:
#         if merged_df[col].isnull().any():  # 检查是否有缺失值
#             if merged_df[col].dtype == 'object':  # 如果是类别型数据
#                 merged_df[col] = merged_df[col].fillna('Unknown')
#             else:  # 如果是数值型数据
#                 merged_df[col] = merged_df[col].fillna(-999)
#     # 删除全为0的列
#     # merged_df = merged_df.loc[:, (merged_df != 0).any(axis=0)]
#     # 对比赛队伍进行独热编码 (One-Hot Encoding)
#     teams_encoded = pd.get_dummies(merged_df[['HomeTeam', 'AwayTeam']], drop_first=True).astype(int)
#     merged_df = pd.concat([merged_df, teams_encoded], axis=1)
#     # 删除编码后的原始 'HomeTeam' 和 'AwayTeam' 列
#     merged_df.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
#
#     # 进行数据归一化
#     scaler = MinMaxScaler()
#     # 获取独热编码生成的列名
#     encoded_columns = teams_encoded.columns.tolist()
#     # 选择数值型特征，排除 FTR, HTR 和独热编码的列
#     columns_to_scale = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     columns_to_scale = [col for col in columns_to_scale if col not in ['FTR', 'HTR'] + encoded_columns]  # 移除不需要归一化的列
#     # 应用归一化
#     merged_df[columns_to_scale] = scaler.fit_transform(merged_df[columns_to_scale])
#     # 保存处理后的数据为新的CSV文件
#     merged_df.to_csv('Processed_merged_output.csv', index=False, encoding='utf-8')  # 指定编码为utf-8
#     print('数据预处理完成，已保存为 Processed_merged_output.csv')
# else:
#     print("没有成功读取任何CSV文件。")


