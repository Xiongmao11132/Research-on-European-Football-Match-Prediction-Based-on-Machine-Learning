import os
import requests
'''
# 创建一个包含各赛季和联赛数据链接的字典
data_links = {
    "2024/2025": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/2425/F2.csv"
    },
    "2023/2024": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/2324/F2.csv"
    },
    "2022/2023": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/2223/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/2223/F2.csv"
    },
    "2021/2022": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/2122/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/2122/F2.csv"
    },
    "2020/2021": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/2021/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/2021/F2.csv"
    },
    "2019/2020": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1920/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1920/F2.csv"
    },
    "2018/2019": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1819/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1819/F2.csv"
    },
    "2017/2018": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1718/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1718/F2.csv"
    },
    "2016/2017": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1617/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1617/F2.csv"
    },
    "2015/2016": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1516/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1516/F2.csv"
    },
    "2014/2015": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1415/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1415/F2.csv"
    },
    "2013/2014": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1314/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1314/F2.csv"
    },
    "2012/2013": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1213/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1213/F2.csv"
    },
    "2011/2012": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1112/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1112/F2.csv"
    },
    "2010/2011": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/1011/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/1011/F2.csv"
    },
    "2009/2010": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/0910/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/0910/F2.csv"
    },
    "2008/2009": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/0809/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/0809/F2.csv"
    },
    "2007/2008": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/0708/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/0708/F2.csv"
    },
    "2006/2007": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/0607/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/0607/F2.csv"
    },
    "2005/2006": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/0506/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/0506/F2.csv"
    },
    "2004/2005": {
        "Le Championnat": "https://www.football-data.co.uk/mmz4281/0405/F1.csv",
        "Division 2": "https://www.football-data.co.uk/mmz4281/0405/F2.csv"
    }
}

download_folder = 'France'
os.makedirs(download_folder, exist_ok=True)

# 下载每个链接
for season, leagues in data_links.items():
    # 创建年份文件夹
    season_folder = os.path.join(download_folder, season.replace('/', '_'))
    os.makedirs(season_folder, exist_ok=True)

    for league, url in leagues.items():
        print(f"Downloading {league} data for {season}...")
        response = requests.get(url)
        if response.status_code == 200:
            # 使用年份文件夹保存文件
            filename = os.path.join(season_folder, f"{league.replace(' ', '_')}.csv")
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filename}")
        else:
            print(f"Failed to download {league} data for {season}: {response.status_code}")

print("All downloads completed.")
'''
import requests

# 设置 URL，并附加 API key 到查询参数中
url = 'https://api.sportradar.com/oddscomparison-prematch/trial/v2/zh/sports?api_key=123'

# 设置请求头
headers = {
    'accept': 'application/json'
}

# 发送 GET 请求
response = requests.get(url, headers=headers)

# 打印响应的状态码和内容
print(f'Status Code: {response.status_code}')
print(f'Response Text: {response.text}')

# 尝试解析 JSON
try:
    response_json = response.json()  # 解析 JSON
    print(f'Response JSON: {response_json}')
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON, the response is not valid JSON format.")


