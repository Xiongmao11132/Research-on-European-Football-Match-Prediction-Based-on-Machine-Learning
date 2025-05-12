import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# 将数据加载为 DataFrame
data = {
    "Feature": ["B365H", "WHH", "B365A", "BWA", "BWH", "PSCH", "AHCh", "WHA", "SBA", "B365D",
                "BWD", "SJH", "AHh", "VCA", "IWH", "PSCA", "BbAvAHA", "VCD", "BbAvA", "LBH"],
    "Importance": [0.601836, 0.118118, 0.059102, 0.026061, 0.020486, 0.015554, 0.014888, 0.011183,
                   0.009983, 0.009436, 0.009197, 0.008059, 0.005119, 0.004315, 0.004302, 0.004055,
                   0.003874, 0.003601, 0.003492, 0.003349]
}
df = pd.DataFrame(data)

# 水平条形图
df = df.sort_values(by="Importance", ascending=True)

# 可视化
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")
plt.title("Feature Importance", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.show()



# 累计特征重要性折线图
df["Cumulative Importance"] = np.cumsum(df["Importance"])

plt.figure(figsize=(10, 6))
plt.plot(df["Feature"], df["Cumulative Importance"], marker="o", color="b")
plt.xticks(rotation=45, ha="right")
plt.title("Cumulative Feature Importance", fontsize=16)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Cumulative Importance", fontsize=14)
plt.grid()
plt.tight_layout()
plt.show()

# 对数缩放柱状图（如果重要性差距悬殊）
df = df.sort_values(by="Importance", ascending=True)
plt.figure(figsize=(10, 8))
sns.heatmap(df[["Importance"]].T, annot=True, fmt=".4f", cmap="Blues", cbar=False, linewidths=0.5)
plt.title("Feature Importance Heatmap", fontsize=16)
plt.xticks(ticks=np.arange(0.5, len(df)), labels=df["Feature"], rotation=45, ha="right", fontsize=10)
plt.yticks([])
plt.tight_layout()
plt.show()
