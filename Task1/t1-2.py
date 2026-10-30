import pandas as pd

# from 中文字体设置 import setup_chinese_font

# setup_chinese_font()
import matplotlib.pyplot as plt

data = {
    "姓名": ["A", "B", "C", "D", "E"],
    "Python": [85, 92, 78, 96, 88],
    "数学": [76, 89, 82, 95, 80],
    "英语": [90, 85, 75, 91, 86],
}

df = pd.DataFrame(data)

print("学生成绩：")
print(df)

df["平均分"] = df[["Python", "数学", "英语"]].mean(axis=1)

print("\n平均成绩：")
print(df[["姓名", "平均分"]])

z = pd.DataFrame([{"姓名": "Z", "Python": 99, "数学": 60, "英语": 70, "平均分": 76.33}])
df2 = pd.concat([df, z], ignore_index=True)

plt.bar(df2["姓名"], df2["平均分"])

plt.title("学生平均成绩")
plt.xlabel("学生")
plt.ylabel("平均分")

plt.ylim(0, 100)
plt.show()
