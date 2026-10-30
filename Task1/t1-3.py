"""
作业 Task1 第 3 题：把示例代码的柱状图改成折线图（plt.plot）
运行方式：python t1-3.py
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


def setup_chinese_font():
    """挑一个系统里真实存在的中文字体。

    为什么不能直接写 plt.rcParams["font.sans-serif"] = ["微软雅黑"]？
    因为 matplotlib 默认字体 DejaVu Sans 里没有汉字字形，缺字就画成方框；
    而如果写了系统里没有的字体名，matplotlib 只会发个警告再静默回退到 DejaVu，
    结果还是满屏方框。所以这里先在"已安装字体列表"里核对，再设置。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in (
        "Microsoft YaHei",     # Windows 微软雅黑
        "SimHei",              # Windows 黑体
        "Noto Sans SC",        # Google Noto / Linux
        "Source Han Sans SC",  # 思源黑体
        "PingFang SC",         # macOS 苹方
        "WenQuanYi Zen Hei",   # Linux 文泉驿
        "DengXian",            # Windows 等线
        "SimSun",              # Windows 宋体
    ):
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams["font.sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False  # 负号也用这个字体，否则 -1 变方框
            return name
    print("警告：系统里没找到中文字体，图中中文可能显示为方框", file=sys.stderr)
    return None


setup_chinese_font()

data = {
    "姓名": ["A", "B", "C", "D", "E"],
    "Python": [85, 92, 78, 96, 88],
    "数学": [76, 89, 82, 95, 80],
    "英语": [90, 85, 75, 91, 86],
}
df = pd.DataFrame(data)

plt.figure(figsize=(7, 4.5))

# 一条线 = 一次 plot；同一次里给的 x 和 y 要能一一对应
# marker：每个数据点上画个记号，方便看出"每个学生"的分数
# label ：给这条线贴个名字，legend() 会把它拿出来放进图例
for subject, marker in (("Python", "o"), ("数学", "s"), ("英语", "^")):
    plt.plot(df["姓名"], df[subject], marker=marker, label=subject)

plt.title("各科成绩折线图")
plt.xlabel("学生")
plt.ylabel("分数")
plt.ylim(0, 100)
plt.grid(alpha=0.3)

# 图例：必须在所有 plot 之后调用，否则此时轴上还没有 label，图例会空
plt.legend()

# 存图必须在 show() 之前，show() 之后画布会被清空，存出来是空白图
plt.savefig("成绩折线图.png", dpi=150, bbox_inches="tight")

plt.show()
