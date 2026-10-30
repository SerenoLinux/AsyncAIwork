# -*- coding: utf-8 -*-
r"""
Task2.1 线性回归（梯度下降）

用梯度下降拟合一条直线 y = w * x + b，数据来自同目录的 data1.csv。

运行方式（本机 PATH 里的 python 是 3.11，没装 numpy，必须用下面这个解释器）：
    & "D:\Software\Python\python.exe" 2.1.py

说明：控制台输出一律用 ASCII（epoch / w / b / MSE），避免中文终端出现乱码；
中文只出现在 matplotlib 画出来的图里（图标题、轴标题）。
"""

import sys

import matplotlib
import matplotlib.pyplot as plt

# import pandas as pd
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
        "Microsoft YaHei",  # Windows 微软雅黑
        "SimHei",  # Windows 黑体
        "Noto Sans SC",  # Google Noto / Linux
        "Source Han Sans SC",  # 思源黑体
        "PingFang SC",  # macOS 苹方
        "WenQuanYi Zen Hei",  # Linux 文泉驿
        "DengXian",  # Windows 等线
        "SimSun",  # Windows 宋体
    ):
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams[
                "font.sans-serif"
            ]
            matplotlib.rcParams["axes.unicode_minus"] = (
                False  # 负号也用这个字体，否则 -1 变方框
            )
            return name
    print("警告：系统里没找到中文字体，图中中文可能显示为方框", file=sys.stderr)
    return None


setup_chinese_font()

from pathlib import Path

import csv

import numpy as np

# import matplotlib.pyplot as plt

# ============================== 可调参数 ==============================
DATA_FILE = "data1.csv"  # 数据文件名（和本脚本放在同一个目录）
COL_X = 0  # 第 0 列（第一个字段）当作 x：学习时间
COL_Y = 1  # 第 1 列（第二个字段）当作 y：考试成绩
LEARNING_RATE = 0.001  # 学习率（步长），太大就会发散成 nan
EPOCHS = 20000  # 训练轮数
PRINT_EVERY = 2000  # 每多少轮打印一次进度，0 表示不打印
# ====================================================================


def _to_float(cell, row_no, col_no):
    """把表格里的一格文本转成浮点数；转不动就报一个能看懂的错误。"""
    try:
        return float(cell)
    except ValueError:
        raise ValueError(
            "row %d col %d is not a number: %r "
            "(wrong delimiter, or this is a header line?)" % (row_no, col_no, cell)
        ) from None


def load_data(path):
    """
    读取 CSV 并返回 (x, y, info)。

    这份 data1.csv 的实际情况：
      - 没有表头，第 1 行就是数据
      - 用 Tab（制表符）分隔，而不是逗号
      - 第 0 列 = 1..18，第 1 列 = 62..97
    所以下面三件事都必须做对：编码、分隔符、表头。
    """
    if not path.exists():
        raise FileNotFoundError("data file not found: %s" % path.name)

    # encoding="utf-8-sig"：顺手吃掉文件开头的 BOM，碰到中文也不会乱码
    # newline=""：交给 csv 模块自己处理换行，避免多出空行
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)  # 先读一小段当"样品"，用来猜分隔符
        f.seek(0)  # 猜完把读指针拨回开头

        try:
            # 让它从 , ; Tab 三种里挑一个最像的
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
            guessed = True
        except csv.Error:
            delimiter = "\t"  # 猜不出来就按本文件的实际情况兜底
            guessed = False

        # csv.reader 把每一行切成字符串列表；跳过全空白行
        rows = [
            r for r in csv.reader(f, delimiter=delimiter) if any(c.strip() for c in r)
        ]

    if not rows:
        raise ValueError("%s contains no data" % path.name)

    # ---- 判断有没有表头：看第一行的目标列能不能直接变成数字 ----
    first = rows[0]
    has_header = False
    if len(first) > max(COL_X, COL_Y):
        try:
            float(first[COL_X])
            float(first[COL_Y])
        except ValueError:
            has_header = True  # 第一行不是数字 => 它是表头，丢掉
    if has_header:
        rows = rows[1:]

    if len(rows) < 2:
        raise ValueError(
            "%s has only %d data row(s) after the header, need at least 2"
            % (path.name, len(rows))
        )

    # ---- 列 -> 变量：靠"位置索引"把某一列绑到 x 或 y 上 ----
    xs, ys = [], []
    for i, row in enumerate(rows, start=1):
        if len(row) <= max(COL_X, COL_Y):
            raise ValueError(
                "row %d has only %d column(s), cannot read column %d"
                % (i, len(row), max(COL_X, COL_Y))
            )
        xs.append(_to_float(row[COL_X], i, COL_X))
        ys.append(_to_float(row[COL_Y], i, COL_Y))

    info = {
        "samples": len(xs),
        "delimiter": delimiter,
        "delimiter_guessed": guessed,
        "header": has_header,
    }
    return np.array(xs, dtype=float), np.array(ys, dtype=float), info


def train(x, y, lr=LEARNING_RATE, epochs=EPOCHS, print_every=PRINT_EVERY):
    """
    梯度下降：完全照题面伪代码来。

        pred = w*x + b
        err  = pred - y          (预测值 - 实际值)
        dw   = mean(2*err*x)
        db   = mean(2*err)
        w   -= lr*dw
        b   -= lr*db

    返回 (w, b, loss_history)。
    """
    w = 0.0  # 权重（斜率）
    b = 0.0  # 偏置（截距）
    loss_history = []

    for epoch in range(1, epochs + 1):
        pred = w * x + b  # 1. 计算预测值
        err = pred - y  # 2. 误差 = 预测值 - 实际值
        dw = np.mean(2 * err * x)  # 3. 损失对 w 的导数
        db = np.mean(2 * err)  #    损失对 b 的导数
        w -= lr * dw  # 4. 朝着让损失变小的方向挪一小步
        b -= lr * db  #    w 和 b 用的是同一轮的梯度（同步更新）

        mse = float(np.mean(err**2))  # 本轮的均方误差，只用来观察
        loss_history.append(mse)
        if print_every and epoch % print_every == 0:
            print(
                "epoch %6d | w = %9.6f | b = %9.6f | MSE = %12.6f" % (epoch, w, b, mse)
            )

    return w, b, loss_history


def plot(x, y, w, b, loss_history, out_png):
    """画两张图：左边散点 + 拟合直线，右边损失随轮次下降。"""
    # 让图里的中文能正常显示（不设置的话会变成一串方块）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- 左图：原始散点 + 学出来的直线 ----
    ax1.scatter(x, y, s=40, color="#1f77b4", label="真实数据")
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    ax1.plot(
        x_line,
        w * x_line + b,
        color="#d62728",
        linewidth=2,
        label="拟合直线：y = %.3fx + %.3f" % (w, b),
    )
    ax1.set_xlabel("学习时间")
    ax1.set_ylabel("考试成绩")
    ax1.set_title("散点与拟合直线")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---- 右图：损失曲线 ----
    ax2.plot(
        np.arange(1, len(loss_history) + 1),
        loss_history,
        color="#2ca02c",
        linewidth=1.5,
    )
    ax2.set_xlabel("轮次 epoch")
    ax2.set_ylabel("MSE（均方误差）")
    ax2.set_title("损失随训练下降")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)  # 先存盘，再弹窗
    plt.show()


def main():
    data_path = Path(__file__).with_name(DATA_FILE)
    x, y, info = load_data(data_path)
    print("file      : %s" % data_path.name)
    print("rows      : %d" % info["samples"])
    print(
        "delimiter : %r (guessed=%s)" % (info["delimiter"], info["delimiter_guessed"])
    )
    print("header    : %s" % info["header"])
    print("columns   : COL_X=%d -> x, COL_Y=%d -> y" % (COL_X, COL_Y))

    w, b, loss_history = train(x, y)

    mse = float(np.mean((w * x + b - y) ** 2))
    print("result    : w = %.6f, b = %.6f, MSE = %.6f" % (w, b, mse))
    print("predict   : x = 10 -> y = %.2f" % (w * 10 + b))

    out_png = Path(__file__).with_name("fit.png")
    plot(x, y, w, b, loss_history, out_png)
    print("saved     : %s" % out_png.name)


if __name__ == "__main__":
    main()
