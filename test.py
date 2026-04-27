import pandas as pd
import numpy as np
import scipy as sp
import re, bisect

import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from package.MyTT import *
import my_quant_lib as MQL

# 设置全局字体
fm.fontManager.addfont(
    "C:/Users/29433/AppData/Local/Microsoft/Windows/Fonts/FZHTJW.TTF"
)
mpl.rcParams["font.sans-serif"] = "FZHei-B01S"  # 此处需要用字体文件真正的名称
mpl.rcParams["axes.unicode_minus"] = False  # 解决负号无法显示的问题
font = fm.FontProperties(
    fname="C:/Users/29433/AppData/Local/Microsoft/Windows/Fonts/FZHTJW.TTF"
)
print("hello world")
