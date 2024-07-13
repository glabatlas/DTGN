import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d

COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange',
          'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
          'tab:gray', 'tab:olive', 'tab:cyan', 'xkcd:sky blue', 'xkcd:coral']


def draw_roc(data: dict, title='',smooth=True, path=None):
    # 设置滑动窗口大小
    window_size = 3
    for i, label in enumerate(sorted(data.keys())):
        fpr, tpr = data[label]
        # print(fpr)
        # print(tpr)
        # 计算移动平均
        if smooth:
            a = 1
            # 移动平均
            # fpr = np.convolve(fpr, np.ones(window_size) / window_size, mode='valid')
            # tpr = np.convolve(tpr, np.ones(window_size) / window_size, mode='valid')
            # Loess
            fpr, tpr = sm.nonparametric.lowess(tpr, fpr, frac=0.1).T
            fpr[0] = 0
            tpr[0] = 0
            fpr[-1] = 1
            tpr[-1] = 1
            # 样条插值
            # fpr,index = np.unique(fpr,return_index=True)
            # print(fpr,tpr)
        plt.plot(fpr, tpr, color=COLOR_LIST[i], label=label)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend()
    if path != None:
        plt.savefig(path)
    plt.show()
