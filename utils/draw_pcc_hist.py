# @Author : CyIce
# @Time : 2024/7/3 9:21

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# To draw the histogram of the delta PCC values.


def remove_outliers(data, m=3):
    mean = np.mean(data)
    std = np.std(data)
    # Keep only the data within m standard deviations from the mean
    filtered_data = data[np.abs(data - mean) < m * std]
    return filtered_data


def draw_hist(delta, title, bins=100, alpha=None, sample_size=None, saved_path=None):
    plt.clf()
    delta_flattened = delta.flatten()
    # Filter out non-finite values
    delta_flattened = delta_flattened[np.isfinite(delta_flattened)]
    # Remove outliers
    delta_flattened = remove_outliers(delta_flattened, m=3)
    if sample_size is not None:
        sample_size = int(sample_size)
        if sample_size > len(delta_flattened):
            sample_size = len(delta_flattened)
        # Sample the specified number of data points
        delta_flattened = np.random.choice(delta_flattened, sample_size, replace=False)

    # Draw frequency distribution histogram
    plt.hist(delta_flattened, bins=bins, density=True, edgecolor='k', alpha=0.7)

    # Calculate the standard normal distribution curve
    mu, std = norm.fit(delta_flattened)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Draw the normal distribution curve in red dashed line
    plt.plot(x, p, 'r-', linewidth=2)
    if alpha is not None:
        # Calculate critical z-values for a two-tailed test with alpha = 0.05
        z_critical_low = norm.ppf(alpha / 2, loc=mu, scale=std)
        z_critical_high = norm.ppf(1 - alpha / 2, loc=mu, scale=std)

        plt.axvline(z_critical_low, color='green', linestyle='solid', linewidth=2,
                    label=f'Critical Low (z={z_critical_low:.2f})')
        plt.axvline(z_critical_high, color='green', linestyle='solid', linewidth=2,
                    label=f'Critical High (z={z_critical_high:.2f})')


    plt.title(title)
    plt.xlabel('Delta PCC')
    plt.ylabel('Density')
    plt.grid(False)
    if saved_path is not None:
        plt.savefig(f"{saved_path}/{title}.pdf", dpi=300)
    else:
        plt.show()
    if alpha is not None:
        return z_critical_low, z_critical_high
    return None,None


def plot_standard_normal_with_highlighted_points(points, sample_size, alpha, title='Standard Normal Distribution',
                                                 saved_path=None):
    """
    绘制标准正态分布，并在指定的横坐标位置标红。

    参数:
    points (list): 要在标准正态分布上标红的横坐标值。
    title (str): 图形的标题。
    """
    plt.clf()
    if isinstance(points, np.ndarray):
        points = points.tolist()
    points_within_range = [point for point in points if -6 <= point <= 6]
    # 随机采样指定数量的点
    sampled_points = np.random.choice(points_within_range, sample_size, replace=False)

    # 定义标准正态分布的参数
    mu = 0
    std = 1

    # 生成 x 轴的值
    x = np.linspace(-6, 6, 1000)

    # 计算标准正态分布的概率密度函数值
    p = norm.pdf(x, mu, std)

    # 绘制标准正态分布曲线
    plt.plot(x, p, 'r-', linewidth=2, label='Standard Normal Distribution')

    # 在指定的横坐标位置标红
    for point in sampled_points:
        # plt.scatter(point, 0, color='red', zorder=5)
        plt.scatter(point, norm.pdf(point, mu, std), s=20, color='blue', zorder=5)

    z_critical_low = norm.ppf(alpha / 2, loc=mu, scale=std)
    z_critical_high = norm.ppf(1 - alpha / 2, loc=mu, scale=std)
    plt.axvline(z_critical_low, color='green', linestyle='solid', linewidth=2,
                label=f'Critical Low (z={z_critical_low:.2f})')
    plt.axvline(z_critical_high, color='green', linestyle='solid', linewidth=2,
                label=f'Critical High (z={z_critical_high:.2f})')

    # 添加标题和标签
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')

    # 显示图例
    plt.legend()
    if saved_path is not None:
        plt.savefig(f"{saved_path}/{title}.pdf", dpi=300)
    else:
        plt.show()
