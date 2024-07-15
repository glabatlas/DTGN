# @Author : CyIce
# @Time : 2024/7/14 20:03
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# def ff(chi_square_data):
#
#     # 绘制卡方分布的直方图
#     count, bins, ignored = plt.hist(chi_square_data, bins=30, density=True, alpha=0.6, color='g', label='Histogram')
#
#     # 拟合卡方分布并估计自由度
#     df_estimated, loc, scale = chi2.fit(chi_square_data, floc=0)
#
#     # 使用拟合的参数绘制卡方分布
#     x = np.linspace(0, np.max(chi_square_data), 1000)
#     fitted_pdf = chi2.pdf(x, df_estimated, loc, scale)
#
#     # 绘制拟合的卡方分布
#     plt.plot(x, fitted_pdf, 'r-', lw=2, label=f'Fitted Chi-Square Distribution\n(df={df_estimated:.2f})')
#
#     # 添加标题和标签
#     plt.title('Chi-Square Distribution Histogram and Fitting')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.legend()
#
#     # 显示图形
#     plt.show()

import numpy as np

def calculate_freedom(X1,X2):
    # 示例数据
    # 计算协方差矩阵
    data = np.stack((X1, X2), axis=0)
    cov_matrix = np.cov(data)
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 计算估计的自由度
    sum_eigenvalues = np.sum(eigenvalues)
    sum_eigenvalues_squared = np.sum(eigenvalues ** 2)
    estimated_degrees_of_freedom = (2 * (sum_eigenvalues ** 2)) / sum_eigenvalues_squared

    print("\n估计的自由度:")
    print(estimated_degrees_of_freedom)
