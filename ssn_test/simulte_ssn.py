import numpy as np
from scipy.stats import norm, pearsonr
import matplotlib.pyplot as plt


def generate_random_series(n, pcc, num_samples):
    mean = [0, 0]
    cov = [[1, pcc], [pcc, 1]]  # 协方差矩阵
    return np.random.multivariate_normal(mean, cov, num_samples)


def calculate_delta_pcc(n, pcc, num_simulations):
    delta_pccs = []
    for _ in range(num_simulations):
        data = generate_random_series(n, pcc, n)
        x, y = data[:, 0], data[:, 1]
        pcc_simulated, _ = pearsonr(x, y)
        delta_pcc = pcc_simulated - pcc
        delta_pccs.append(delta_pcc)
    return np.array(delta_pccs)


def get_significance_thresholds(delta_pccs, alpha=0.05):
    lower_threshold = np.percentile(delta_pccs, alpha / 2 * 100)
    upper_threshold = np.percentile(delta_pccs, (1 - alpha / 2) * 100)
    return lower_threshold, upper_threshold


def theoretical_delta_pcc_threshold(n, pcc, alpha=0.05):
    sigma_delta_pcc = np.sqrt((1 - pcc ** 2) / (n - 1))
    z_score = norm.ppf(1 - alpha / 2)
    return z_score * sigma_delta_pcc


# 参数设置
n_values = [8,10,18]  # 特定的样本数量
# n_values = np.arange(3, 200, 1)  # 特定的样本数量
# pcc_values = np.arange(0, 1.0, 0.1)  # PCC从0到0.9，间隔为0.1
pcc_values = [0]  # PCC从0到0.9，间隔为0.1
num_simulations = 2000  # 模拟次数
alpha = 0.05  # 显著性水平

# 创建子图
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
axs = axs.ravel()  # 将子图数组展平

for i, pcc in enumerate(pcc_values):
    simulated_thresholds = []
    theoretical_thresholds = []

    for n in n_values:
        print(f"Running simulations for n={n}, PCC={pcc}")
        delta_pccs = calculate_delta_pcc(n, pcc, num_simulations)
        _, upper_threshold = get_significance_thresholds(delta_pccs, alpha)
        simulated_thresholds.append(upper_threshold)
        theoretical_threshold = theoretical_delta_pcc_threshold(n, pcc, alpha)
        theoretical_thresholds.append(theoretical_threshold)
    print(simulated_thresholds)
    print(theoretical_thresholds)
    # 绘制每个PCC值对应的图
    axs[i].plot(n_values, simulated_thresholds, 'r--', label='Simulated curve')
    axs[i].plot(n_values, theoretical_thresholds, 'b-', label='Theoretical curve')
    axs[i].set_xlabel('Number of reference samples')
    axs[i].set_ylabel('Significant value of ΔPCC')
    axs[i].set_title(f'Significant value of ΔPCC (p-value = 0.05, PCC={pcc})')
    axs[i].legend()

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('significant_value_of_delta_pcc.png')

# 显示图片
plt.show()
