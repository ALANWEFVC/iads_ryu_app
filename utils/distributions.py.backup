# utils/distributions.py
"""概率分布工具类"""

import numpy as np
from math import log, pi, e


class BetaDistribution:
    """Beta分布处理类"""

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def get_confidence(self):
        """获取P(UP)置信度"""
        return self.alpha / (self.alpha + self.beta)

    def update(self, success):
        """更新分布参数
        Args:
            success: True表示UP，False表示DOWN
        """
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def entropy(self):
        """计算熵H(i,m)"""
        p_up = self.get_confidence()
        p_down = 1 - p_up

        # 避免log(0)
        if p_up == 0 or p_down == 0:
            return 0

        return -(p_up * log(p_up) + p_down * log(p_down))

    def expected_entropy_after_probe(self):
        """计算探测后的预期熵"""
        p_up = self.get_confidence()
        p_down = 1 - p_up

        # 如果探测结果为UP
        alpha_up = self.alpha + 1
        beta_up = self.beta
        p_up_after_up = alpha_up / (alpha_up + beta_up)
        p_down_after_up = 1 - p_up_after_up

        if p_up_after_up > 0 and p_down_after_up > 0:
            h_after_up = -(p_up_after_up * log(p_up_after_up) +
                           p_down_after_up * log(p_down_after_up))
        else:
            h_after_up = 0

        # 如果探测结果为DOWN
        alpha_down = self.alpha
        beta_down = self.beta + 1
        p_up_after_down = alpha_down / (alpha_down + beta_down)
        p_down_after_down = 1 - p_up_after_down

        if p_up_after_down > 0 and p_down_after_down > 0:
            h_after_down = -(p_up_after_down * log(p_up_after_down) +
                             p_down_after_down * log(p_down_after_down))
        else:
            h_after_down = 0

        # 预期熵是加权平均
        return p_up * h_after_up + p_down * h_after_down

    def to_dict(self):
        """转换为字典"""
        return {
            'type': 'beta',
            'alpha': self.alpha,
            'beta': self.beta,
            'confidence': self.get_confidence()
        }


class GaussianDistribution:
    """高斯分布处理类"""

    def __init__(self, mu=0.0, sigma2=1000.0):
        self.mu = mu
        self.sigma2 = sigma2

    def update(self, measurement, noise_var=1.0):
        """贝叶斯更新
        Args:
            measurement: 测量值
            noise_var: 测量噪声方差
        """
        # 贝叶斯更新公式
        sigma2_new = 1 / (1 / self.sigma2 + 1 / noise_var)
        mu_new = sigma2_new * (self.mu / self.sigma2 + measurement / noise_var)

        self.mu = mu_new
        self.sigma2 = sigma2_new

    def entropy(self):
        """计算熵H(i,m)"""
        return 0.5 * log(2 * pi * e * self.sigma2)

    def expected_entropy_after_probe(self, noise_var=1.0):
        """计算探测后的预期熵"""
        # 探测后的方差
        sigma2_new = 1 / (1 / self.sigma2 + 1 / noise_var)
        # 探测后的熵
        return 0.5 * log(2 * pi * e * sigma2_new)

    def to_dict(self):
        """转换为字典"""
        return {
            'type': 'gaussian',
            'mu': self.mu,
            'sigma2': self.sigma2,
            'std': np.sqrt(self.sigma2)
        }


class StabilityCalculator:
    """稳定性计算器"""

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.measurements = []
        self.timestamps = []

    def add_measurement(self, value, timestamp):
        """添加测量值"""
        self.measurements.append(value)
        self.timestamps.append(timestamp)

        # 移除窗口外的数据
        current_time = timestamp
        cutoff_time = current_time - self.window_size

        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.pop(0)
            self.measurements.pop(0)

    def calculate_stability(self):
        """计算稳定性S(i,m)"""
        if len(self.measurements) < 2:
            return 0.0

        # 计算方差
        measurements_array = np.array(self.measurements)
        variance = np.var(measurements_array)

        return variance

    def get_normalized_stability(self, max_stability=10.0):
        """获取归一化稳定性"""
        stability = self.calculate_stability()
        return min(stability / max_stability, 1.0)