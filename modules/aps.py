# modules/aps.py
"""主动探测调度器(APS)模块"""

import numpy as np
import time
from collections import defaultdict
from config import *


class CMAB:
    """上下文多臂赌博机(Contextual Multi-Armed Bandit)"""

    def __init__(self, n_features=4, n_arms=4):
        self.n_features = n_features
        self.n_arms = n_arms
        self.strategies = list(CMAB_STRATEGIES.values())

        # 初始化参数：每个臂的参数是高斯分布
        # 均值向量 mu[arm] = n_features维向量
        # 协方差矩阵 Sigma[arm] = n_features x n_features矩阵
        self.mu = {}
        self.Sigma = {}
        self.Sigma_inv = {}

        for strategy in self.strategies:
            self.mu[strategy] = np.zeros(n_features)
            self.Sigma[strategy] = np.eye(n_features)
            self.Sigma_inv[strategy] = np.eye(n_features)

        # 历史记录
        self.history = []
        self.selected_strategy = None

    def select_strategy(self, context):
        """使用Thompson Sampling选择策略

        Args:
            context: 上下文向量c (4维)

        Returns:
            str: 选择的策略
        """
        scores = {}
        sampled_theta = {}

        # 对每个臂进行Thompson采样
        for strategy in self.strategies:
            # 从后验分布N(mu, Sigma)中采样参数theta
            theta = np.random.multivariate_normal(
                self.mu[strategy],
                self.Sigma[strategy]
            )
            sampled_theta[strategy] = theta

            # 计算得分: score = c^T * theta
            scores[strategy] = np.dot(context, theta)

        # 选择得分最高的策略
        self.selected_strategy = max(scores, key=scores.get)

        # 记录选择
        self.history.append({
            'context': context.copy(),
            'strategy': self.selected_strategy,
            'scores': scores,
            'sampled_theta': sampled_theta
        })

        return self.selected_strategy

    def update(self, context, reward, noise_var=1.0):
        """更新参数

        Args:
            context: 上下文向量
            reward: 奖励值
            noise_var: 噪声方差
        """
        if self.selected_strategy is None:
            return

        strategy = self.selected_strategy

        # 贝叶斯更新
        # Sigma_new^(-1) = Sigma_old^(-1) + (1/noise_var) * c * c^T
        c = context.reshape(-1, 1)
        self.Sigma_inv[strategy] = self.Sigma_inv[strategy] + (1 / noise_var) * np.dot(c, c.T)
        self.Sigma[strategy] = np.linalg.inv(self.Sigma_inv[strategy])

        # mu_new = Sigma_new * (Sigma_old^(-1) * mu_old + (1/noise_var) * c * r)
        self.mu[strategy] = np.dot(
            self.Sigma[strategy],
            np.dot(self.Sigma_inv[strategy], self.mu[strategy]) + (1 / noise_var) * c.flatten() * reward
        )

    def get_strategy_stats(self):
        """获取策略统计信息"""
        strategy_counts = defaultdict(int)
        for record in self.history:
            strategy_counts[record['strategy']] += 1

        return {
            'total_selections': len(self.history),
            'strategy_counts': dict(strategy_counts),
            'current_parameters': {
                strategy: {
                    'mu': self.mu[strategy].tolist(),
                    'Sigma_diag': np.diag(self.Sigma[strategy]).tolist()
                }
                for strategy in self.strategies
            }
        }


class CTLC:
    """控制理论逻辑(Control Theory Logic Controller)"""

    def __init__(self, kp=0.1, target_stability=1.0):
        self.kp = kp
        self.target_stability = target_stability
        self.min_interval = SYSTEM_CONFIG['probe_interval_min']
        self.max_interval = SYSTEM_CONFIG['probe_interval_max']

    def adjust_probe_interval(self, current_interval, stability):
        """调整探测间隔

        Args:
            current_interval: 当前探测间隔
            stability: 稳定性S(i,m)

        Returns:
            float: 新的探测间隔
        """
        # 比例控制: T_new = T_old * (1 + Kp * (1 - S/S_target))
        adjustment_factor = 1 + self.kp * (1 - stability / self.target_stability)
        new_interval = current_interval * adjustment_factor

        # 限制在合理范围内
        new_interval = max(self.min_interval, min(self.max_interval, new_interval))

        return new_interval

    def batch_adjust(self, esm):
        """批量调整所有实体的探测间隔"""
        adjustments = []

        for (entity_id, metric), state in esm.state_table.items():
            old_interval = state.probe_interval
            stability = state.get_stability()
            new_interval = self.adjust_probe_interval(old_interval, stability)

            if new_interval != old_interval:
                esm.set_probe_interval(entity_id, metric, new_interval)
                adjustments.append({
                    'entity_id': entity_id,
                    'metric': metric,
                    'old_interval': old_interval,
                    'new_interval': new_interval,
                    'stability': stability
                })

        return adjustments


class PRIO:
    """优先级引擎(Priority Engine)"""

    def __init__(self, weights=None):
        if weights is None:
            weights = APS_CONFIG['priority_weights']

        self.weights = weights

    def calculate_policy_match(self, task, selected_strategy, esm, event_triggered=False):
        """计算策略匹配度

        Args:
            task: 任务对象
            selected_strategy: CMAB选择的策略
            esm: ESM引用
            event_triggered: 是否有事件触发

        Returns:
            float: 策略匹配度
        """
        state = esm.get_state(task.entity_id, task.metric)
        if not state:
            return 0.0

        if selected_strategy == CMAB_STRATEGIES['FOCUS_UNCERTAINTY']:
            # 不确定性越高，匹配度越高
            uncertainty = state.get_uncertainty()
            return uncertainty / APS_CONFIG['max_uncertainty']

        elif selected_strategy == CMAB_STRATEGIES['HIGHFREQ_UNSTABLE']:
            # 稳定性越低（不稳定），匹配度越高
            stability = state.get_stability()
            return stability  # 已经归一化

        elif selected_strategy == CMAB_STRATEGIES['COVERAGE_BALANCER']:
            # 所有任务匹配度相同
            return 1.0

        elif selected_strategy == CMAB_STRATEGIES['EVENT_TRIGGER']:
            # 事件触发的任务匹配度高
            return 1.0 if event_triggered else 0.0

        return 0.0

    def calculate_priority(self, task, eig, urgency, policy_match, event_trig):
        """计算优先级

        Args:
            task: 任务对象
            eig: 预期信息增益
            urgency: 紧急度
            policy_match: 策略匹配度
            event_trig: 事件触发信号

        Returns:
            float: 优先级分数
        """
        priority = (
                self.weights['eig'] * eig +
                self.weights['urgency'] * urgency +
                self.weights['policy_match'] * policy_match +
                self.weights['event_trig'] * event_trig
        )

        return priority

    def select_top_k(self, task_pool, esm, em, selected_strategy, k=10):
        """选择Top-K任务

        Args:
            task_pool: 任务池（来自UQ）
            esm: ESM引用
            em: EM引用
            selected_strategy: CMAB选择的策略
            k: 选择任务数

        Returns:
            list: Top-K任务列表
        """
        # 计算每个任务的优先级
        task_priorities = []

        for task in task_pool:
            # 获取各项指标
            eig = task.eig

            state = esm.get_state(task.entity_id, task.metric)
            if not state:
                continue

            urgency = state.get_urgency()

            # 获取事件触发信号（如果EM可用）
            event_trig = 0.0
            if em is not None:
                event_trig = em.get_event_trigger(task.entity_id, task.metric)

            # 计算策略匹配度
            policy_match = self.calculate_policy_match(
                task, selected_strategy, esm, event_trig > 0
            )

            # 计算优先级
            priority = self.calculate_priority(
                task, eig, urgency, policy_match, event_trig
            )

            task_priorities.append({
                'task': task,
                'priority': priority,
                'components': {
                    'eig': eig,
                    'urgency': urgency,
                    'policy_match': policy_match,
                    'event_trig': event_trig
                }
            })

        # 按优先级排序
        task_priorities.sort(key=lambda x: x['priority'], reverse=True)

        # 返回Top-K
        return task_priorities[:k]


class ActiveProbingScheduler:
    """主动探测调度器"""

    def __init__(self, esm, uq, em=None):
        self.esm = esm
        self.uq = uq
        self.em = em

        # 初始化子模块
        self.cmab = CMAB()
        self.ctlc = CTLC(
            kp=APS_CONFIG['kp'],
            target_stability=APS_CONFIG['target_stability']
        )
        self.prio = PRIO()

        # 统计信息
        self.stats = {
            'total_rounds': 0,
            'total_tasks_selected': 0,
            'strategy_history': []
        }

    def select_tasks(self, k=None):
        """选择探测任务

        Args:
            k: 任务数，默认使用配置值

        Returns:
            list: 选中的任务及其优先级信息
        """
        if k is None:
            k = SYSTEM_CONFIG['top_k']

        # 1. 获取上下文向量
        context = self.esm.get_context_vector()

        # 2. CMAB选择策略
        selected_strategy = self.cmab.select_strategy(context)

        # 3. CTLC调整探测间隔
        adjustments = self.ctlc.batch_adjust(self.esm)

        # 4. PRIO选择Top-K任务
        selected_tasks = self.prio.select_top_k(
            self.uq.task_pool,
            self.esm,
            self.em,
            selected_strategy,
            k
        )

        # 更新统计信息
        self.stats['total_rounds'] += 1
        self.stats['total_tasks_selected'] += len(selected_tasks)
        self.stats['strategy_history'].append(selected_strategy)

        return {
            'tasks': selected_tasks,
            'strategy': selected_strategy,
            'context': context.tolist(),
            'interval_adjustments': adjustments
        }

    def update_reward(self, reward):
        """更新奖励（供RFU调用）"""
        context = self.esm.get_context_vector()
        self.cmab.update(context, reward)

    def get_statistics(self):
        """获取统计信息"""
        return {
            'aps_stats': self.stats,
            'cmab_stats': self.cmab.get_strategy_stats(),
            'recent_strategies': self.stats['strategy_history'][-10:]
        }