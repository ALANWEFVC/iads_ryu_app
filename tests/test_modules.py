# tests/test_modules.py
"""IADS模块单元测试"""

import pytest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iads_ryu_app.modules.esm import EntityStateManager, EntityState
from iads_ryu_app.modules.uq import UncertaintyQuantifier, Task
from iads_ryu_app.modules.aps import CMAB, CTLC, PRIO, ActiveProbingScheduler  # 添加ActiveProbingScheduler
from iads_ryu_app.modules.em import EventManager, Event
from iads_ryu_app.utils.distributions import BetaDistribution, GaussianDistribution
from iads_ryu_app.config import *



class TestDistributions:
    """测试概率分布类"""

    def test_beta_distribution(self):
        """测试Beta分布"""
        beta = BetaDistribution(alpha=2, beta=1)

        # 测试置信度计算
        assert beta.get_confidence() == 2 / 3

        # 测试更新
        beta.update(True)  # UP
        assert beta.alpha == 3
        assert beta.beta == 1

        beta.update(False)  # DOWN
        assert beta.alpha == 3
        assert beta.beta == 2

        # 测试熵计算
        entropy = beta.entropy()
        assert entropy > 0

    def test_gaussian_distribution(self):
        """测试高斯分布"""
        gaussian = GaussianDistribution(mu=10, sigma2=100)

        # 测试更新
        gaussian.update(15, noise_var=1)
        assert gaussian.mu != 10  # 均值应该改变
        assert gaussian.sigma2 < 100  # 方差应该减小

        # 测试熵计算
        entropy = gaussian.entropy()
        assert entropy > 0


class TestESM:
    """测试实体状态管理器"""

    def test_entity_state(self):
        """测试单个实体状态"""
        state = EntityState('link1', 'rtt')

        # 测试初始状态
        assert state.get_uncertainty() > 0
        assert state.get_stability() == 0

        # 测试更新
        state.update(10.5)
        state.update(11.0)
        state.update(10.8)

        # 稳定性应该增加
        assert state.get_stability() > 0

    def test_esm_manager(self):
        """测试ESM管理器"""
        esm = EntityStateManager()

        # 添加实体
        esm.add_entity('link1')
        assert 'link1' in esm.entities

        # 检查是否为所有指标创建了状态
        for metric in esm.metrics:
            state = esm.get_state('link1', metric)
            assert state is not None

        # 测试上下文向量
        context = esm.get_context_vector()
        assert len(context) == 4
        assert all(0 <= x <= 1 for x in context)


class TestUQ:
    """测试不确定性量化器"""

    def test_task_creation(self):
        """测试任务创建"""
        task = Task('link1', 'rtt')
        assert task.entity_id == 'link1'
        assert task.metric == 'rtt'
        assert task.eig == 0.0

    def test_eig_calculation(self):
        """测试EIG计算"""
        esm = EntityStateManager()
        esm.add_entity('link1')

        uq = UncertaintyQuantifier(esm)

        # 获取一个任务
        task = uq.task_pool[0]

        # 计算EIG
        eig = uq.calculate_eig(task)
        assert eig >= 0  # EIG应该非负


class TestAPS:
    """测试主动探测调度器"""

    def test_cmab(self):
        """测试CMAB"""
        cmab = CMAB(n_features=4, n_arms=4)

        # 测试策略选择
        context = np.array([0.5, 0.2, 0.3, 0.1])
        strategy = cmab.select_strategy(context)
        assert strategy in CMAB_STRATEGIES.values()

        # 测试更新
        cmab.update(context, reward=0.5)

        # 检查参数是否更新
        assert len(cmab.history) == 1

    def test_ctlc(self):
        """测试CTLC"""
        ctlc = CTLC(kp=0.1, target_stability=1.0)

        # 测试间隔调整
        # 高稳定性，应该增加间隔
        new_interval = ctlc.adjust_probe_interval(10, stability=0.5)
        assert new_interval > 10

        # 低稳定性，应该减少间隔
        new_interval = ctlc.adjust_probe_interval(10, stability=2.0)
        assert new_interval < 10

    def test_prio(self):
        """测试优先级引擎"""
        prio = PRIO()

        # 测试优先级计算
        priority = prio.calculate_priority(
            task=None,  # 这里简化
            eig=0.5,
            urgency=1.0,
            policy_match=0.8,
            event_trig=0.0
        )

        expected = (
                0.4 * 0.5 +  # EIG
                0.3 * 1.0 +  # Urgency
                0.2 * 0.8 +  # Policy Match
                0.1 * 0.0  # Event Trig
        )
        assert abs(priority - expected) < 0.001


class TestEM:
    """测试事件管理器"""

    def test_event_creation(self):
        """测试事件创建"""
        event = Event('link1', 'rtt', 'rtt_spike', severity=0.8)

        assert event.entity_id == 'link1'
        assert event.metric == 'rtt'
        assert event.event_type == 'rtt_spike'
        assert event.severity == 0.8

    def test_event_detection(self):
        """测试事件检测"""
        esm = EntityStateManager()
        esm.add_entity('link1')

        em = EventManager(esm)

        # 更新状态使其触发事件
        state = esm.get_state('link1', 'liveness')
        # 模拟低置信度
        state.distribution.alpha = 1
        state.distribution.beta = 10

        # 检测事件
        em.check_and_detect_events()

        # 应该有事件触发
        trigger = em.get_event_trigger('link1', 'liveness')
        assert trigger == 1.0


@pytest.fixture
def setup_iads():
    """设置IADS测试环境"""
    esm = EntityStateManager()
    esm.add_entity('link1')
    esm.add_entity('link2')

    uq = UncertaintyQuantifier(esm)
    em = EventManager(esm)
    aps = ActiveProbingScheduler(esm, uq, em)

    return {
        'esm': esm,
        'uq': uq,
        'em': em,
        'aps': aps
    }


def test_integration(setup_iads):
    """集成测试"""
    esm = setup_iads['esm']
    uq = setup_iads['uq']
    em = setup_iads['em']
    aps = setup_iads['aps']

    # 执行一轮完整的选择
    selection = aps.select_tasks(k=5)

    assert 'tasks' in selection
    assert 'strategy' in selection
    assert len(selection['tasks']) <= 5

    # 检查选中的任务
    for task_info in selection['tasks']:
        assert 'task' in task_info
        assert 'priority' in task_info
        assert task_info['priority'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
