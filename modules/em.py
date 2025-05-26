# modules/em.py
"""事件管理器(EM)模块"""

import time
from collections import defaultdict, deque
import numpy as np
from config import *


class Event:
    """事件类"""

    def __init__(self, entity_id, metric, event_type, severity, timestamp=None):
        self.entity_id = entity_id
        self.metric = metric
        self.event_type = event_type
        self.severity = severity  # 0.0 - 1.0
        self.timestamp = timestamp or time.time()
        self.details = {}

    def to_dict(self):
        return {
            'entity_id': self.entity_id,
            'metric': self.metric,
            'event_type': self.event_type,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'details': self.details
        }


class EventManager:
    """事件管理器"""

    def __init__(self, esm):
        self.esm = esm

        # 事件存储
        self.events = deque(maxlen=10000)  # 最多保存10000个事件
        self.recent_events = deque()  # 最近的事件（滑动窗口）

        # 事件触发状态
        self.event_triggers = {}  # {(entity_id, metric): 0.0 or 1.0}

        # 历史测量值（用于异常检测）
        self.measurement_history = defaultdict(lambda: deque(maxlen=100))

        # 统计信息
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_entity': defaultdict(int),
            'events_by_metric': defaultdict(int)
        }

        # 核心实体列表（可配置）
        self.core_entities = set()

    def add_core_entity(self, entity_id):
        """添加核心实体"""
        self.core_entities.add(entity_id)

    def check_and_detect_events(self):
        """检查并检测事件"""
        current_time = time.time()

        # 清理旧的最近事件
        self._clean_recent_events(current_time)

        # 重置事件触发
        self.event_triggers.clear()

        # 检测每个实体-指标的异常
        for (entity_id, metric), state in self.esm.state_table.items():
            events = self._detect_anomalies(entity_id, metric, state)

            for event in events:
                self._add_event(event)
                # 设置事件触发
                self.event_triggers[(entity_id, metric)] = 1.0

                # 如果是核心实体，扩展触发
                if entity_id in self.core_entities and metric == METRICS['RTT']:
                    # 扩展到其他指标
                    for other_metric in [METRICS['PLR'], METRICS['BANDWIDTH']]:
                        self.event_triggers[(entity_id, other_metric)] = 1.0

    def _detect_anomalies(self, entity_id, metric, state):
        """检测异常

        Args:
            entity_id: 实体ID
            metric: 指标
            state: EntityState对象

        Returns:
            list: 检测到的事件列表
        """
        events = []

        if metric == METRICS['LIVENESS']:
            # Liveness异常检测
            p_up = state.distribution.get_confidence()
            if p_up < EVENT_THRESHOLDS['liveness_threshold']:
                event = Event(
                    entity_id, metric,
                    'liveness_low',
                    severity=1.0 - p_up
                )
                event.details['p_up'] = p_up
                events.append(event)

        else:
            # RTT/PLR/Bandwidth异常检测
            # 1. 稳定性异常
            stability = state.get_stability()
            if stability > EVENT_THRESHOLDS['stability_threshold'] / APS_CONFIG['max_stability']:
                event = Event(
                    entity_id, metric,
                    'high_instability',
                    severity=min(stability, 1.0)
                )
                event.details['stability'] = stability
                events.append(event)

            # 2. 测量值异常（需要历史数据）
            if hasattr(state.distribution, 'mu'):
                current_value = state.distribution.mu
                history_key = (entity_id, metric)

                # 添加到历史
                self.measurement_history[history_key].append(current_value)

                # 如果有足够的历史数据
                if len(self.measurement_history[history_key]) >= 10:
                    history = list(self.measurement_history[history_key])
                    hist_mean = np.mean(history[:-1])  # 不包括当前值
                    hist_std = np.std(history[:-1])

                    if hist_std > 0:
                        # 检查是否偏离历史均值
                        if metric == METRICS['RTT']:
                            threshold = EVENT_THRESHOLDS['rtt_spike_factor'] * hist_std
                            if abs(current_value - hist_mean) > threshold:
                                event = Event(
                                    entity_id, metric,
                                    'rtt_spike',
                                    severity=min(abs(current_value - hist_mean) / (5 * hist_std), 1.0)
                                )
                                event.details['current'] = current_value
                                event.details['historical_mean'] = hist_mean
                                event.details['deviation'] = abs(current_value - hist_mean)
                                events.append(event)

        return events

    def _add_event(self, event):
        """添加事件"""
        self.events.append(event)
        self.recent_events.append(event)

        # 更新统计
        self.stats['total_events'] += 1
        self.stats['events_by_type'][event.event_type] += 1
        self.stats['events_by_entity'][event.entity_id] += 1
        self.stats['events_by_metric'][event.metric] += 1

    def _clean_recent_events(self, current_time):
        """清理旧的最近事件"""
        cutoff_time = current_time - SYSTEM_CONFIG['sliding_window']

        while self.recent_events and self.recent_events[0].timestamp < cutoff_time:
            self.recent_events.popleft()

    def get_event_trigger(self, entity_id, metric):
        """获取事件触发信号

        Args:
            entity_id: 实体ID
            metric: 指标

        Returns:
            float: 0.0 或 1.0
        """
        return self.event_triggers.get((entity_id, metric), 0.0)

    def get_num_recent_events(self):
        """获取最近事件数"""
        return len(self.recent_events)

    def get_num_recent_events_normalized(self):
        """获取归一化的最近事件数"""
        return min(len(self.recent_events) / EVENT_THRESHOLDS['max_recent_events'], 1.0)

    def update_context_in_esm(self):
        """更新ESM中的事件相关上下文"""
        # 这个方法允许ESM获取num_recent_events
        # 实际实现中，ESM的get_context_vector会调用EM
        pass

    def get_recent_events(self, limit=20):
        """获取最近的事件"""
        events = list(self.recent_events)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in events[:limit]]

    def get_events_by_entity(self, entity_id, limit=10):
        """获取特定实体的事件"""
        entity_events = [e for e in self.events if e.entity_id == entity_id]
        entity_events.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in entity_events[:limit]]

    def get_statistics(self):
        """获取统计信息"""
        # 计算事件频率
        event_rate = 0
        if self.recent_events:
            time_span = time.time() - self.recent_events[0].timestamp
            if time_span > 0:
                event_rate = len(self.recent_events) / time_span

        # 获取最活跃的实体
        top_entities = sorted(
            self.stats['events_by_entity'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_events': self.stats['total_events'],
            'recent_events_count': len(self.recent_events),
            'event_rate': event_rate,  # 事件/秒
            'events_by_type': dict(self.stats['events_by_type']),
            'events_by_metric': dict(self.stats['events_by_metric']),
            'top_entities': top_entities,
            'active_triggers': sum(1 for v in self.event_triggers.values() if v > 0)
        }

    def get_anomaly_summary(self):
        """获取异常摘要"""
        summary = {
            'liveness_issues': [],
            'instability_issues': [],
            'value_spikes': []
        }

        # 分析最近的事件
        for event in self.recent_events:
            event_dict = event.to_dict()

            if event.event_type == 'liveness_low':
                summary['liveness_issues'].append({
                    'entity': event.entity_id,
                    'p_up': event.details.get('p_up', 0),
                    'time': event.timestamp
                })

            elif event.event_type == 'high_instability':
                summary['instability_issues'].append({
                    'entity': event.entity_id,
                    'metric': event.metric,
                    'stability': event.details.get('stability', 0),
                    'time': event.timestamp
                })

            elif event.event_type in ['rtt_spike', 'plr_spike']:
                summary['value_spikes'].append({
                    'entity': event.entity_id,
                    'metric': event.metric,
                    'current': event.details.get('current', 0),
                    'deviation': event.details.get('deviation', 0),
                    'time': event.timestamp
                })

        return summary