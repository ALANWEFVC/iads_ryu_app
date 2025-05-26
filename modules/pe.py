# modules/pe.py
"""探测执行器(PE)模块"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, lldp, icmp, ipv4, arp
from ryu.ofproto import ether
import struct
import random
from config import *


class ProbeResult:
    """探测结果"""

    def __init__(self, task, success=False, value=None, timestamp=None):
        self.task = task
        self.success = success
        self.value = value
        self.timestamp = timestamp or time.time()
        self.error = None

    def to_dict(self):
        return {
            'entity_id': self.task.entity_id,
            'metric': self.task.metric,
            'success': self.success,
            'value': self.value,
            'timestamp': self.timestamp,
            'error': self.error
        }


class ProbeExecutor:
    """探测执行器"""

    def __init__(self, app):
        self.app = app  # Ryu应用引用
        self.datapath = None  # OpenFlow数据路径

        # 探测相关
        self.pending_probes = {}  # 待处理的探测
        self.probe_results = []  # 探测结果

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=10)

        # 统计信息
        self.stats = {
            'total_probes': 0,
            'successful_probes': 0,
            'failed_probes': 0,
            'probe_times': defaultdict(list)
        }

        # LLDP探测
        self.lldp_sent_time = {}

        # ICMP探测
        self.icmp_id_counter = 0
        self.icmp_pending = {}  # {icmp_id: (task, sent_time)}

    def set_datapath(self, datapath):
        """设置数据路径"""
        self.datapath = datapath

    def execute_batch(self, task_priorities):
        """批量执行探测任务

        Args:
            task_priorities: APS返回的任务优先级列表

        Returns:
            list: 探测结果列表
        """
        if not self.datapath:
            return []

        # 清空之前的结果
        self.probe_results = []

        # 按指标类型分组
        tasks_by_metric = defaultdict(list)
        for item in task_priorities:
            task = item['task']
            tasks_by_metric[task.metric].append(task)

        # 并行执行不同类型的探测
        futures = []

        for metric, tasks in tasks_by_metric.items():
            if metric == METRICS['LIVENESS']:
                future = self.executor.submit(self._probe_liveness_batch, tasks)
            elif metric == METRICS['RTT']:
                future = self.executor.submit(self._probe_rtt_batch, tasks)
            elif metric == METRICS['PLR']:
                future = self.executor.submit(self._probe_plr_batch, tasks)
            elif metric == METRICS['BANDWIDTH']:
                future = self.executor.submit(self._probe_bandwidth_batch, tasks)

            futures.append(future)

        # 等待所有探测完成
        for future in as_completed(futures, timeout=30):
            try:
                future.result()
            except Exception as e:
                self.app.logger.error(f"Probe execution error: {e}")

        # 更新统计信息
        self.stats['total_probes'] += len(task_priorities)

        return self.probe_results

    def _probe_liveness_batch(self, tasks):
        """批量执行Liveness探测（使用LLDP）"""
        for task in tasks:
            try:
                # 解析实体ID（假设格式为 "dpid1-port1:dpid2-port2"）
                parts = task.entity_id.split(':')
                if len(parts) != 2:
                    continue

                src_parts = parts[0].split('-')
                if len(src_parts) != 2:
                    continue

                src_dpid = int(src_parts[0])
                src_port = int(src_parts[1])

                # 发送LLDP包
                self._send_lldp(src_dpid, src_port)

                # 记录发送时间
                self.lldp_sent_time[(src_dpid, src_port)] = time.time()

                # 暂时假设成功（实际结果由LLDP响应决定）
                result = ProbeResult(task, success=True, value=True)
                self.probe_results.append(result)
                self.stats['successful_probes'] += 1

            except Exception as e:
                result = ProbeResult(task, success=False)
                result.error = str(e)
                self.probe_results.append(result)
                self.stats['failed_probes'] += 1

    def _probe_rtt_batch(self, tasks):
        """批量执行RTT探测（使用ICMP）"""
        for task in tasks:
            try:
                # 解析实体ID获取目标IP
                # 这里简化处理，实际应根据拓扑获取
                target_ip = self._get_target_ip_from_entity(task.entity_id)
                if not target_ip:
                    continue

                # 生成ICMP ID
                icmp_id = self._get_next_icmp_id()

                # 记录待处理的ICMP
                self.icmp_pending[icmp_id] = (task, time.time())

                # 发送ICMP Echo Request
                self._send_icmp_echo(target_ip, icmp_id)

                # 等待响应（这里简化处理，实际应异步）
                time.sleep(0.1)

                # 检查是否收到响应
                if icmp_id not in self.icmp_pending:
                    # 收到响应
                    rtt = self.stats['probe_times'][task.entity_id][-1] if task.entity_id in self.stats[
                        'probe_times'] else 10.0
                    result = ProbeResult(task, success=True, value=rtt)
                    self.stats['successful_probes'] += 1
                else:
                    # 超时
                    result = ProbeResult(task, success=False)
                    result.error = "Timeout"
                    self.stats['failed_probes'] += 1
                    del self.icmp_pending[icmp_id]

                self.probe_results.append(result)

            except Exception as e:
                result = ProbeResult(task, success=False)
                result.error = str(e)
                self.probe_results.append(result)
                self.stats['failed_probes'] += 1

    def _probe_plr_batch(self, tasks):
        """批量执行PLR探测"""
        for task in tasks:
            try:
                # 发送多个包计算丢包率
                num_packets = 10
                received = 0

                for i in range(num_packets):
                    # 这里简化处理，实际应发送测试包
                    if random.random() > 0.1:  # 假设10%丢包率
                        received += 1

                plr = 1.0 - (received / num_packets)

                result = ProbeResult(task, success=True, value=plr)
                self.probe_results.append(result)
                self.stats['successful_probes'] += 1

            except Exception as e:
                result = ProbeResult(task, success=False)
                result.error = str(e)
                self.probe_results.append(result)
                self.stats['failed_probes'] += 1

    def _probe_bandwidth_batch(self, tasks):
        """批量执行带宽探测"""
        for task in tasks:
            try:
                # 通过端口统计信息估算带宽
                # 这里简化处理，实际应读取OpenFlow统计
                bandwidth = random.uniform(100, 1000)  # Mbps

                result = ProbeResult(task, success=True, value=bandwidth)
                self.probe_results.append(result)
                self.stats['successful_probes'] += 1

            except Exception as e:
                result = ProbeResult(task, success=False)
                result.error = str(e)
                self.probe_results.append(result)
                self.stats['failed_probes'] += 1

    def _send_lldp(self, dpid, port):
        """发送LLDP包"""
        if not self.datapath or self.datapath.id != dpid:
            return

        pkt = packet.Packet()

        # 添加以太网头
        eth = ethernet.ethernet(
            dst=lldp.LLDP_MAC_NEAREST_BRIDGE,
            src=self._get_port_mac(dpid, port),
            ethertype=ether.ETH_TYPE_LLDP
        )
        pkt.add_protocol(eth)

        # 添加LLDP
        lldp_pkt = lldp.lldp()

        # Chassis ID TLV
        chassis_id = lldp.ChassisID(
            subtype=lldp.ChassisID.SUB_MAC_ADDRESS,
            chassis_id=struct.pack('!Q', dpid)[2:]
        )
        lldp_pkt.add_tlv(chassis_id)

        # Port ID TLV
        port_id = lldp.PortID(
            subtype=lldp.PortID.SUB_PORT_COMPONENT,
            port_id=struct.pack('!H', port)
        )
        lldp_pkt.add_tlv(port_id)

        # TTL TLV
        ttl = lldp.TTL(ttl=120)
        lldp_pkt.add_tlv(ttl)

        # End TLV
        end = lldp.End()
        lldp_pkt.add_tlv(end)

        pkt.add_protocol(lldp_pkt)

        # 发送
        self._send_packet(self.datapath, port, pkt)

    def _send_icmp_echo(self, target_ip, icmp_id):
        """发送ICMP Echo Request"""
        # 这里简化处理，实际实现需要完整的包构造
        pass

    def _send_packet(self, datapath, port, pkt):
        """发送数据包"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()
        data = pkt.data

        actions = [parser.OFPActionOutput(port=port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    def _get_port_mac(self, dpid, port):
        """获取端口MAC地址"""
        # 简化处理，实际应从端口信息获取
        return f"00:00:00:00:{dpid:02x}:{port:02x}"

    def _get_target_ip_from_entity(self, entity_id):
        """从实体ID获取目标IP"""
        # 简化处理，实际应根据拓扑映射
        return "10.0.0.1"

    def _get_next_icmp_id(self):
        """获取下一个ICMP ID"""
        self.icmp_id_counter = (self.icmp_id_counter + 1) % 65536
        return self.icmp_id_counter

    def handle_lldp_packet(self, dpid, port, lldp_pkt):
        """处理LLDP响应"""
        key = (dpid, port)
        if key in self.lldp_sent_time:
            rtt = time.time() - self.lldp_sent_time[key]
            del self.lldp_sent_time[key]
            # 更新相关任务的结果
            # 这里需要映射回具体的任务

    def handle_icmp_reply(self, icmp_id, src_ip):
        """处理ICMP响应"""
        if icmp_id in self.icmp_pending:
            task, sent_time = self.icmp_pending[icmp_id]
            rtt = (time.time() - sent_time) * 1000  # 转换为毫秒
            del self.icmp_pending[icmp_id]

            # 记录RTT
            self.stats['probe_times'][task.entity_id].append(rtt)
            if len(self.stats['probe_times'][task.entity_id]) > 100:
                self.stats['probe_times'][task.entity_id].pop(0)

    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_probes': self.stats['total_probes'],
            'successful_probes': self.stats['successful_probes'],
            'failed_probes': self.stats['failed_probes'],
            'success_rate': self.stats['successful_probes'] / self.stats['total_probes']
            if self.stats['total_probes'] > 0 else 0,
            'pending_lldp': len(self.lldp_sent_time),
            'pending_icmp': len(self.icmp_pending)
        }