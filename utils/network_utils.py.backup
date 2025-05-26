# utils/network_utils.py
"""网络工具函数"""

import struct
import socket
from ryu.lib import mac


def dpid_to_str(dpid):
    """将DPID转换为字符串格式"""
    return '%016x' % dpid


def str_to_dpid(dpid_str):
    """将字符串转换为DPID"""
    return int(dpid_str, 16)


def mac_to_int(mac_str):
    """将MAC地址字符串转换为整数"""
    return int(mac_str.replace(':', ''), 16)


def int_to_mac(mac_int):
    """将整数转换为MAC地址字符串"""
    mac_str = '%012x' % mac_int
    return ':'.join([mac_str[i:i + 2] for i in range(0, 12, 2)])


def ip_to_int(ip_str):
    """将IP地址字符串转换为整数"""
    return struct.unpack('!I', socket.inet_aton(ip_str))[0]


def int_to_ip(ip_int):
    """将整数转换为IP地址字符串"""
    return socket.inet_ntoa(struct.pack('!I', ip_int))


def create_entity_id(src_dpid, src_port, dst_dpid, dst_port):
    """创建实体ID（链路ID）"""
    return f"{src_dpid}-{src_port}:{dst_dpid}-{dst_port}"


def parse_entity_id(entity_id):
    """解析实体ID

    Returns:
        tuple: (src_dpid, src_port, dst_dpid, dst_port)
    """
    try:
        parts = entity_id.split(':')
        if len(parts) != 2:
            return None

        src_parts = parts[0].split('-')
        dst_parts = parts[1].split('-')

        if len(src_parts) != 2 or len(dst_parts) != 2:
            return None

        return (
            int(src_parts[0]),
            int(src_parts[1]),
            int(dst_parts[0]),
            int(dst_parts[1])
        )
    except:
        return None


def get_link_bandwidth_mbps(port_stats):
    """从端口统计信息计算带宽（Mbps）

    Args:
        port_stats: OpenFlow端口统计

    Returns:
        float: 带宽（Mbps）
    """
    # 这是简化的计算，实际应该基于时间间隔
    tx_bytes = port_stats.tx_bytes
    rx_bytes = port_stats.rx_bytes

    # 假设统计间隔为1秒
    bandwidth_bps = (tx_bytes + rx_bytes) * 8
    bandwidth_mbps = bandwidth_bps / 1000000.0

    return bandwidth_mbps


def calculate_rtt_from_timestamps(sent_time, received_time):
    """计算RTT（毫秒）"""
    return (received_time - sent_time) * 1000


class TopologyHelper:
    """拓扑辅助类"""

    def __init__(self):
        self.adjacency = {}  # {dpid: {port: (neighbor_dpid, neighbor_port)}}
        self.hosts = {}  # {mac: (dpid, port)}
        self.switch_ports = {}  # {dpid: set(ports)}

    def add_link(self, src_dpid, src_port, dst_dpid, dst_port):
        """添加链路"""
        if src_dpid not in self.adjacency:
            self.adjacency[src_dpid] = {}
        if dst_dpid not in self.adjacency:
            self.adjacency[dst_dpid] = {}

        self.adjacency[src_dpid][src_port] = (dst_dpid, dst_port)
        self.adjacency[dst_dpid][dst_port] = (src_dpid, src_port)

        # 更新端口信息
        if src_dpid not in self.switch_ports:
            self.switch_ports[src_dpid] = set()
        if dst_dpid not in self.switch_ports:
            self.switch_ports[dst_dpid] = set()

        self.switch_ports[src_dpid].add(src_port)
        self.switch_ports[dst_dpid].add(dst_port)

    def add_host(self, mac, dpid, port):
        """添加主机"""
        self.hosts[mac] = (dpid, port)

        if dpid not in self.switch_ports:
            self.switch_ports[dpid] = set()
        self.switch_ports[dpid].add(port)

    def get_neighbor(self, dpid, port):
        """获取邻居"""
        if dpid in self.adjacency and port in self.adjacency[dpid]:
            return self.adjacency[dpid][port]
        return None

    def get_all_links(self):
        """获取所有链路"""
        links = []
        seen = set()

        for src_dpid, ports in self.adjacency.items():
            for src_port, (dst_dpid, dst_port) in ports.items():
                link_id = tuple(sorted([(src_dpid, src_port), (dst_dpid, dst_port)]))
                if link_id not in seen:
                    seen.add(link_id)
                    links.append({
                        'src': {'dpid': src_dpid, 'port': src_port},
                        'dst': {'dpid': dst_dpid, 'port': dst_port}
                    })

        return links

    def get_host_links(self):
        """获取主机链路"""
        links = []
        for mac, (dpid, port) in self.hosts.items():
            links.append({
                'host': mac,
                'switch': {'dpid': dpid, 'port': port}
            })
        return links