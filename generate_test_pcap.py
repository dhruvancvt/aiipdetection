#!/usr/bin/env python3
"""
Create a simple test PCAP file using raw packet writing.
This avoids complex dependencies.
"""

import struct
import time
import random

def write_pcap_header(f):
    """Write PCAP file header."""
    # PCAP global header
    # magic_number: 0xa1b2c3d4
    # version_major: 2
    # version_minor: 4
    # thiszone: 0
    # sigfigs: 0
    # snaplen: 65535
    # network: 1 (Ethernet)
    f.write(struct.pack('IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))

def write_pcap_packet(f, timestamp, data):
    """Write a single packet to PCAP file."""
    # Packet header
    # ts_sec: timestamp seconds
    # ts_usec: timestamp microseconds
    # incl_len: number of octets of packet saved
    # orig_len: actual length of packet
    ts_sec = int(timestamp)
    ts_usec = int((timestamp - ts_sec) * 1000000)
    packet_len = len(data)

    f.write(struct.pack('IIII', ts_sec, ts_usec, packet_len, packet_len))
    f.write(data)

def create_ethernet_frame(src_mac, dst_mac, ethertype, payload):
    """Create an Ethernet frame."""
    src_mac_bytes = bytes.fromhex(src_mac.replace(':', ''))
    dst_mac_bytes = bytes.fromhex(dst_mac.replace(':', ''))
    return dst_mac_bytes + src_mac_bytes + struct.pack('!H', ethertype) + payload

def create_ipv4_packet(src_ip, dst_ip, protocol, payload):
    """Create a simplified IPv4 packet."""
    # IP header (simplified)
    version_ihl = (4 << 4) | 5  # Version 4, IHL 5 (20 bytes)
    tos = 0
    total_length = 20 + len(payload)
    identification = random.randint(0, 65535)
    flags_fragment = 0
    ttl = 64
    checksum = 0  # Simplified: not calculating real checksum

    src_ip_bytes = bytes([int(x) for x in src_ip.split('.')])
    dst_ip_bytes = bytes([int(x) for x in dst_ip.split('.')])

    header = struct.pack('!BBHHHBBH',
                        version_ihl, tos, total_length,
                        identification, flags_fragment,
                        ttl, protocol, checksum)
    header += src_ip_bytes + dst_ip_bytes

    return header + payload

def create_tcp_segment(src_port, dst_port, payload):
    """Create a simplified TCP segment."""
    seq = random.randint(0, 0xFFFFFFFF)
    ack = random.randint(0, 0xFFFFFFFF)
    offset_flags = (5 << 12) | 0x002  # Offset 5, SYN flag
    window = 65535
    checksum = 0  # Simplified
    urgent = 0

    header = struct.pack('!HHIIHHHH',
                        src_port, dst_port, seq, ack,
                        offset_flags, window, checksum, urgent)

    return header + payload

def create_udp_datagram(src_port, dst_port, payload):
    """Create a simplified UDP datagram."""
    length = 8 + len(payload)
    checksum = 0  # Simplified

    header = struct.pack('!HHHH', src_port, dst_port, length, checksum)
    return header + payload

def generate_test_pcap(output_file="test_capture.pcap", num_packets=1000):
    """Generate a test PCAP file."""
    print(f"Creating test PCAP file: {output_file}")
    print(f"Generating {num_packets} packets...")

    with open(output_file, 'wb') as f:
        write_pcap_header(f)

        # Normal traffic sources
        normal_sources = [
            "192.168.1.10",
            "192.168.1.11",
            "192.168.1.12",
            "192.168.1.13",
            "192.168.1.14"
        ]

        # Attacker IP (will show anomalous behavior)
        attacker_ip = "10.128.1.100"

        # Destinations
        normal_destinations = [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "151.101.1.69",
            "172.217.14.206"
        ]

        current_time = time.time()

        # Generate 80% normal traffic
        normal_count = int(num_packets * 0.8)

        for i in range(normal_count):
            timestamp = current_time + i * 0.01
            src_ip = random.choice(normal_sources)
            dst_ip = random.choice(normal_destinations)
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 53])

            # Create packet
            payload = b"X" * random.randint(50, 200)

            if i % 2 == 0:
                transport = create_tcp_segment(src_port, dst_port, payload)
                ip_packet = create_ipv4_packet(src_ip, dst_ip, 6, transport)  # 6 = TCP
            else:
                transport = create_udp_datagram(src_port, dst_port, payload)
                ip_packet = create_ipv4_packet(src_ip, dst_ip, 17, transport)  # 17 = UDP

            eth_frame = create_ethernet_frame(
                "00:11:22:33:44:55",
                "aa:bb:cc:dd:ee:ff",
                0x0800,  # IPv4
                ip_packet
            )

            write_pcap_packet(f, timestamp, eth_frame)

        # Generate 20% anomalous traffic (attacker)
        anomalous_count = num_packets - normal_count

        # Create many unique destinations
        attacker_destinations = []
        for i in range(1, 255):
            attacker_destinations.append(f"10.0.0.{i}")
            attacker_destinations.append(f"192.168.2.{i}")

        for i in range(anomalous_count):
            timestamp = current_time + (normal_count + i) * 0.01
            dst_ip = random.choice(attacker_destinations)
            src_port = random.randint(1024, 65535)
            dst_port = random.randint(1, 1024)  # Port scanning

            # Small packets (typical scanning behavior)
            payload = b"X" * random.randint(40, 80)

            transport = create_tcp_segment(src_port, dst_port, payload)
            ip_packet = create_ipv4_packet(attacker_ip, dst_ip, 6, transport)

            eth_frame = create_ethernet_frame(
                "00:11:22:33:44:55",
                "aa:bb:cc:dd:ee:ff",
                0x0800,
                ip_packet
            )

            write_pcap_packet(f, timestamp, eth_frame)

    print(f"\n✓ Test PCAP file created successfully!")
    print(f"  File: {output_file}")
    print(f"  Total packets: {num_packets}")
    print(f"  Normal traffic: {normal_count} packets from {len(normal_sources)} IPs")
    print(f"  Anomalous traffic: {anomalous_count} packets from attacker IP: {attacker_ip}")
    print(f"\nExpected behavior:")
    print(f"  - Attacker IP {attacker_ip} should be detected as anomalous")
    print(f"  - It has many packets to many unique destinations (scanning pattern)")
    print(f"\nYou can now test with:")
    print(f"  python model.py {output_file}")
    print(f"  python convpcaptopy.py {output_file}")
    print(f"  python anon.py {output_file}")
    print(f"  python visualize.py unsupervised_processed_output.csv")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create simple test PCAP file')
    parser.add_argument(
        '-o', '--output',
        default='test_capture.pcap',
        help='Output PCAP file (default: test_capture.pcap)'
    )
    parser.add_argument(
        '-n', '--num-packets',
        type=int,
        default=1000,
        help='Number of packets (default: 1000)'
    )

    args = parser.parse_args()
    generate_test_pcap(args.output, args.num_packets)
