#!/usr/bin/env python3
"""
PCAP Anonymization Tool

This module anonymizes MAC and IP addresses in PCAP files while preserving
attacker IPs for security analysis. Useful for sharing network captures while
protecting sensitive information.
"""

import argparse
import logging
import sys
import gc
from pathlib import Path
from typing import Dict, List

from scapy.all import rdpcap, wrpcap, Ether, IP
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PcapAnonymizer:
    """Anonymizes MAC and IP addresses in PCAP files."""

    def __init__(
        self,
        mac_prefix: str = "ff:ff:ff:ff",
        ip_network: str = "192.168",
        preserve_attacker_subnet: str = "10.128"
    ):
        """
        Initialize the anonymizer.

        Args:
            mac_prefix: Prefix for anonymized MAC addresses
            ip_network: Network prefix for anonymized IPs (e.g., "192.168")
            preserve_attacker_subnet: Subnet to preserve (e.g., "10.128")
        """
        self.macs: Dict[str, str] = {}
        self.ips: Dict[str, str] = {}
        self.new_ips = [0, 1]
        self.new_mac = [0, 1]
        self.mac_prefix = mac_prefix
        self.ip_network = ip_network
        self.preserve_attacker_subnet = preserve_attacker_subnet
        self.pkt_counter = 0
        self.attacker_ips: List[str] = []

    def _anonymize_mac(self, mac: str) -> str:
        """
        Anonymize a MAC address.

        Args:
            mac: Original MAC address

        Returns:
            Anonymized MAC address
        """
        if mac in self.macs:
            return self.macs[mac]

        tmp_mac = f"{self.mac_prefix}:{str(self.new_mac[0]).zfill(2)}:{str(self.new_mac[1]).zfill(2)}"
        self.macs[mac] = tmp_mac
        self.new_mac[1] += 1

        if self.new_mac[1] >= 99:
            self.new_mac[0] += 1
            self.new_mac[1] = 0

        return tmp_mac

    def _is_attacker_ip(self, ip: str) -> bool:
        """
        Check if IP belongs to attacker subnet.

        Args:
            ip: IP address to check

        Returns:
            True if IP is in attacker subnet
        """
        parts = ip.split('.')
        subnet_parts = self.preserve_attacker_subnet.split('.')

        try:
            return (
                len(parts) >= len(subnet_parts) and
                all(int(parts[i]) == int(subnet_parts[i]) for i in range(len(subnet_parts)))
            )
        except (ValueError, IndexError):
            return False

    def _anonymize_ip(self, ip: str) -> str:
        """
        Anonymize an IP address (unless it's an attacker IP).

        Args:
            ip: Original IP address

        Returns:
            Anonymized IP address or original if attacker
        """
        if self._is_attacker_ip(ip):
            if ip not in self.attacker_ips:
                self.attacker_ips.append(ip)
                logger.info(f"Found attacker IP (preserving): {ip}")
            return ip

        if ip in self.ips:
            return self.ips[ip]

        anon_ip = f"{self.ip_network}.{self.new_ips[0]}.{self.new_ips[1]}"
        self.ips[ip] = anon_ip
        self.new_ips[1] += 1

        if self.new_ips[1] == 255:
            self.new_ips[0] += 1
            self.new_ips[1] = 0

        return anon_ip

    def anonymize_file(self, input_file: str, output_file: str = None) -> None:
        """
        Anonymize a single PCAP file.

        Args:
            input_file: Path to input PCAP file
            output_file: Path to output file (default: input_file-anon.pcap)
        """
        if not Path(input_file).exists():
            raise FileNotFoundError(f"PCAP file not found: {input_file}")

        if output_file is None:
            output_file = f"{input_file}-anon.pcap"

        logger.info(f"Reading PCAP file: {input_file}")

        try:
            pcap = rdpcap(input_file)
            logger.info(f"Loaded {len(pcap)} packets, starting anonymization")

            anonymized_packets = []

            for pkt in tqdm(pcap, desc="Anonymizing packets", unit="pkt"):
                self.pkt_counter += 1

                # Anonymize MAC addresses
                if Ether in pkt:
                    if pkt[Ether].src:
                        pkt[Ether].src = self._anonymize_mac(pkt[Ether].src)
                    if pkt[Ether].dst:
                        pkt[Ether].dst = self._anonymize_mac(pkt[Ether].dst)

                # Anonymize IP addresses
                if IP in pkt:
                    if pkt[IP].src:
                        pkt[IP].src = self._anonymize_ip(pkt[IP].src)
                    if pkt[IP].dst:
                        pkt[IP].dst = self._anonymize_ip(pkt[IP].dst)

                anonymized_packets.append(pkt)

            # Write anonymized packets to file
            logger.info(f"Writing anonymized PCAP to: {output_file}")
            wrpcap(output_file, anonymized_packets)

            # Print summary
            logger.info(f"Anonymization complete!")
            logger.info(f"  MAC addresses anonymized: {len(self.macs)}")
            logger.info(f"  IP addresses anonymized: {len(self.ips)}")
            logger.info(f"  Attacker IPs preserved: {len(self.attacker_ips)}")
            logger.info(f"  Total packets processed: {self.pkt_counter}")
            logger.info(f"  Output file: {output_file}")

            # Clean up
            del pcap
            gc.collect()

        except Exception as e:
            logger.error(f"Error anonymizing {input_file}: {e}")
            raise

    def anonymize_files(self, input_files: List[str], output_dir: str = None) -> None:
        """
        Anonymize multiple PCAP files.

        Args:
            input_files: List of input PCAP file paths
            output_dir: Directory for output files (default: same as input)
        """
        for file_path in input_files:
            try:
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    filename = Path(file_path).name
                    output_file = str(Path(output_dir) / f"{filename}-anon.pcap")
                else:
                    output_file = None

                self.anonymize_file(file_path, output_file)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue


def main():
    """Command-line interface for PCAP anonymization."""
    parser = argparse.ArgumentParser(
        description='PCAP Anonymization Tool - Anonymize MAC and IP addresses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s capture.pcap
  %(prog)s capture1.pcap capture2.pcap -o ./anonymized
  %(prog)s *.pcap --preserve-subnet 10.128 --ip-network 172.16
        """
    )

    parser.add_argument(
        'pcap_files',
        nargs='+',
        help='PCAP file(s) to anonymize'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for anonymized files (default: same as input)'
    )
    parser.add_argument(
        '--mac-prefix',
        default='ff:ff:ff:ff',
        help='MAC address prefix for anonymization (default: ff:ff:ff:ff)'
    )
    parser.add_argument(
        '--ip-network',
        default='192.168',
        help='IP network prefix for anonymization (default: 192.168)'
    )
    parser.add_argument(
        '--preserve-subnet',
        default='10.128',
        help='Subnet to preserve (attacker IPs, default: 10.128)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create anonymizer
    anonymizer = PcapAnonymizer(
        mac_prefix=args.mac_prefix,
        ip_network=args.ip_network,
        preserve_attacker_subnet=args.preserve_subnet
    )

    # Process files
    anonymizer.anonymize_files(args.pcap_files, args.output_dir)


if __name__ == "__main__":
    main()
