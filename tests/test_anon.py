"""Unit tests for anon.py"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anon import PcapAnonymizer


class TestPcapAnonymizer:
    """Test PCAP anonymization functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = PcapAnonymizer(
            mac_prefix="ff:ff:ff:ff",
            ip_network="192.168",
            preserve_attacker_subnet="10.128"
        )

    def test_mac_anonymization(self):
        """Test MAC address anonymization."""
        mac1 = "aa:bb:cc:dd:ee:ff"
        mac2 = "11:22:33:44:55:66"

        # Anonymize MACs
        anon_mac1 = self.anonymizer._anonymize_mac(mac1)
        anon_mac2 = self.anonymizer._anonymize_mac(mac2)

        # Check format
        assert anon_mac1.startswith("ff:ff:ff:ff:")
        assert anon_mac2.startswith("ff:ff:ff:ff:")

        # Check consistency
        assert self.anonymizer._anonymize_mac(mac1) == anon_mac1
        assert self.anonymizer._anonymize_mac(mac2) == anon_mac2

        # Check uniqueness
        assert anon_mac1 != anon_mac2

    def test_ip_anonymization(self):
        """Test IP address anonymization."""
        ip1 = "8.8.8.8"
        ip2 = "1.1.1.1"

        # Anonymize IPs
        anon_ip1 = self.anonymizer._anonymize_ip(ip1)
        anon_ip2 = self.anonymizer._anonymize_ip(ip2)

        # Check format
        assert anon_ip1.startswith("192.168.")
        assert anon_ip2.startswith("192.168.")

        # Check consistency
        assert self.anonymizer._anonymize_ip(ip1) == anon_ip1
        assert self.anonymizer._anonymize_ip(ip2) == anon_ip2

        # Check uniqueness
        assert anon_ip1 != anon_ip2

    def test_attacker_ip_preservation(self):
        """Test that attacker IPs are preserved."""
        attacker_ip = "10.128.1.100"
        normal_ip = "8.8.8.8"

        # Anonymize
        anon_attacker = self.anonymizer._anonymize_ip(attacker_ip)
        anon_normal = self.anonymizer._anonymize_ip(normal_ip)

        # Attacker IP should be unchanged
        assert anon_attacker == attacker_ip

        # Normal IP should be anonymized
        assert anon_normal != normal_ip
        assert anon_normal.startswith("192.168.")

    def test_is_attacker_ip(self):
        """Test attacker IP detection."""
        assert self.anonymizer._is_attacker_ip("10.128.1.1") is True
        assert self.anonymizer._is_attacker_ip("10.128.255.255") is True
        assert self.anonymizer._is_attacker_ip("10.129.1.1") is False
        assert self.anonymizer._is_attacker_ip("192.168.1.1") is False
        assert self.anonymizer._is_attacker_ip("8.8.8.8") is False

    def test_custom_subnets(self):
        """Test custom subnet configuration."""
        custom_anonymizer = PcapAnonymizer(
            mac_prefix="aa:aa:aa:aa",
            ip_network="172.16",
            preserve_attacker_subnet="172.31"
        )

        # Test MAC prefix
        anon_mac = custom_anonymizer._anonymize_mac("11:22:33:44:55:66")
        assert anon_mac.startswith("aa:aa:aa:aa:")

        # Test IP network
        anon_ip = custom_anonymizer._anonymize_ip("8.8.8.8")
        assert anon_ip.startswith("172.16.")

        # Test attacker subnet
        assert custom_anonymizer._is_attacker_ip("172.31.1.1") is True
        assert custom_anonymizer._is_attacker_ip("10.128.1.1") is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
