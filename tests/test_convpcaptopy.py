"""Unit tests for convpcaptopy.py"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from convpcaptopy import pcap_to_csv


class TestPcapToCsv:
    """Test PCAP to CSV conversion functions."""

    @patch('convpcaptopy.pyshark.FileCapture')
    def test_pcap_to_csv_basic(self, mock_capture, tmp_path):
        """Test basic PCAP to CSV conversion."""
        # Mock packets
        mock_packet1 = Mock()
        mock_packet1.number = 1
        mock_packet1.sniff_time = "2024-01-01 12:00:00"
        mock_packet1.ip.src = "192.168.1.1"
        mock_packet1.ip.dst = "8.8.8.8"
        mock_packet1.highest_layer = "TCP"
        mock_packet1.length = 100
        mock_packet1.__str__ = Mock(return_value="Test packet info")

        mock_packet2 = Mock()
        mock_packet2.number = 2
        mock_packet2.sniff_time = "2024-01-01 12:00:01"
        mock_packet2.ip.src = "192.168.1.2"
        mock_packet2.ip.dst = "1.1.1.1"
        mock_packet2.highest_layer = "UDP"
        mock_packet2.length = 200
        mock_packet2.__str__ = Mock(return_value="Test packet info 2")

        # Setup mock capture
        mock_capture_instance = MagicMock()
        mock_capture_instance.__iter__ = Mock(return_value=iter([mock_packet1, mock_packet2]))
        mock_capture.return_value = mock_capture_instance

        # Create test input/output paths
        input_pcap = str(tmp_path / "test.pcap")
        output_csv = str(tmp_path / "test_output.csv")

        # Create dummy input file
        Path(input_pcap).touch()

        # Run conversion
        df = pcap_to_csv(input_pcap, output_csv, include_info=True)

        # Verify results
        assert len(df) == 2
        assert 'Source' in df.columns
        assert 'Destination' in df.columns
        assert 'Protocol' in df.columns
        assert 'Length' in df.columns
        assert df['Source'].iloc[0] == "192.168.1.1"
        assert df['Protocol'].iloc[1] == "UDP"

    def test_pcap_to_csv_file_not_found(self):
        """Test handling of non-existent PCAP file."""
        with pytest.raises(FileNotFoundError):
            pcap_to_csv("nonexistent.pcap")

    @patch('convpcaptopy.pyshark.FileCapture')
    def test_pcap_to_csv_no_info(self, mock_capture, tmp_path):
        """Test conversion without info column."""
        mock_packet = Mock()
        mock_packet.number = 1
        mock_packet.sniff_time = "2024-01-01 12:00:00"
        mock_packet.ip.src = "192.168.1.1"
        mock_packet.ip.dst = "8.8.8.8"
        mock_packet.highest_layer = "TCP"
        mock_packet.length = 100

        mock_capture_instance = MagicMock()
        mock_capture_instance.__iter__ = Mock(return_value=iter([mock_packet]))
        mock_capture.return_value = mock_capture_instance

        input_pcap = str(tmp_path / "test.pcap")
        output_csv = str(tmp_path / "test_output.csv")
        Path(input_pcap).touch()

        df = pcap_to_csv(input_pcap, output_csv, include_info=False)

        assert 'Info' not in df.columns

    @patch('convpcaptopy.pyshark.FileCapture')
    def test_pcap_to_csv_max_packets(self, mock_capture, tmp_path):
        """Test max_packets limit."""
        # Create 100 mock packets
        mock_packets = []
        for i in range(100):
            mock_packet = Mock()
            mock_packet.number = i
            mock_packet.sniff_time = f"2024-01-01 12:00:{i:02d}"
            mock_packet.ip.src = f"192.168.1.{i % 255}"
            mock_packet.ip.dst = "8.8.8.8"
            mock_packet.highest_layer = "TCP"
            mock_packet.length = 100
            mock_packet.__str__ = Mock(return_value="Test")
            mock_packets.append(mock_packet)

        mock_capture_instance = MagicMock()
        mock_capture_instance.__iter__ = Mock(return_value=iter(mock_packets))
        mock_capture.return_value = mock_capture_instance

        input_pcap = str(tmp_path / "test.pcap")
        output_csv = str(tmp_path / "test_output.csv")
        Path(input_pcap).touch()

        # Limit to 10 packets
        df = pcap_to_csv(input_pcap, output_csv, max_packets=10)

        assert len(df) == 10

    @patch('convpcaptopy.pyshark.FileCapture')
    def test_pcap_to_csv_no_ip_packets(self, mock_capture, tmp_path):
        """Test handling of packets without IP layer."""
        # Mock non-IP packet
        mock_packet = Mock()
        mock_packet.number = 1
        mock_packet.sniff_time = "2024-01-01 12:00:00"
        mock_packet.highest_layer = "ARP"
        mock_packet.length = 60
        mock_packet.__str__ = Mock(return_value="ARP packet")
        # No ip attribute
        del mock_packet.ip

        mock_capture_instance = MagicMock()
        mock_capture_instance.__iter__ = Mock(return_value=iter([mock_packet]))
        mock_capture.return_value = mock_capture_instance

        input_pcap = str(tmp_path / "test.pcap")
        output_csv = str(tmp_path / "test_output.csv")
        Path(input_pcap).touch()

        df = pcap_to_csv(input_pcap, output_csv)

        # Should still process packet, but with None for Source/Destination
        assert len(df) == 1
        assert pd.isna(df['Source'].iloc[0])
        assert pd.isna(df['Destination'].iloc[0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
