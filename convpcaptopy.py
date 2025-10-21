#!/usr/bin/env python3
"""
PCAP to CSV Converter

This module converts PCAP files to CSV format for easier analysis and
processing with standard data tools.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pyshark
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pcap_to_csv(
    input_pcap: str,
    output_csv: str = None,
    max_packets: Optional[int] = None,
    include_info: bool = True
) -> pd.DataFrame:
    """
    Convert a PCAP file to CSV format.

    Args:
        input_pcap: Path to input PCAP file
        output_csv: Path to output CSV file (default: input_name.csv)
        max_packets: Maximum number of packets to process
        include_info: Include packet info summary column

    Returns:
        DataFrame containing packet information

    Raises:
        FileNotFoundError: If input PCAP file doesn't exist
        Exception: If conversion fails
    """
    if not Path(input_pcap).exists():
        raise FileNotFoundError(f"PCAP file not found: {input_pcap}")

    if output_csv is None:
        output_csv = Path(input_pcap).stem + ".csv"

    logger.info(f"Reading PCAP file: {input_pcap}")

    try:
        capture = pyshark.FileCapture(input_pcap)
        packets_list = []
        packet_count = 0

        # Create progress bar
        pbar = tqdm(desc="Converting packets", unit="pkt")

        for packet in capture:
            try:
                packet_info = {
                    "No.": packet.number,
                    "Time": packet.sniff_time,
                    "Source": packet.ip.src if hasattr(packet, 'ip') else None,
                    "Destination": packet.ip.dst if hasattr(packet, 'ip') else None,
                    "Protocol": packet.highest_layer,
                    "Length": packet.length,
                }

                if include_info:
                    packet_info["Info"] = str(packet)[:100]

                packets_list.append(packet_info)
                packet_count += 1
                pbar.update(1)

                if max_packets and packet_count >= max_packets:
                    logger.info(f"Reached maximum packet limit: {max_packets}")
                    break

            except Exception as e:
                logger.debug(f"Error processing packet {packet_count}: {e}")
                continue

        pbar.close()
        capture.close()

        if not packets_list:
            logger.warning("No packets were successfully converted")
            return pd.DataFrame()

        logger.info(f"Converting {len(packets_list)} packets to DataFrame")
        df = pd.DataFrame(packets_list)

        logger.info(f"Saving to CSV: {output_csv}")
        df.to_csv(output_csv, index=False)

        logger.info(f"Conversion complete! {len(df)} packets saved to {output_csv}")

        # Print summary statistics
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Input file:       {input_pcap}")
        print(f"Output file:      {output_csv}")
        print(f"Packets converted: {len(df)}")
        if 'Source' in df.columns:
            print(f"Unique sources:    {df['Source'].nunique()}")
            print(f"Unique destinations: {df['Destination'].nunique()}")
        if 'Protocol' in df.columns:
            print(f"Protocols found:   {df['Protocol'].nunique()}")
            print("\nTop 5 protocols:")
            print(df['Protocol'].value_counts().head())
        print("="*60 + "\n")

        return df

    except Exception as e:
        logger.error(f"Error converting PCAP to CSV: {e}")
        raise


def main():
    """Command-line interface for PCAP to CSV conversion."""
    parser = argparse.ArgumentParser(
        description='PCAP to CSV Converter - Convert network captures to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s capture.pcap
  %(prog)s capture.pcap -o output.csv
  %(prog)s capture.pcap --max-packets 10000 --no-info
        """
    )

    parser.add_argument(
        'pcap_file',
        help='Path to input PCAP file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file path (default: input_name.csv)'
    )
    parser.add_argument(
        '--max-packets',
        type=int,
        help='Maximum number of packets to convert (default: all)'
    )
    parser.add_argument(
        '--no-info',
        action='store_true',
        help='Exclude packet info summary column'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        pcap_to_csv(
            input_pcap=args.pcap_file,
            output_csv=args.output,
            max_packets=args.max_packets,
            include_info=not args.no_info
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
