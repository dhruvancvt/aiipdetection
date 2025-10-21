#!/usr/bin/env python3
"""
AI-driven Network Anomaly Detection System

This module analyzes PCAP files to identify potentially malicious network behavior
using unsupervised machine learning (Isolation Forest algorithm).
"""

import argparse
import logging
import sys
from pathlib import Path
import pickle
from typing import Optional, Tuple

from tqdm import tqdm
import pyshark
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('anomaly_detection.log')
    ]
)
logger = logging.getLogger(__name__)


def read_pcap(file_path: str, max_packets: Optional[int] = None) -> pd.DataFrame:
    """
    Read and parse a PCAP file into a pandas DataFrame.

    Args:
        file_path: Path to the PCAP file
        max_packets: Maximum number of packets to read (None for all)

    Returns:
        DataFrame containing packet information (Source, Destination, Length)

    Raises:
        FileNotFoundError: If the PCAP file doesn't exist
        Exception: If the file cannot be read
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"PCAP file not found: {file_path}")

    logger.info(f"Reading PCAP file: {file_path}")

    try:
        capture = pyshark.FileCapture(file_path, use_json=True, include_raw=False)
        data = []

        for i, packet in enumerate(tqdm(capture, desc="Reading packets", unit="packet")):
            if 'IP' in packet:
                try:
                    data.append({
                        'Source': packet.ip.src,
                        'Destination': packet.ip.dst,
                        'Length': int(packet.length)
                    })
                except AttributeError:
                    continue
            if max_packets and i >= max_packets - 1:
                break

        capture.close()
        logger.info(f"Successfully read {len(data)} packets")
        return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"Error reading PCAP file: {e}")
        raise


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from packet data for anomaly detection.

    Features created:
    - packet_count: Number of packets per source IP
    - average_length: Average packet length per source IP
    - unique_destinations: Number of unique destination IPs per source

    Args:
        dataframe: DataFrame with packet data

    Returns:
        DataFrame with additional feature columns
    """
    logger.info("Engineering features from packet data")
    tqdm.pandas(desc="Adding features")

    dataframe['packet_count'] = dataframe.groupby('Source')['Source'].transform('count')
    dataframe['average_length'] = dataframe.groupby('Source')['Length'].transform('mean')
    dataframe['unique_destinations'] = dataframe.groupby('Source')['Destination'].transform('nunique')

    return dataframe


def train_anomaly_detector(
    X: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Train an Isolation Forest model for anomaly detection.

    Args:
        X: Feature matrix
        n_estimators: Number of trees in the forest
        contamination: Expected proportion of anomalies
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    logger.info(f"Training Isolation Forest with {n_estimators} estimators, "
                f"contamination={contamination}")

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(X_scaled)

    logger.info("Model training completed")
    return model, scaler


def save_model(model: IsolationForest, scaler: StandardScaler, output_path: str):
    """
    Save trained model and scaler to disk.

    Args:
        model: Trained Isolation Forest model
        scaler: Fitted StandardScaler
        output_path: Path to save the model
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        logger.info(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


def load_model(model_path: str) -> Tuple[IsolationForest, StandardScaler]:
    """
    Load a trained model and scaler from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Tuple of (model, scaler)
    """
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return data['model'], data['scaler']
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def detect_anomalies(
    pcap_file: str,
    output_csv: str = "unsupervised_processed_output.csv",
    max_packets: Optional[int] = None,
    n_estimators: int = 100,
    contamination: float = 0.05,
    save_model_path: Optional[str] = None
):
    """
    Main function to detect network anomalies in a PCAP file.

    Args:
        pcap_file: Path to input PCAP file
        output_csv: Path to output CSV file
        max_packets: Maximum packets to process (None for all)
        n_estimators: Number of trees in Isolation Forest
        contamination: Expected anomaly proportion
        save_model_path: Path to save trained model (optional)
    """
    try:
        # Read PCAP file
        df = read_pcap(pcap_file, max_packets)

        if df.empty:
            logger.warning("No packets found in PCAP file")
            return

        # Add features
        df = add_features(df)

        # Prepare features for model
        df['log_packet_count'] = np.log1p(df['packet_count'])
        df['log_average_length'] = np.log1p(df['average_length'])
        features = ['log_packet_count', 'log_average_length', 'unique_destinations']
        X = df[features].fillna(0)

        # Train model
        model, scaler = train_anomaly_detector(X, n_estimators, contamination)

        # Predict anomalies
        X_scaled = scaler.transform(X)
        df['anomaly_score'] = model.predict(X_scaled)
        df['is_anomalous'] = (df['anomaly_score'] == -1).astype(int)

        # Identify malicious IPs
        malicious_ips = df[df['is_anomalous'] == 1]['Source'].unique()

        logger.info(f"\nDetected {len(malicious_ips)} potentially malicious IPs")
        print("\n" + "="*60)
        print("POTENTIAL MALICIOUS IPs DETECTED:")
        print("="*60)
        for ip in malicious_ips:
            print(f"  ip.src == {ip}")
        print("="*60 + "\n")

        # Save results
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")

        # Save model if requested
        if save_model_path:
            save_model(model, scaler, save_model_path)

        # Print summary statistics
        total_packets = len(df)
        anomalous_packets = df['is_anomalous'].sum()
        anomaly_percentage = (anomalous_packets / total_packets) * 100

        logger.info(f"Summary: {total_packets} total packets, "
                   f"{anomalous_packets} anomalous ({anomaly_percentage:.2f}%)")

    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise


def main():
    """Command-line interface for the anomaly detection system."""
    parser = argparse.ArgumentParser(
        description='AI-driven Network Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.pcap
  %(prog)s input.pcap -o results.csv --max-packets 10000
  %(prog)s input.pcap --contamination 0.1 --save-model model.pkl
        """
    )

    parser.add_argument(
        'pcap_file',
        help='Path to input PCAP file'
    )
    parser.add_argument(
        '-o', '--output',
        default='unsupervised_processed_output.csv',
        help='Output CSV file path (default: unsupervised_processed_output.csv)'
    )
    parser.add_argument(
        '--max-packets',
        type=int,
        help='Maximum number of packets to process (default: all)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Isolation Forest (default: 100)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of anomalies (default: 0.05)'
    )
    parser.add_argument(
        '--save-model',
        help='Path to save trained model'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    detect_anomalies(
        pcap_file=args.pcap_file,
        output_csv=args.output,
        max_packets=args.max_packets,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        save_model_path=args.save_model
    )


if __name__ == "__main__":
    main()
