#!/usr/bin/env python3
"""
Visualization Tool for Anomaly Detection Results

This module provides visualization capabilities for analyzing network anomaly
detection results from the AI IP Detection system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Load anomaly detection results from CSV.

    Args:
        csv_path: Path to results CSV file

    Returns:
        DataFrame with results
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    logger.info(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} records")
    return df


def plot_anomaly_distribution(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot distribution of anomalous vs normal traffic.

    Args:
        df: Results DataFrame
        output_path: Optional path to save plot
    """
    if 'is_anomalous' not in df.columns:
        logger.warning("No 'is_anomalous' column found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    anomaly_counts = df['is_anomalous'].value_counts()
    labels = ['Normal', 'Anomalous']
    colors = ['#2ecc71', '#e74c3c']

    axes[0].pie(
        anomaly_counts.values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    axes[0].set_title('Traffic Distribution: Normal vs Anomalous', fontsize=14, fontweight='bold')

    # Bar chart
    axes[1].bar(labels, anomaly_counts.values, color=colors, alpha=0.7)
    axes[1].set_ylabel('Packet Count', fontsize=12)
    axes[1].set_title('Anomaly Detection Results', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(anomaly_counts.values):
        axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_distributions(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot distributions of engineered features.

    Args:
        df: Results DataFrame
        output_path: Optional path to save plot
    """
    features = ['packet_count', 'average_length', 'unique_destinations']
    available_features = [f for f in features if f in df.columns]

    if not available_features:
        logger.warning("No feature columns found for visualization")
        return

    fig, axes = plt.subplots(len(available_features), 2, figsize=(14, 4 * len(available_features)))

    if len(available_features) == 1:
        axes = axes.reshape(1, -1)

    for idx, feature in enumerate(available_features):
        # Histogram
        axes[idx, 0].hist(df[feature], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx, 0].set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        axes[idx, 0].set_ylabel('Frequency', fontsize=11)
        axes[idx, 0].set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx, 0].grid(alpha=0.3)

        # Box plot by anomaly status
        if 'is_anomalous' in df.columns:
            df_plot = df.copy()
            df_plot['Status'] = df_plot['is_anomalous'].map({0: 'Normal', 1: 'Anomalous'})

            sns.boxplot(
                data=df_plot,
                x='Status',
                y=feature,
                ax=axes[idx, 1],
                palette={'Normal': '#2ecc71', 'Anomalous': '#e74c3c'}
            )
            axes[idx, 1].set_title(f'{feature.replace("_", " ").title()} by Status', fontsize=12, fontweight='bold')
            axes[idx, 1].set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
            axes[idx, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_top_sources(df: pd.DataFrame, top_n: int = 10, output_path: Optional[str] = None):
    """
    Plot top source IPs by packet count.

    Args:
        df: Results DataFrame
        top_n: Number of top sources to show
        output_path: Optional path to save plot
    """
    if 'Source' not in df.columns:
        logger.warning("No 'Source' column found")
        return

    # Aggregate by source
    source_stats = df.groupby('Source').agg({
        'Source': 'count',
        'is_anomalous': 'max' if 'is_anomalous' in df.columns else 'first'
    }).rename(columns={'Source': 'packet_count'})

    source_stats = source_stats.sort_values('packet_count', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e74c3c' if x == 1 else '#3498db' for x in source_stats['is_anomalous']]

    bars = ax.barh(range(len(source_stats)), source_stats['packet_count'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(source_stats)))
    ax.set_yticklabels(source_stats.index, fontsize=10)
    ax.set_xlabel('Packet Count', fontsize=12)
    ax.set_title(f'Top {top_n} Source IPs by Packet Count', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(source_stats.iterrows()):
        ax.text(row['packet_count'], i, f" {int(row['packet_count'])}", va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Anomalous'),
        Patch(facecolor='#3498db', alpha=0.7, label='Normal')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot correlation matrix of features.

    Args:
        df: Results DataFrame
        output_path: Optional path to save plot
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove index-like columns
    numeric_cols = [col for col in numeric_cols if col not in ['Unnamed: 0', 'index']]

    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation matrix")
        return

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def generate_summary_report(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Generate a text summary report of the analysis.

    Args:
        df: Results DataFrame
        output_path: Optional path to save report
    """
    report_lines = [
        "="*60,
        "ANOMALY DETECTION SUMMARY REPORT",
        "="*60,
        "",
        f"Total Packets Analyzed: {len(df):,}",
    ]

    if 'is_anomalous' in df.columns:
        anomalous_count = df['is_anomalous'].sum()
        normal_count = len(df) - anomalous_count
        anomaly_pct = (anomalous_count / len(df)) * 100

        report_lines.extend([
            f"Anomalous Packets: {anomalous_count:,} ({anomaly_pct:.2f}%)",
            f"Normal Packets: {normal_count:,} ({100-anomaly_pct:.2f}%)",
            ""
        ])

    if 'Source' in df.columns:
        unique_sources = df['Source'].nunique()
        report_lines.append(f"Unique Source IPs: {unique_sources:,}")

        if 'is_anomalous' in df.columns:
            malicious_ips = df[df['is_anomalous'] == 1]['Source'].nunique()
            report_lines.append(f"Potentially Malicious IPs: {malicious_ips:,}")

    if 'Destination' in df.columns:
        unique_dests = df['Destination'].nunique()
        report_lines.append(f"Unique Destination IPs: {unique_dests:,}")

    report_lines.append("")
    report_lines.append("-"*60)
    report_lines.append("TOP FEATURE STATISTICS")
    report_lines.append("-"*60)

    for feature in ['packet_count', 'average_length', 'unique_destinations']:
        if feature in df.columns:
            report_lines.append(f"\n{feature.replace('_', ' ').title()}:")
            report_lines.append(f"  Mean: {df[feature].mean():.2f}")
            report_lines.append(f"  Median: {df[feature].median():.2f}")
            report_lines.append(f"  Std Dev: {df[feature].std():.2f}")
            report_lines.append(f"  Min: {df[feature].min():.2f}")
            report_lines.append(f"  Max: {df[feature].max():.2f}")

    report_lines.append("")
    report_lines.append("="*60)

    report = "\n".join(report_lines)
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to: {output_path}")


def main():
    """Command-line interface for visualization."""
    parser = argparse.ArgumentParser(
        description='Visualization Tool for Anomaly Detection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.csv
  %(prog)s results.csv --output-dir ./plots
  %(prog)s results.csv --top-n 20 --no-show
        """
    )

    parser.add_argument(
        'csv_file',
        help='Path to anomaly detection results CSV'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Directory to save plots (default: show plots interactively)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top sources to display (default: 10)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots interactively (only save)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to: {output_dir}")

    try:
        # Load results
        df = load_results(args.csv_file)

        # Generate summary report
        report_path = output_dir / "summary_report.txt" if output_dir else None
        generate_summary_report(df, report_path)

        # Generate plots
        plot_anomaly_distribution(
            df,
            output_dir / "anomaly_distribution.png" if output_dir else None
        )

        plot_feature_distributions(
            df,
            output_dir / "feature_distributions.png" if output_dir else None
        )

        plot_top_sources(
            df,
            top_n=args.top_n,
            output_path=output_dir / f"top_{args.top_n}_sources.png" if output_dir else None
        )

        plot_correlation_matrix(
            df,
            output_dir / "correlation_matrix.png" if output_dir else None
        )

        logger.info("Visualization complete!")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
