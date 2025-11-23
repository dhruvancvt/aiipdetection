"""Unit tests for visualize.py"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize import (
    load_results,
    plot_anomaly_distribution,
    plot_feature_distributions,
    plot_top_sources,
    plot_correlation_matrix,
    generate_summary_report
)


class TestLoadResults:
    """Test result loading functions."""

    def test_load_results_success(self, tmp_path):
        """Test successful loading of results CSV."""
        # Create test CSV
        df = pd.DataFrame({
            'Source': ['192.168.1.1', '192.168.1.2', '10.128.1.100'],
            'Destination': ['8.8.8.8', '1.1.1.1', '10.0.0.1'],
            'Length': [100, 200, 50],
            'packet_count': [10, 5, 100],
            'average_length': [150, 180, 60],
            'unique_destinations': [2, 3, 50],
            'is_anomalous': [0, 0, 1]
        })

        csv_path = tmp_path / "test_results.csv"
        df.to_csv(csv_path, index=False)

        # Load results
        loaded_df = load_results(str(csv_path))

        assert len(loaded_df) == 3
        assert 'Source' in loaded_df.columns
        assert 'is_anomalous' in loaded_df.columns

    def test_load_results_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_results("nonexistent.csv")


class TestPlotFunctions:
    """Test plotting functions."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'Source': ['192.168.1.1'] * 50 + ['10.128.1.100'] * 20,
            'Destination': [f'10.0.0.{i}' for i in range(70)],
            'Length': np.random.randint(50, 1500, 70),
            'packet_count': [50] * 50 + [20] * 20,
            'average_length': np.random.randint(100, 1000, 70),
            'unique_destinations': [10] * 50 + [20] * 20,
            'is_anomalous': [0] * 50 + [1] * 20
        })

    @patch('visualize.plt.savefig')
    @patch('visualize.plt.show')
    def test_plot_anomaly_distribution(self, mock_show, mock_savefig, sample_df, tmp_path):
        """Test anomaly distribution plotting."""
        output_path = tmp_path / "test_plot.png"

        # Should not raise any errors
        plot_anomaly_distribution(sample_df, str(output_path))

        # Verify savefig was called
        mock_savefig.assert_called_once()

    @patch('visualize.plt.savefig')
    @patch('visualize.plt.show')
    def test_plot_feature_distributions(self, mock_show, mock_savefig, sample_df, tmp_path):
        """Test feature distribution plotting."""
        output_path = tmp_path / "test_features.png"

        plot_feature_distributions(sample_df, str(output_path))

        mock_savefig.assert_called_once()

    @patch('visualize.plt.savefig')
    @patch('visualize.plt.show')
    def test_plot_top_sources(self, mock_show, mock_savefig, sample_df, tmp_path):
        """Test top sources plotting."""
        output_path = tmp_path / "test_top_sources.png"

        plot_top_sources(sample_df, top_n=5, output_path=str(output_path))

        mock_savefig.assert_called_once()

    @patch('visualize.plt.savefig')
    @patch('visualize.plt.show')
    def test_plot_correlation_matrix(self, mock_show, mock_savefig, sample_df, tmp_path):
        """Test correlation matrix plotting."""
        output_path = tmp_path / "test_correlation.png"

        plot_correlation_matrix(sample_df, str(output_path))

        mock_savefig.assert_called_once()

    def test_plot_with_missing_columns(self):
        """Test plotting with missing required columns."""
        # DataFrame without is_anomalous column
        df = pd.DataFrame({
            'Source': ['192.168.1.1', '192.168.1.2'],
            'Destination': ['8.8.8.8', '1.1.1.1']
        })

        # Should handle gracefully (log warning, not crash)
        plot_anomaly_distribution(df)
        plot_feature_distributions(df)


class TestSummaryReport:
    """Test summary report generation."""

    def test_generate_summary_report(self, tmp_path, capsys):
        """Test summary report generation."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1', '192.168.1.2', '10.128.1.100'],
            'Destination': ['8.8.8.8', '1.1.1.1', '10.0.0.1'],
            'packet_count': [10, 5, 100],
            'average_length': [150, 180, 60],
            'unique_destinations': [2, 3, 50],
            'is_anomalous': [0, 0, 1]
        })

        report_path = tmp_path / "test_report.txt"

        generate_summary_report(df, str(report_path))

        # Check report file was created
        assert report_path.exists()

        # Check report content
        report_content = report_path.read_text()
        assert "ANOMALY DETECTION SUMMARY REPORT" in report_content
        assert "Total Packets Analyzed:" in report_content
        assert "Unique Source IPs:" in report_content

        # Check console output
        captured = capsys.readouterr()
        assert "ANOMALY DETECTION SUMMARY REPORT" in captured.out

    def test_generate_summary_report_minimal_data(self, capsys):
        """Test summary with minimal DataFrame."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        })

        generate_summary_report(df)

        captured = capsys.readouterr()
        assert "Total Packets Analyzed: 3" in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
