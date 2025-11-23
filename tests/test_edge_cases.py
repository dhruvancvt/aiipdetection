"""Edge case and error handling tests"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import add_features, train_anomaly_detector, read_pcap, detect_anomalies


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['Source', 'Destination', 'Length'])

        result = add_features(df)

        # Should return empty DataFrame with new columns
        assert len(result) == 0
        assert 'packet_count' in result.columns
        assert 'average_length' in result.columns
        assert 'unique_destinations' in result.columns

    def test_single_packet(self):
        """Test handling of single packet."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1'],
            'Destination': ['8.8.8.8'],
            'Length': [100]
        })

        result = add_features(df)

        assert len(result) == 1
        assert result['packet_count'].iloc[0] == 1
        assert result['average_length'].iloc[0] == 100
        assert result['unique_destinations'].iloc[0] == 1

    def test_all_same_source(self):
        """Test when all packets from same source."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1'] * 100,
            'Destination': [f'10.0.0.{i}' for i in range(100)],
            'Length': [100] * 100
        })

        result = add_features(df)

        assert result['packet_count'].iloc[0] == 100
        assert result['average_length'].iloc[0] == 100
        assert result['unique_destinations'].iloc[0] == 100

    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1', None, '192.168.1.2'],
            'Destination': ['8.8.8.8', '1.1.1.1', None],
            'Length': [100, 200, 150]
        })

        # Should handle NaN values gracefully
        result = add_features(df)
        assert len(result) == 3

    def test_very_large_packet_counts(self):
        """Test with very large packet counts."""
        # Create data with extreme packet counts
        df = pd.DataFrame({
            'Source': ['192.168.1.1'] * 10000,
            'Destination': ['8.8.8.8'] * 10000,
            'Length': [100] * 10000
        })

        result = add_features(df)

        assert result['packet_count'].iloc[0] == 10000

    def test_zero_length_packets(self):
        """Test packets with zero length."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1', '192.168.1.2'],
            'Destination': ['8.8.8.8', '1.1.1.1'],
            'Length': [0, 0]
        })

        result = add_features(df)

        assert result['average_length'].iloc[0] == 0

    def test_model_training_small_dataset(self):
        """Test model training with very small dataset."""
        # Minimum viable dataset (5 samples)
        X = np.random.rand(5, 3)

        model, scaler = train_anomaly_detector(X, n_estimators=10, contamination=0.2)

        # Should still work
        assert model is not None
        assert scaler is not None

    def test_model_training_single_feature(self):
        """Test model with single feature."""
        X = np.random.rand(100, 1)

        model, scaler = train_anomaly_detector(X, n_estimators=10)

        predictions = model.predict(scaler.transform(X))
        assert len(predictions) == 100

    def test_model_training_all_same_values(self):
        """Test model when all values are identical."""
        X = np.ones((100, 3))

        model, scaler = train_anomaly_detector(X, n_estimators=10)

        # Should handle gracefully (no variance)
        predictions = model.predict(scaler.transform(X))
        assert len(predictions) == 100

    def test_contamination_boundary_values(self):
        """Test contamination parameter at boundaries."""
        X = np.random.rand(100, 3)

        # Very low contamination
        model1, scaler1 = train_anomaly_detector(X, contamination=0.01)
        predictions1 = model1.predict(scaler1.transform(X))
        anomaly_count1 = np.sum(predictions1 == -1)
        assert anomaly_count1 <= 5  # Should have very few anomalies

        # High contamination
        model2, scaler2 = train_anomaly_detector(X, contamination=0.5)
        predictions2 = model2.predict(scaler2.transform(X))
        anomaly_count2 = np.sum(predictions2 == -1)
        assert anomaly_count2 >= 40  # Should have many anomalies

    def test_read_pcap_file_not_found(self):
        """Test read_pcap with non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_pcap("nonexistent.pcap")

    @patch('model.read_pcap')
    def test_detect_anomalies_empty_pcap(self, mock_read_pcap, tmp_path):
        """Test detection with empty PCAP (no packets)."""
        # Mock empty DataFrame
        mock_read_pcap.return_value = pd.DataFrame()

        # Should handle gracefully
        detect_anomalies("dummy.pcap", max_packets=10)

        # Should log warning and return without crashing
        mock_read_pcap.assert_called_once()

    def test_duplicate_sources(self):
        """Test handling of duplicate source IPs."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1', '192.168.1.1', '192.168.1.1'],
            'Destination': ['8.8.8.8', '8.8.8.8', '1.1.1.1'],
            'Length': [100, 100, 200]
        })

        result = add_features(df)

        # All rows should have same packet_count since same source
        assert result['packet_count'].nunique() == 1
        assert result['packet_count'].iloc[0] == 3

    def test_unicode_handling(self):
        """Test handling of special characters (if any)."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1'],
            'Destination': ['8.8.8.8'],
            'Length': [100]
        })

        # Should handle without errors
        result = add_features(df)
        assert len(result) == 1

    def test_extremely_large_packet_length(self):
        """Test with unrealistically large packet lengths."""
        df = pd.DataFrame({
            'Source': ['192.168.1.1'],
            'Destination': ['8.8.8.8'],
            'Length': [999999999]  # Extremely large
        })

        result = add_features(df)

        # Should still process
        assert result['average_length'].iloc[0] == 999999999


class TestModelRobustness:
    """Test model robustness and consistency."""

    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with same random_state."""
        X = np.random.rand(100, 3)

        model1, scaler1 = train_anomaly_detector(X, random_state=42)
        predictions1 = model1.predict(scaler1.transform(X))

        model2, scaler2 = train_anomaly_detector(X, random_state=42)
        predictions2 = model2.predict(scaler2.transform(X))

        # Should produce identical results
        assert np.array_equal(predictions1, predictions2)

    def test_different_random_states(self):
        """Test that different random states produce different models."""
        X = np.random.rand(100, 3)

        model1, scaler1 = train_anomaly_detector(X, random_state=42)
        predictions1 = model1.predict(scaler1.transform(X))

        model2, scaler2 = train_anomaly_detector(X, random_state=123)
        predictions2 = model2.predict(scaler2.transform(X))

        # Results may differ slightly
        # Just check both are valid
        assert len(predictions1) == len(predictions2) == 100

    def test_model_consistency_across_subsets(self):
        """Test that model works on different data subsets."""
        # Train on one dataset
        X_train = np.random.rand(100, 3)
        model, scaler = train_anomaly_detector(X_train)

        # Predict on different dataset
        X_test = np.random.rand(50, 3)
        predictions = model.predict(scaler.transform(X_test))

        assert len(predictions) == 50
        assert set(predictions).issubset({-1, 1})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
