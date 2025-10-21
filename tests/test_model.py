"""Unit tests for model.py"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import add_features, train_anomaly_detector, save_model, load_model


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_add_features(self):
        """Test feature addition to DataFrame."""
        # Create sample data
        df = pd.DataFrame({
            'Source': ['192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.2'],
            'Destination': ['10.0.0.1', '10.0.0.2', '10.0.0.1', '10.0.0.1'],
            'Length': [100, 200, 150, 150]
        })

        result = add_features(df)

        # Check that new columns exist
        assert 'packet_count' in result.columns
        assert 'average_length' in result.columns
        assert 'unique_destinations' in result.columns

        # Check values
        assert result[result['Source'] == '192.168.1.1']['packet_count'].iloc[0] == 2
        assert result[result['Source'] == '192.168.1.2']['packet_count'].iloc[0] == 2
        assert result[result['Source'] == '192.168.1.1']['average_length'].iloc[0] == 150
        assert result[result['Source'] == '192.168.1.1']['unique_destinations'].iloc[0] == 2
        assert result[result['Source'] == '192.168.1.2']['unique_destinations'].iloc[0] == 1


class TestAnomalyDetector:
    """Test anomaly detection functions."""

    def test_train_anomaly_detector(self):
        """Test model training."""
        # Create sample feature matrix
        X = np.random.rand(100, 3)

        model, scaler = train_anomaly_detector(X, n_estimators=10, contamination=0.1)

        # Check that model and scaler are returned
        assert model is not None
        assert scaler is not None

        # Check that model can predict
        predictions = model.predict(scaler.transform(X))
        assert len(predictions) == 100
        assert set(predictions).issubset({-1, 1})

    def test_model_persistence(self, tmp_path):
        """Test saving and loading model."""
        # Create and train a simple model
        X = np.random.rand(50, 3)
        model, scaler = train_anomaly_detector(X, n_estimators=10)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        save_model(model, scaler, str(model_path))

        # Check file exists
        assert model_path.exists()

        # Load model
        loaded_model, loaded_scaler = load_model(str(model_path))

        # Check loaded model works
        predictions = loaded_model.predict(loaded_scaler.transform(X))
        assert len(predictions) == 50


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete feature engineering and prediction pipeline."""
        # Create sample packet data
        df = pd.DataFrame({
            'Source': ['192.168.1.1'] * 50 + ['10.0.0.1'] * 10,
            'Destination': ['10.0.0.' + str(i % 20) for i in range(60)],
            'Length': np.random.randint(50, 1500, 60)
        })

        # Add features
        df = add_features(df)

        # Prepare features for model
        df['log_packet_count'] = np.log1p(df['packet_count'])
        df['log_average_length'] = np.log1p(df['average_length'])
        features = ['log_packet_count', 'log_average_length', 'unique_destinations']
        X = df[features].fillna(0)

        # Train model
        model, scaler = train_anomaly_detector(X, n_estimators=10, contamination=0.1)

        # Predict
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        # Check results
        assert len(predictions) == 60
        assert predictions.min() == -1  # At least some anomalies
        assert predictions.max() == 1   # At least some normal


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
