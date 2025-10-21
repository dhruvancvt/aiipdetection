# AI IP Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **AI-driven Network Anomaly Detection System** that analyzes PCAP (packet capture) files to identify potentially malicious network behavior using unsupervised machine learning. The system uses Isolation Forest algorithm to detect anomalous IP addresses and network patterns.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Anomaly Detection](#anomaly-detection)
  - [PCAP Anonymization](#pcap-anonymization)
  - [PCAP to CSV Conversion](#pcap-to-csv-conversion)
  - [Results Visualization](#results-visualization)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Unsupervised Anomaly Detection**: Identify malicious IPs without labeled training data using Isolation Forest
- **PCAP Processing**: Efficient reading and processing of network packet captures
- **Feature Engineering**: Automatic extraction of network behavior features:
  - Packet count per source IP
  - Average packet length
  - Unique destination count
  - Log-transformed features for skewed distributions
- **PCAP Anonymization**: Privacy-preserving tool to anonymize MAC and IP addresses while preserving attacker IPs
- **Data Visualization**: Comprehensive visualization of detection results and network statistics
- **Model Persistence**: Save and load trained models for consistent analysis
- **Command-line Interface**: Full CLI support for all tools with flexible configuration
- **Logging**: Comprehensive logging for debugging and audit trails
- **Unit Tests**: Test suite for core functionality

---

## Installation

### Prerequisites

- Python 3.8 or higher
- tshark (for pyshark): `sudo apt-get install tshark` (Linux) or `brew install wireshark` (macOS)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/dhruvancvt/aiipdetection.git
cd aiipdetection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

Analyze a PCAP file for anomalies:

```bash
python model.py capture.pcap
```

This will:
- Read the PCAP file
- Extract and engineer features
- Train an Isolation Forest model
- Identify potentially malicious IPs
- Save results to `unsupervised_processed_output.csv`

---

## Usage

### Anomaly Detection

Detect network anomalies in PCAP files using machine learning.

**Basic usage:**

```bash
python model.py capture.pcap
```

**Advanced options:**

```bash
# Specify output file
python model.py capture.pcap -o results.csv

# Limit packets processed
python model.py capture.pcap --max-packets 10000

# Adjust model parameters
python model.py capture.pcap --n-estimators 200 --contamination 0.1

# Save trained model
python model.py capture.pcap --save-model trained_model.pkl

# Enable verbose logging
python model.py capture.pcap -v
```

**Parameters:**

- `pcap_file`: Path to input PCAP file (required)
- `-o, --output`: Output CSV file path (default: `unsupervised_processed_output.csv`)
- `--max-packets`: Maximum number of packets to process (default: all)
- `--n-estimators`: Number of trees in Isolation Forest (default: 100)
- `--contamination`: Expected proportion of anomalies (default: 0.05 = 5%)
- `--save-model`: Path to save trained model
- `-v, --verbose`: Enable verbose logging

**Output:**

The script generates a CSV file with columns:
- `Source`: Source IP address
- `Destination`: Destination IP address
- `Length`: Packet length
- `packet_count`: Total packets from source
- `average_length`: Average packet length from source
- `unique_destinations`: Number of unique destinations
- `log_packet_count`: Log-transformed packet count
- `log_average_length`: Log-transformed average length
- `anomaly_score`: -1 (anomalous) or 1 (normal)
- `is_anomalous`: Binary flag (1 = anomalous, 0 = normal)

---

### PCAP Anonymization

Anonymize MAC and IP addresses in PCAP files while preserving attacker IPs for analysis.

**Basic usage:**

```bash
python anon.py capture.pcap
```

**Advanced options:**

```bash
# Anonymize multiple files
python anon.py capture1.pcap capture2.pcap

# Specify output directory
python anon.py capture.pcap -o ./anonymized

# Custom anonymization settings
python anon.py capture.pcap --mac-prefix "aa:aa:aa:aa" --ip-network "172.16"

# Preserve different attacker subnet
python anon.py capture.pcap --preserve-subnet "172.31"
```

**Parameters:**

- `pcap_files`: One or more PCAP files to anonymize
- `-o, --output-dir`: Output directory (default: same as input)
- `--mac-prefix`: MAC address prefix (default: `ff:ff:ff:ff`)
- `--ip-network`: IP network prefix (default: `192.168`)
- `--preserve-subnet`: Subnet to preserve (default: `10.128`)
- `-v, --verbose`: Enable verbose logging

**Features:**

- MAC addresses anonymized to format: `ff:ff:ff:ff:XX:XX`
- IP addresses anonymized to `192.168.X.X` range
- Attacker IPs (configurable subnet) preserved for identification
- Memory-efficient processing with garbage collection

---

### PCAP to CSV Conversion

Convert PCAP files to CSV format for analysis with standard data tools.

**Basic usage:**

```bash
python convpcaptopy.py capture.pcap
```

**Advanced options:**

```bash
# Specify output file
python convpcaptopy.py capture.pcap -o output.csv

# Limit packets converted
python convpcaptopy.py capture.pcap --max-packets 5000

# Exclude packet info summary
python convpcaptopy.py capture.pcap --no-info
```

**Parameters:**

- `pcap_file`: Path to input PCAP file
- `-o, --output`: Output CSV file (default: `input_name.csv`)
- `--max-packets`: Maximum packets to convert
- `--no-info`: Exclude packet info column
- `-v, --verbose`: Enable verbose logging

**Output CSV columns:**

- `No.`: Packet number
- `Time`: Timestamp
- `Source`: Source IP address
- `Destination`: Destination IP address
- `Protocol`: Protocol type
- `Length`: Packet length
- `Info`: Packet summary (optional)

---

### Results Visualization

Visualize anomaly detection results with comprehensive plots and statistics.

**Basic usage:**

```bash
python visualize.py unsupervised_processed_output.csv
```

**Advanced options:**

```bash
# Save plots to directory
python visualize.py results.csv -o ./plots

# Show top 20 sources
python visualize.py results.csv --top-n 20

# Save plots without displaying
python visualize.py results.csv -o ./plots --no-show
```

**Parameters:**

- `csv_file`: Path to anomaly detection results CSV
- `-o, --output-dir`: Directory to save plots
- `--top-n`: Number of top sources to display (default: 10)
- `--no-show`: Do not display plots interactively
- `-v, --verbose`: Enable verbose logging

**Generated visualizations:**

1. **Anomaly Distribution**: Pie and bar charts showing normal vs anomalous traffic
2. **Feature Distributions**: Histograms and box plots for all engineered features
3. **Top Sources**: Bar chart of most active source IPs
4. **Correlation Matrix**: Feature correlation heatmap
5. **Summary Report**: Text report with key statistics

---

## Configuration

### Configuration File

Edit `config.yaml` to customize default parameters:

```yaml
# Model Parameters
model:
  n_estimators: 100
  contamination: 0.05
  random_state: 42

# Processing Parameters
processing:
  max_packets: null
  output_dir: "./results"
  output_csv: "unsupervised_processed_output.csv"

# Logging Configuration
logging:
  level: INFO
  log_file: "anomaly_detection.log"

# Feature Engineering
features:
  enabled:
    - log_packet_count
    - log_average_length
    - unique_destinations

# Anonymization Settings
anonymization:
  preserve_attacker_subnet: "10.128"
  mac_prefix: "ff:ff:ff:ff"
  ip_network: "192.168"
```

---

## Project Structure

```
aiipdetection/
├── model.py                 # Main anomaly detection module
├── anon.py                  # PCAP anonymization tool
├── convpcaptopy.py          # PCAP to CSV converter
├── visualize.py             # Results visualization tool
├── importdataset.py         # Dataset import utility
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_model.py        # Model tests
│   └── test_anon.py         # Anonymization tests
└── .gitignore
```

---

## Testing

### Generate Test Data

Create a test PCAP file with simulated network traffic:

```bash
# Generate test PCAP with 1000 packets
python generate_test_pcap.py

# Or specify custom size
python generate_test_pcap.py -n 5000 -o custom_test.pcap
```

This creates a PCAP file with:
- 80% normal traffic from 5 source IPs
- 20% anomalous traffic from attacker IP `10.128.1.100`
- The attacker exhibits port scanning behavior (many packets to many destinations)

### Run Tests

Test the system with generated data:

```bash
# Test anomaly detection
python model.py test_capture.pcap

# Expected output: Attacker IP 10.128.1.100 detected as anomalous

# Test other tools
python convpcaptopy.py test_capture.pcap
python anon.py test_capture.pcap
python visualize.py unsupervised_processed_output.csv -o ./plots
```

### Run Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run with verbose output
pytest -v
```

**Test coverage:**

- Feature engineering functions
- Anomaly detection model training
- Model persistence (save/load)
- PCAP anonymization
- IP/MAC address handling
- Attacker subnet preservation

For detailed testing instructions, see [TESTING.md](TESTING.md).

---

## How It Works

### Anomaly Detection Pipeline

1. **PCAP Reading**: Parse network packets using pyshark
2. **Feature Extraction**: Extract source, destination, and packet length
3. **Feature Engineering**:
   - Aggregate packets by source IP
   - Calculate packet count, average length, unique destinations
   - Apply log transformation to reduce skewness
4. **Normalization**: StandardScaler for feature scaling
5. **Model Training**: Isolation Forest with configurable parameters
6. **Prediction**: Identify anomalous IPs based on model scores
7. **Output**: CSV file with all features and anomaly flags

### Isolation Forest Algorithm

Isolation Forest is an unsupervised learning algorithm that:
- Builds random decision trees
- Anomalies are easier to isolate (fewer splits needed)
- Assigns anomaly score based on path length
- No labeled data required
- Efficient for high-dimensional data

---

## Examples

### Example 1: Basic Anomaly Detection

```bash
# Detect anomalies in network capture
python model.py network_traffic.pcap

# Output:
# ============================================================
# POTENTIAL MALICIOUS IPs DETECTED:
# ============================================================
#   ip.src == 10.128.1.45
#   ip.src == 192.168.1.123
# ============================================================
```

### Example 2: Anonymize and Analyze

```bash
# Step 1: Anonymize PCAP file
python anon.py sensitive_capture.pcap -o ./safe

# Step 2: Analyze anonymized file
python model.py ./safe/sensitive_capture.pcap-anon.pcap

# Step 3: Visualize results
python visualize.py unsupervised_processed_output.csv -o ./plots
```

### Example 3: Custom Analysis Pipeline

```bash
# High-sensitivity detection (expect 10% anomalies)
python model.py capture.pcap \
  --contamination 0.1 \
  --n-estimators 200 \
  --save-model high_sensitivity.pkl \
  -o high_sensitivity_results.csv

# Visualize with top 20 sources
python visualize.py high_sensitivity_results.csv \
  --top-n 20 \
  -o ./analysis_plots
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and add tests
4. Run the test suite:
   ```bash
   pytest
   ```
5. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```
6. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a pull request

**Development Guidelines:**

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed
- Use type hints where appropriate

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'pyshark'`**

Solution: Install tshark system dependency:
```bash
# Ubuntu/Debian
sudo apt-get install tshark

# macOS
brew install wireshark
```

**Issue: `Permission denied` when reading PCAP**

Solution: Run with appropriate permissions or adjust file permissions:
```bash
chmod +r capture.pcap
```

**Issue: `Empty DataFrame` or no packets detected**

Solution: Verify PCAP file contains IP packets:
```bash
tshark -r capture.pcap -c 10
```

---

## Roadmap

Future enhancements:

- [ ] Real-time packet capture and analysis
- [ ] Support for additional ML algorithms (LOF, One-Class SVM)
- [ ] Web-based dashboard for visualization
- [ ] Export to SIEM formats (CEF, LEEF)
- [ ] Integration with threat intelligence feeds
- [ ] Deep learning models for advanced pattern detection
- [ ] Automated report generation

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{aiipdetection2024,
  author = {dhruvancvt},
  title = {AI IP Detection: Network Anomaly Detection System},
  year = {2024},
  url = {https://github.com/dhruvancvt/aiipdetection}
}
```

---

## Contact

For questions, suggestions, or issues:

- Open an issue on GitHub
- Contact the repository maintainer

---

**Disclaimer**: This tool is for educational and defensive security purposes only. Use responsibly and in accordance with applicable laws and regulations.
