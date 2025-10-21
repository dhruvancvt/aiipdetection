# Testing Guide

This document explains how to test the AI IP Detection system.

## Quick Start

### 1. Generate Test PCAP File

Create a test PCAP file with simulated network traffic:

```bash
python generate_test_pcap.py
```

This creates `test_capture.pcap` with:
- **800 packets** from 5 normal IPs (legitimate traffic)
- **200 packets** from 1 attacker IP (scanning behavior)
- Total: **1000 packets**

**Expected behavior:** The attacker IP `10.128.1.100` should be detected as anomalous because it sends many packets to many unique destinations (port scanning pattern).

### 2. Run Anomaly Detection

Test the main detection model:

```bash
python model.py test_capture.pcap
```

Expected output:
```
============================================================
POTENTIAL MALICIOUS IPs DETECTED:
============================================================
  ip.src == 10.128.1.100
============================================================
```

### 3. Run All Tools

Test all components:

```bash
# Convert PCAP to CSV
python convpcaptopy.py test_capture.pcap -o test_output.csv

# Anonymize PCAP
python anon.py test_capture.pcap -o ./anonymized

# Analyze with model
python model.py test_capture.pcap -o test_results.csv

# Visualize results
python visualize.py test_results.csv -o ./test_plots
```

---

## Test PCAP File Details

### Traffic Patterns

#### Normal Traffic (80%)
- **Source IPs:** 5 different IPs in 192.168.1.0/24 subnet
  - 192.168.1.10
  - 192.168.1.11
  - 192.168.1.12
  - 192.168.1.13
  - 192.168.1.14
- **Destinations:** 5 common public IPs (DNS, web services)
- **Ports:** Standard ports (80, 443, 53)
- **Behavior:** Regular web browsing and DNS lookups

#### Anomalous Traffic (20%)
- **Source IP:** 10.128.1.100 (attacker)
- **Destinations:** 500+ unique IPs (scanning multiple subnets)
- **Ports:** Wide range (1-1024) indicating port scanning
- **Behavior:** Typical network reconnaissance pattern

### Why This Tests Anomaly Detection

The Isolation Forest algorithm detects the attacker based on:

1. **High packet count:** Attacker sends many packets
2. **Many unique destinations:** Scans hundreds of IPs
3. **Abnormal ratio:** High packets-to-destinations ratio
4. **Different pattern:** Deviates from normal traffic baseline

---

## Custom Test Data

### Create Custom PCAP

Generate different traffic volumes:

```bash
# Small test (100 packets)
python generate_test_pcap.py -n 100 -o small_test.pcap

# Medium test (1000 packets) - default
python generate_test_pcap.py -n 1000 -o medium_test.pcap

# Large test (10000 packets)
python generate_test_pcap.py -n 10000 -o large_test.pcap
```

### Using Real PCAP Files

If you have real PCAP files:

```bash
# Test with your own PCAP
python model.py /path/to/your/capture.pcap

# Limit packets for faster testing
python model.py /path/to/your/capture.pcap --max-packets 5000

# Adjust sensitivity
python model.py /path/to/your/capture.pcap --contamination 0.1
```

---

## Unit Tests

### Run Unit Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_model.py -v
pytest tests/test_anon.py -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Test Coverage

Current test coverage includes:

- ✅ Feature engineering (`add_features()`)
- ✅ Model training (`train_anomaly_detector()`)
- ✅ Model persistence (save/load)
- ✅ PCAP anonymization (`PcapAnonymizer`)
- ✅ MAC address anonymization
- ✅ IP address anonymization
- ✅ Attacker subnet preservation
- ✅ Full pipeline integration

---

## Prerequisites for Testing

### Required Software

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **TShark** (for pyshark)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tshark

   # macOS
   brew install wireshark

   # Verify installation
   tshark --version
   ```

3. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Installation Check

Verify all dependencies are installed:

```python
python -c "import pyshark, pandas, numpy, sklearn, tqdm; print('All dependencies OK')"
```

---

## Troubleshooting Tests

### Issue: TShark Not Found

**Error:**
```
TSharkNotFoundException: TShark not found
```

**Solution:**
```bash
# Install tshark
sudo apt-get install tshark

# Or on macOS
brew install wireshark
```

### Issue: Permission Denied

**Error:**
```
Permission denied when reading PCAP
```

**Solution:**
```bash
# Make PCAP readable
chmod +r test_capture.pcap

# Or run with sudo (not recommended)
sudo python model.py test_capture.pcap
```

### Issue: Empty Results

**Error:**
```
No packets found in PCAP file
```

**Solution:**
- Verify PCAP file exists and is not empty
- Check PCAP contains IP packets:
  ```bash
  tshark -r test_capture.pcap -c 10
  ```
- Regenerate test PCAP:
  ```bash
  python generate_test_pcap.py -n 1000
  ```

### Issue: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Or install specific module
pip install <module_name>
```

---

## Test Scenarios

### Scenario 1: Basic Functionality Test

**Objective:** Verify all tools work correctly

```bash
# 1. Generate test data
python generate_test_pcap.py -n 500

# 2. Run detection
python model.py test_capture.pcap

# 3. Verify output file exists
ls -lh unsupervised_processed_output.csv
```

**Expected Result:** Output CSV file created with anomaly flags

### Scenario 2: Anonymization Test

**Objective:** Verify PCAP anonymization preserves attacker IPs

```bash
# 1. Anonymize test PCAP
python anon.py test_capture.pcap

# 2. Check anonymized file
tshark -r test_capture.pcap-anon.pcap -c 20 -T fields -e ip.src

# 3. Verify attacker IP preserved
tshark -r test_capture.pcap-anon.pcap -Y "ip.src == 10.128.1.100" -c 5
```

**Expected Result:** Normal IPs anonymized to 192.168.X.X, attacker IP unchanged

### Scenario 3: Performance Test

**Objective:** Test with large PCAP files

```bash
# 1. Generate large test file
python generate_test_pcap.py -n 10000 -o large_test.pcap

# 2. Time the analysis
time python model.py large_test.pcap

# 3. Check memory usage
/usr/bin/time -v python model.py large_test.pcap
```

**Expected Result:** Completes within reasonable time, acceptable memory usage

### Scenario 4: Visualization Test

**Objective:** Verify plots are generated correctly

```bash
# 1. Run detection
python model.py test_capture.pcap -o test_results.csv

# 2. Generate visualizations
python visualize.py test_results.csv -o ./test_plots

# 3. Check output
ls -lh test_plots/
```

**Expected Result:** 4-5 PNG files created in test_plots directory

---

## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install tshark
      run: |
        sudo apt-get update
        sudo apt-get install -y tshark

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Generate test PCAP
      run: python generate_test_pcap.py -n 500

    - name: Run unit tests
      run: pytest -v

    - name: Test anomaly detection
      run: python model.py test_capture.pcap
```

---

## Performance Benchmarks

Expected performance on standard hardware:

| Packets | Processing Time | Memory Usage |
|---------|----------------|--------------|
| 100     | < 5 seconds    | ~50 MB       |
| 1,000   | < 15 seconds   | ~100 MB      |
| 10,000  | < 2 minutes    | ~500 MB      |
| 100,000 | < 20 minutes   | ~2 GB        |

*Note: Times vary based on hardware and PCAP complexity*

---

## Contributing Tests

When adding new features, please:

1. **Add unit tests** in `tests/` directory
2. **Update test documentation** in this file
3. **Ensure all tests pass** before submitting PR
4. **Maintain >80% code coverage**

Example test structure:

```python
def test_new_feature():
    """Test description."""
    # Arrange
    input_data = ...

    # Act
    result = new_feature(input_data)

    # Assert
    assert result == expected_value
```

---

## Questions?

If you encounter issues with testing:

1. Check this guide first
2. Review the main [README.md](README.md)
3. Check existing issues on GitHub
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - System information (`python --version`, `tshark --version`)
