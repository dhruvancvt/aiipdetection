from tqdm import tqdm
import pyshark
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Optimized function for reading PCAP files
def read_pcap(file_path, max_packets=None):
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
    return pd.DataFrame(data)

# Read PCAP file
df = read_pcap('outputs and datasets/wrccdc.2024-02-17.084915.pcap')

# Feature engineering
def add_features(dataframe):
    tqdm.pandas(desc="Adding features")
    dataframe['packet_count'] = dataframe.groupby('Source')['Source'].transform('count')
    dataframe['average_length'] = dataframe.groupby('Source')['Length'].transform('mean')
    dataframe['unique_destinations'] = dataframe.groupby('Source')['Destination'].transform('nunique')
    return dataframe

df = add_features(df)

# Define features for anomaly detection
df['log_packet_count'] = np.log1p(df['packet_count'])
df['log_average_length'] = np.log1p(df['average_length'])
features = ['log_packet_count', 'log_average_length', 'unique_destinations']
X = df[features].fillna(0)

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(X_scaled)
df['is_anomalous'] = (df['anomaly_score'] == -1).astype(int)

# Identify potentially malicious IPs
malicious_ips = df[df['is_anomalous'] == 1]['Source'].unique()
print("\nPotential Malicious IPs:")
for ip in malicious_ips:
    print(f"ip.src == {ip}")

# Save processed data
df.to_csv("unsupervised_processed_output.csv", index=False)
