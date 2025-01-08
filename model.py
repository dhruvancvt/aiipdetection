from tqdm import tqdm
import pyshark
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from lightgbm import LGBMClassifier
import optuna


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


df = read_pcap('outputs and datasets/SUEE1.pcap', max_packets=1000000)


# Optimized feature engineering
def add_features(dataframe):
    tqdm.pandas(desc="Adding features")
    dataframe['packet_count'] = dataframe.groupby('Source')['Source'].transform('count')
    dataframe['average_length'] = dataframe.groupby('Source')['Length'].transform('mean')
    dataframe['unique_destinations'] = dataframe.groupby('Source')['Destination'].transform('nunique')
    return dataframe


df = add_features(df)

# Define malicious IPs
malicious_ips = ['192.168.1.100', '10.0.0.1']
df['is_attacker'] = df['Source'].apply(lambda x: 1 if x in malicious_ips else 0)

# Define features
df['log_packet_count'] = np.log1p(df['packet_count'])
df['log_average_length'] = np.log1p(df['average_length'])
features = ['log_packet_count', 'log_average_length', 'unique_destinations']
X = df[features].fillna(0)
y = df['is_attacker']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# Optimized hyperparameter tuning with Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best parameters:", best_params)

# Train the final model
model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
model.fit(X_resampled, y_resampled)

# Evaluate the model
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

# Save processed data and predictions
df['predicted_attacker'] = model.predict(X)
attacker_ips = df[df['predicted_attacker'] == 1]['Source'].unique()
print("\nDetected Attacker IPs:")
for ip in attacker_ips:
    print("ip.src ==", ip)

df.to_csv("processed_output.csv", index=False)
