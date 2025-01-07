import pyshark
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def read_pcap(file_path):
    capture = pyshark.FileCapture(file_path)
    data = []
    for packet in capture:
        if 'IP' in packet:
            try:
                data.append({
                    'Source': packet.ip.src,
                    'Destination': packet.ip.dst,
                    'Length': int(packet.length)
                })
            except AttributeError:
                continue
    capture.close()
    return pd.DataFrame(data)


df = read_pcap('outputs and datasets/SUEE1.pcap')


def add_features(dataframe):
    dataframe['packet_count'] = dataframe.groupby('Source')['Source'].transform('count')
    dataframe['average_length'] = dataframe.groupby('Source')['Length'].transform('mean')
    return dataframe

df = add_features(df)

malicious_ips = ['192.168.1.100', '10.0.0.1']
df['is_attacker'] = df['Source'].apply(lambda x: 1 if x in malicious_ips else 0)

features = ['packet_count', 'average_length']
X = df[features].fillna(0)
y = df['is_attacker']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

def optimize_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                               param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Average F1 Score: {np.mean(cv_scores):.4f}")

df['predicted_attacker'] = model.predict(X)
attacker_ips = df[df['predicted_attacker'] == 1]['Source'].unique()
print("\nDetected Attacker IPs:")
for ip in attacker_ips:
    print(f"- {ip}")

df.to_csv("processed_output.csv", index=False)
