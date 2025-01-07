import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("output.csv")


df['packet_count'] = df.groupby('Source')['Source'].transform('count')
df['average_length'] = df.groupby('Source')['Length'].transform('mean')


df['is_attacker'] = 0  
malicious_ips = ['192.168.1.100', '10.0.0.1']  # Replace with actual attacker IPs
df.loc[df['Source'].isin(malicious_ips), 'is_attacker'] = 1

# Drop non-numeric columns and irrelevant data
features = ['packet_count', 'average_length']
X = df[features]
y = df['is_attacker']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Identify attacker IPs
df['predicted_attacker'] = model.predict(X[features])
attacker_ips = df[df['predicted_attacker'] == 1]['Source'].unique()
print("Detected Attacker IPs:", attacker_ips)
