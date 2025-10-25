import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


ticker = input("Enter stock symbol (e.g. AAPL, RELIANCE.NS): ").upper()
df = yf.download(ticker, start="2018-01-01", end="2025-01-01")
if df.empty:
    raise ValueError("Invalid ticker or no data found!")

df['SMA_10'] = df['Close'].rolling(10).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df.dropna()


features = ['SMA_10', 'SMA_50', 'RSI']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {acc:.2f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))


df_test = df.iloc[len(X_train):].copy()
df_test['Predicted'] = y_pred

plt.figure(figsize=(12,5))
plt.plot(df_test.index, df_test['Close'], label='Close Price', color='blue')
plt.scatter(df_test.index[df_test['Predicted']==1],
            df_test[df_test['Predicted']==1]['Close'],
            color='green', label='Predicted Up', marker='^', alpha=0.8)
plt.scatter(df_test.index[df_test['Predicted']==0],
            df_test[df_test['Predicted']==0]['Close'],
            color='red', label='Predicted Down', marker='v', alpha=0.8)
plt.title(f"{ticker} — Predicted Up/Down Signals")
plt.legend()
plt.show()
