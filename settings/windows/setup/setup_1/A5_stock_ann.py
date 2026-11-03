# Assignment 5: Predict stock market trend using an Artificial Neural Network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. Stock prices (synthetic). To use real data instead:
#    import pandas as pd; prices = pd.read_csv('stock.csv')['Close'].values
np.random.seed(42)
prices = 1000 + np.cumsum(np.random.randn(500) * 10)

# 2. Normalize to 0-1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

# 3. Sliding window: last 5 days -> next day
window = 5
X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i - window:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

# 4. Train/test split in time order (no shuffling)
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# 5. ANN: 2 hidden layers, ReLU, Adam
model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu',
                     solver='adam', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. Predict and convert back to prices
pred   = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
today  = scaler.inverse_transform(X_test[:, -1].reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(actual, pred))
trend_acc = np.mean((pred > today) == (actual > today)) * 100
print(f"RMSE           : {rmse:.2f}")
print(f"Trend accuracy : {trend_acc:.2f} %")

# 7. Next-day trend
next_price = scaler.inverse_transform(model.predict(scaled[-window:].reshape(1, -1)).reshape(-1, 1))[0, 0]
print(f"Last price: {prices[-1]:.2f}, Predicted next: {next_price:.2f} -> "
      f"Trend: {'UP' if next_price > prices[-1] else 'DOWN'}")

plt.plot(actual, label='Actual'); plt.plot(pred, label='Predicted')
plt.title('Stock Price Prediction using ANN'); plt.legend(); plt.show()
