import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

# 1. 시계열 데이터 로딩
df = pd.read_csv('gpu_usage_30min.csv')
data = df['gpu_usage'].values.reshape(-1, 1)

# 2. 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. 시퀀스 생성 함수
def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

sequence_length = 10  # 최근 10개 사용량으로 다음 사용량 예측
X, y = create_sequences(scaled_data, sequence_length)

# 4. dataset 분리
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. LSTM 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 6. 학습 시간 측정 시작
start_time = time.time()

# 7. 모델 학습
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 8. 학습 시간 측정 종료
end_time = time.time()
elapsed = end_time - start_time

print(f"\n 학습 시간: {elapsed:.2f}초")

# 9. 예측
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# 10. 결과 시각화
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title('LSTM GPU Usage Prediction')
plt.xlabel('Time')
plt.ylabel('GPU Usage')
plt.show()
