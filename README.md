# AI-IOT Activity Predictor

Dự án này được thiết kế để dự đoán các hoạt động của con người trong thời gian thực bằng cách sử dụng dữ liệu cảm biến từ thiết bị ESP32. Dữ liệu được xử lý bằng mô hình Transformer được triển khai trong TensorFlow.

## Yêu cầu

Trước khi chạy mã, đảm bảo bạn đã cài đặt các phần mềm sau:

- Python 3.6 hoặc cao hơn
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Tkinter

Bạn có thể cài đặt các gói Python cần thiết bằng pip:

```sh
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Hướng dẫn

### 1. Làm mịn dữ liệu

Trước tiên, bạn cần làm mịn dữ liệu cảm biến để loại bỏ nhiễu và chuẩn hóa dữ liệu. Bạn có thể sử dụng các công cụ như Pandas và Scikit-learn để thực hiện việc này.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ file CSV
data = pd.read_csv('path/to/your/sensor_data.csv')

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Lưu các tham số của scaler để sử dụng sau này
import numpy as np
np.savez('processed_data/scaler_params.npz', mean=scaler.mean_, scale=scaler.scale_)
```
```sh
python smooth_data.py
```
### 2. Train mô hình

Sau khi dữ liệu đã được làm mịn, bạn có thể bắt đầu huấn luyện mô hình Transformer. Đảm bảo bạn đã chuẩn bị dữ liệu huấn luyện và nhãn tương ứng.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Định nghĩa lớp TransformerBlock
class TransformerBlock(Layer):
    # ...existing code...

# Xây dựng mô hình Transformer
def build_transformer_model(input_shape, num_classes):
    # ...existing code...

# Chuẩn bị dữ liệu huấn luyện
X_train = ...  # Dữ liệu đầu vào
y_train = ...  # Nhãn tương ứng

# Xây dựng và huấn luyện mô hình
input_shape = (sequence_length, 6)
num_classes = len(activity_labels)
model = build_transformer_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Lưu trọng số của mô hình
model.save_weights('processed_data/transformer_model.h5')
```
```sh
python train_transformer.py
```
### 3. Dự đoán hoạt động

Sau khi mô hình đã được huấn luyện, bạn có thể sử dụng nó để dự đoán các hoạt động trong thời gian thực. Chạy script `predict.py` để bắt đầu dự đoán.

```sh
python predict.py
```

Điều này sẽ mở một cửa sổ GUI hiển thị các dự đoán hoạt động của con người theo thời gian thực dựa trên dữ liệu cảm biến nhận được từ thiết bị ESP32.

### 4. Lấy địa chỉ IP của host

Để lấy địa chỉ IP của host (máy tính của bạn), bạn có thể sử dụng lệnh sau trong terminal hoặc command prompt:

- Trên Windows:

  ```sh
  ipconfig
  ```

  Tìm dòng có tên "IPv4 Address" trong kết quả.

- Trên macOS/Linux:

  ```sh
  ifconfig
  ```

  Tìm dòng có tên "inet" trong kết quả.

Sử dụng địa chỉ IP này để cấu hình thiết bị ESP32 của bạn gửi dữ liệu cảm biến đến host.

## Lưu ý
- Đảm bảo Esp32 và máy tính dùng chung 1 mạng.
- Đảm bảo thiết bị ESP32 và predict.py của bạn cùng chung 1 host và port để truyền dữ liệu theo thời gian thực.
- Host mặc định là `192.168.52.147` và dải port là `8080-8090`. Bạn có thể thay đổi các thiết lập này trong lớp `ActivityPredictor`.

## Khắc phục sự cố

- Nếu bạn gặp vấn đề khi load mô hình hoặc scaler, đảm bảo các file được đặt đúng trong thư mục `processed_data`.
- Đối với các vấn đề khác, tham khảo các thông báo lỗi in ra trong console để gỡ lỗi.
