import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from scipy import ndimage
import random

# Callback để xử lý việc dừng training an toàn
class SafeInterruptCallback(Callback):
    def __init__(self):
        super().__init__()
        self.interrupted = False
        
    def on_epoch_end(self, epoch, logs=None):
        try:
            if self.interrupted:
                print('\nĐang dừng training một cách an toàn...')
                self.model.stop_training = True
        except KeyboardInterrupt:
            self.interrupted = True
            print('\nNhấn Ctrl+C lần nữa để dừng ngay lập tức.')

def augment_data(X, y, augmentation_factor=2):
    """Augment data using various techniques"""
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Add original data
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Add augmented data
        for _ in range(augmentation_factor - 1):
            # Random noise
            noise = np.random.normal(0, 0.1, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])
            
            # Time warping
            if random.random() < 0.5:
                warped = ndimage.gaussian_filter(X[i], sigma=random.uniform(0.1, 0.3))
                augmented_X.append(warped)
                augmented_y.append(y[i])
            
            # Magnitude scaling
            if random.random() < 0.5:
                scale = random.uniform(0.8, 1.2)
                augmented_X.append(X[i] * scale)
                augmented_y.append(y[i])
            
            # Rotation (for accelerometer data)
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                rotated = ndimage.rotate(X[i].reshape(2, 3), angle, reshape=False).flatten()
                augmented_X.append(rotated)
                augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

# Đọc tất cả dữ liệu đã xử lý
print("Đang đọc dữ liệu...")
PROCESSED_DATA_FOLDER = "processed_data"
processed_data = []

# Đọc tất cả file processed data
for file in os.listdir(PROCESSED_DATA_FOLDER):
    if file.startswith("processed_") and file.endswith(".csv"):
        file_path = os.path.join(PROCESSED_DATA_FOLDER, file)
        df = pd.read_csv(file_path)
        processed_data.append(df)
        print(f"Đã đọc file: {file} ({len(df)} mẫu)")

# Gộp tất cả dữ liệu
df = pd.concat(processed_data, ignore_index=True)
print(f"\nTổng số mẫu: {len(df)}")

print("\nPhân bố ActivityLabel:")
print(df['ActivityLabel'].value_counts())

# Chuẩn bị dữ liệu
X = df[['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']].values
y = df['ActivityLabel'].values

# Chuẩn hóa dữ liệu với StandardScaler
print("\nChuẩn hóa dữ liệu...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia dữ liệu thành tập train và test với tỷ lệ 80/20
print("\nChuẩn bị dữ liệu...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Augment training data
print("\nAugment dữ liệu training...")
X_train_aug, y_train_aug = augment_data(X_train, y_train, augmentation_factor=3)
print(f"Số mẫu training sau khi augment: {len(X_train_aug)}")

# In thông tin về shape của dữ liệu
print(f"Shape của dữ liệu train: {X_train_aug.shape}")
print(f"Shape của dữ liệu test: {X_test.shape}")

# In phân bố các lớp
print("\nPhân bố các lớp trong tập train:")
unique_labels = np.unique(y)
for label in unique_labels:
    count = np.sum(y_train_aug == label)
    print(f"Lớp {label}: {count} mẫu ({count/len(y_train_aug)*100:.2f}%)")

# Xây dựng mô hình Dense với kiến trúc mới
print("\nXây dựng mô hình...")
input_shape = (X_train.shape[1],)
num_classes = len(unique_labels)

model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=input_shape),
    
    # Dense layers with batch normalization and dropout
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    # Output layer
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Biên dịch mô hình với learning rate thấp hơn
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks với patience cao hơn
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

safe_interrupt = SafeInterruptCallback()

# Huấn luyện mô hình với batch size nhỏ hơn
print("\nBắt đầu huấn luyện... (Nhấn Ctrl+C để dừng an toàn)")
try:
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=64,
        callbacks=[early_stop, reduce_lr, safe_interrupt],
        verbose=1
    )
    
    # Lưu model sau khi training hoàn tất
    print("\nLưu mô hình...")
    model.save('processed_data/combined_model.h5')
    print("✅ Đã lưu mô hình!")
    
except KeyboardInterrupt:
    print("\nĐã dừng training.")
    
    if len(history.history['loss']) > 0:
        print("Lưu mô hình trạng thái cuối cùng...")
        model.save('processed_data/combined_model.h5')
        print("✅ Đã lưu mô hình!")

# Đánh giá mô hình
print("\nĐánh giá mô hình...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Dự đoán và in ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nBáo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred_classes))

# In thông tin về các nhãn
activity_mapping = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'activity_mapping.csv'))
print("\nMapping các hoạt động:")
print(activity_mapping)
