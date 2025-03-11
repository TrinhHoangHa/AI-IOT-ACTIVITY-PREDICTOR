import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

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

def create_sequences(data, seq_length, stride=64):  # Giảm stride
    """Create sequences with moderate stride"""
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def add_noise(data, noise_factor=0.1):  # Giảm nhiễu
    """Add moderate random noise to sequences"""
    noise = np.random.normal(loc=0, scale=noise_factor, size=data.shape)
    return data + noise

def time_warp(data, sigma=0.3):  # Giảm biến dạng
    """Apply moderate time warping to sequences"""
    warped_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        time_steps = np.arange(data.shape[1])
        random_warps = 1.0 + np.random.normal(loc=0, scale=sigma, size=data.shape[1])
        warped_steps = np.cumsum(random_warps)
        warped_steps = (data.shape[1] - 1) * (warped_steps - warped_steps.min()) / (warped_steps.max() - warped_steps.min())
        
        for j in range(data.shape[2]):
            warped_data[i, :, j] = np.interp(time_steps, warped_steps, data[i, :, j])
    
    return warped_data

def rotate_data(data, max_angle=15):
    """Apply random rotation to sequences"""
    rotated_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        # Random angle in degrees
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle)
        
        # Rotation matrix
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        rot_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Apply rotation to accelerometer and gyroscope data separately
        for j in range(data.shape[1]):
            # Rotate accelerometer data (X, Y)
            accel_xy = np.dot(rot_matrix, data[i, j, :2])
            rotated_data[i, j, 0] = accel_xy[0]
            rotated_data[i, j, 1] = accel_xy[1]
            rotated_data[i, j, 2] = data[i, j, 2]  # Keep Z unchanged
            
            # Rotate gyroscope data (X, Y)
            gyro_xy = np.dot(rot_matrix, data[i, j, 3:5])
            rotated_data[i, j, 3] = gyro_xy[0]
            rotated_data[i, j, 4] = gyro_xy[1]
            rotated_data[i, j, 5] = data[i, j, 5]  # Keep Z unchanged
    
    return rotated_data

def magnitude_scale(data, sigma=0.15):  # Increased scaling variation
    """Apply stronger random magnitude scaling to sequences"""
    scales = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], 1, data.shape[2]))
    return data * scales

def augment_data(X, y):
    """Augment sequence data with multiple techniques"""
    print("Augmenting data...")
    X_aug = []
    y_aug = []
    
    # Original data 
    X_aug.append(X)
    y_aug.append(y)
    
    # Noisy data
    X_noise = add_noise(X.copy())
    X_aug.append(X_noise)
    y_aug.append(y)
    
    # Time warped data
    X_warp = time_warp(X.copy())
    X_aug.append(X_warp)
    y_aug.append(y)
    
    # Rotated data
    X_rot = rotate_data(X.copy())
    X_aug.append(X_rot)
    y_aug.append(y)
    
    # Magnitude scaled data
    X_scale = magnitude_scale(X.copy())
    X_aug.append(X_scale)
    y_aug.append(y)
    
    # Combined augmentations
    X_combined = np.concatenate(X_aug)
    y_combined = np.concatenate(y_aug)
    
    # Shuffle
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    
    return X_combined[indices], y_combined[indices]

def load_and_preprocess_data(seq_length=32):  # Giảm sequence length
    """Load and preprocess data"""
    print("Đang đọc dữ liệu...")
    PROCESSED_DATA_FOLDER = "processed_data"
    processed_data = []

    for file in os.listdir(PROCESSED_DATA_FOLDER):
        if file.startswith("processed_") and file.endswith(".csv"):
            file_path = os.path.join(PROCESSED_DATA_FOLDER, file)
            df = pd.read_csv(file_path)
            processed_data.append(df)
            print(f"Đã đọc file: {file} ({len(df)} mẫu)")

    df = pd.concat(processed_data, ignore_index=True)
    print(f"\nTổng số mẫu sau khi gộp: {len(df)}")

    print("\nCân bằng dữ liệu...")
    min_samples = df['ActivityLabel'].value_counts().min()
    balanced_data = []

    for label in df['ActivityLabel'].unique():
        label_data = df[df['ActivityLabel'] == label]
        if len(label_data) > min_samples:
            sampled_data = label_data.sample(n=min_samples, random_state=42)
            balanced_data.append(sampled_data)
        else:
            balanced_data.append(label_data)

    df = pd.concat(balanced_data, ignore_index=True)
    print(f"Số mẫu sau khi cân bằng: {len(df)}")
    print("\nPhân bố ActivityLabel sau khi cân bằng:")
    print(df['ActivityLabel'].value_counts())

    X = df[['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']].values
    y = df['ActivityLabel'].values

    print("\nChuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\nTạo sequences (độ dài = {seq_length}, stride = 64)...")
    X = create_sequences(X, seq_length, stride=64)
    y = y[seq_length-1::64]  # Adjust labels according to stride

    # Apply data augmentation
    print("\nÁp dụng data augmentation...")
    X_aug, y_aug = augment_data(X, y)

    print(f"Shape của dữ liệu sau khi augment: {X_aug.shape}")
    return X_aug, y_aug

def build_lstm_model(input_shape, num_classes):
    """Build LSTM model with balanced regularization"""
    model = tf.keras.Sequential([
        # Input layer with moderate noise
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.GaussianNoise(0.1),  # Giảm nhiễu
        
        # First LSTM layer - tăng units
        tf.keras.layers.LSTM(64, return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(5e-4),  # Giảm L2
                           recurrent_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),  # Giảm dropout
        
        # Second LSTM layer - tăng units
        tf.keras.layers.LSTM(32, return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                           recurrent_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Third LSTM layer - tăng units
        tf.keras.layers.LSTM(16, return_sequences=False,
                           kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                           recurrent_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Dense layers with moderate regularization
        tf.keras.layers.Dense(16, activation='relu',  # Tăng units
                            kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
    ])
    
    return model

def train_lstm_model():
    """Train LSTM model với quy trình được tối ưu hóa"""
    
    # 1. Load và tiền xử lý dữ liệu
    print("\n1. Đang tải và xử lý dữ liệu...")
    X, y = load_and_preprocess_data(seq_length=32)
    
    # 2. Chia dữ liệu thành 3 tập với stratification
    print("\n2. Chia dữ liệu thành tập train/validation/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.15,  # Giảm test size
        stratify=y,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.15,  # Giảm validation size
        stratify=y_temp,
        random_state=42
    )
    
    print(f"Kích thước dữ liệu:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # 3. Xây dựng mô hình với kiến trúc cải tiến
    print("\n3. Xây dựng mô hình LSTM...")
    model = build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=len(np.unique(y))
    )
    
    # 4. Compile với optimizer được tối ưu
    print("\n4. Biên dịch mô hình...")
    initial_learning_rate = 1e-3
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0,  # Gradient clipping
        weight_decay=1e-5  # L2 regularization
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. Callbacks được cải thiện
    print("\n5. Thiết lập callbacks...")
    
    # Early stopping với patience cao hơn
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Tăng patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction với các tham số tối ưu
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Giảm learning rate một nửa
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Model checkpoint để lưu model tốt nhất
    checkpoint = ModelCheckpoint(
        'processed_data/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Tensorboard callback để theo dõi quá trình training
    tensorboard = TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # 6. Training với các tham số tối ưu
    print("\n6. Bắt đầu training...")
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,  # Tăng số epochs
            batch_size=32,
            callbacks=[early_stop, reduce_lr, checkpoint, tensorboard],
            verbose=1,
            shuffle=True
        )
        
        # 7. Đánh giá model
        print("\n7. Đánh giá model...")
        # Đánh giá trên validation set
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation accuracy: {val_accuracy:.4f}")
        
        # Đánh giá trên test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # 8. Lưu model cuối cùng
        print("\n8. Lưu mô hình...")
        model.save('processed_data/lstm_model_final.h5')
        print("✅ Đã lưu mô hình!")
        
        # 9. Vẽ đồ thị training history
        print("\n9. Vẽ đồ thị training history...")
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('processed_data/training_history.png')
        print("✅ Đã lưu đồ thị training history!")
        
    except KeyboardInterrupt:
        print("\nĐã dừng training theo yêu cầu.")
        
        if len(history.history['loss']) > 0:
            print("Lưu mô hình trạng thái cuối cùng...")
            model.save('processed_data/lstm_model_interrupted.h5')
            print("✅ Đã lưu mô hình!")
    
    return history, model

if __name__ == "__main__":
    # Thiết lập seed cho tính tái lập
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Thiết lập GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Chạy training
    history, model = train_lstm_model() 