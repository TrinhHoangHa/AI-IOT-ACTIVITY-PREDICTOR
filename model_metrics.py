import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import time
import json
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.layers import Layer

class TransformerBlock(Layer):
    def __init__(self, embed_dim=6, num_heads=4, ff_dim=32, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        # Khởi tạo các layer ngay trong __init__
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

def build_transformer_model(input_shape, num_classes):
    """Xây dựng lại model transformer như khi training"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Position encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_encoding = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    x = inputs + pos_encoding
    
    # Transformer blocks
    transformer_block1 = TransformerBlock(input_shape[1], 4, 32)
    transformer_block2 = TransformerBlock(input_shape[1], 4, 32)
    
    x = transformer_block1(x)
    x = transformer_block2(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_sequences(data, seq_length=32, stride=16):
    """Tạo sequences từ dữ liệu time series với stride=16 như transformer"""
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def load_and_preprocess_data():
    """Load và tiền xử lý dữ liệu cho transformer model"""
    print("\nĐang đọc dữ liệu...")
    
    processed_data = []
    total_samples = 0
    
    # Đọc tất cả file dữ liệu đã xử lý
    for file in os.listdir('processed_data'):
        if file.startswith('processed_') and file.endswith('.csv'):
            df = pd.read_csv(f'processed_data/{file}')
            total_samples += len(df)
            print(f"Đã đọc file: {file} ({len(df):,} mẫu)")
            processed_data.append(df)
    
    # Gộp dữ liệu
    df = pd.concat(processed_data, ignore_index=True)
    print(f"\nTổng số mẫu: {total_samples:,}")
    
    # Chuẩn bị features và labels
    features = df[['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']].values
    labels = df['ActivityLabel'].values
    
    # Chuẩn hóa features
    print("\nChuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Tạo sequences
    print("\nTạo sequences...")
    X = create_sequences(features_scaled)
    y = []
    for i in range(0, len(labels) - 32 + 1, 16):
        y.append(labels[i + 32 - 1])  # Lấy nhãn của điểm cuối cùng trong sequence
    y = np.array(y)
    
    print(f"Shape của dữ liệu sau khi tạo sequences: X={X.shape}, y={y.shape}")
    return X, y

def calculate_flops(model, input_shape):
    """Tính toán số FLOPS của model"""
    try:
        # Tạo concrete function
        concrete = tf.function(lambda x: model(x))
        concrete_func = concrete.get_concrete_function(
            tf.TensorSpec(input_shape, tf.float32))
        
        # Chuyển đổi model sang frozen graph
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()
        
        # Tính toán FLOPS
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)
            return flops.total_float_ops
    except Exception as e:
        print(f"Không thể tính FLOPS: {str(e)}")
        return None

def measure_inference_time(model, X_test, num_runs=100):
    """Đo thời gian inference"""
    print("\nĐo thời gian inference...")
    
    # Chuẩn bị sample data cho inference
    sample_data = X_test[0:1]  # Lấy một mẫu
    
    # Warmup
    print("Warmup...")
    _ = model.predict(sample_data, verbose=0)
    
    # Đo thời gian
    print(f"Đang đo {num_runs} lần...")
    times = []
    for i in range(num_runs):
        start_time = time.time()
        _ = model.predict(sample_data, verbose=0)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Chuyển sang ms
    
    return {
        'average_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times)
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Vẽ và lưu confusion matrix"""
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    
    # Tính phần trăm
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Vẽ heatmap
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Ma trận nhầm lẫn (%)')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cm, cm_percent

def plot_tsne(features, labels, class_names, save_path):
    """Vẽ và lưu t-SNE visualization"""
    print("\nĐang tính toán t-SNE...")
    
    # Giảm số chiều xuống 2D
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    features_2d = tsne.fit_transform(features)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(class_names):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=label, alpha=0.6, s=50)
    
    plt.title('Biểu đồ t-SNE')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return features_2d

def get_model_parameters(model):
    """Lấy thông số của model"""
    try:
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_variables])
        
        return {
            'total_parameters': int(trainable_params + non_trainable_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params)
        }
    except Exception as e:
        print(f"Không thể tính số tham số của model: {str(e)}")
        return {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'non_trainable_parameters': 0
        }

def save_model_metrics():
    """Lưu tất cả metrics của model"""
    # Tạo thư mục để lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'model_metrics_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nKết quả sẽ được lưu trong thư mục: {results_dir}")
    
    # Load và tiền xử lý dữ liệu trước để biết kích thước input
    print("\n1. Loading và tiền xử lý dữ liệu...")
    X_test, y_test = load_and_preprocess_data()
    
    # Xây dựng lại model với cấu trúc giống hệt lúc training
    print("\n2. Xây dựng lại model...")
    input_shape = (32, 6)  # sequence_length=32, features=6
    num_classes = len(np.unique(y_test))
    model = build_transformer_model(input_shape, num_classes)
    
    # Load weights từ model đã lưu
    print("\n3. Loading weights...")
    model.load_weights('processed_data/transformer_model.h5')
    
    # Dự đoán
    print("\n4. Thực hiện dự đoán...")
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Đọc mapping hoạt động
    activity_mapping = pd.read_csv('processed_data/activity_mapping.csv')
    class_names = activity_mapping['activity'].tolist()
    
    # Tính toán các metric
    print("\n5. Tính toán các metric...")
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
    accuracy = np.mean(y_test == y_pred_classes)
    
    # Tạo báo cáo phân loại chi tiết
    classification_rep = classification_report(y_test, y_pred_classes, 
                                            target_names=class_names, 
                                            output_dict=True)
    
    # Đo thời gian inference
    print("\n6. Đo thời gian inference...")
    inference_times = measure_inference_time(model, X_test)
    
    # Tính FLOPS
    print("\n7. Tính toán FLOPS...")
    flops = calculate_flops(model, (None, 32, 6))
    
    # Lấy thông số model
    print("\n8. Lấy thông số model...")
    model_params = get_model_parameters(model)
    
    # Tạo visualizations
    print("\n9. Tạo các biểu đồ...")
    cm, cm_percent = plot_confusion_matrix(
        y_test, y_pred_classes, class_names,
        f'{results_dir}/confusion_matrix.png'
    )
    
    # Tạo báo cáo chi tiết
    report = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'raw': cm.tolist(),
            'percentage': cm_percent.tolist()
        },
        'classification_report': classification_rep,
        'inference_time': inference_times,
        'model_parameters': model_params,
        'flops': int(flops) if flops is not None else "Không thể tính toán",
        'class_names': class_names,
        'data_shape': {
            'X_test': X_test.shape,
            'y_test': y_test.shape
        }
    }
    
    # Lưu báo cáo
    print("\n10. Lưu báo cáo...")
    with open(f'{results_dir}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nĐã lưu tất cả kết quả vào thư mục: {results_dir}")
    print("\nTóm tắt kết quả chính:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"\nThời gian inference trung bình: {inference_times['average_ms']:.2f}ms")
    print(f"Số tham số model: {model_params['total_parameters']:,}")

if __name__ == "__main__":
    save_model_metrics() 