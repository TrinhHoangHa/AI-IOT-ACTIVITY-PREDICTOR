

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

RAW_DATA_FOLDER = "raw_data"
PROCESSED_DATA_FOLDER = "preprocessing/actual_data"
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

def smooth_data(data, window_size=5, poly_order=2):
    return savgol_filter(data, window_size, poly_order)

def extract_features(df):
    features = []
    for axis in ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]:
        values = smooth_data(df[axis].values)
        features.extend([ 
            np.mean(values), 
            np.std(values), 
            skew(values), 
            kurtosis(values),
        ])
    return features

feature_list = []
labels = []

for file in os.listdir(RAW_DATA_FOLDER):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(RAW_DATA_FOLDER, file))
        if df.empty:
            print(f"⚠️ Bỏ qua file rỗng: {file}")
            continue

        activity = df["ActivityLabel"].iloc[0]
        features = extract_features(df)
        feature_list.append(features)
        labels.append(activity)

features_np = np.array(feature_list)
labels_np = np.array(labels)

scaler = StandardScaler()
features_np = scaler.fit_transform(features_np)

np.save(os.path.join(PROCESSED_DATA_FOLDER, "sensor_data.npy"), features_np)
np.save(os.path.join(PROCESSED_DATA_FOLDER, "sensor_labels.npy"), labels_np)

print(f"✅ Dữ liệu đã tiền xử lý và lưu tại {PROCESSED_DATA_FOLDER}")
