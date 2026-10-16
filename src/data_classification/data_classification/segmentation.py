import pandas as pd
import os
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

class Segmentation:
    def __init__(self):
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.cur_path, "..", "data")
        self.train_dir = os.path.join(self.data_path, "train")
        self.test_dir = os.path.join(self.data_path, "test")


        self.fs = 500
        self.window_ms = 200 #might need to come back later to check if good size
        self.overlap = 0.5
        self.window = int(self.fs * self.window_ms / 1000) #To get 100 samples 
        self.step = int(self.window * (1 - self.overlap))

    def cutting(self, name):
        
        raw = pd.read_csv(os.path.join(self.data_path, f"{name}.csv"))

        split = int(len(raw) * 0.8)
        train = raw.iloc[:split]
        test  = raw.iloc[split:]

        train.to_csv(os.path.join(self.train_dir, f"{name}_train.csv"), index=False)
        test.to_csv(os.path.join(self.test_dir, f"{name}_test.csv"), index=False)
        return f"{name}: split into {len(train)} train and {len(test)} test rows"


    def make_windows(self, df):
        data = df.to_numpy()
        windows = []
        for start in range(0, len(data) - self.window + 1, self.step):
            windows.append(data[start:start + self.window])
        return np.array(windows)
    
    def signal_segmentation(self, name):
        train = pd.read_csv(os.path.join(self.train_dir, f"{name}_train.csv"))
        test = pd.read_csv(os.path.join(self.test_dir, f"{name}_test.csv"))


        train_window = self.make_windows(train)
        test_window = self.make_windows(test)


        print(f"{name} windows  train: {train_window.shape}  test: {test_window.shape}")
        return train_window, test_window
    
    def compute_threshold(self, rest_train_path, factor=2):
        rest = pd.read_csv(rest_train_path).to_numpy()
        return factor * np.std(np.diff(rest, axis=0), axis=0)
    
    def feature_extraction(self, windows, threshold):
        x = windows - windows.mean(axis=1, keepdims=True)
        diffs = np.diff(x, axis=1)                     
 
        mean_absolute_value = np.mean(np.abs(x), axis=1)
        willison_amplitude = np.sum(np.abs(diffs) >= threshold, axis=1)
        zero_crossing = np.sum((x[:, :-1] * x[:, 1:] < 0) & (np.abs(diffs) >= threshold), axis=1)
        slope_sign_changes = np.sum((diffs[:, :-1] * -diffs[:, 1:]) >= threshold**2, axis=1)
        waveform_length = np.sum(np.abs(diffs), axis=1)
 
        return np.concatenate([mean_absolute_value, willison_amplitude, zero_crossing,
                               slope_sign_changes, waveform_length], axis=1)

    def build_dataset(self, gestures, rest_name):
        # 1. split every recording 80/20
        for name in gestures:
            print(self.cutting(name))
 
        # 2. threshold from the training part of the rest recording only
        threshold = self.compute_threshold(os.path.join(self.train_dir, f"{rest_name}_train.csv"))
 
        # 3. windows -> features, with a label per window
        X_train, y_train, X_test, y_test = [], [], [], []
        for name, label in gestures.items():
            train_windows, test_windows = self.signal_segmentation(name)
            X_train.append(self.feature_extraction(train_windows, threshold))
            X_test.append(self.feature_extraction(test_windows, threshold))
            y_train.append(np.full(len(train_windows), label))
            y_test.append(np.full(len(test_windows), label))
 
        return (np.vstack(X_train), np.concatenate(y_train),
                np.vstack(X_test), np.concatenate(y_test), threshold)

        

def main(args=None):
    segmen_ = Segmentation()
 
    # gesture file name (without .csv) -> label number
    gestures = {"relaxed": 0, "fisting": 1, "hand_down": 2, "hand_up": 3}
 
    X_train, y_train, X_test, y_test, threshold = segmen_.build_dataset(gestures, rest_name="relaxed")
    print("features  train:", X_train.shape, " test:", X_test.shape)
 
    out_path = os.path.join(segmen_.data_path, "features.npz")
    np.savez(out_path,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             threshold=threshold,
             label_names=np.array(list(gestures.keys())))
    print("saved to", out_path)



if __name__ == "__main__":
    main()