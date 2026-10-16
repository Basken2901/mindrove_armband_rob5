import pandas as pd
import os
import numpy as np

class Segmentation:
    def __init__(self):
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.cur_path, "..", "data")
        self.import_data = os.path.join(self.data_path, "test.csv")
        self.train_data = os.path.join(self.data_path, "train", "first_train.csv")
        self.test_data = os.path.join(self.data_path, "test","first_test.csv")

        self.fs = 500
        self.window_ms = 200 #might need to come back later to check if good size
        self.overlap = 0.5
        self.window = int(self.fs * self.window_ms / 1000) #To get 100 samples 
        self.step = int(self.window * (1 - self.overlap))

    def cutting(self):
        
        raw = pd.read_csv(self.import_data)  

        split = int(len(raw) * 0.8)        
        train = raw.iloc[:split]           
        test  = raw.iloc[split:] 

        train.to_csv(self.train_data, index=False)
        test.to_csv(self.test_data, index=False)
        return "Split into 80% training and 20% test"

    def make_windows(self, df):
        data = df.to_numpy()
        windows = []
        for start in range(0, len(data) - self.window + 1, self.step):
            windows.append(data[start:start + self.window])
        return np.array(windows)
    
    def signal_segmentation(self):
        train = pd.read_csv(self.train_data)
        test = pd.read_csv(self.test_data)

        train_window = self.make_windows(train)
        test_window = self.make_windows(test)

        print("train: ", train_window)
        print("test: ", test_window)

        return train_window, test_window
    def feature_extraction(self, data):
        

        

def main(args=None):
    segmen_ = Segmentation()
    print(segmen_.cutting())
    train_windows, test_windows = segmen_.signal_segmentation()


if __name__ == "__main__":
    main()