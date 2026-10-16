import os
 
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main(args=None):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_path, "..", "data")

    data = np.load(os.path.join(data_path, "features.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    label_names = [str(n) for n in data["label_names"]]

    print("train: ", X_train.shape, " test: ", X_test.shape, " classes: ", label_names)

    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"\n=== {name} ===")
        print(f"accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("confusion matrix (rows = true, columns = predicted):")
        print("labels:", label_names)
        print(confusion_matrix(y_test, predictions))

        joblib.dump(model, os.path.join(data_path, f"model_{name.lower()}.joblib"))

    print("\nmodels saved in", data_path)
    
if __name__ == "__main__":
    main()
