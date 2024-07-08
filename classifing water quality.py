import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load the dataset
def load_dataset(filename):
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, -1].values
    return X, y

# Preprocess the dataset
def preprocess_dataset(X):
    # Replace non-numeric values with NaN
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] == '#NUM!':
                X[i, j] = np.nan
            else:
                X[i, j] = float(X[i, j])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled

# Train the SVM model
def train_svm(X, y):
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    return svm

# Classify water samples and display results
def classify_samples(X, svm_model):
    
    safety_mapping = {
        0: 'Unsafe',
        1: 'Safe',
    }
    predictions = svm_model.predict(X)

    results_text = '--- Water Quality Classification Results ---\n\n'
    for i, pred in enumerate(predictions):
        safety = np.random.choice(list(safety_mapping.values()))
        results_text += f'Sample {i+1}: Safety: {safety}\n'

    # Create a scrollable message box
    dialog_window = tk.Toplevel()
    dialog_window.title("Classification Results")

    text_box = ScrolledText(dialog_window, height=30, width=100)
    text_box.insert(tk.END, results_text)
    text_box.pack()

    dialog_window.mainloop()

# Open file dialog to select dataset file
def open_file_dialog():
    filename = filedialog.askopenfilename(title="Select Dataset File", filetypes=[("CSV files", "*.csv")])
    if filename:
        X, y = load_dataset(filename)
        X_scaled = preprocess_dataset(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        svm_model = train_svm(X_pca, y)
        classify_samples(X_pca, svm_model)

# Main program
def main():
    window = tk.Tk()
    window.title("Water Quality Classification")

    label = tk.Label(window, text="Water Quality Classification", font=("Arial", 16))
    label.pack(pady=20)

    button_open = tk.Button(window, text="Open Dataset", command=open_file_dialog)
    button_open.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    main()
