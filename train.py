import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob

FEATURE_LENGTH = 257  # Define the expected length of feature vectors

# Global label-to-algorithm mapping
GLOBAL_LABEL_TO_ALGORITHM = {
    0: 'AES-CBC',
    1: 'AES-CFB',
    2: 'AES-CTR',
    3: 'AES-ECB',
    4: 'AES-OFB',
    5: 'AES-OPENPGP',
    6: 'ARC2-CBC',
    7: 'ARC2-CFB',
    8: 'ARC2-CTR',
    9: 'ARC2-ECB',
    10: 'ARC2-OFB',
    11: 'ARC2-OPENPGP',
    12: 'ARC4',
    13: 'BLOWFISH-CBC',
    14: 'Blowfish-CFB',
    15: 'Blowfish-CTR',
    16: 'Blowfish-ECB',
    17: 'Blowfish-OFB',
    18: 'Blowfish-OPENPGP',
    19: 'CAST-CBC',
    20: 'CAST-CFB',
    21: 'CAST-CTR',
    22: 'CAST-ECB',
    23: 'CAST-OFB',
    24: 'CAST-OPENPGP',
    25: 'ChaCha20',
    26: 'DES-CBC',
    27: 'DES-CFB',
    28: 'DES-CTR',
    29: 'DES-ECB',
    30: 'DES-OFB',
    31: 'DES-OPENPGP',
    32: 'DES3-CBC',
    33: 'DES3-CFB',
    34: 'DES3-CTR',
    35: 'DES3-ECB',
    36: 'DES3-OFB',
    37: 'DES3-OPENPGP',
    38: 'Salsa20'
}

def is_valid_hex_string(s):
    """Check if a string is a valid hexadecimal string."""
    s = s.strip().lower()
    if len(s) % 2 != 0 or any(c not in '0123456789abcdef' for c in s):
        return False
    return True

def hex_to_bytes(hex_str):
    """Converts a hexadecimal string to a list of byte values."""
    if is_valid_hex_string(hex_str):
        try:
            return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
        except ValueError:
            return []
    else:
        return []

def calculate_entropy(data):
    """Calculates the entropy of the data."""
    if not data:
        return 0
    prob = [float(i) / len(data) for i in Counter(data).values()]
    entropy = -sum(p * math.log(p) / math.log(2.0) for p in prob)
    return entropy

def extract_features(ciphertext):
    """Extracts features from the ciphertext."""
    bytes_data = hex_to_bytes(ciphertext)
    if not bytes_data:
        return np.zeros(FEATURE_LENGTH)  # Return a zero vector if byte conversion failed

    # Byte frequency (normalized)
    byte_freq = np.bincount(bytes_data, minlength=256) / len(bytes_data)

    # Calculate entropy
    entropy = calculate_entropy(bytes_data)

    # Combine features
    features = np.concatenate([byte_freq, [entropy]])

    # Ensure the feature length is consistent
    if len(features) != FEATURE_LENGTH:
        features = np.pad(features, (0, FEATURE_LENGTH - len(features)), 'constant')

    return features

def process_data(file_path):
    """Processes a single CSV file and returns a pandas DataFrame."""
    # Load CSV with the appropriate column names
    df = pd.read_csv(file_path, usecols=['ciphertext', 'algorithm'])

    # Rename columns to match expected names
    df.rename(columns={'ciphertext': 'ciphertext', 'algorithm': 'algorithm'}, inplace=True)

    df['features'] = df['ciphertext'].apply(extract_features)

    # Debug: Print unique algorithms in the dataset
    unique_algorithms = df['algorithm'].unique()
    print(f"Unique algorithms in {file_path}:", unique_algorithms)

    # Ensure algorithm names match the global mapping
    df['algorithm'] = df['algorithm'].str.upper()

    def get_label(algorithm):
        try:
            return list(GLOBAL_LABEL_TO_ALGORITHM.values()).index(algorithm)
        except ValueError:
            print(f"Warning: Algorithm '{algorithm}' is not in the global label mapping.")
            return None

    df['label'] = df['algorithm'].apply(get_label)

    # Filter out rows where labels are None (unmapped algorithms)
    df.dropna(subset=['label'], inplace=True)

    return df

def main():
    # Assuming CSV files are in a directory named 'dataset'
    csv_files = glob.glob('datatypes/*.csv')

    dataframes = []
    for file in csv_files:
        df = process_data(file)
        dataframes.append(df)

    combined_df = pd.concat(dataframes)

    # Split data into features and labels
    X = np.vstack(combined_df['features'])
    y = combined_df['label'].astype(int)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'encryption_model3,4.pkl')
    print("Model saved as 'encryption_model3,4.pkl'")

if __name__ == "__main__":
    main()
