import numpy as np
import joblib
from collections import Counter
import math
import gc
import gradio as gr
import pandas as pd

FEATURE_LENGTH = 257  # Must match the length used during training

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
    13: 'Blowfish-CBC',
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

# Mapping of models to the algorithms they were trained on
MODEL_ALGORITHM_MAPS = {
    'encryption_model24,25.pkl': [23,24],    # Model trained on AES-CBC and ARC2-CBC
    'encryption_model38,39.pkl': [31,38], # Model trained on Blowfish-CBC and CAST-CBC
    'encryption_model32,33.pkl': [37,26]
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
        print("Error: Invalid hexadecimal string.")
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

def load_model(model_file):
    """Load a single model from the provided file."""
    model = joblib.load(model_file)
    print(f"Model loaded from {model_file}")
    return model

def predict_with_model(model, features, model_file):
    """Predict using a single model and return the algorithm with the highest probability."""
    predictions = {}

    # Predict with the model and get probabilities
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    pred_label = prediction[0]
    trained_algos = MODEL_ALGORITHM_MAPS.get(model_file, [])

    # Debugging information
    print(f"Model: {model_file}")
    print(f"Predicted label: {pred_label}")
    print(f"Predicted probabilities: {probabilities}")

    # Get the index of the predicted label in the classes array
    label_index = np.where(model.classes_ == pred_label)[0][0]

    # Get the probability for the predicted label
    prob = probabilities[0][label_index]

    # Only consider the algorithm if it matches the trained algorithms of the model
    if pred_label in trained_algos:
        algo_name = GLOBAL_LABEL_TO_ALGORITHM.get(pred_label, "Unknown algorithm")
        predictions[model_file] = (algo_name, prob)
    else:
        print(f"Label {pred_label} not in trained algorithms for {model_file}")

    return predictions

def evaluate_best_prediction(predictions):
    """Evaluate and select the algorithm with the highest probability."""
    if not predictions:
        return "Not matched"

    best_algo, best_prob = None, 0.0

    for algo, prob in predictions.values():
        if prob > best_prob:
            best_algo = algo
            best_prob = prob

    return best_algo if best_algo else "Not matched"

def process_ciphertext(ciphertext):
    """Process a single ciphertext input."""
    features = extract_features(ciphertext)
    features = features.reshape(1, -1)  # Reshape for a single sample
    all_predictions = {}

    model_files = ['encryption_model24,25.pkl', 'encryption_model38,39.pkl','encryption_model32,33.pkl']
    for model_file in model_files:
        model = load_model(model_file)
        predictions = predict_with_model(model, features, model_file)
        all_predictions.update(predictions)
        del model
        gc.collect()

    best_prediction = evaluate_best_prediction(all_predictions)
    return best_prediction

def process_csv(file):
    """Process a CSV file with ciphertexts."""
    df = pd.read_csv(file.name)
    results = []

    if 'ciphertext' in df.columns:
        for index, row in df.iterrows():
            ciphertext = row['ciphertext']
            algo = process_ciphertext(ciphertext)
            results.append({"ciphertext": ciphertext, "algorithm": algo})
    else:
        return "CSV file must contain a 'ciphertext' column."

    return pd.DataFrame(results)

def gradio_interface(ciphertext, csv_file):
    """Gradio interface function."""
    if csv_file:
        return process_csv(csv_file)
    else:
        return process_ciphertext(ciphertext)

# Define custom CSS
custom_css = """
    .input_textbox, .output_text {
        font-size: 20px; /* Adjust the font size as needed */
    }
"""

# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter ciphertext here...", label="Single Ciphertext"),
        gr.File(label="Upload CSV File with Ciphertexts")
    ],
    outputs="dataframe",
    title="Encryption Algorithm Detection",
    description="Enter a ciphertext (in hexadecimal format) to detect the encryption algorithm used. Alternatively, upload a CSV file containing ciphertexts to process multiple entries.",
    css=custom_css
)

# Launch the Gradio app on port 7000
if __name__ == "__main__":
    iface.launch(server_port=7000)
