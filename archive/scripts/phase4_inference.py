#!/usr/bin/env python3
"""PHASE 4: Analyze GW150914 with trained classifier"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import joblib
from datetime import datetime

print("=" * 80)
print("PHASE 4: GW150914 Event Analysis")
print("=" * 80)

# Load trained model
print("\n[1/4] Loading trained model...")
classifier = joblib.load("models/qnim_theory_classifier.pkl")
pca = joblib.load("models/qnim_pca.pkl")
label_encoder = joblib.load("models/qnim_label_encoder.pkl")
print("[OK] Model loaded")
print("     Classes: {}".format(label_encoder.classes_))

# Load GW150914 data
print("\n[2/4] Loading real GW150914 data...")
h1_file = Path("data/raw/H-H1_LOSC_4_V2-1126259446-32.hdf5")
l1_file = Path("data/raw/L-L1_LOSC_4_V2-1126259446-32.hdf5")

if not h1_file.exists() or not l1_file.exists():
    print("[ERROR] GW150914 data not found in data/raw/")
    sys.exit(1)

try:
    with h5py.File(h1_file, 'r') as h1f:
        h1_strain = h1f['/strain/Strain'][()]
    with h5py.File(l1_file, 'r') as l1f:
        l1_strain = l1f['/strain/Strain'][()]
    
    print("[OK] Loaded H1 data: {} samples".format(len(h1_strain)))
    print("     Loaded L1 data: {} samples".format(len(l1_strain)))
except Exception as e:
    print("[ERROR] Failed to load data: {}".format(e))
    sys.exit(1)

# Prepare data for inference
print("\n[3/4] Preprocessing for inference...")

# Use H1 channel (same as training)
strain_data = h1_strain
if len(strain_data) < 16384:
    # Pad if too short
    strain_data = np.pad(strain_data, (0, 16384 - len(strain_data)))
elif len(strain_data) > 16384:
    # Trim to 16384 samples
    strain_data = strain_data[:16384]

# Reshape for PCA (1 sample, 16384 features)
X = strain_data.reshape(1, -1)

# Apply PCA
X_pca = pca.transform(X)

print("[OK] Preprocessed data shape: {}".format(X_pca.shape))

# Run inference
print("\n[4/4] Running inference...")

# Predict class
prediction = classifier.predict(X_pca)[0]
probabilities = classifier.predict_proba(X_pca)[0]

# Get class name
pred_class = label_encoder.classes_[prediction]
if isinstance(pred_class, bytes):
    pred_class = pred_class.decode('utf-8')

print("[OK] Prediction complete")

# Display results
print("\n" + "=" * 80)
print("GW150914 ANALYSIS RESULTS")
print("=" * 80)

print("\nPredicted Theory Family: {}".format(pred_class))
print("\nProbability Distribution:")

for i, class_name in enumerate(label_encoder.classes_):
    if isinstance(class_name, bytes):
        class_name = class_name.decode('utf-8')
    prob = probabilities[i] * 100
    bar = "=" * int(prob / 2)
    print("  {:<20} {:.1f}%  {}".format(class_name, prob, bar))

# Save results
print("\nSaving results...")
results = {
    'timestamp': datetime.now().isoformat(),
    'event': 'GW150914',
    'predicted_theory': pred_class,
    'probabilities': {
        label_encoder.classes_[i].decode('utf-8') if isinstance(label_encoder.classes_[i], bytes) else label_encoder.classes_[i]: float(probabilities[i])
        for i in range(len(label_encoder.classes_))
    },
    'h1_samples': len(h1_strain),
    'l1_samples': len(l1_strain)
}

import json
results_file = "reports/gw150914_analysis_results.json"
Path("reports").mkdir(exist_ok=True)

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print("[OK] Results saved to {}".format(results_file))

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE - GW150914 Analysis Finished")
print("=" * 80)
print("\nNext: PHASE 5 - Generate comprehensive reports")
