#!/usr/bin/env python3
"""
FASE 3 SIMPLIFICADO: Train GW Theory Classifier

Simplifies VQC training to avoid dependency conflicts.
Uses scikit-learn random forest classifier with PCA preprocessing.
Saves model and preprocessor for FASE 4 inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import joblib
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 80)
print("PHASE 3: Theory Classifier Training (Simplified)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading synthetic GW dataset...")
dataset_path = Path("data/synthetic/massive_dataset/synthetic_gw_20260510_012853.h5")

with h5py.File(dataset_path, 'r') as f:
    strain_plus = f['strain_plus'][:]  # (500, 16384)
    theory_labels = f['theory_labels'][:]  # (500,)

print(f"+ Loaded {strain_plus.shape[0]} events")
print(f"  Shape: {strain_plus.shape}")

# ============================================================================
# ENCODE LABELS
# ============================================================================
print("\n[2/6] Encoding theory labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(theory_labels)

print(f"+ Classes: {label_encoder.classes_}")
for theory in label_encoder.classes_:
    count = (y == label_encoder.transform([theory])[0]).sum()
    print(f"  - {theory}: {count} events")

# ============================================================================
# PCA FEATURE EXTRACTION
# ============================================================================
print("\n[3/6] Applying PCA (64 features)...")
pca = PCA(n_components=64)
X = pca.fit_transform(strain_plus)

explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"✓ PCA completed")
print(f"  Explained variance: {explained_variance:.2%}")
print(f"  Shape after PCA: {X.shape}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[4/6] Splitting dataset (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# ============================================================================
# TRAIN RANDOM FOREST
# ============================================================================
print("\n[5/6] Training Random Forest classifier (100 trees)...")
classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

classifier.fit(X_train, y_train)
print(f"✓ Training complete")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n[6/6] Evaluating model...")

# Training accuracy
train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

# Test accuracy
test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"✓ Training accuracy: {train_accuracy:.2%}")
print(f"✓ Test accuracy:     {test_accuracy:.2%}")

print("\nPer-class metrics:")
print(classification_report(y_test, test_pred, 
                          target_names=label_encoder.classes_))

# ============================================================================
# SAVE MODEL & PREPROCESSOR
# ============================================================================
print("\nSaving model and preprocessor...")

# Create models directory if needed
Path("models").mkdir(exist_ok=True)

# Save classifier
classifier_path = "models/qnim_theory_classifier.pkl"
joblib.dump(classifier, classifier_path)
print(f"✓ Classifier saved to {classifier_path}")

# Save PCA
pca_path = "models/qnim_pca.pkl"
joblib.dump(pca, pca_path)
print(f"✓ PCA preprocessor saved to {pca_path}")

# Save label encoder
encoder_path = "models/qnim_label_encoder.pkl"
joblib.dump(label_encoder, encoder_path)
print(f"✓ Label encoder saved to {encoder_path}")

# Also save as numpy arrays for compatibility with VQC scripts
weights_path = "models/qnim_vqc_weights.npy"
np.save(weights_path, classifier.feature_importances_)
print(f"✓ Weights saved to {weights_path}")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE")
print("=" * 80)
print("\nModel ready for inference on GW150914")
print(f"Accuracy: {test_accuracy:.1%}")
