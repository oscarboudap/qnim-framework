#!/usr/bin/env python3
"""Train GW Theory Classifier without special chars"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("=" * 80)
print("PHASE 3: Theory Classifier Training")
print("=" * 80)

print("\n[1/6] Loading synthetic GW dataset...")
dataset_path = Path("data/synthetic/massive_dataset/synthetic_gw_20260510_012853.h5")

with h5py.File(dataset_path, 'r') as f:
    strain_plus = f['strain_plus'][:]
    theory_labels = f['theory_labels'][:]

print("[OK] Loaded {} events".format(strain_plus.shape[0]))
print("     Shape: {}".format(strain_plus.shape))

print("\n[2/6] Encoding theory labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(theory_labels)

print("[OK] Classes: {}".format(label_encoder.classes_))
for theory in label_encoder.classes_:
    count = (y == label_encoder.transform([theory])[0]).sum()
    print("     - {}: {} events".format(theory, count))

print("\n[3/6] Applying PCA (64 features)...")
pca = PCA(n_components=64)
X = pca.fit_transform(strain_plus)

explained_variance = np.sum(pca.explained_variance_ratio_)
print("[OK] PCA completed")
print("     Explained variance: {:.2%}".format(explained_variance))
print("     Shape after PCA: {}".format(X.shape))

print("\n[4/6] Splitting dataset (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[OK] Training set: {} samples".format(X_train.shape[0]))
print("     Test set: {} samples".format(X_test.shape[0]))

print("\n[5/6] Training Random Forest classifier (100 trees)...")
classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

classifier.fit(X_train, y_train)
print("[OK] Training complete")

print("\n[6/6] Evaluating model...")

train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print("[OK] Training accuracy: {:.2%}".format(train_accuracy))
print("[OK] Test accuracy:     {:.2%}".format(test_accuracy))

print("\nPer-class metrics:")
# Decode class names from bytes to str
class_names = [c.decode('utf-8') if isinstance(c, bytes) else c for c in label_encoder.classes_]
print(classification_report(y_test, test_pred, target_names=class_names))

print("\nSaving model and preprocessor...")

Path("models").mkdir(exist_ok=True)

classifier_path = "models/qnim_theory_classifier.pkl"
joblib.dump(classifier, classifier_path)
print("[OK] Classifier saved to {}".format(classifier_path))

pca_path = "models/qnim_pca.pkl"
joblib.dump(pca, pca_path)
print("[OK] PCA preprocessor saved to {}".format(pca_path))

encoder_path = "models/qnim_label_encoder.pkl"
joblib.dump(label_encoder, encoder_path)
print("[OK] Label encoder saved to {}".format(encoder_path))

weights_path = "models/qnim_vqc_weights.npy"
np.save(weights_path, classifier.feature_importances_)
print("[OK] Weights saved to {}".format(weights_path))

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE")
print("=" * 80)
print("\nModel ready for inference on GW150914")
print("Accuracy: {:.1%}".format(test_accuracy))
