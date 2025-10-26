"""
Retrain the kidney stone prediction model with current scikit-learn version
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("=" * 60)
print("KIDNEY STONE MODEL RETRAINING")
print("=" * 60)

# Load the dataset
print("\n1. Loading dataset...")
df = pd.read_csv('../data/kidney_stone_urine_analysis_extended.csv')
print(f"   ✓ Loaded {len(df)} samples")
print(f"   ✓ Features: {list(df.columns)}")

# Prepare features and target
print("\n2. Preparing features and target...")
# Drop both 'target' (label) and 'id' (not a feature for prediction)
X = df.drop(['target', 'id'], axis=1)
y = df['target']
print(f"   ✓ Features shape: {X.shape}")
print(f"   ✓ Features: {list(X.columns)}")
print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")

# Split the data
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✓ Training samples: {len(X_train)}")
print(f"   ✓ Testing samples: {len(X_test)}")

# Train the model
print("\n4. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
model.fit(X_train, y_train)
print("   ✓ Model trained successfully")

# Evaluate the model
print("\n5. Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   ✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n   Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Feature Importances:")
for idx, row in feature_importance.iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

# Save the model
print("\n6. Saving model...")
model_path = 'models/kidney_stone_model_extended.joblib'
joblib.dump(model, model_path)
print(f"   ✓ Model saved to: {model_path}")

# Verify the saved model can be loaded
print("\n7. Verifying saved model...")
loaded_model = joblib.load(model_path)
test_prediction = loaded_model.predict(X_test[:1])
print(f"   ✓ Model loaded and tested successfully")
print(f"   ✓ Test prediction: {test_prediction[0]}")

# Print scikit-learn version for reference
import sklearn
print(f"\n8. Environment info:")
print(f"   ✓ scikit-learn version: {sklearn.__version__}")
print(f"   ✓ Model class: {type(model).__name__}")
print(f"   ✓ Model parameters: {model.get_params()}")

print("\n" + "=" * 60)
print("MODEL RETRAINING COMPLETE!")
print("=" * 60)
print("\nYou can now restart your FastAPI server:")
print("  cd ml_api")
print("  uvicorn main:app --reload --port 8000")
print("=" * 60)
