from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

df=pd.read_csv('vocal_gender_features_new.csv')
df_scaled=df.copy()
feature_cols = [col for col in df.columns if col != 'label']
# Prepare features and label
X = df_scaled[feature_cols]
y = df_scaled['label']

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Initialize and train SVM with default parameters and probability enabled for future use
# svm_model = SVC(probability=True, random_state=42)
# svm_model.fit(X_train, y_train)

# # Predictions
# y_pred_svm = svm_model.predict(X_test)

# # Evaluation metrics
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# precision_svm = precision_score(y_test, y_pred_svm)
# recall_svm = recall_score(y_test, y_pred_svm)
# f1_svm = f1_score(y_test, y_pred_svm)
# conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# print(f"SVM Classifier Performance:")
# print(f"Accuracy: {accuracy_svm:.4f}")
# print(f"Precision: {precision_svm:.4f}")
# print(f"Recall: {recall_svm:.4f}")
# print(f"F1 Score: {f1_svm:.4f}")
# print(f"Confusion Matrix:\n{conf_matrix_svm}")

# Save the trained model to a file
joblib_file = "rf_voice_classifier.pkl"
joblib.dump(rf_model, joblib_file)

print("Random Forest model saved successfully.")

#Example: fit scaler on all feature columns of your training data
scaler = StandardScaler()
scaler.fit(df[feature_cols])  # Use your original unscaled training features here

# Save the scaler for later use
joblib.dump(scaler, 'feature_scaler.pkl')
print("Scaler saved as 'feature_scaler.pkl'")