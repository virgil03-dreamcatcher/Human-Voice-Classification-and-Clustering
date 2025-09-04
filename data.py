import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

df=pd.read_csv('vocal_gender_features_new.csv')
# print(df.head(5))
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# print(df.describe) 
# Convert all columns except label to float, label to int
feature_cols = [col for col in df.columns if col != 'label']
df[feature_cols] = df[feature_cols].astype(float)
df['label'] = df['label'].astype(int)

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

# Preview scaled features
print(df_scaled.head())

# Plot histograms for a few selected features
features_to_plot = feature_cols[:6]  # Example: first 6 features
df_scaled[features_to_plot].hist(bins=20, figsize=(12,8))
plt.suptitle('Feature Distributions after Scaling')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(14, 12))
corr_matrix = df_scaled.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Correlation of features with target label
label_corr = df_scaled.corr()['label'].sort_values()
print("Correlation of features with label (gender):")
print(label_corr)

# Prepare features (exclude label)
X = df_scaled[feature_cols]

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Evaluate K-Means with silhouette score
kmeans_sil_score = silhouette_score(X, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_sil_score:.4f}")

# DBSCAN clustering (tune eps as needed)
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Evaluate DBSCAN (ignore noise points labeled -1)
dbscan_core_labels = dbscan_labels[dbscan_labels != -1]
dbscan_core_samples = X[dbscan_labels != -1]

if len(set(dbscan_core_labels)) > 1:
    dbscan_sil_score = silhouette_score(dbscan_core_samples, dbscan_core_labels)
    print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_sil_score:.4f}")
else:
    print("DBSCAN found less than 2 clusters (excluding noise), silhouette score not computed.")



# with open('feature_cols.json', 'w') as f:
#     json.dump(feature_cols, f)