import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

df = pd.read_csv('data.csv')
print(f"Samples: {df.shape[0]}, Features: {df.shape[1]}")
print(f"Features: {', '.join(df.columns)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

k_range = range(2, 11)
results = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    
    results.append({'k': k, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch})
    print(f"k={k}: sil={sil:.4f}, db={db:.4f}, ch={ch:.1f}")

best_by_silhouette = max(results, key=lambda x: x['silhouette'])
best_k = best_by_silhouette['k']

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nOptimal k: {best_k}")
print(f"Silhouette: {best_by_silhouette['silhouette']:.4f}")

print("\nCluster distribution:")
for cluster, count in df['Cluster'].value_counts().sort_index().items():
    print(f"  Cluster {cluster}: {count} ({count/len(df)*100:.1f}%)")

df.to_csv('data_clustered.csv', index=False)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = plt.cm.Set1(np.linspace(0, 1, best_k))
for i in range(best_k):
    mask = df['Cluster'] == i
    axes[0,0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], label=f'Cluster {i}', alpha=0.5, s=10)
axes[0,0].set_title(f'K-Means Clustering Results (k={best_k})')
axes[0,0].set_xlabel('PC1')
axes[0,0].set_ylabel('PC2')
axes[0,0].legend(fontsize=8)

k_vals = [r['k'] for r in results]
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
axes[0,1].plot(k_vals, inertias, 'bo-', linewidth=2)
axes[0,1].set_title('Elbow Method for Optimal k')
axes[0,1].set_xlabel('Number of clusters (k)')
axes[0,1].set_ylabel('Inertia')
axes[0,1].axvline(x=best_k, color='r', linestyle='--')
axes[0,1].grid(True, alpha=0.3)

sil_scores = [r['silhouette'] for r in results]
axes[1,0].plot(k_vals, sil_scores, 'ro-', linewidth=2)
axes[1,0].set_title('Silhouette Score vs Number of Clusters')
axes[1,0].set_xlabel('Number of clusters (k)')
axes[1,0].set_ylabel('Silhouette Score')
axes[1,0].axvline(x=best_k, color='r', linestyle='--')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10)
axes[1,1].set_title('Original Data (No Clustering)')
axes[1,1].set_xlabel('PC1')
axes[1,1].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('clustering_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved: data_clustered.csv, clustering_analysis.png")