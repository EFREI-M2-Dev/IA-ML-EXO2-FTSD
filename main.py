import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Charger le CSV
df = pd.read_csv("./datasets/medical_dataset.csv")

# Supprimer la colonne "Result" (c'est la cible d'un modèle supervisé, donc on l'ignore ici)
df.drop(columns=["Result"], inplace=True, errors='ignore')

# Nettoyage : suppression des lignes incomplètes
df.dropna(inplace=True)

# Normalisation des données
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Clustering avec KMeans
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Affichage des résultats
print("\n🫀 Patients groupés par similarité de profil clinique :")
for cluster_num in range(n_clusters):
    print(f"\nCluster {cluster_num} :")
    cluster_patients = df[df['cluster'] == cluster_num]
    print(cluster_patients.head(5))  #expl

# Score de silhouette
sil_score = silhouette_score(X_scaled, df['cluster'])
print(f"\nSilhouette Score : {sil_score:.4f}")
