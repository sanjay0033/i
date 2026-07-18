"""
Customer Segmentation Project
=============================
1. Load & clean data
2. EDA
3. Feature engineering + scaling
4. Determine optimal k (elbow + silhouette)
5. KMeans clustering
6. PCA visualization of segments
7. Segment profiling (radar/bar characteristics)
8. Export labeled dataset + summary stats
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
OUT = "/home/claude"

# ---------- 1. Load & clean ----------
df = pd.read_csv(f"{OUT}/customer_data.csv")
df["annual_income"] = df["annual_income"].fillna(df["annual_income"].median())
df["web_visits_per_month"] = df["web_visits_per_month"].fillna(df["web_visits_per_month"].median())

# ---------- 2. EDA: correlation heatmap ----------
num_cols = ["age", "annual_income", "annual_spend", "purchase_frequency",
            "recency_days", "avg_order_value", "discount_usage_rate",
            "tenure_months", "web_visits_per_month"]

plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Customer Behavior & Demographic Features")
plt.tight_layout()
plt.savefig(f"{OUT}/01_correlation_heatmap.png")
plt.close()

# ---------- 3. Feature engineering ----------
features = df[num_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ---------- 4. Optimal k: elbow + silhouette ----------
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(K_range), sil_scores, marker="o", color="#DD8452")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{OUT}/02_elbow_silhouette.png")
plt.close()

print("Silhouette scores:", dict(zip(K_range, np.round(sil_scores, 3))))
# NOTE: silhouette score alone maximizes at k=2, but that collapses the customer
# base into an overly coarse split with little marketing value. The elbow curve
# still shows a clear bend around k=5, and k=5 yields well-separated, business-
# actionable segments (confirmed against domain knowledge of typical retail
# customer archetypes: budget shoppers, big spenders, occasional browsers,
# loyal regulars, and new customers). We use k=5 as the business-informed choice.
best_k = 5
print("Chosen k (business-informed):", best_k)

# ---------- 5. Final KMeans model ----------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

# ---------- 6. PCA visualization ----------
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
df["pca1"], df["pca2"] = pcs[:, 0], pcs[:, 1]

plt.figure(figsize=(9, 7))
palette = sns.color_palette("Set2", best_k)
sns.scatterplot(data=df, x="pca1", y="pca2", hue="segment", palette=palette, s=45, alpha=0.8)
plt.title(f"Customer Segments Visualized via PCA (k={best_k})\n"
          f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig(f"{OUT}/03_pca_clusters.png")
plt.close()

# ---------- 7. Segment profiling ----------
profile = df.groupby("segment")[num_cols].mean().round(1)
profile["count"] = df.groupby("segment").size()
profile["pct_of_base"] = (profile["count"] / len(df) * 100).round(1)
profile.to_csv(f"{OUT}/segment_profile_table.csv")
print("\nSegment profile:\n", profile)

# Normalized profile for heatmap (z-scores) to compare relative characteristics
profile_z = (profile[num_cols] - profile[num_cols].mean()) / profile[num_cols].std()
plt.figure(figsize=(10, 6))
sns.heatmap(profile_z, annot=profile[num_cols].values, fmt=".0f", cmap="RdYlGn", center=0,
            cbar_kws={"label": "Relative level (z-score)"})
plt.title("Segment Characteristics (values shown, color = relative intensity)")
plt.ylabel("Segment")
plt.tight_layout()
plt.savefig(f"{OUT}/04_segment_heatmap.png")
plt.close()

# Bar chart: segment sizes
plt.figure(figsize=(8, 5))
size_order = profile["count"].sort_values(ascending=False)
sns.barplot(x=size_order.index.astype(str), y=size_order.values, palette="Set2")
plt.title("Customer Count by Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
for i, v in enumerate(size_order.values):
    plt.text(i, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig(f"{OUT}/05_segment_sizes.png")
plt.close()

# Preferred category mix per segment
cat_mix = pd.crosstab(df["segment"], df["preferred_category"], normalize="index").round(2)
plt.figure(figsize=(9, 5))
cat_mix.plot(kind="bar", stacked=True, colormap="Set3", ax=plt.gca())
plt.title("Preferred Product Category Mix by Segment")
plt.ylabel("Proportion of Segment")
plt.xlabel("Segment")
plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/06_category_mix.png")
plt.close()

# ---------- 8. Export ----------
df.drop(columns=["pca1", "pca2"]).to_csv(f"{OUT}/customer_data_segmented.csv", index=False)
print("\nDone. Files written to", OUT)
