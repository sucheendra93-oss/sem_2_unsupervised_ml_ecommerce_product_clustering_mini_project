"""
E-Commerce Product Clustering (Unsupervised ML)

Pipeline:
  1. Load products            -> products.csv
  2. Feature engineering      -> text (TF-IDF + SVD) + numbers (log + StandardScaler)
  3. Choose K                 -> Elbow method + Silhouette score
  4. Cluster                  -> K-Means
  5. Compare algorithms       -> K-Means vs Agglomerative vs DBSCAN
  6. Interpret & visualise    -> top keywords per cluster, PCA 2-D plot

Run:  python clustering.py      (saves plots + clustered CSV in outputs/)
"""
import os

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

DATA = "products.csv"
OUT = "outputs"
SKEWED = ["price", "num_reviews"]       # long-tailed -> log first
NORMAL = ["rating"]
TEXT_WEIGHT, NUM_WEIGHT = 1.0, 0.1      # text decides "what" a product is; numbers only nudge


# ---------- 1. Load ----------
def load_data(path=DATA):
    if not os.path.exists(path):
        raise SystemExit("products.csv not found - run:  python download_data.py")
    return add_text(pd.read_csv(path))


def add_text(df):
    """Title + description in one column, the model reads this."""
    return df.assign(text=df["title"].fillna("") + " " + df["description"].fillna(""))


# ---------- 2. Features ----------
def make_features():
    """Turns a product table into a numeric matrix the algorithms can use."""
    text = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,   # words -> importance scores
                                  sublinear_tf=True, max_df=0.5)),                # damp repeats, drop too-common words
        ("svd", TruncatedSVD(n_components=50, random_state=42)),                          # 1000s of words -> 50 "topics"
        ("norm", Normalizer()),                                                           # equal length vectors
    ])
    return ColumnTransformer(
        [("text", text, "text"),
         ("skewed", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), SKEWED),
         ("normal", StandardScaler(), NORMAL)],
        transformer_weights={"text": TEXT_WEIGHT, "skewed": NUM_WEIGHT, "normal": NUM_WEIGHT},
    )


def make_model(k):
    """Full model = features + K-Means. Fit once, then .predict() works on new products too."""
    return Pipeline([("features", make_features()),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42))])


# ---------- 3. Choose K ----------
def evaluate_k(df, k_range=range(2, 11)):
    X = make_features().fit_transform(df)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_)))
    return pd.DataFrame(rows)


# ---------- 5. Compare algorithms ----------
def compare_algorithms(df, k, eps=0.6):
    X = make_features().fit_transform(df)
    algos = {
        "K-Means": KMeans(n_clusters=k, n_init=10, random_state=42),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=k),
        "DBSCAN": DBSCAN(eps=eps, min_samples=10),
    }
    rows = []
    for name, algo in algos.items():
        labels = algo.fit_predict(X)
        found = len(set(labels) - {-1})          # DBSCAN marks noise as -1
        mask = labels != -1
        ok = found >= 2
        rows.append({
            "Algorithm": name,
            "Clusters found": found,
            "Noise points": int((~mask).sum()),
            "Silhouette (higher better)": round(silhouette_score(X[mask], labels[mask]), 3) if ok else None,
            "Davies-Bouldin (lower better)": round(davies_bouldin_score(X[mask], labels[mask]), 3) if ok else None,
            # ARI uses the hidden true category ONLY to grade the result, never to train
            "ARI vs true category": round(adjusted_rand_score(df["true_category"], labels), 3)
                                    if "true_category" in df else None,
        })
    return pd.DataFrame(rows)


# ---------- 6. Interpret ----------
def top_keywords(model, df, labels, n=6):
    tfidf = model.named_steps["features"].named_transformers_["text"].named_steps["tfidf"]
    M, terms = tfidf.transform(df["text"]), tfidf.get_feature_names_out()
    return {c: [terms[i] for i in np.asarray(M[labels == c].mean(axis=0)).ravel().argsort()[::-1][:n]]
            for c in sorted(set(labels))}


def cluster_profiles(model, df):
    labels = model.named_steps["kmeans"].labels_
    kw = top_keywords(model, df, labels)
    prof = df.assign(cluster=labels).groupby("cluster").agg(
        products=("product_id", "count"),
        avg_price=("price", "mean"),
        avg_rating=("rating", "mean"),
        avg_reviews=("num_reviews", "median"),
    ).round(1)
    prof["top_keywords"] = [", ".join(kw[c]) for c in prof.index]
    if "true_category" in df:  # sanity check only
        prof["mostly"] = df.assign(cluster=labels).groupby("cluster")["true_category"].agg(lambda s: s.mode()[0])
    return prof


def pca_2d(model, df):
    X = model.named_steps["features"].transform(df)
    return PCA(n_components=2, random_state=42).fit_transform(X)


# ---------- Run everything ----------
def main(k=6):
    os.makedirs(OUT, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} products\n")

    scores = evaluate_k(df)
    print("Choosing K:\n", scores.round(3).to_string(index=False), "\n")
    best_k = int(scores.loc[scores.silhouette.idxmax(), "k"])
    print(f"Best K by silhouette = {best_k} (using k={k})\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(scores.k, scores.inertia, "o-"); ax[0].set(title="Elbow Method", xlabel="K", ylabel="Inertia (WCSS)")
    ax[1].plot(scores.k, scores.silhouette, "o-", color="green"); ax[1].set(title="Silhouette Score", xlabel="K")
    plt.tight_layout(); plt.savefig(f"{OUT}/choose_k.png", dpi=120); plt.close()

    model = make_model(k).fit(df)
    df["cluster"] = model.named_steps["kmeans"].labels_
    print("Cluster profiles:\n", cluster_profiles(model, df).to_string(), "\n")

    xy = pca_2d(model, df)
    plt.figure(figsize=(7, 6))
    plt.scatter(xy[:, 0], xy[:, 1], c=df.cluster, cmap="tab10", s=12)
    plt.title(f"Product clusters (K-Means, k={k}) - PCA 2D view"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{OUT}/clusters_pca.png", dpi=120); plt.close()

    comparison = compare_algorithms(df, k)
    print("Algorithm comparison:\n", comparison.to_string(index=False), "\n")

    df.drop(columns="text").to_csv(f"{OUT}/products_clustered.csv", index=False)
    print(f"Saved plots + products_clustered.csv in {OUT}/")

    # Self-check: clusters should agree with Amazon's categories far better than random (ARI 0)
    ari = comparison.set_index("Algorithm").loc["K-Means", "ARI vs true category"]
    assert ari > 0.2, f"K-Means ARI too low: {ari}"
    new = add_text(pd.DataFrame([dict(title="Dog chew toy for puppies", description="Durable rubber toy for pets",
                                      price=12.99, rating=4.5, num_reviews=300)]))
    print("New product 'Dog chew toy for puppies' -> cluster", model.predict(new)[0])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
