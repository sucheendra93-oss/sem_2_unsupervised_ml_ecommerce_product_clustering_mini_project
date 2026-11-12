"""
Interactive demo for the professor.
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import clustering as C

st.set_page_config(page_title="E-Commerce Product Clustering", layout="wide")
st.title("🛒 E-Commerce Product Clustering")
st.caption("Unsupervised ML: grouping 6,000 real Amazon products by text + price + rating, with no labels")


@st.cache_data
def load():
    return C.load_data()


@st.cache_data
def k_scores(_df):
    return C.evaluate_k(_df)


@st.cache_resource
def fit(k, _df):
    model = C.make_model(k).fit(_df)
    return model, C.pca_2d(model, _df)


df = load()
k = st.sidebar.slider("Number of clusters (K)", 2, 10, 6)
st.sidebar.markdown("**Pipeline**\n1. TF-IDF on title + description\n2. SVD → 50 topics\n"
                    "3. log + scale price, reviews, rating\n4. K-Means\n5. PCA → 2-D plot")
model, xy = fit(k, df)
labels = model.named_steps["kmeans"].labels_
view = df.assign(cluster=labels.astype(str), PC1=xy[:, 0], PC2=xy[:, 1])

t1, t2, t3, t4, t5 = st.tabs(["1️⃣ Data", "2️⃣ Choose K", "3️⃣ Clusters", "4️⃣ Compare algorithms", "5️⃣ Try a product"])

with t1:
    st.subheader("Dataset: Amazon Reviews 2023 product metadata (McAuley Lab, UCSD)")
    a, b, c = st.columns(3)
    a.metric("Products", len(df)); b.metric("Median price", f"${df.price.median():.2f}"); c.metric("Avg rating", f"{df.rating.mean():.2f}")
    st.dataframe(df.drop(columns="text").head(50), width="stretch")
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(df, x="price", nbins=60, title="Price is skewed → we use log(price)"), width="stretch")
    r.plotly_chart(px.histogram(df, x="true_category", title="Amazon category (hidden from model, used only to check)"),
                   width="stretch")

with t2:
    st.subheader("How many clusters? Elbow method + Silhouette score")
    s = k_scores(df)
    l, r = st.columns(2)
    l.plotly_chart(px.line(s, x="k", y="inertia", markers=True, title="Elbow: look for the bend"), width="stretch")
    r.plotly_chart(px.line(s, x="k", y="silhouette", markers=True, title="Silhouette: higher = better separated"),
                   width="stretch")
    st.info("Inertia always falls as K grows, so we look for where it stops falling fast. "
            "Silhouette is low for text data (high-dimensional, overlapping products); K=6 is a good trade-off "
            "between score and clusters a business can actually use.")

with t3:
    st.subheader(f"K-Means with K = {k}")
    st.plotly_chart(px.scatter(view, x="PC1", y="PC2", color="cluster", hover_data=["title", "price", "true_category"],
                               title="Each dot = a product (PCA squeezes 53 features into 2 for viewing)", height=550),
                    width="stretch")
    st.markdown("**Cluster profiles** (what is each group?)")
    st.dataframe(C.cluster_profiles(model, df), width="stretch")
    st.plotly_chart(px.imshow(pd.crosstab(view.cluster, view.true_category), text_auto=True, aspect="auto",
                              title="Clusters vs Amazon's real categories (check only)"), width="stretch")
    c = st.selectbox("Show products in cluster", sorted(set(labels)))
    st.dataframe(view[labels == c][["title", "price", "rating", "true_category"]].head(20), width="stretch")

with t4:
    st.subheader("K-Means vs Agglomerative vs DBSCAN")
    eps = st.slider("DBSCAN eps (neighbourhood radius)", 0.3, 1.2, 0.6, 0.05)
    st.dataframe(C.compare_algorithms(df, k, eps), width="stretch")
    st.markdown("- **Silhouette**: -1 to 1, higher = tighter, better separated clusters\n"
                "- **Davies-Bouldin**: lower = better\n"
                "- **ARI**: agreement with Amazon's categories (0 = random, 1 = perfect)\n"
                "- DBSCAN's high silhouette is misleading: it throws most products away as *noise* (-1), "
                "because text data has no clear dense regions.")

with t5:
    st.subheader("Which cluster does a new product belong to?")
    with st.form("new"):
        title = st.text_input("Title", "Stainless steel dog food bowl, non-slip")
        desc = st.text_area("Description", "Durable bowl for small and medium dogs and cats")
        a, b, c = st.columns(3)
        price = a.number_input("Price ($)", 0.5, 2000.0, 14.99)
        rating = b.slider("Rating", 1.0, 5.0, 4.5)
        reviews = c.number_input("Number of reviews", 0, 100000, 120)
        go = st.form_submit_button("Predict cluster")
    if go:
        new = C.add_text(pd.DataFrame([dict(title=title, description=desc, price=price, rating=rating, num_reviews=reviews)]))
        cl = int(model.predict(new)[0])
        st.success(f"→ Cluster **{cl}** · keywords: {C.cluster_profiles(model, df).loc[cl, 'top_keywords']}")
        feats = model.named_steps["features"]
        sims = cosine_similarity(feats.transform(new), feats.transform(df))[0]
        st.markdown("**Most similar products** (useful for *'customers also viewed'*)")
        st.dataframe(df.assign(similarity=sims.round(3)).nlargest(5, "similarity")[["title", "price", "similarity"]],
                     width="stretch")
