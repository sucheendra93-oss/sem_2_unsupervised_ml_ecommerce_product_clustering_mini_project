# E-Commerce Product Clustering (Unsupervised ML)

Group 6,000 **real Amazon products** into meaningful clusters **without labels**, using their
title, description, price, rating and review count.

---

## 1. Problem statement

Online stores have millions of products. Categories are often missing, wrong, or too broad.
**Goal:** automatically find groups of similar products so the store can
- organise the catalogue and fix wrong categories
- recommend similar products ("customers also viewed")
- spot hidden segments (we found a separate **lighting** group hidden inside Amazon's broad *Tools & Home Improvement* category)

Why *unsupervised*? We don't give the model any category labels. It finds structure by itself.

---

## 2. Dataset

| | |
|---|---|
| Source | Amazon Reviews 2023, product metadata (McAuley Lab, UC San Diego), via Hugging Face `milistu/AMAZON-Products-2023` |
| Size used | 6,000 products (1,000 from each of 6 Amazon categories, balanced; the smallest category, Pet Supplies, has 1,338) |
| Columns | `title`, `description`, `price`, `rating`, `num_reviews`, `true_category` |
| Free? | Yes, no login. `download_data.py` streams only the 6 columns we need |

`true_category` is **never given to the model**. We only use it at the end to check
whether clusters make sense (like an answer key).

Cleaning done: removed missing price/rating, prices ≤ 0 or ≥ $2000, HTML tags, duplicates, very short titles.

---

## 3. How to run

```bash
pip install -r requirements.txt
python download_data.py      # once, ~2-5 min, needs internet -> products.csv
python clustering.py         # full pipeline, prints results, saves plots in outputs/
streamlit run app.py         # interactive demo in the browser (works offline)
```

---

## 4. Method (the pipeline)

```
products.csv
   │
   ├─ text = title + description ──► TF-IDF ──► SVD (50 topics) ──► normalise ─┐
   ├─ price, num_reviews ──► log ──► StandardScaler ──────────────────────(×0.1)─┼─► 53 features ─► K-Means ─► cluster
   └─ rating ──► StandardScaler ──────────────────────────────────────────(×0.1)─┘                      │
                                                                                        PCA ─► 2-D plot ◄┘
```

| Step | What it does | Why |
|---|---|---|
| **TF-IDF** | Turns text into numbers: a word scores high if frequent in *this* product but rare overall | "dog" says a lot, "the" says nothing |
| `sublinear_tf`, `max_df=0.5` | Repeated words count less; words in >50% of products dropped | stops long descriptions / common words dominating |
| **Truncated SVD (LSA)** | Compresses ~thousands of word columns into 50 "topics" | removes noise, groups synonyms, faster |
| **log(price)** | Prices are skewed ($5 … $1999) | stops a few expensive items dominating distances |
| **StandardScaler** | Mean 0, std 1 | puts all numbers on the same scale (K-Means uses distances!) |
| **Weights 1.0 / 0.1** | Text decides *what* a product is, numbers just nudge | with weight 0.25 the model made a "badly rated products" cluster instead of product types |
| **K-Means** | Picks K centres, assigns every product to the nearest, moves centres to the mean, repeats | simple, fast, scales well |
| **PCA** | Squeezes 53 features to 2, only for drawing | humans can't see 53-D |

Everything is one scikit-learn `Pipeline`, so `model.predict(new_product)` works for products never seen before.

---

## 5. Choosing K

- **Elbow method:** plot inertia (within-cluster sum of squares) vs K; pick where the curve bends.
- **Silhouette score:** −1 … 1, how much closer a point is to its own cluster than the next one. Higher is better.

On our data the silhouette rises slowly (0.06 → 0.10) with no sharp peak. That is normal for text data
(very high-dimensional, products overlap). We chose **K = 6** because the elbow flattens around there and the
6 clusters are easy for a business to interpret.

---

## 6. Results (K = 6)

| Cluster | Products | Avg price | Top keywords | Interpretation |
|---|---|---|---|---|
| 0 | 832 | $27.2 | dog, dogs, cat, pet, cats, toy | **Pet supplies** |
| 4 | 876 | $27.1 | case, phone, galaxy, samsung, cover | **Phone accessories** |
| 1 | 932 | $43.9 | women, men, sleeve, shirt, casual | **Clothing** |
| 2 | 257 | $40.1 | light, led, lamp, solar, lighting | **Lighting** (new segment, 85% from *Tools & Home Improvement*) |
| 3 | 1898 | $59.9 | steel, water, bag, stainless | **Hard goods: bottles, bags, tools, outdoor gear** (priciest) |
| 5 | 1205 | $41.7 | storage, home, set, room, gift | **Home & storage / gifts** |

The same 6 themes appeared with 1,800 products and with 6,000, so the clusters are **stable**, not luck.

### Algorithm comparison

| Algorithm | Clusters | Noise | Silhouette ↑ | Davies-Bouldin ↓ | ARI vs Amazon category ↑ |
|---|---|---|---|---|---|
| **K-Means** | 6 | 0 | **0.080** | **3.51** | **0.412** |
| Agglomerative (Ward) | 6 | 0 | 0.067 | 3.56 | 0.331 |
| DBSCAN (eps 0.6) | 35 | 2464 | 0.114* | 1.56* | 0.186 |

\* DBSCAN looks good on silhouette only because it labelled 41% of products as **noise**, scored only the
leftovers, and split them into 35 tiny clusters. Text data has no clear dense regions, so DBSCAN is a poor fit here. **K-Means wins** on every
metric among methods that keep all products.

**ARI (Adjusted Rand Index):** 0 = random, 1 = perfect match with Amazon's categories. 0.41 is clearly better than random.
It's not higher because Amazon's own categories overlap (a water bottle can be in *Sports*, *Home* or *Tools*),
and clustering found groups that make sense but differ from Amazon's (e.g. a separate lighting group).

---

## 7. Limitations & future work

- TF-IDF only matches exact words ("sofa" ≠ "couch"). **Sentence embeddings** (e.g. BERT / sentence-transformers) would capture meaning.
- One broad "home / general" cluster: try more K, or cluster that group again (hierarchical drill-down).
- Use product images (CNN features) together with text.
- K-Means assumes round, similar-size clusters; Gaussian Mixture Models allow soft membership.

---

## 8. Files

| File | Purpose |
|---|---|
| `download_data.py` | Downloads + cleans the real dataset → `products.csv` |
| `clustering.py` | The whole ML pipeline + plots + a self-check (`assert` ARI > 0.2) |
| `Product_Clustering.ipynb` | Same pipeline step by step with explanations, outputs saved (best for learning / slides) |
| `app.py` | Streamlit demo with 5 tabs |
| `outputs/` | `choose_k.png`, `clusters_pca.png`, `products_clustered.csv` |
| `HOW_TO_RUN.md` | Full setup, run steps, settings, troubleshooting, demo-day checklist |
| `submission/` | `Product_Clustering_Report.docx` (report) and `Product_Clustering_Presentation.pptx` (15 slides with speaker notes) |
| `.venv/` | Local Python environment. Don't include it when zipping the project for submission |

---

## 9. Presentation outline (10 slides)

1. **Title** — E-Commerce Product Clustering using Unsupervised ML
2. **Problem** — millions of products, messy categories; need automatic grouping
3. **Why unsupervised** — no labels needed; supervised vs unsupervised in one line each
4. **Dataset** — Amazon 2023, 6,000 products, columns, cleaning steps
5. **Feature engineering** — TF-IDF → SVD, log + scaling, weights (use the pipeline diagram)
6. **Choosing K** — `outputs/choose_k.png`, elbow + silhouette
7. **Clusters** — `outputs/clusters_pca.png` + results table with names
8. **Algorithm comparison** — K-Means vs Agglomerative vs DBSCAN table, why DBSCAN fails
9. **Business use** — catalogue cleanup, recommendations, new segment discovery → **live demo**
10. **Limitations & future work** + Thank you

## 10. Demo script (≈5 min)

1. `streamlit run app.py`
2. **Data tab:** "Real Amazon data. Price is skewed, so we take log."
3. **Choose K tab:** "Elbow bends near 6; silhouette is low because text data overlaps, that's expected."
4. **Clusters tab:** hover dots, show profiles table, then the heatmap: "Pet cluster ≈ Pet Supplies, phone cluster ≈ Cell Phones, without ever seeing labels." Move the K slider to show it live.
5. **Compare tab:** move the DBSCAN eps slider: "See how noise points explode, that's why we use K-Means."
6. **Try a product tab:** type *"Stainless steel dog food bowl"* → pet cluster + 5 similar products. Then let the professor type one.

---

## 11. Likely viva questions

**Q: What is unsupervised learning?**
Learning patterns from data without labels. Clustering = grouping similar items.

**Q: How does K-Means work?**
1) pick K random centres 2) assign each point to its nearest centre 3) move each centre to the mean of its points 4) repeat until nothing changes. It minimises inertia (sum of squared distances to centres). `n_init=10` runs it 10 times with different starts and keeps the best.

**Q: Why scale features?**
K-Means uses distance. Unscaled, price (up to 2000) would completely outweigh rating (1–5).

**Q: What is TF-IDF?**
Term Frequency × Inverse Document Frequency. Word importance = how often it appears in this product × how rare it is across all products.

**Q: Why SVD and not PCA on text?**
TF-IDF is a huge sparse matrix. TruncatedSVD works directly on sparse data; PCA needs to centre (densify) it. SVD on TF-IDF is called **LSA** (Latent Semantic Analysis).

**Q: Why is the silhouette score so low?**
Text data is high-dimensional and products genuinely overlap. Low silhouette ≠ useless clusters; the keywords and the category heatmap show they're meaningful.

**Q: What is ARI and isn't using labels cheating?**
Adjusted Rand Index measures agreement between two groupings (chance-corrected). Labels are used **only after** clustering, for evaluation, never for training.

**Q: Why not DBSCAN?**
DBSCAN needs dense regions. In 53-D text space distances are similar everywhere, so most points become noise. It's great for spatial data / outlier detection.

**Q: K-Means vs Hierarchical?**
Hierarchical builds a tree (dendrogram), no need to choose K upfront, but O(n²) memory; slower on big data. K-Means is fast but needs K and assumes roundish clusters.

**Q: How would this work for a new product?**
The whole thing is one pipeline: `model.predict(new_product)` applies the same TF-IDF/scaling and assigns the nearest centre (shown in the demo's last tab).

**Q: How to improve?**
Sentence embeddings (BERT), image features, GMM for soft clusters, re-cluster the mixed group.
