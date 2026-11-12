# How to Set Up and Run: E-Commerce Product Clustering

This guide takes you from a fresh computer to a running demo. Follow the steps in order.
Tested on macOS with Python 3.10; works the same on Windows and Linux.

---

## 1. What you need

| Requirement | Version | Check with |
|---|---|---|
| Python | 3.9 or newer | `python3 --version` (Windows: `python --version`) |
| pip | any recent | `python3 -m pip --version` |
| Internet | only for the **one-time** data download | |
| Disk space | ~10 MB for the project + ~300 MB for Python libraries | |
| VS Code or Jupyter | only to open the notebook | |

No GPU, no API keys, no Kaggle account needed.

---

## 2. Project structure

```
Ecommerce_Product_Clustering/
├── download_data.py          # Step 1: downloads real Amazon data -> products.csv
├── clustering.py             # Step 2: full ML pipeline, prints results, saves plots
├── Product_Clustering.ipynb  # Same pipeline step by step with explanations (for learning)
├── app.py                    # Step 3: interactive Streamlit demo
├── requirements.txt          # Python libraries
├── products.csv              # The dataset (6,000 products), created by Step 1
├── outputs/                  # Created by Step 2
│   ├── choose_k.png          #   elbow + silhouette charts
│   ├── clusters_pca.png      #   2-D cluster plot
│   └── products_clustered.csv#   every product with its cluster number
├── README.md                 # Project explanation, results, viva questions
└── HOW_TO_RUN.md             # This file
```

---

## 3. One-time setup

### 3.1 Open a terminal in the project folder

```bash
cd ~/Desktop/Personal/Unsupervised_Learning/Ecommerce_Product_Clustering
```
(Windows: `cd %USERPROFILE%\Desktop\Personal\Unsupervised_Learning\Ecommerce_Product_Clustering`)

### 3.2 (Recommended) Create a virtual environment

A virtual environment keeps this project's libraries separate from the rest of your computer.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Run the activate command again every time you open a new terminal.

> Skipping the virtual environment is fine if your system Python already has the libraries.

### 3.3 Install the libraries

```bash
pip install -r requirements.txt
```

| Library | Used for |
|---|---|
| pandas, numpy | loading and handling the table of products |
| scikit-learn | TF-IDF, SVD, scaling, K-Means, Agglomerative, DBSCAN, PCA, metrics |
| matplotlib | plots saved to `outputs/` and in the notebook |
| plotly, streamlit | interactive demo app |
| pyarrow, huggingface_hub | streaming the dataset from Hugging Face |

For the notebook, also install and register the Jupyter kernel, so VS Code/Jupyter use **this** environment:
```bash
pip install ipykernel notebook
python -m ipykernel install --user --name ecommerce-clustering --display-name "Python (Ecommerce Clustering)"
```
The notebook is already set to use the kernel named **Python (Ecommerce Clustering)**.

### 3.4 Check the setup

```bash
python3 -c "import sklearn, pandas, streamlit, plotly, pyarrow, huggingface_hub; print('Setup OK')"
```

---

## 4. Running the project

### Step 1: Download the dataset (once, needs internet)

```bash
python3 download_data.py
```

- Takes **2–5 minutes**. It streams only 6 needed columns from the 1.6 GB source, not the whole thing.
- If you see `HTTP Error 429 ... Rate limited. Waiting ...`, that is normal: it waits and retries by itself.
- Expected end of output:
  ```
  Saved products.csv with 6000 products
  ```
- After this, **everything works offline**. `products.csv` is already in the folder, so you only need this step to re-create it.

**Source:** Amazon Reviews 2023 product metadata, McAuley Lab (UC San Diego),
https://huggingface.co/datasets/milistu/AMAZON-Products-2023

### Step 2: Run the full pipeline

```bash
python3 clustering.py
```

Takes ~10 seconds. It prints, in order:
1. Elbow + silhouette scores for K = 2…10
2. Profile of each of the 6 clusters (size, average price/rating, top keywords)
3. K-Means vs Agglomerative vs DBSCAN comparison table
4. Predicted cluster for a sample new product

It saves `outputs/choose_k.png`, `outputs/clusters_pca.png` and `outputs/products_clustered.csv`.
It ends with a built-in self-check: if the clustering quality drops below a minimum, it stops with an `AssertionError`.

### Step 3: Open the notebook (for learning / explaining the code)

**VS Code:** open `Product_Clustering.ipynb` → *Select Kernel* (top right) → choose your Python / `.venv` → **Run All**.

**Jupyter:**
```bash
jupyter notebook Product_Clustering.ipynb
```
Then *Kernel → Restart & Run All*. It takes ~30 seconds. Outputs are already saved inside, so you can show it without running.

### Step 4: Launch the demo app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501** (first load takes ~10 seconds). Stop it with `Ctrl + C` in the terminal.

| Tab | What to show |
|---|---|
| 1 · Data | real products, price distribution, Amazon categories |
| 2 · Choose K | elbow + silhouette charts |
| 3 · Clusters | interactive 2-D plot (hover = product name), cluster profiles, heatmap vs Amazon categories, product list per cluster. **Move the K slider in the sidebar** to re-cluster live |
| 4 · Compare algorithms | metrics table; move the DBSCAN `eps` slider to see noise points change |
| 5 · Try a product | type any product → predicted cluster + 5 most similar products |

---

## 5. Changing settings

| What | Where | Default |
|---|---|---|
| Products per category | `PER_CATEGORY` in `download_data.py` (re-run Step 1) | 1000 |
| Which Amazon categories | `CATEGORIES` in `download_data.py` | 6 categories |
| Number of clusters K | `main(k=6)` in `clustering.py`; slider in the app | 6 |
| Text vs number importance | `TEXT_WEIGHT, NUM_WEIGHT` in `clustering.py` | 1.0, 0.1 |
| Number of text "topics" | `TruncatedSVD(n_components=50)` in `clustering.py` | 50 |

After changing `clustering.py`, restart the Streamlit app (`Ctrl + C`, then run it again).

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named ...` | Activate the venv (3.2), then `pip install -r requirements.txt` |
| `products.csv not found - run: python download_data.py` | Run Step 1 (needs internet once) |
| `HTTP Error 429 ... Rate limited` | Nothing to do: it waits and retries. If it fails after 5 retries, wait 5 minutes and run again |
| `streamlit: command not found` | Use `python3 -m streamlit run app.py` |
| `Port 8501 is already in use` | Another app is still running: close it, or `streamlit run app.py --server.port 8502` |
| Notebook: `ModuleNotFoundError: No module named 'matplotlib'` (or any library) | The notebook is using a different Python. Click the kernel name (top right) → *Select Another Kernel* → **Python (Ecommerce Clustering)** or `.venv/bin/python`, then **Restart** and **Run All** |
| Notebook says "select a kernel" / kernel won't start | `pip install ipykernel`, then pick the `.venv` Python in VS Code |
| `python3` not found (Windows) | Use `python` instead of `python3` |
| Results differ slightly after re-downloading | Normal if the source changes; the random seed (42) keeps runs identical on the same `products.csv` |

---

## 7. Demo-day checklist

- [ ] `products.csv` exists in the folder (so no internet needed)
- [ ] `python3 clustering.py` runs without errors
- [ ] `streamlit run app.py` opened **before** the presentation (first load is the slowest)
- [ ] Notebook open in VS Code as a backup (outputs already saved)
- [ ] `outputs/*.png` added to the slides
- [ ] Example product ready for tab 5, e.g. *"Stainless steel dog food bowl"*, and let the professor type their own
