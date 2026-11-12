"""
Step 0: Download a REAL e-commerce product dataset (free, no login needed).

Source: Amazon Reviews 2023 product metadata (McAuley Lab, UC San Diego),
        mirrored on Hugging Face as "milistu/AMAZON-Products-2023".
        https://huggingface.co/datasets/milistu/AMAZON-Products-2023

The full dataset is ~1.6 GB because it contains images + embeddings.
We stream ONLY the 7 columns we need (~ a few MB), so the download is quick.

Output: products.csv  (run once; after that everything works offline)

`true_category` = Amazon's own category. The clustering model NEVER sees it,
we only use it at the end to check whether our clusters make sense.
"""
import re

import pandas as pd

REPO = "hf://datasets/milistu/AMAZON-Products-2023@~parquet/default/train/{:04d}.parquet"
COLS = ["title", "description", "main_category", "average_rating", "rating_number", "price"]
CATEGORIES = ["AMAZON FASHION", "Amazon Home", "Tools & Home Improvement",
              "Cell Phones & Accessories", "Pet Supplies", "Sports & Outdoors"]
PER_CATEGORY = 1000


def clean_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if not isinstance(x, str):            # description is stored as a list of sentences
        x = " ".join(map(str, x))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x)).strip()[:500]  # strip HTML, cap length


def download():
    parts = []
    for i in range(4):
        print(f"Downloading shard {i + 1}/4 ...")
        parts.append(pd.read_parquet(REPO.format(i), columns=COLS))
    df = pd.concat(parts, ignore_index=True)
    print(f"Raw rows: {len(df)}")

    # Cleaning
    df = df[df.main_category.isin(CATEGORIES)]
    df = df.dropna(subset=["title", "price", "average_rating"])
    df = df[(df.price > 0) & (df.price < 2000)]                  # drop junk / extreme prices
    df["title"] = df.title.map(clean_text)
    df["description"] = df.description.map(clean_text)
    df = df[df.title.str.len() > 10].drop_duplicates("title")

    # Balanced sample so no category dominates
    print("Clean products available per category:\n", df.main_category.value_counts(), "\n")
    df = df.sample(frac=1, random_state=42)                         # shuffle
    df = df.groupby("main_category").head(PER_CATEGORY).reset_index(drop=True)  # up to N per category

    out = pd.DataFrame({
        "product_id": range(1, len(df) + 1),
        "title": df.title,
        "description": df.description,
        "price": df.price.round(2),
        "rating": df.average_rating,
        "num_reviews": df.rating_number.fillna(0).astype(int),
        "true_category": df.main_category,
    })
    return out


if __name__ == "__main__":
    df = download()
    df.to_csv("products.csv", index=False)
    print(f"\nSaved products.csv with {len(df)} products")
    print(df.true_category.value_counts())
