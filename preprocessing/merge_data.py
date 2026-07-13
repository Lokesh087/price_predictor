"""
==========================================================
Merge Olist datasets into a single dataset
==========================================================
"""

from pathlib import Path
import pandas as pd


# ======================================================
# Project Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ORDERS_FILE = RAW_DIR / "olist_orders_dataset.csv"
ORDER_ITEMS_FILE = RAW_DIR / "olist_order_items_dataset.csv"
PRODUCTS_FILE = RAW_DIR / "olist_products_dataset.csv"

OUTPUT_FILE = PROCESSED_DIR / "merged_dataset.csv"


# ======================================================
# Check File Exists
# ======================================================

def check_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"\nFile Not Found:\n{file_path}")


# ======================================================
# Load Dataset
# ======================================================

def load_data():

    print("\nLoading datasets...\n")

    check_file(ORDERS_FILE)
    check_file(ORDER_ITEMS_FILE)
    check_file(PRODUCTS_FILE)

    orders = pd.read_csv(
        ORDERS_FILE,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )
    order_items = pd.read_csv(
        ORDER_ITEMS_FILE,
        parse_dates=[
            "shipping_limit_date"
        ]
    )
    products = pd.read_csv(PRODUCTS_FILE)
    print("Datasets Loaded Successfully")
    return orders, order_items, products


# ======================================================
# Merge Dataset
# ======================================================

def merge_dataset():

    orders, order_items, products = load_data()

    print("\nOrders Shape :", orders.shape)
    print("Order Items Shape :", order_items.shape)
    print("Products Shape :", products.shape)

    print("\nMerging Orders + Order Items...")

    merged = pd.merge(
        orders,
        order_items,
        on="order_id",
        how="inner"
    )
    print("Shape :", merged.shape)
    print("\nMerging Products...")
    merged = pd.merge(
        merged,
        products,
        on="product_id",
        how="left"
    )
    print("Shape :", merged.shape)
    merged.drop_duplicates(inplace=True)
    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )
    print("\nMerged Dataset Saved Successfully")
    print(OUTPUT_FILE)
    print("\nColumns")
    print(merged.columns.tolist())


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    try:
        merge_dataset()
        print("\nSUCCESS")
    except Exception as e:
        print("\nERROR")
        print(e)