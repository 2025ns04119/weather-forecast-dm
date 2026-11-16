"""
Feature store demo: versioned retrieval for training and for inference.

    python -m feature_store.demo_retrieval
"""
import pandas as pd

from feature_store.store import FeatureStore, feature_names, load_registry

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 12)


def main():
    reg = load_registry()
    print("=" * 70)
    print("1. Registry: feature versions")
    for v, text in reg["versions"].items():
        print(f"   {v}: {len(feature_names(v, reg)):>2} features - {text}")

    fs = FeatureStore()
    print("\n2. Offline store check (registry vs code, rows per version)")
    fs.check()

    print("\n3. TRAINING retrieval - version v3, two cities, last 3 days")
    end = pd.Timestamp.now(tz="UTC").floor("h")
    train = fs.get_training_features("v3", ["Bengaluru", "London"], start=end - pd.Timedelta(days=3))
    print(train.groupby("location").size().rename("rows").to_string())
    print(train.head(3).to_string(index=False))

    print("\n4. TRAINING retrieval - older version v2 (same API, different feature set)")
    v2 = fs.get_training_features("v2", ["Bengaluru"])
    print(f"   rows={len(v2)}  columns={list(v2.columns)}")

    print("\n5. ONLINE retrieval - latest 24 hours of Bengaluru, version v3 (used by inference)")
    online = fs.get_online_features("Bengaluru", "v3", n_hours=24)
    print(online.tail(3).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
