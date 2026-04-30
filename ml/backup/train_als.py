from ml.als_implicit import ALSConfig, train_and_save


if __name__ == "__main__":
    # You can tune these for speed/quality trade-offs
    config = ALSConfig(
        factors=64,
        iterations=15,
        reg=0.1,
        alpha=40.0,
        seed=42,
    )
    train_and_save(config=config)
    print("ALS implicit model trained and saved.")

