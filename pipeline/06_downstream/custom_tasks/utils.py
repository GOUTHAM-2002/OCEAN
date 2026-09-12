def process_docs(dataset):
    # Deterministic fixed subset: first 1000 stories (dataset order is fixed).
    n = min(1000, len(dataset))
    return dataset.select(range(n))
