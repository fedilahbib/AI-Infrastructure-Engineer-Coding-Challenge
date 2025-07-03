def normalize_vector(vec):
    norm = (vec**2).sum() ** 0.5
    return vec / norm if norm != 0 else vec