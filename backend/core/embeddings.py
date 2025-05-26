# app/core/embeddings.py

import os
import json
import numpy as np

import json
import numpy as np
import os


def load_embeddings(file_path="embed.json"):
    if not os.path.exists(file_path):
        print(f"Embedding file {file_path} not found.")
        return {}

    with open(file_path, "r") as f:
        data = json.load(f)

    known_faces = {k: [np.array(e) for e in v] for k, v in data.items()}
    return known_faces


def save_embeddings(embeddings, file_path="embed.json"):
    """
    Save embeddings to a JSON file.
    """
    json_data = {}

    for name, emb_list in embeddings.items():
        # Convert numpy arrays to lists for JSON serialization
        json_data[name] = [emb.tolist() for emb in emb_list]

    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)
