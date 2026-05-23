import os
import pickle


def save_model(model, save_path="model.pkl"):
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
