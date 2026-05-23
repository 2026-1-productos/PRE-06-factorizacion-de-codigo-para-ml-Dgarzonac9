import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(url):
    # descarga de datos
    df = pd.read_csv(url, sep=";")

    # preparacion de datos
    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    return train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )
