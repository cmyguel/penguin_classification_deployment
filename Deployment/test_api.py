from fastapi.testclient import TestClient
import pandas as pd
from io import BytesIO
from sklearn.metrics import accuracy_score
import pytest


from main import app

home = TestClient(app)


def test_get_func():
    r = home.get("/?cl=1.0&cd=1.0&fl=1.0")
    assert r.status_code == 200
    assert 'features' in r.json()
    assert 'prediction' in r.json()

def test_post_json():
    json = {
    "culmen_length": 1.0,
    "culmen_depth": 1.0,
    "flipper_length": 2.0,
    }
    r = home.post("/json", json=json)
    print(r.json())
    assert r.status_code == 200
    assert 'features' in r.json()
    assert 'prediction' in r.json()

def test_post_file():
    file = open("../ML Model/test sample.csv", "rb")
    files = {"file": file}
    r = home.post("/file", files = files)
    df_pred = pd.read_csv(BytesIO(r.content), header=None)
    assert r.status_code == 200
    assert type(r.content) == bytes

    df_test = pd.read_csv("../ML Model/test sample 2.csv")
    assert df_pred.shape == df_test.shape

    print(df_pred)


def test_post_file_acc():
    file = open("../ML Model/test sample.csv", "rb")
    files = {"file": file}
    r = home.post("/file", files = files)
    df_pred = pd.read_csv(BytesIO(r.content), header=None)
    df_test = pd.read_csv("../ML Model/test sample 2.csv")

    acc = accuracy_score(df_test.species, df_pred[3] )
    assert acc >= 0.95

def test_post_file_acc_stratified():
    file = open("../ML Model/test sample.csv", "rb")
    files = {"file": file}
    r = home.post("/file", files = files)
    df_pred = pd.read_csv(BytesIO(r.content), header=None)
    df_test = pd.read_csv("../ML Model/test sample 2.csv")

    for specie in set( df_test.species ):
        idx = (df_test.species == specie)

        df_pred_chunk = df_pred[idx]
        df_test_chunk = df_test[idx]

        acc = accuracy_score(df_test_chunk.species,  df_pred_chunk[3])

        print(f"accuracy {specie}: {acc}")
        assert acc >= 0.90





