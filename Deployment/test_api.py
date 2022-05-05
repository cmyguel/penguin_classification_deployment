from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse
import pytest
from sklearn.metrics import accuracy_score
import pandas as pd
from io import BytesIO

from main import app


home = TestClient(app)

# Load data test data
@pytest.fixture(scope="session")
def df_true():
    df = pd.read_csv("../ML Model/test sample 2.csv")
    return df

def test_get_func_response():
    r = home.get("/?cl=1.0&cd=2.0&fl=3.0")
    assert r.status_code == 200
    assert 'features' in r.json()
    assert 'prediction' in r.json()

def test_post_json_response():
    json = {
    "culmen_length": 1.0,
    "culmen_depth": 2.0,
    "flipper_length": 2.0,
    }
    r = home.post("/json", json=json)
    assert r.status_code == 200
    assert 'features' in r.json()
    assert 'prediction' in r.json()

def test_post_file_response(df_true):
    files = {'file': open("../ML Model/test sample.csv", "rb")}
    r = home.post("/file", files=files)

    assert r.status_code == 200
    assert isinstance(r.content, bytes)

    df_pred = pd.read_csv(BytesIO(r.content), header=None)
    assert df_true.shape == df_pred.shape


def test_post_file_acc(df_true):
    files = {'file': open("../ML Model/test sample.csv", "rb")}
    r = home.post("/file", files=files)
    df_pred = pd.read_csv(BytesIO(r.content), header=None)

    acc = accuracy_score(df_true['species'], df_pred[3])
    print("general accuracy:", acc)
    assert acc >=0.95

def test_post_file_stratified_acc(df_true):
    files = {'file': open("../ML Model/test sample.csv", "rb")}
    r = home.post("/file", files=files)
    df_pred = pd.read_csv(BytesIO(r.content), header=None)

    for specie in set(df_true.species):
        index = df_true.species == specie

        df_true_chunk = df_true[index]
        df_pred_chunk = df_pred[index]

        acc = accuracy_score(df_true_chunk.species, df_pred_chunk[3])
        print(f"stratified accuracy ({specie}):", acc)
        assert acc >=0.90