"""Demo 測試 fixture：以小子集快速建出測試用模型包，並從資料集取各類序列。"""
import pandas as pd
import pytest

from Demo import bundle as B
from Demo.predict import Predictor
from Research.QuSpike_PQK import features as F


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo_bundle")
    return B.build_bundle(seed=0, out=d / "b", cache=d / "cache", limit=4000, log=lambda *_: None)


@pytest.fixture(scope="session")
def predictor(bundle):
    return Predictor(bundle)


@pytest.fixture(scope="session")
def samples():
    df = pd.read_csv(F.DEFAULT_DATA, sep="\t", dtype={"date": str})
    df = df[df.split == "test"].sample(60, random_state=1).reset_index(drop=True)
    return df  # 欄位 lineage, sequence


@pytest.fixture(scope="session")
def seqs(samples):
    return samples.sequence.tolist()
