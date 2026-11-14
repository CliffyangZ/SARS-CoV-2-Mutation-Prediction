"""時間切分邊界。"""
from Research.StyleGAN.FE.time_split import split_labels


def test_boundaries():
    s = split_labels(["2021-02-28", "2021-03-01", "2021-03-31", "2021-04-01", "2019-12-30"])
    assert list(s) == ["train", "gap", "gap", "test", "train"]
