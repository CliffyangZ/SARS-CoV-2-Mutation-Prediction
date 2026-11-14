"""小型資料下 Jaccard UMAP 可執行並輸出圖檔。"""
import numpy as np
from Research.StyleGAN.FE import umap_latent as u


def test_umap(tmp_path):
    rng = np.random.default_rng(0)
    Xd = (rng.random((150, 64)) < 0.1).astype(np.uint8); Xg = (rng.random((50, 64)) < 0.1).astype(np.uint8)
    Z = u.embed(Xd, Xg)
    assert Z.shape == (200, 2)
    u.plot_umap(Z, 150, np.array(["a"] * 75 + ["b"] * 75), np.array(["train"] * 100 + ["test"] * 50), tmp_path / "u.png")
    assert (tmp_path / "u.png").exists()
