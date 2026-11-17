"""有限取樣與去極化噪聲的掉點曲線（T033–T036）。

在本階段不送機的決定下，這是**唯一**能回答「硬體上還成立嗎」的證據來源。

兩個實作約束（研究 D5）：
- 有限取樣必須以 `qml.set_shots` 轉換施加（device 層級設 shots 在 0.45.1 已棄用）；
  這點由 pqk.features 負責，本模組只傳參數。
- 密度矩陣模擬慢 33–70 倍（實測 46.6 ms/樣本），故只在**分層子集**上執行。
"""
import numpy as np

from . import criteria, features as F, pipeline


def stratified_subset(y, size, seed=0):
    """分層抽樣：每類別按比例取樣，至少 1 筆。回傳索引。"""
    y = np.asarray(y)
    rng = np.random.default_rng(int(seed))
    classes, counts = np.unique(y, return_counts=True)
    size = min(int(size), len(y))
    take = np.maximum((counts / counts.sum() * size).round().astype(int), 1)
    out = []
    for c, k in zip(classes, take):
        idx = np.where(y == c)[0]
        out.append(rng.permutation(idx)[:min(k, len(idx))])
    return np.sort(np.concatenate(out))


def subset_featureset(fs, size, seed=0):
    """把 FeatureSet 縮成分層子集（train 與 test 各自分層）。"""
    tr = stratified_subset(fs.y_train, size, seed)
    te = stratified_subset(fs.y_test, max(size // 2, 1), seed + 1)
    return F.FeatureSet(**{**fs.__dict__, "x_train": fs.x_train[tr], "y_train": fs.y_train[tr],
                           "x_test": fs.x_test[te], "y_test": fs.y_test[te],
                           "month_train": fs.month_train[tr] if fs.month_train is not None else None})


def scan(fs, groups, cfg, cache_dir, shots_list=(None, 8192, 1024, 256),
         noise_list=(0.0, 0.005, 0.02), controls=("MAIN", "C3", "C0"), seeds=(0,),
         subset=2000, tune=False):
    """掉點掃描。回傳 (每個設定點的紀錄清單, 基準判準)。"""
    from Research.lineage_baselines.metrics import compute_metrics

    sub = subset_featureset(fs, subset)
    sub_groups = F.make_groups(sub)
    rows, by_point = [], {}
    for shots in shots_list:
        for p in noise_list:
            c = dict(cfg, shots=shots, noise_p=p)
            scores = {}
            for ctrl in controls:
                vals = []
                for seed in seeds:
                    pred, proba, _ = pipeline.evaluate_control(ctrl, sub, sub_groups, c, seed,
                                                               cache_dir, tune=tune)
                    vals.append(compute_metrics(sub.y_test, pred, proba, len(sub.labels))["f1_macro"])
                scores[ctrl] = vals
            recs = criteria.evaluate(scores)
            key = (shots, p)
            by_point[key] = recs
            for ctrl, vals in scores.items():
                m, s = criteria.mean_std(vals)
                rows.append(dict(shots=shots, noise_p=p, control=ctrl, f1_macro=m, std=s,
                                 subset_size=int(len(sub.y_test))))
    return rows, by_point


def drop_curve(rows, baseline=(None, 0.0)):
    """相對無噪聲基準的掉點。"""
    base = {r["control"]: r["f1_macro"] for r in rows
            if (r["shots"], r["noise_p"]) == baseline}
    out = []
    for r in rows:
        b = base.get(r["control"])
        out.append(dict(r, drop=None if b is None else r["f1_macro"] - b))
    return out


def min_shots_for_same_verdict(by_point, claim="量子性有用（糾纏有作用）", noise_p=0.0):
    """維持與無噪聲相同判定所需的最低取樣次數（SC-011）。

    回傳 (最低 shots 或 None, 說明)。達不到時回傳 None 並說明，不得靜默略過。
    """
    def v(recs):
        """從判準紀錄中取出指定宣稱的判定。"""
        for r in recs:
            if r["claim"] == claim:
                return r["verdict"]
        return None

    ref = v(by_point.get((None, noise_p), []))
    if ref is None:
        return None, "缺少無噪聲基準，無法判定"
    cands = sorted((s for (s, p) in by_point if p == noise_p and s is not None))
    for s in cands:
        if v(by_point[(s, noise_p)]) == ref:
            return int(s), f"在 shots={s} 時判定與無噪聲一致（{ref}）"
    top = cands[-1] if cands else None
    return None, (f"即使在最高取樣次數 shots={top} 下，判定仍與無噪聲不同"
                  f"（無噪聲為 {ref}）——此為必須明確說明的情況（SC-011）")
