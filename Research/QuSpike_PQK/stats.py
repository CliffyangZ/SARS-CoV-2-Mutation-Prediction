"""多種子彙整與雜訊下限重估（FR-003／FR-004，研究 D3）。

論文的雜訊下限 δ=0.02 來自早期別的實驗。本模組以**本資料集**的多種子變異重新估計，
並與 δ 並列；判準使用兩者中較大者。種子清單不一致的結果不得合併（避免事後挑種子）。
"""
import numpy as np

DELTA = 0.02  # 論文事前訂定的雜訊下限

IMPROVEMENT_FIELDS = ("routing", "soft_threshold", "supplement", "kernel_mode", "nystrom_m")


def aggregate(vals):
    """(均值, 樣本標準差)；單一樣本時標準差為 0。"""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def estimate_noise_floor(scores_by_control, delta=DELTA):
    """由各設定的多種子分數重估雜訊下限。

    回傳 dict：sigma_pooled（各設定變異數的平均再開根號）、sigma_c0、delta、
    floor = max(delta, sigma_pooled)、floor_source（'delta' 或 'seed_variance'）。
    """
    stds = {k: aggregate(v)[1] for k, v in scores_by_control.items()
            if len([x for x in v if np.isfinite(x)]) > 1}
    pooled = float(np.sqrt(np.mean([s ** 2 for s in stds.values()]))) if stds else 0.0
    floor = max(float(delta), pooled)
    return dict(sigma_pooled=pooled, sigma_c0=stds.get("C0"), delta=float(delta), floor=floor,
                floor_source="seed_variance" if pooled > delta else "delta")


def assert_same_seeds(results):
    """所有結果的 config.seeds 必須完全相同，否則拒絕合併（D3）。results: [dict]。"""
    seen = {tuple(sorted(r.get("config", {}).get("seeds") or [r.get("seed")])) for r in results}
    if len(seen) > 1:
        raise ValueError(f"種子清單不一致，拒絕合併：{sorted(seen)}")


def assert_same_improvements(configs):
    """FR-010：MAIN 與各對照組的古典改進欄位必須相同。configs: {設定名: config dict}。"""
    ref_name, ref = next(iter(configs.items()))
    for name, c in configs.items():
        for f in IMPROVEMENT_FIELDS:
            if c.get(f) != ref.get(f):
                raise ValueError(f"古典改進欄位 {f} 在 {ref_name} 與 {name} 之間不一致："
                                 f"{ref.get(f)!r} vs {c.get(f)!r}")
