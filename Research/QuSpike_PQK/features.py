"""特徵載入、分群與群內降維（T008–T010）。

三件事必須嚴格遵守，否則整個評估失效：
1. 所有對照組共用同一個 FeatureSet（FR-002）——比較才只反映特徵圖的差異。
2. 降維只能用 train 擬合（FR-003）——否則 test 標籤洩漏進特徵。
3. 縮放後落在 [−π/2, π/2]——避免編碼角度在週期上首尾重疊。
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from Research.QuSpike_JEPA import dataset as D
from Research.QuSpike_JEPA.ceiling import GROUPS
from Research.QuSpike_JEPA.mutations import UNKNOWN

# 難群：Delta 母譜系 + 其子譜系、B.1 系。其餘歸 others，不進量子路徑。
QUANTUM_GROUPS = {
    "delta": GROUPS["delta_parent"] + GROUPS["delta_sub"],
    "b1": GROUPS["b1_family"],
}
REFERENCE_SCORE = 0.747  # 監督線性參考線（22 類 test macro F1），見研究報告 §5.1
NOISE_FLOOR = 0.02       # 事前訂定的雜訊下限，見研究報告 §4
DEFAULT_DATA = D.ROOT / "Data" / "spike-lineage" / "dataset_v2.tsv"


@dataclass
class FeatureSet:
    """載入後的輸入特徵；所有對照組共用同一個實例。"""

    source: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    labels: list
    data_hash: str
    split_version: str
    selection_reason: str
    month_train: np.ndarray = None  # 供 validation 切分使用（沿用 JEPA 的時間規則）
    ood_fraction: float = None      # 測試序列含訓練詞彙外突變的比例（邊界情況；特徵向量忽略該突變）


@dataclass
class Group:
    """類別分組。is_quantum=False 者直接走參考線。"""

    name: str
    class_ids: tuple
    n_train: int
    n_test: int
    is_quantum: bool
    train_idx: np.ndarray = field(default=None, repr=False)
    test_idx: np.ndarray = field(default=None, repr=False)


@dataclass
class Projection:
    """群內 train-only 降維與縮放的結果。

    `transform(x)` 可對**任意列**投影（FR-001：推論時要對所有評估列算特徵，
    而不是只對「真實標籤屬於該群」的列）。轉換器只用訓練列擬合。
    """

    group: str
    method: str
    n_components: int
    n_padded: int
    clip_lo: np.ndarray
    clip_hi: np.ndarray
    z_train: np.ndarray
    z_test: np.ndarray
    supplement: str = "zero"
    n_supplement: int = 0
    supplement_note: str = ""
    _tf: object = field(default=None, repr=False)
    _width: int = 0
    _sup: object = field(default=None, repr=False)
    _basis: object = field(default=None, repr=False)
    _sup_lo: object = field(default=None, repr=False)
    _sup_hi: object = field(default=None, repr=False)

    @staticmethod
    def _scale(u, lo, hi):
        """以 train 的分位數截斷後線性映射到 [−π/2, π/2]。"""
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        c = np.clip(np.asarray(u, np.float64), lo, hi)
        return (c - lo) / span * np.pi - np.pi / 2

    def transform(self, x):
        """對任意列做投影＋縮放＋補維，輸出形狀 (n, 2·n_qubits)。"""
        x = np.asarray(x)
        u = self._tf.transform(x) if self._tf is not None else x[:, :self.n_components]
        z = self._scale(u, self.clip_lo, self.clip_hi)
        if self._sup is not None:
            xr = x - (x @ self._basis) @ self._basis.T
            z = np.hstack([z, self._scale(self._sup.transform(xr), self._sup_lo, self._sup_hi)])
        pad = self._width - z.shape[1]
        if pad > 0:
            z = np.hstack([z, np.zeros((len(z), pad))])
        return z


def mutation_matrix(ts, vocab):
    """0/1 突變指標矩陣（排除 X），與 baselines.mutset_linear 使用同一組特徵。"""
    m = ts.valid & (ts.sym != UNKNOWN)
    rows, cols = np.nonzero(m)
    tok = ts.pos[rows, cols].astype(np.int64) * D.N_SYM + ts.sym[rows, cols].astype(np.int64)
    j = np.searchsorted(vocab, tok)
    ok = (j < len(vocab)) & (vocab[np.minimum(j, len(vocab) - 1)] == tok)
    out = np.zeros((len(ts), len(vocab)), np.float32)
    out[rows[ok], j[ok]] = 1.0
    return out


def ood_fraction(ts, vocab):
    """含有訓練詞彙外突變的序列比例。這類突變在特徵向量中被忽略，報告必須揭露其比例。"""
    m = ts.valid & (ts.sym != UNKNOWN)
    tok = ts.pos.astype(np.int64) * D.N_SYM + ts.sym.astype(np.int64)
    j = np.searchsorted(vocab, tok)
    known = (j < len(vocab)) & (vocab[np.minimum(j, len(vocab) - 1)] == tok)
    return float((m & ~known).any(1).mean())


def load_features(source="mutation", data=DEFAULT_DATA, embedding_cache=None, reason=None):
    """載入特徵。source='mutation' 為 0/1 突變指標；'lm' 為凍結語言模型嵌入快取。"""
    train, test, split_version = D.load_splits(data)
    if source == "mutation":
        keep = train.valid & (train.sym != UNKNOWN)
        vocab = np.unique(train.pos[keep].astype(np.int64) * D.N_SYM + train.sym[keep].astype(np.int64))
        x_tr, x_te = mutation_matrix(train, vocab), mutation_matrix(test, vocab)
        ood = ood_fraction(test, vocab)
        name = "mutation_indicator"
        default_reason = (f"22 類凍結語言模型嵌入 test macro F1 0.679 未超過參考線 {REFERENCE_SCORE} "
                          f"達 {NOISE_FLOOR}，依 FR-002 規則採用突變指標")
    elif source == "lm":
        if embedding_cache is None:
            raise ValueError("source='lm' 需要 embedding_cache 路徑")
        z = np.load(Path(embedding_cache))
        x_tr, x_te = np.asarray(z["train"], np.float32), np.asarray(z["test"], np.float32)
        ood = None
        name = "lm_embedding"
        default_reason = "由呼叫端指定使用凍結語言模型嵌入"
    else:
        raise ValueError(f"未知的 source: {source}")
    return FeatureSet(source=name, x_train=x_tr, x_test=x_te,
                      y_train=np.asarray(train.y, np.int64), y_test=np.asarray(test.y, np.int64),
                      labels=list(train.labels), data_hash=D.file_hash(data),
                      split_version=split_version, selection_reason=reason or default_reason,
                      month_train=np.asarray(train.month, np.int32), ood_fraction=ood)


def split_validation(fs, frac=0.15):
    """train 內每類別日期最晚的 frac 作為 validation（沿用 JEPA 的 split_validation 規則）。

    超參數（專家混合權重）必須在此選定，不得看 test——否則比較會偏向調得最細的設定。
    """
    month = fs.month_train
    if month is None:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(fs.y_train))
        cut = int(len(perm) * (1 - frac))
        return perm[:cut], perm[cut:]
    val = np.zeros(len(fs.y_train), bool)
    for k in np.unique(fs.y_train):
        ids = np.where((fs.y_train == k) & (month >= 0))[0]
        if len(ids) < 2:
            continue
        n_val = max(1, int(round(len(ids) * frac)))
        order = ids[np.argsort(month[ids], kind="stable")]
        val[order[len(order) - n_val:]] = True
    return np.where(~val)[0], np.where(val)[0]


def make_groups(fs, min_train=50):
    """切出 delta／b1／others 三群，並附各群的索引。"""
    idx = {l: i for i, l in enumerate(fs.labels)}
    out, claimed = [], set()
    for name, names in QUANTUM_GROUPS.items():
        ids = tuple(sorted(idx[x] for x in names if x in idx))
        claimed |= set(ids)
        out.append(_build_group(name, ids, fs, True))
    rest = tuple(sorted(set(range(len(fs.labels))) - claimed))
    out.append(_build_group("others", rest, fs, False))
    return [degrade_if_too_small(g, min_train) for g in out]


def _build_group(name, ids, fs, is_quantum):
    tr = np.where(np.isin(fs.y_train, ids))[0]
    te = np.where(np.isin(fs.y_test, ids))[0]
    return Group(name=name, class_ids=ids, n_train=len(tr), n_test=len(te),
                 is_quantum=is_quantum, train_idx=tr, test_idx=te)


def degrade_if_too_small(g, min_train=50):
    """類別數 < 2 或 train 樣本過少時，降級為不走量子（edge case）。"""
    if g.is_quantum and (len(g.class_ids) < 2 or g.n_train < min_train):
        return Group(**{**g.__dict__, "is_quantum": False})
    return g


def _safe_transform(tf, x, k):
    """空矩陣時回傳 (0, k)（sklearn 對 0 列會丟例外；群內無測試列是合法的 edge case）。"""
    return tf.transform(x) if len(x) else np.zeros((0, k))


def decoupled_qubits(z_train, n_qubits):
    """回傳完全解耦的量子位元索引（FR-009）。

    環上第 e 條邊的耦合強度取自 z 的第 n+e 維（n=2 時環上只有一條邊）。
    若某位元相連的邊，其耦合維度在訓練集上變異皆為 0（零填補），該位元就與其他位元完全解耦，
    線路對它而言等於單位元演化——量子特徵圖在該位元上沒有「量子性」可言。
    """
    n = int(n_qubits)
    z = np.asarray(z_train, np.float64)
    if n < 2 or z.shape[1] < 2 * n:
        return list(range(n))
    edges = [(0, 1, n)] if n == 2 else [(j, (j + 1) % n, n + j) for j in range(n)]
    live = z.std(0) > 1e-12
    out = []
    for q in range(n):
        inc = [d for a, b, d in edges if q in (a, b)]
        if not any(live[d] for d in inc):
            out.append(q)
    return out


def _fit_pls_supplement(xt, yt, lda, k, pad):
    """LDA 子空間正交補上的有監督補充維度（PLS-DA）。回傳 (pls, basis, 說明)。"""
    from sklearn.cross_decomposition import PLSRegression

    basis, _ = np.linalg.qr(np.asarray(lda.scalings_[:, :k], np.float64))
    xr = xt - (xt @ basis) @ basis.T
    classes = np.unique(yt)
    onehot = (yt[:, None] == classes[None, :]).astype(np.float64)
    nc = int(min(pad, onehot.shape[1], xr.shape[1]))
    if nc < 1:
        return None, None, "無可用的補充成分"
    pls = PLSRegression(n_components=nc, scale=False).fit(xr, onehot)
    return pls, basis, ""


def fit_projection(fs, group, n_qubits, method="lda", q=(1.0, 99.0), supplement="zero"):
    """群內降維 + 縮放。**只用 train 擬合**（FR-003）。回傳 Projection。

    supplement='zero'：維度不足時以 0 填補（v1 行為）；
    supplement='pls' ：維度不足時改以有監督的 PLS-DA 成分補足（FR-009）。
    """
    xt, yt = fs.x_train[group.train_idx], fs.y_train[group.train_idx]
    xe = fs.x_test[group.test_idx]
    width = 2 * n_qubits
    # 以**實際出現**的類別數決定 LDA 維度：小樣本下群內可能缺類別，
    # 用宣告的 class_ids 會讓 n_components 超出 sklearn 的上界（群內樣本過少的 edge case）。
    n_present = int(len(np.unique(yt)))
    if method == "lda" and n_present < 2:
        method = "pca"  # 只剩一個類別時 LDA 無定義，退回 PCA
    k = min(n_present - 1, width) if method == "lda" else min(width, xt.shape[1])
    k = max(int(min(k, xt.shape[1])), 1)

    lda = None
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        try:
            tf = lda = LinearDiscriminantAnalysis(n_components=k).fit(xt, yt)
        except np.linalg.LinAlgError:
            # float32 的 SVD 偶爾不收斂（例如 v2-select 的二次切分）；只在失敗時改用 float64 重試，不改動既有結果
            tf = lda = LinearDiscriminantAnalysis(n_components=k).fit(np.asarray(xt, np.float64), yt)
        ut, ue = tf.transform(xt), _safe_transform(tf, xe, k)
    elif method == "pca":
        from sklearn.decomposition import PCA

        tf = PCA(n_components=k, random_state=0).fit(xt)
        ut, ue = tf.transform(xt), _safe_transform(tf, xe, k)
    elif method == "pls":
        # 探索性（exp_ref_proj）：純 PLS-DA，不受 LDA 的 C−1 維上限限制
        from sklearn.cross_decomposition import PLSRegression

        classes = np.unique(yt)
        onehot = (yt[:, None] == classes[None, :]).astype(np.float64)
        k = max(int(min(width, len(classes), xt.shape[1])), 1)
        tf = PLSRegression(n_components=k, scale=False).fit(np.asarray(xt, np.float64), onehot)
        ut, ue = tf.transform(xt), _safe_transform(tf, xe, k)
    elif method == "raw":
        tf = None
        ut, ue = xt[:, :k], xe[:, :k]
    else:
        raise ValueError(f"未知的 method: {method}")

    lo = np.percentile(ut, q[0], axis=0).astype(np.float64)
    hi = np.percentile(ut, q[1], axis=0).astype(np.float64)
    zt, ze = Projection._scale(ut, lo, hi), Projection._scale(ue, lo, hi)
    pad = width - k

    sup = basis = sup_lo = sup_hi = None
    n_sup, note, used = 0, "", "zero"
    if supplement not in ("zero", "pls"):
        raise ValueError(f"未知的 supplement: {supplement}")
    if supplement == "pls" and pad > 0 and lda is not None:
        sup, basis, note = _fit_pls_supplement(xt, yt, lda, k, pad)
        if sup is not None:
            st = sup.transform(xt - (xt @ basis) @ basis.T)
            sup_lo = np.percentile(st, q[0], axis=0).astype(np.float64)
            sup_hi = np.percentile(st, q[1], axis=0).astype(np.float64)
            se = (sup.transform(xe - (xe @ basis) @ basis.T) if len(xe)
                  else np.zeros((0, st.shape[1])))
            zt = np.hstack([zt, Projection._scale(st, sup_lo, sup_hi)])
            ze = np.hstack([ze, Projection._scale(se, sup_lo, sup_hi)])
            n_sup, used = int(st.shape[1]), "pls"
    left = width - zt.shape[1]
    if left > 0:
        zt = np.hstack([zt, np.zeros((len(zt), left))])
        ze = np.hstack([ze, np.zeros((len(ze), left))])
    return Projection(group=group.name, method=method, n_components=k, n_padded=left,
                      clip_lo=lo, clip_hi=hi, z_train=zt, z_test=ze, supplement=used,
                      n_supplement=n_sup, supplement_note=note, _tf=tf, _width=width,
                      _sup=sup, _basis=basis, _sup_lo=sup_lo, _sup_hi=sup_hi)
