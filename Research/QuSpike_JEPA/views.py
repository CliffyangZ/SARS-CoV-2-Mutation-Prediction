"""View 產生：區段保留（最少突變數）、突變層級／位置層級／連續缺失遮蔽，以及三種正樣本配對。

隨機性：batch() 以 (seed, epoch, 樣本索引) 為隨機源，故相同設定可重現、續跑結果等價。
位置層級遮蔽的數量為 round(mask_pos × 保留區間內突變數)（偏離文件「遮 15% 位置」：
1273 位的 15% 約 190 個 token，與每條 20–40 個突變不成比例），遮蔽處以 MASK_SYM 保留，不丟棄。
"""
import numpy as np

from . import pairs as P
from .dataset import MASK_SYM, N_SYM
from .mutations import N_POS, UNKNOWN


class ViewMaker:
    """對 TokenSet 產生成對 view。variant：VA 同譜系、VB 親緣鄰近、VC 單序列破壞。"""

    def __init__(self, ts, variant="VA", mask_mut=0.15, mask_pos=0.15, trim=(0.5, 1.0), local_trim=(0.15, 0.5),
                 min_mut=3, mask_prior="P0", span_prob=0.0, span_len=(20, 100), neighbor_dist=4, neighbor_months=1,
                 max_retry=20, seed=0, vd_trim=0.02):
        self.ts, self.variant = ts, P.canonical_variant(variant)
        self.mask_mut, self.mask_pos, self.trim, self.local_trim = mask_mut, mask_pos, trim, local_trim
        self.min_mut, self.mask_prior, self.span_prob, self.span_len = min_mut, mask_prior, span_prob, span_len
        self.max_retry, self.seed, self.vd_trim = max_retry, seed, vd_trim
        self.rng = np.random.default_rng(seed)
        self.stats = dict(pairs=0, fallback_self=0, retry_fail=0, empty_views=0, cross_lineage=0)
        self._groups = P.lineage_groups(ts.y) if self.variant == "VA" else None
        self._nbr = P.neighbor_lists(ts, neighbor_dist, neighbor_months, seed=seed) if self.variant == "VB" else None
        ids = (ts.pos.astype(np.int64) * N_SYM + ts.sym.astype(np.int64))[ts.valid]
        u, c = np.unique(ids, return_counts=True)
        self._freq = dict(zip(u.tolist(), (c / max(len(ts), 1)).tolist()))
        self._mean_freq = float(np.average(c / max(len(ts), 1), weights=c)) if len(c) else 1.0
        if self.variant == "VD":
            self._xrun_count, self._xrun_len = x_run_stats(ts)

    # --- 配對 ---
    def partner(self, i, rng=None):
        """回傳 i 的正樣本伙伴索引；樣本不足時退回自己並計數。"""
        rng = rng or self.rng
        if self.variant in ("VC", "VD"):
            return i
        pool = self._groups[int(self.ts.y[i])] if self.variant == "VA" else self._nbr[i]
        pool = pool[pool != i]
        if len(pool) == 0:
            self.stats["fallback_self"] += 1
            return i
        return int(rng.choice(pool))

    # --- 遮蔽 ---
    def _mask_rate(self, token_ids):
        """各 token 被遮蔽的機率：P0 均勻；P1 依出現比例（高頻多遮），期望遮蔽數與 P0 相近。"""
        token_ids = np.asarray(token_ids)
        if self.mask_prior == "P0":
            return np.full(len(token_ids), self.mask_mut)
        f = np.array([self._freq.get(int(t), 0.0) for t in token_ids])
        return np.minimum(0.9, self.mask_mut * f / self._mean_freq)

    def _window(self, rng, local):
        lo, hi = self.local_trim if local else self.trim
        w = int(round(rng.uniform(lo, hi) * N_POS))
        w = min(max(w, 1), N_POS)
        start = int(rng.integers(1, N_POS - w + 2))
        return start, start + w - 1

    def nuisance_view(self, i, rng):
        """V-D：只加定序 nuisance，不遮蔽或丟棄任何突變的身分（§3.4 修訂）。

        - 連續缺失：段數與段長都從 train 實際的 X 連續段分佈抽樣（至少 1 段）；段內原有 token 換成逐位的 X，
          與真實資料中 amplicon dropout 的表示方式相同。
        - 首尾截斷：兩端各截去至多 vd_trim 比例的位置（未觀測，不產生 token），與 mutations_of 對首尾短缺的處理一致。
        """
        v = self.ts.valid[i]
        toks = dict(zip(self.ts.pos[i][v].astype(int).tolist(), self.ts.sym[i][v].astype(int).tolist()))
        if len(self._xrun_len):
            k = max(1, int(rng.choice(self._xrun_count)))
            for _ in range(k):
                ln = int(min(rng.choice(self._xrun_len), N_POS))
                s0 = int(rng.integers(1, N_POS - ln + 2))
                for p in range(s0, s0 + ln):
                    toks[p] = UNKNOWN
        a = 1 + int(rng.uniform(0, self.vd_trim) * N_POS)
        b = N_POS - int(rng.uniform(0, self.vd_trim) * N_POS)
        out = sorted((p, s) for p, s in toks.items() if a <= p <= b)
        if not out:
            self.stats["empty_views"] += 1
        return out

    def one_view(self, i, local=False, rng=None):
        """對第 i 條序列產生一個 view：[(位置, 符號)]，被遮蔽者符號為 MASK_SYM。"""
        rng = rng or self.rng
        if self.variant == "VD":
            return self.nuisance_view(i, rng)
        v = self.ts.valid[i]
        pos, sym = self.ts.pos[i][v].astype(int), self.ts.sym[i][v].astype(int)
        need = min(self.min_mut, len(pos))
        for _ in range(self.max_retry):
            a, b = self._window(rng, local)
            keep = (pos >= a) & (pos <= b)
            if keep.sum() >= need:
                break
        else:  # 重取失敗：改取包含某個突變的區間，並計數
            self.stats["retry_fail"] += 1
            if len(pos) == 0:
                a, b = 1, N_POS
            else:
                lo, hi = self.local_trim if local else self.trim
                w = min(max(int(round(rng.uniform(lo, hi) * N_POS)), 1), N_POS)
                c = int(rng.choice(pos))
                a = int(np.clip(c - rng.integers(0, w), 1, N_POS - w + 1))
                b = a + w - 1
            keep = (pos >= a) & (pos <= b)
        kp, ks = pos[keep], sym[keep].copy()
        if len(kp):
            tid = kp.astype(np.int64) * N_SYM + ks
            ks[rng.random(len(kp)) < self._mask_rate(tid)] = MASK_SYM
        out = list(zip(kp.tolist(), ks.tolist()))
        n_pm = int(round(self.mask_pos * len(kp)))
        if n_pm > 0:
            free = np.setdiff1d(np.arange(a, b + 1), kp)
            if len(free):
                pm = rng.choice(free, min(n_pm, len(free)), replace=False)
                out += [(int(p), MASK_SYM) for p in pm]
        if self.span_prob > 0 and rng.random() < self.span_prob:
            ln = int(rng.integers(self.span_len[0], self.span_len[1] + 1))
            s0 = int(rng.integers(1, N_POS - ln + 2))
            out = [(p, MASK_SYM if s0 <= p < s0 + ln else s) for p, s in out]
        out.sort()
        if not out:
            self.stats["empty_views"] += 1
        return out

    # --- 批次 ---
    @staticmethod
    def _pad(views):
        T = max(1, max(len(v) for v in views))
        pos = np.zeros((len(views), T), np.int64)
        sym = np.zeros((len(views), T), np.int64)
        valid = np.zeros((len(views), T), bool)
        for r, v in enumerate(views):
            for c, (p, s) in enumerate(v):
                pos[r, c], sym[r, c], valid[r, c] = p, s, True
        return pos, sym, valid

    def batch(self, idx, epoch=0):
        """回傳 (pos_a, sym_a, valid_a, pos_b, sym_b, valid_b)：a 為 global view，b 為伙伴的 local view。"""
        va, vb = [], []
        for i in np.asarray(idx):
            rng = np.random.default_rng([self.seed, int(epoch), int(i)])
            j = self.partner(int(i), rng)
            self.stats["pairs"] += 1
            self.stats["cross_lineage"] += int(self.ts.y[j] != self.ts.y[i])
            va.append(self.one_view(int(i), False, rng))
            vb.append(self.one_view(j, True, rng))
        return (*self._pad(va), *self._pad(vb))

    def full_view(self, idx):
        """不遮蔽、不取區段的完整突變集合（推論／probe 用），回傳 (pos, sym, valid)。"""
        idx = np.asarray(idx)
        return (self.ts.pos[idx].astype(np.int64), self.ts.sym[idx].astype(np.int64), self.ts.valid[idx])


def x_run_stats(ts):
    """train 中每條序列的 X 連續段數，以及所有段的長度（V-D 的缺失分佈）。"""
    counts, lens = [], []
    for i in range(len(ts)):
        v = ts.valid[i]
        xp = np.sort(ts.pos[i][v][ts.sym[i][v] == UNKNOWN].astype(int))
        if len(xp) == 0:
            counts.append(0)
            continue
        breaks = np.where(np.diff(xp) != 1)[0]
        seg = np.split(xp, breaks + 1)
        counts.append(len(seg))
        lens += [len(s) for s in seg]
    return np.array(counts, int), np.array(lens, int)
