"""v2 的事前登錄與測試集防護（US3，FR-011／FR-014，SC-007）。

停止規則禁止為了翻轉結論而事後調參。要讓「v2 是新假設」可被稽核，防護必須做進物件模型：
- 設計階段只拿得到 `ValidationOnlyView`——物件上**根本沒有** x_test／y_test；
- `evaluate_on_test` 第一步就是 `assert_locked`，未鎖定時不會載入任何測試資料；
- 每次測試評估都追加到 `test_evals.jsonl`，正式報告只採 index==1。

登錄檔位置預設為 `Doc/registry/pqk_v2/`，格式見 specs/008-pqk-optimization/contracts/preregistration.md。
"""
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import features as F

DEFAULT_ROOT = F.D.ROOT / "Doc" / "registry" / "pqk_v2"
HASHED = ("hypothesis", "design", "selection_log", "controls", "criteria", "stop_rule", "seeds")
REQUIRED = ("version", "hypothesis", "design", "controls", "criteria", "stop_rule", "seeds")
EVALS = "test_evals.jsonl"


class RegistryError(RuntimeError):
    """登錄檔不存在、未鎖定、已被改動或違反鎖定規則。"""


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def content_hash(rec):
    """對受保護欄位的正規化 JSON 取 SHA-256。"""
    payload = json.dumps({k: rec.get(k) for k in HASHED}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(version, root):
    return Path(root) / f"{version}.json"


def validate_draft(draft):
    """草稿必須含假設、超參數、判準、對照組、停止規則與種子。"""
    missing = [k for k in REQUIRED if not draft.get(k)]
    if missing:
        raise RegistryError(f"登錄草稿缺少欄位：{missing}")
    if len(draft["seeds"]) < 5:
        raise RegistryError("登錄的種子數必須 ≥ 5（FR-003）")
    return True


def lock(draft, root=DEFAULT_ROOT):
    """鎖定草稿（dict 或 JSON 路徑）。同版本已存在 → RegistryError。回傳寫入的路徑。"""
    if not isinstance(draft, dict):
        draft = json.loads(Path(draft).read_text(encoding="utf-8"))
    validate_draft(draft)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = _path(draft["version"], root)
    if path.exists():
        raise RegistryError(f"版本 {draft['version']} 已存在；修正只能另存新版本（supersedes）")
    rec = dict(draft, status="locked", locked_at=_now(), content_sha256=content_hash(draft))
    rec.setdefault("supersedes", None)
    rec.setdefault("selection_log", [])
    rec["content_sha256"] = content_hash(rec)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path


def load(version, root=DEFAULT_ROOT):
    """讀取並驗證雜湊。不存在或內容被改動 → RegistryError。"""
    path = _path(version, root)
    if not path.exists():
        raise RegistryError(f"找不到登錄檔 {version}（{path}）；請先 v2-register --lock")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("content_sha256") != content_hash(rec):
        raise RegistryError(f"登錄檔 {version} 的內容與雜湊不符（被改動過）")
    return rec


def assert_locked(version, root=DEFAULT_ROOT):
    """未鎖定或不存在即拒絕。回傳登錄紀錄。"""
    rec = load(version, root)
    if rec.get("status") != "locked":
        raise RegistryError(f"登錄 {version} 的狀態是 {rec.get('status')!r}，不是 locked")
    return rec


def _evals(root):
    p = Path(root) / EVALS
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def record_test_eval(version, run_id, root=DEFAULT_ROOT):
    """追加一筆測試評估紀錄，回傳該版本的第幾次評估（從 1 起）。"""
    assert_locked(version, root)
    n = sum(1 for e in _evals(root) if e["version"] == version) + 1
    with open(Path(root) / EVALS, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(version=version, utc=_now(), run_id=run_id, index=n),
                           ensure_ascii=False) + "\n")
    return n


def timing_ok(version, root=DEFAULT_ROOT):
    """SC-007：locked_at 必須早於該版本第一次測試評估。尚無評估時回傳 None。"""
    rec = load(version, root)
    ev = sorted((e for e in _evals(root) if e["version"] == version), key=lambda e: e["index"])
    if not ev:
        return None
    return rec["locked_at"] < ev[0]["utc"]


def all_versions(root=DEFAULT_ROOT):
    """所有登錄版本（報告需列出，邊界情況）。"""
    root = Path(root)
    return sorted(p.stem for p in root.glob("v*.json")) if root.exists() else []


def evaluate_on_test(version, loader, runner, run_id="run", root=DEFAULT_ROOT):
    """唯一能碰測試集的入口：先驗證登錄，再延遲載入測試資料。

    未鎖定時 `loader` 完全不會被呼叫。回傳 (評估次數, runner 的回傳值)。
    """
    rec = assert_locked(version, root)
    idx = record_test_eval(version, run_id, root)
    return idx, runner(loader(), rec)


class ValidationOnlyView:
    """設計階段專用的資料視圖：只含 train 切出的 fit／validation，**沒有** x_test／y_test。

    `as_featureset()` 回傳的 FeatureSet 其「測試位置」放的是 validation 切分，
    管線因此可以原封不動地在 validation 上評估，而原始測試陣列從未進入此物件。
    """

    def __init__(self, fs):
        fit_idx, val_idx = F.split_validation(fs)
        self.x_fit, self.y_fit = fs.x_train[fit_idx], fs.y_train[fit_idx]
        self.x_val, self.y_val = fs.x_train[val_idx], fs.y_train[val_idx]
        self.month_fit = fs.month_train[fit_idx] if fs.month_train is not None else None
        self.labels, self.data_hash, self.split_version = fs.labels, fs.data_hash, fs.split_version
        self.source, self.selection_reason = fs.source, fs.selection_reason

    def as_featureset(self):
        """x_train=fit、x_test=validation 的 FeatureSet。"""
        return F.FeatureSet(source=self.source, x_train=self.x_fit, x_test=self.x_val,
                            y_train=self.y_fit, y_test=self.y_val, labels=self.labels,
                            data_hash=self.data_hash + "_vsel", split_version=self.split_version,
                            selection_reason=self.selection_reason, month_train=self.month_fit)
