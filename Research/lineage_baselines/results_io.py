"""結果檔讀寫：每個（模型，表示，種子）一個 JSON；已完成者可跳過。"""
import json
import os
import platform
from pathlib import Path


def result_path(out_dir, model, representation, seed):
    """依命名規則回傳結果檔路徑。"""
    return Path(out_dir) / f"{model}__{representation}__seed{seed}.json"


def is_complete(path, min_version=None):
    """結果檔存在且 status 為 complete；給定 min_version 時，recipe_version 還需不小於它（舊版配方的結果視為未完成）。"""
    path = Path(path)
    if not path.exists():
        return False
    try:
        r = json.loads(path.read_text())
        return r.get("status") == "complete" and (min_version is None or r.get("recipe_version", 1) >= min_version)
    except (json.JSONDecodeError, OSError):
        return False


def read_signal_mode(path):
    """讀取結果檔記錄的訊號模式；舊結果沒有此欄位，視為 raw。"""
    try:
        return json.loads(Path(path).read_text()).get("hyperparameters", {}).get("signal_mode", "raw")
    except (json.JSONDecodeError, OSError):
        return "raw"


def write_result(path, result):
    """原子寫入結果檔（先寫暫存檔再改名，避免中斷造成半個檔案）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def read_results(out_dir):
    """讀取目錄中所有結果檔（忽略無法解析者）。"""
    res = []
    for p in sorted(Path(out_dir).glob("*__seed*.json")):
        try:
            res.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return res


def environment_info():
    """記錄 Python、torch、sklearn 版本與 GPU 名稱。"""
    import sklearn

    info = {"python": platform.python_version(), "sklearn": sklearn.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    return info
