"""實驗追蹤：wandb（online/offline/off）＋ 本機 losses.csv（FR-032、FR-033）。

金鑰只從環境變數 / .env 讀取，絕不印出或寫入檔案。無金鑰或連線失敗自動降為離線。
"""
import csv
import os
from pathlib import Path

ENTITY, PROJECT = "PhysicalAI-Cat", "qustylegan-repro"
CSV_FIELDS = ["step", "block", "fade_alpha", "loss_g", "loss_d"]


class Tracker:
    """同時寫 losses.csv 與 wandb；wandb 任何失敗都不會中斷訓練。"""

    def __init__(self, run_dir, config_dict, mode="online", name=None):
        run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = run_dir / "losses.csv"
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._f); self._w.writerow(CSV_FIELDS)
        self.run, self.mode, self.url = None, "off", ""
        if mode != "off":
            self._init_wandb(run_dir, config_dict, mode, name)

    def _init_wandb(self, run_dir, config_dict, mode, name):
        """初始化 wandb；沒有金鑰或登入失敗就降為離線。"""
        try:
            import wandb
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("WANDB_API_KEY")
            if mode == "online" and key:
                try:
                    wandb.login(key=key, verify=False)
                except Exception:
                    mode = "offline"
            else:
                mode = "offline"
            os.environ["WANDB_MODE"] = mode
            os.environ.setdefault("WANDB_SILENT", "true")
            self.run = wandb.init(entity=ENTITY, project=PROJECT, name=name, config=config_dict,
                                  dir=str(run_dir), mode=mode, reinit=True)
            self.mode = mode
            self.url = getattr(self.run, "url", "") or ""
        except Exception:
            self.run, self.mode = None, "off"

    def log(self, step, block, fade_alpha, loss_g, loss_d, **extra):
        """記錄一個點到 losses.csv 與 wandb。"""
        self._w.writerow([step, block, f"{fade_alpha:.4f}", f"{loss_g:.6f}", f"{loss_d:.6f}"])
        self._f.flush()
        if self.run is not None:
            try:
                self.run.log(dict(loss_g=loss_g, loss_d=loss_d, block=block, fade_alpha=fade_alpha, **extra), step=step)
            except Exception:
                pass

    def close(self):
        """關閉檔案與 wandb。"""
        self._f.close()
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
