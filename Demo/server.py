"""本機網頁：標準函式庫 http.server。只綁 localhost、不記錄輸入內容。契約見 specs/009-*/contracts/web-api.md。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import bundle as B, parse
from .predict import Predictor

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 1_000_000
LOCAL = ("127.0.0.1", "localhost", "::1")


def make_handler(predictor, lock=None):
    lock = lock or threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 不記錄請求（含輸入內容）
            pass

        def _send(self, code, obj=None, body=None, ctype="application/json; charset=utf-8"):
            data = body if body is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, body=(STATIC / "index.html").read_bytes(), ctype="text/html; charset=utf-8")
            if self.path == "/static/virus.svg":  # 白名單：只提供這個檔案
                return self._send(200, body=(STATIC / "virus.svg").read_bytes(), ctype="image/svg+xml")
            if self.path == "/api/health":
                if predictor is None:
                    return self._send(503, {"ready": False, "error": "找不到模型包；請先執行 python -m Demo.cli build"})
                return self._send(200, {"ready": True, "s_default": predictor.b.s_default})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/predict":
                return self._send(404, {"error": "not found"})
            if predictor is None:
                return self._send(503, {"error": "找不到模型包；請先執行 python -m Demo.cli build"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY:
                    return self._send(400, {"error": "請求太大", "details": []})
                req = json.loads(self.rfile.read(n) or b"{}")
                k, s = req.get("k", 5), req.get("s")
                if isinstance(k, bool) or not isinstance(k, int):
                    raise parse.InputError("k 必須是整數")
                if s is not None and (isinstance(s, bool) or not isinstance(s, (int, float))):
                    raise parse.InputError("s 必須是數字或 null")
                with lock:
                    pred = predictor.predict(req.get("input"), req.get("mode", "auto"), k=k,
                                             s=None if s is None else float(s))
                self._send(200, pred.to_dict())
            except parse.InputError as e:
                self._send(400, {"error": str(e), "details": e.details})
            except (json.JSONDecodeError, ValueError) as e:
                self._send(400, {"error": f"請求格式錯誤：{e}", "details": []})

    return H


def serve(host="127.0.0.1", port=8000, bundle_path=B.DEFAULT_BUNDLE):
    if host not in LOCAL:
        print("拒絕：Demo 只允許綁 localhost")
        return 2
    try:
        pred = Predictor(B.load_bundle(bundle_path))
    except B.BundleMissing as e:
        print(e)
        return 3
    srv = ThreadingHTTPServer((host, port), make_handler(pred))
    print(f"即時譜系預測器：http://{host}:{port}  （Ctrl+C 結束）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
