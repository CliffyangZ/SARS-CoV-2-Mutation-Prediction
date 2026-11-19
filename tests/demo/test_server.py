import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from Demo import server


@pytest.fixture(scope="module")
def url(predictor):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(predictor))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def post(url, body):
    req = urllib.request.Request(url + "/api/predict", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_health_and_page(url):
    assert json.load(urllib.request.urlopen(url + "/api/health"))["ready"] is True
    assert b"<html" in urllib.request.urlopen(url + "/").read()


def test_predict_ok(url, seqs):
    code, d = post(url, {"input": seqs[0], "k": 3, "s": None})
    assert code == 200 and len(d["top_k"]) == 3
    assert {"routing", "s_used", "top1_changed_by_expert", "parsed", "disclaimer"} <= set(d)


def test_errors(url, seqs):
    for body in ({"input": ""}, {"input": seqs[0], "k": 0}, {"input": seqs[0], "s": -1}, {"input": "A452R"}):
        code, d = post(url, body)
        assert code == 400 and "error" in d and "details" in d


def test_503_without_model():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert post(f"http://127.0.0.1:{srv.server_port}", {"input": "x"})[0] == 503
    finally:
        srv.shutdown()


def test_refuses_non_localhost():
    assert server.serve("0.0.0.0", 0) == 2
