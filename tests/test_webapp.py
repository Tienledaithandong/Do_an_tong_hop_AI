import json
import math
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from hive import webapp


@pytest.fixture(scope="module")
def server():
    srv = webapp.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _request(server, path, method="GET", payload=None, raw=None, timeout=60):
    port = server.server_address[1]
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()


def _json(server, path, method="GET", payload=None):
    status, _, body = _request(server, path, method=method, payload=payload)
    return status, json.loads(body.decode("utf-8"))


def _wait(server, job_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, job = _json(server, f"/api/jobs/{job_id}")
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def _run(server, payload):
    status, job = _json(server, "/api/jobs", method="POST", payload=payload)
    assert status == 202, job
    finished = _wait(server, job["id"])
    assert finished["status"] == "done", finished
    status, result = _json(server, f"/api/jobs/{job['id']}/result")
    assert status == 200
    return job["id"], finished, result


# --------------------------------------------------------------------- basics


def test_health_and_static_assets(server):
    status, health = _json(server, "/api/health")
    assert status == 200 and health["ok"] is True

    status, content_type, body = _request(server, "/")
    assert status == 200
    assert content_type == "text/html"
    text = body.decode("utf-8")
    assert "HiveMindRL" in text
    assert "/app.js" in text

    status, content_type, body = _request(server, "/app.js")
    assert status == 200
    assert content_type in {"text/javascript", "application/javascript"}
    assert b"runJob" in body

    status, content_type, body = _request(server, "/styles.css")
    assert status == 200
    assert content_type == "text/css"
    assert b"--honey" in body


def test_options_expose_the_ui_contract(server):
    status, options = _json(server, "/api/options")
    assert status == 200
    families = {family["id"] for family in options["families"]}
    assert {"complete", "ws", "rgg", "two_room", "ba", "er"} <= families
    assert set(options["scenarios"]) == {"low", "middle", "high"}
    assert {"quick", "medium", "full"} <= set(options["presets"])
    assert len(options["default_compare"]) >= 3


def test_invalid_payloads_are_rejected(server):
    cases = [
        {"kind": "graph", "N": 5},
        {"kind": "graph", "family": "nope", "param": 1},
        {"kind": "graph", "family": "ws", "param": 2.0},
        {"kind": "reproduction", "scenario": "unknown"},
        {"kind": "compare", "configs": []},
        {"kind": "compare", "configs": [{"family": "ws", "param": 0.1}] * 15},
        {"kind": "unknown"},
    ]
    for payload in cases:
        status, body = _json(server, "/api/jobs", method="POST", payload=payload)
        assert status == 400, (payload, status, body)
        assert body["error"]


def test_malformed_json_is_rejected(server):
    status, _, body = _request(server, "/api/jobs", method="POST", raw=b"{not json")
    assert status == 400
    assert "JSON" in json.loads(body)["error"]


def test_unknown_job_and_result_are_404(server):
    status, _ = _json(server, "/api/jobs/doesnotexist")
    assert status == 404
    status, _ = _json(server, "/api/jobs/doesnotexist/result")
    assert status == 404


def test_malformed_payload_body_is_rejected(server):
    status, _, body = _request(server, "/api/jobs", method="POST", payload=[])
    assert status == 400
    assert "object" in json.loads(body)["error"]


# ----------------------------------------------------------------------- jobs


def test_reproduction_job_returns_curves_and_png(server):
    job_id, job, result = _run(
        server,
        {
            "kind": "reproduction",
            "N": 60,
            "steps": 20,
            "seeds": 2,
            "n": 5,
            "scenario": "middle",
            "delta": 0.1,
        },
    )
    assert set(result["finals"]) == {"well_mixed", "pmcl", "mrd"}
    assert len(result["means"]["well_mixed"]) == 21
    assert math.isfinite(result["max_deviation"])
    assert "curves" in job["plots"]

    status, content_type, body = _request(server, f"/api/jobs/{job_id}/plots/curves.png")
    assert status == 200 and content_type == "image/png"
    assert body.startswith(b"\x89PNG\r\n\x1a\n")


def test_graph_job_returns_metrics_state_and_two_plots(server):
    job_id, job, result = _run(
        server,
        {
            "kind": "graph",
            "family": "ws",
            "param": 0.1,
            "N": 60,
            "steps": 20,
            "seeds": 2,
            "n": 5,
            "scenario": "middle",
            "delta": 0.1,
        },
    )
    for key in ("tau", "lambda2", "path_length", "degree_cv", "edges", "connected"):
        assert key in result["metrics"]
    assert result["metrics"]["edges"] > 0
    assert result["summary"]["hit_rate"] >= 0.0
    assert len(result["graph"]["types"]) == 60
    assert len(result["graph"]["layout"]) == 60
    assert set(job["plots"]) == {"curves", "graph"}

    for name in ("curves", "graph"):
        status, content_type, body = _request(
            server, f"/api/jobs/{job_id}/plots/{name}.png"
        )
        assert status == 200 and content_type == "image/png"
        assert body.startswith(b"\x89PNG\r\n\x1a\n")

    status, _ = _json(server, f"/api/jobs/{job_id}/plots/nope.png")
    assert status == 404


def test_compare_job_fits_the_law_and_returns_a_plot(server):
    job_id, job, result = _run(
        server,
        {
            "kind": "compare",
            "N": 50,
            "steps": 40,
            "seeds": 3,
            "n": 5,
            "configs": [
                {"family": "complete", "param": 1.0},
                {"family": "ws", "param": 0.7},
                {"family": "ws", "param": 1.0},
                {"family": "two_room", "param": 2.0},
            ],
        },
    )
    assert len(result["rows"]) == 4
    assert {row["family"] for row in result["rows"]} == {"complete", "ws", "two_room"}
    assert result["fit"]["n_train"] == 3
    assert result["fit"]["slope"] is not None
    assert "ci" in result["fit"]
    assert "warnings" in result["fit"]
    assert "compare" in job["plots"]

    status, content_type, body = _request(server, f"/api/jobs/{job_id}/plots/compare.png")
    assert status == 200 and content_type == "image/png"
    assert body.startswith(b"\x89PNG\r\n\x1a\n")


def test_graph_well_mixed_mode_runs(server):
    _, _, result = _run(
        server,
        {
            "kind": "graph",
            "family": "complete",
            "param": 1.0,
            "mode": "well_mixed",
            "N": 40,
            "steps": 10,
            "seeds": 2,
            "n": 5,
        },
    )
    assert result["mode"] == "well_mixed"
    assert result["metrics"]["connected"] is True


# -------------------------------------------------------------------- stored


def test_stored_results_endpoint_and_whitelist(server):
    status, listing = _json(server, "/api/results")
    assert status == 200
    assert isinstance(listing["items"], list)

    if listing["items"]:
        name = listing["items"][0]["name"]
        status, payload = _json(server, f"/api/results/{name}")
        assert status == 200
        assert payload["rows"] >= 0
        assert payload["columns"]

        status, content_type, body = _request(server, f"/api/results/{name}.csv")
        assert status == 200
        assert content_type == "text/csv"
        assert b"," in body
    else:
        assert listing["items"] == []

    status, _ = _json(server, "/api/results/not_a_real_result")
    assert status == 404
    status, _ = _json(server, "/api/results/..%2Fsecret")
    assert status == 404


# ------------------------------------------------------------- unit helpers


def test_json_safe_handles_numpy_and_non_finite_values():
    safe = webapp.json_safe(
        {
            "nan": np.float64("nan"),
            "inf": np.inf,
            "array": np.arange(3),
            "flag": np.bool_(True),
            "count": np.int64(7),
        }
    )
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["array"] == [0, 1, 2]
    assert safe["flag"] is True
    assert safe["count"] == 7


def test_common_validation_rejects_out_of_range_values():
    with pytest.raises(webapp.ValidationError):
        webapp._common({"N": 5})
    with pytest.raises(webapp.ValidationError):
        webapp._common({"steps": 5})
    with pytest.raises(webapp.ValidationError):
        webapp._common({"seeds": 0})
    with pytest.raises(webapp.ValidationError):
        webapp._common({"scenario": "very-high"})
    with pytest.raises(webapp.ValidationError):
        webapp._common({"delta": 0.9})


def test_family_parameters_are_clamped_to_the_graph_size():
    assert webapp._family_param("complete", 99, 50) == ("complete", 1.0)
    with pytest.raises(webapp.ValidationError):
        webapp._family_param("two_room", 20, 20)
    with pytest.raises(webapp.ValidationError):
        webapp._family_param("ws", 2.0, 100)
    with pytest.raises(webapp.ValidationError):
        webapp._family_param("ba", 2.5, 100)


def test_integer_validation_rejects_fractional_values():
    with pytest.raises(webapp.ValidationError):
        webapp._integer({"N": 50.5}, "N", 50, 20, 800)


def test_ui_script_only_touches_handles_exposed_by_make_ui():
    """Regression: renderers used ui.metrics before makeUi exposed it."""
    import re

    script = (webapp.UI_DIR / "app.js").read_text(encoding="utf-8")
    body = re.search(r"function makeUi\(prefix\) \{(.*?)\n\}", script, re.S).group(1)
    exposed = set(re.findall(r"^\s*([A-Za-z_]\w*):", body, re.M))
    used = set(re.findall(r"\bui\.([A-Za-z_]\w*)", script))
    missing = used - exposed - {"originalText"}
    assert not missing, f"app.js reads undefined UI handles: {sorted(missing)}"
