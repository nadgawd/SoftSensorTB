"""The plant connector against the real live API, and its OPC UA / MQTT / PI Web API sources with fakes."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api import deployments as deployments_api
from backend.deployment import service
from test_deployment_api import deploy, train
from test_deployment_replay import build_pipeline, plant_csv
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures

_PATH = Path(__file__).resolve().parent.parent / "backend" / "deployment" / "plant_connector" / "softsensor_connector.py"
_spec = importlib.util.spec_from_file_location("softsensor_connector", _PATH)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


@pytest.fixture(autouse=True)
def _fresh_limits():
    deployments_api.live_limiter.reset()
    service.clear_caches()


def api_transport(test_client):
    """The connector's HTTP layer, routed into the FastAPI test client."""

    def transport(method, url, headers, body, timeout):
        path = url.split("://", 1)[-1]
        path = path[path.index("/"):]
        resp = test_client.request(method, path, headers=headers, json=body)
        return resp.status_code, resp.json(), dict(resp.headers)

    return transport


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def test_csv_replay_lab_and_write_back_through_the_live_api(client, tmp_path):  # noqa: F811
    alice = new_user()
    version = build_pipeline(client, alice)
    model_id = train(client, alice, version)
    created = deploy(client, alice, version, model_id)
    dep_id, key = created["deployment"]["id"], created["api_key"]

    frame = pd.read_csv(io.BytesIO(plant_csv()))
    write_csv(tmp_path / "live.csv", frame.drop(columns=["quality"]).iloc[:120])
    labs = frame.dropna(subset=["quality"]).iloc[5:15][["time", "quality"]].rename(columns={"time": "sample_time"})
    write_csv(tmp_path / "lab.csv", labs)

    conn_cfg = client.get(f"/deployments/{dep_id}/connector", headers=as_user(alice))
    import zipfile

    config = json.loads(zipfile.ZipFile(io.BytesIO(conn_cfg.content)).read("softsensor_connector/config.json"))
    config["source"] = {"type": "csv", "csv": {"path": str(tmp_path / "live.csv"), "follow": False, "batch_rows": 50}}
    config["lab"] = {"type": "csv", "csv": {"path": str(tmp_path / "lab.csv"), "time_column": "sample_time",
                                            "value_column": "quality"}}
    config["write_back"] = {"type": "csv", "csv_path": str(tmp_path / "predictions.csv")}

    api = sc.Client(config["server"], dep_id, key, transport=api_transport(client))
    assert api.info()["inputs"]
    source = sc.build_source(config)
    connector = sc.Connector(config, api, source, sc.CsvLabSource(config["lab"]["csv"]),
                             sc.WriteBack(config, source), sleep=lambda s: None)
    connector.run(interval=0)

    with (tmp_path / "predictions.csv").open() as fh:
        written = list(csv.DictReader(fh))
    assert len(written) == 120
    assert sum(1 for r in written if r["status"] in ("ok", "warning")) > 90
    mon = client.get(f"/deployments/{dep_id}/monitor", headers=as_user(alice)).json()
    assert len(mon["records"]) == 120
    assert len(mon["labs"]) == 10 and all(lab["corrected"] is not None for lab in mon["labs"])
    assert mon["deployment"]["bias"] != 0.0


def test_unsent_rows_wait_in_order_and_survive_a_restart(tmp_path):
    sent: list[list[dict]] = []
    state = {"down": True}

    def transport(method, url, headers, body, timeout):
        if state["down"]:
            raise ConnectionError("network unreachable")
        sent.append(body["rows"])
        return 200, {"predictions": [{"time": r["time"], "status": "ok", "prediction": 1.0, "corrected": 1.0}
                                     for r in body["rows"]]}, {}

    path = tmp_path / "feed.csv"
    path.write_text("time,feed\n")
    config = {"spool_path": str(tmp_path / "unsent.jsonl"), "write_back": {"type": "none"}, "source": {"type": "csv"}}
    source = sc.CsvSource({"path": str(path), "follow": True, "start_at_end": False}, {"feed": "feed"}, "time")
    waits: list[float] = []
    connector = sc.Connector(config, sc.Client("http://x", "d", "k", transport=transport), source, sleep=waits.append)

    with path.open("a") as fh:
        fh.write("2024-01-01 00:00:00,1.5\n2024-01-01 00:01:00,2.5\n2024-01-01 00:02:00,")  # last line unfinished
    connector.cycle()
    assert len(connector.queue) == 2 and waits == [5.0]
    restarted = sc.Connector(config, connector.client, source, sleep=waits.append)
    assert len(restarted.queue) == 2  # from the spool file

    with path.open("a") as fh:
        fh.write("3.5\n")
    state["down"] = False
    received = restarted.cycle()
    assert [r["time"] for r in received] == ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00"]
    assert sent == [[{"feed": 1.5, "time": "2024-01-01 00:00:00"}, {"feed": 2.5, "time": "2024-01-01 00:01:00"},
                     {"feed": 3.5, "time": "2024-01-01 00:02:00"}]]
    assert len(restarted.queue) == 0 and (tmp_path / "unsent.jsonl").read_text() == ""


def test_rejected_rows_are_skipped_and_a_bad_key_stops_the_connector(tmp_path):
    calls = {"n": 0}

    def transport(method, url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 422, {"detail": "bad rows"}, {}
        return 401, {"detail": "Invalid API key for this deployment."}, {}

    source = SimpleNamespace(read=lambda: [{"time": "2024-01-01", "feed": 1}], close=lambda: None)
    connector = sc.Connector({"source": {"type": "csv"}}, sc.Client("http://x", "d", "k", transport=transport),
                             source, sleep=lambda s: None)
    connector.cycle()
    assert len(connector.queue) == 0
    with pytest.raises(SystemExit):
        connector.cycle()


def test_busy_server_retry_after_is_respected():
    def transport(method, url, headers, body, timeout):
        return 429, {"detail": "Too many requests"}, {"Retry-After": "17"}

    waits: list[float] = []
    source = SimpleNamespace(read=lambda: [{"time": "2024-01-01", "feed": 1}], close=lambda: None)
    connector = sc.Connector({"source": {"type": "csv"}}, sc.Client("http://x", "d", "k", transport=transport),
                             source, sleep=waits.append)
    connector.cycle()
    assert waits == [17.0] and len(connector.queue) == 1


def test_startup_waits_for_an_unreachable_server_but_not_for_a_bad_key():
    replies = [ConnectionError("network down"), (503, {"detail": "starting"}, {}),
               (429, {"detail": "busy"}, {"Retry-After": "7"}), (200, {"name": "Column quality", "inputs": []}, {})]

    def transport(method, url, headers, body, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    waits: list[float] = []
    info = sc.fetch_info(sc.Client("http://x", "d", "k", transport=transport), keep_trying=True, sleep=waits.append)
    assert info["name"] == "Column quality" and waits == [5.0, 10.0, 7.0]

    refused = lambda *a: (401, {"detail": "Invalid API key for this deployment."}, {})  # noqa: E731
    assert sc.fetch_info(sc.Client("http://x", "d", "k", transport=refused), keep_trying=True, sleep=waits.append) is None
    down = lambda *a: (_ for _ in ()).throw(ConnectionError("network down"))  # noqa: E731
    assert sc.fetch_info(sc.Client("http://x", "d", "k", transport=down), keep_trying=False, sleep=waits.append) is None
    assert waits == [5.0, 10.0, 7.0]


class FakeNode:
    def __init__(self, store, node_id):
        self.store, self.node_id = store, node_id

    def read_data_value(self):
        value, good, stamp = self.store[self.node_id]
        return SimpleNamespace(Value=SimpleNamespace(Value=value), StatusCode=SimpleNamespace(is_good=lambda: good),
                               SourceTimestamp=stamp, ServerTimestamp=None)


class FakeOpcClient:
    def __init__(self, store):
        self.store = store

    def get_node(self, node_id):
        return FakeNode(self.store, node_id)

    def disconnect(self):
        pass


def test_opcua_source_reads_nodes_and_drops_bad_quality():
    t1 = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 5, 1, 8, 0, 5)
    store = {"ns=2;s=FIC101.PV": (51.2, True, t1), "ns=2;s=TI202.PV": (300.4, False, t2)}
    source = sc.OpcUaSource({"nodes": {"feed": "ns=2;s=FIC101.PV", "temp": "ns=2;s=TI202.PV"}},
                            {"feed": "feed", "temp": "temp"}, "time", client_factory=lambda: FakeOpcClient(store))
    [row] = source.read()
    assert row == {"feed": 51.2, "temp": None, "time": "2024-05-01T08:00:05+00:00"}
    with pytest.raises(ValueError, match="no node id"):
        sc.OpcUaSource({"nodes": {}}, {"feed": "feed"}, "time", client_factory=lambda: None)


def test_mqtt_source_holds_the_latest_value_per_topic():
    published = []
    fake = SimpleNamespace(publish=lambda topic, payload, qos, retain: published.append((topic, json.loads(payload), retain)),
                           loop_stop=lambda: None, disconnect=lambda: None)
    source = sc.MqttSource({"topics": {"feed": "plant/feed", "temp": "plant/temp"}, "value_key": "v", "time_key": "ts"},
                           {"feed": "feed", "temp": "temp"}, "time", client=fake)
    assert source.read() == []
    source.on_message("plant/feed", b'{"v": 50.5, "ts": "2024-05-01T08:00:00Z"}')
    source.on_message("plant/temp", b"301.25")
    source.on_message("plant/other", b"1")
    source.on_message("plant/feed", b'{"v": "51.0", "ts": "2024-05-01T08:01:00Z"}')
    [row] = source.read()
    assert row == {"feed": 51.0, "temp": 301.25, "time": "2024-05-01T08:01:00Z"}
    source.publish("plant/softsensor/quality", {"time": "t", "value": 1.0, "status": "ok"})
    assert published == [("plant/softsensor/quality", {"time": "t", "value": 1.0, "status": "ok"}, True)]


def test_pi_web_api_source_reads_a_stream_set_and_writes_back():
    calls = []

    def transport(method, url, params=None, body=None):
        calls.append((method, url, params, body))
        if method == "GET":
            return {"Items": [
                {"WebId": "W1", "Value": {"Timestamp": "2024-05-01T08:00:00Z", "Value": 50.5, "Good": True}},
                {"WebId": "W2", "Value": {"Timestamp": "2024-05-01T08:00:30Z", "Value": {"Name": "I/O Timeout", "IsSystem": True}, "Good": False}},
                {"WebId": "W3", "Value": {"Timestamp": "2024-05-01T07:59:00Z", "Value": {"Name": "B", "Value": 2}, "Good": True}},
            ]}
        return {}

    source = sc.PiWebSource({"base_url": "https://pi/piwebapi/", "web_ids": {"feed": "W1", "temp": "W2", "grade": "W3"}},
                            {"feed": "feed", "temp": "temp", "grade": "grade"}, "time", transport=transport)
    [row] = source.read()
    assert row == {"feed": 50.5, "temp": None, "grade": "B", "time": "2024-05-01T08:00:30Z"}
    assert calls[0][1] == "https://pi/piwebapi/streamsets/value"
    assert calls[0][2] == [("webId", "W1"), ("webId", "W2"), ("webId", "W3")]

    config = {"source": {"type": "piweb"}, "write_back": {"type": "piweb", "piweb_web_id": "OUT"}}
    sc.WriteBack(config, source)([{"time": "2024-05-01T08:01:00Z", "corrected": 12.5, "status": "ok"}])
    assert calls[-1] == ("POST", "https://pi/piwebapi/streams/OUT/value", None,
                         {"Timestamp": "2024-05-01T08:01:00Z", "Value": 12.5})
    with pytest.raises(ValueError, match="needs source.type"):
        sc.WriteBack({"source": {"type": "csv"}, "write_back": {"type": "opcua"}}, source)


def test_values_are_sent_as_numbers_when_they_are_numbers():
    assert [sc.to_number(v) for v in ["1.5", " 2 ", "", "abc", None, float("nan"), 3, True]] == [
        1.5, 2.0, None, "abc", None, None, 3, True,
    ]
