#!/usr/bin/env python3
"""Soft Sensor Toolbox plant connector.

Runs next to the plant (on a server that can reach the historian or the OPC UA
/ MQTT network) and pushes readings out to a deployed soft sensor over HTTPS.
Only outbound connections are made, so no firewall port has to be opened
toward the plant.

Every cycle it:
  1. reads the latest values of the model's inputs from the configured source
     (a CSV export, an OPC UA server, an MQTT broker or the AVEVA PI Web API);
  2. sends them to the deployment and receives the prediction;
  3. optionally writes the prediction back (OPC UA node, MQTT topic, PI tag or CSV);
  4. optionally sends new lab results, which correct the prediction's bias.

Rows that cannot be sent (network down, server busy) are kept and sent later,
in order, so no reading is lost; with ``spool_path`` set they also survive a
restart of the connector.

Usage:
    export SOFTSENSOR_API_KEY=ssk_...
    python softsensor_connector.py --config config.json            # run forever
    python softsensor_connector.py --config config.json --once     # one cycle
    python softsensor_connector.py --config config.json --check    # test the key and settings

Only ``requests`` is required. ``asyncua`` is needed for OPC UA and
``paho-mqtt`` for MQTT (see requirements.txt).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

log = logging.getLogger("softsensor_connector")

MAX_ROWS_PER_REQUEST = 1000
MAX_QUEUE_ROWS = 100_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_number(value: Any) -> Any:
    """Numbers stay numbers, numeric text becomes a number, anything else is sent as it is."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            number = float(text)
        except ValueError:
            return text
        return number if math.isfinite(number) else None
    return value


# --------------------------------------------------------------------------- HTTP


class ApiError(Exception):
    def __init__(self, status: int, detail: str, retry_after: Optional[float] = None) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


Transport = Callable[[str, str, Dict[str, str], Optional[dict], float], "tuple[int, dict, Dict[str, str]]"]


def requests_transport(method: str, url: str, headers: Dict[str, str], body: Optional[dict], timeout: float):
    import requests

    resp = requests.request(method, url, headers=headers, json=body, timeout=timeout)
    try:
        payload = resp.json()
    except ValueError:
        payload = {"detail": resp.text[:500]}
    return resp.status_code, payload, dict(resp.headers)


class Client:
    """The deployment's live API."""

    def __init__(self, server: str, deployment_id: str, api_key: str, transport: Transport = requests_transport,
                 timeout: float = 30.0) -> None:
        self.base = f"{server.rstrip('/')}/live/{deployment_id}"
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.transport = transport
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        status, payload, headers = self.transport(method, self.base + path, self.headers, body, self.timeout)
        if status >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
            retry = headers.get("Retry-After") or headers.get("retry-after")
            raise ApiError(status, str(detail), float(retry) if retry else None)
        return payload

    def info(self) -> dict:
        return self._call("GET", "")

    def predict(self, rows: List[dict]) -> List[dict]:
        return self._call("POST", "/predict", {"rows": rows})["predictions"]

    def lab(self, samples: List[dict]) -> dict:
        return self._call("POST", "/lab", {"samples": samples})


def fetch_info(client: Client, keep_trying: bool, sleep: Callable[[float], None] = time.sleep) -> Optional[dict]:
    """The deployment's description, or None when the key or the deployment is refused.

    With ``keep_trying`` an unreachable or busy server (network down, the API
    still starting) is retried with a growing delay instead of giving up.
    """
    delay = 5.0
    while True:
        try:
            return client.info()
        except ApiError as exc:
            if exc.status in (401, 403, 404) or not keep_trying:
                log.error("The server refused the request: %s", exc)
                return None
            wait, reason = exc.retry_after or delay, str(exc)
        except Exception as exc:  # noqa: BLE001 - network errors
            if not keep_trying:
                log.error("Could not reach the server: %s", exc)
                return None
            wait, reason = delay, f"could not reach the server ({exc})"
        log.warning("Deployment check failed: %s - retrying in %.0f s", reason, wait)
        sleep(wait)
        delay = min(delay * 2, 300.0)


# --------------------------------------------------------------------------- sources


class Source:
    """Yields rows: {time column: ISO time, input: value, ...}."""

    def read(self) -> List[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CsvSource(Source):
    """A CSV file: replayed in batches, or followed as a historian appends to it (``follow``)."""

    def __init__(self, cfg: dict, inputs: Dict[str, str], time_column: Optional[str]) -> None:
        self.path = Path(cfg["path"])
        self.follow = bool(cfg.get("follow", True))
        self.batch_rows = int(cfg.get("batch_rows", 1 if self.follow else 60))
        self.source_time = cfg.get("time_column") or time_column
        self.time_column = time_column or "time"
        self.inputs = inputs
        self.delimiter = cfg.get("delimiter", ",")
        self.offset = 0
        self.header: Optional[List[str]] = None
        self.finished = False
        if self.follow and cfg.get("start_at_end", True) and self.path.exists():
            self._read_header()
            self.offset = self.path.stat().st_size

    def _read_header(self) -> None:
        with self.path.open("r", newline="", encoding="utf-8-sig") as fh:
            first = fh.readline()
            self.header = next(csv.reader([first], delimiter=self.delimiter))
            if self.offset == 0:
                self.offset = len(first.encode("utf-8")) + (3 if self._has_bom() else 0)

    def _has_bom(self) -> bool:
        with self.path.open("rb") as fh:
            return fh.read(3) == b"\xef\xbb\xbf"

    def read(self) -> List[dict]:
        if not self.path.exists():
            log.warning("CSV source %s does not exist yet", self.path)
            return []
        if self.header is None:
            self._read_header()
        if self.path.stat().st_size < self.offset:
            log.warning("CSV source %s was truncated or replaced; reading it from the start", self.path)
            self.offset = 0
            self.header = None
            self._read_header()
        rows: List[dict] = []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            while len(rows) < self.batch_rows or self.follow:
                line = fh.readline()
                if not line or not line.endswith(b"\n"):
                    break  # nothing new, or a line the writer has not finished
                self.offset += len(line)
                text = line.decode("utf-8").rstrip("\r\n")
                if not text.strip():
                    continue
                values = next(csv.reader([text], delimiter=self.delimiter))
                record = dict(zip(self.header or [], values))
                rows.append(self._row(record))
                if not self.follow and len(rows) >= self.batch_rows:
                    break
        if not rows and not self.follow:
            self.finished = True
        return rows

    def _row(self, record: Dict[str, str]) -> dict:
        row = {name: to_number(record.get(column)) for name, column in self.inputs.items()}
        stamp = record.get(self.source_time) if self.source_time else None
        row[self.time_column] = stamp if stamp else utc_now()
        return row


class OpcUaSource(Source):
    """Reads the configured nodes of an OPC UA server once per cycle; bad-quality values are sent as missing."""

    def __init__(self, cfg: dict, inputs: Dict[str, str], time_column: Optional[str], client_factory=None) -> None:
        self.nodes = {name: cfg["nodes"][name] for name in inputs if name in cfg.get("nodes", {})}
        missing = [name for name in inputs if name not in self.nodes]
        if missing:
            raise ValueError(f"opcua.nodes has no node id for: {missing}")
        self.time_column = time_column or "time"
        self.cfg = cfg
        self.client_factory = client_factory or self._default_client
        self.client = None

    def _default_client(self):
        from asyncua.sync import Client as OpcClient  # pip install asyncua

        client = OpcClient(self.cfg["endpoint"])
        if self.cfg.get("username"):
            client.set_user(self.cfg["username"])
            client.set_password(os.environ.get(self.cfg.get("password_env") or "", ""))
        if self.cfg.get("security_string"):
            client.set_security_string(self.cfg["security_string"])
        client.connect()
        return client

    def _connected(self):
        if self.client is None:
            self.client = self.client_factory()
        return self.client

    def read(self) -> List[dict]:
        try:
            client = self._connected()
            names = list(self.nodes)
            row: dict = {}
            stamps = []
            for name in names:
                data_value = client.get_node(self.nodes[name]).read_data_value()
                good = getattr(getattr(data_value, "StatusCode", None), "is_good", lambda: True)()
                value = getattr(getattr(data_value, "Value", None), "Value", None)
                row[name] = to_number(value) if good else None
                stamp = getattr(data_value, "SourceTimestamp", None) or getattr(data_value, "ServerTimestamp", None)
                if stamp is not None:
                    stamps.append(stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc))
        except Exception:
            self.close()
            raise
        row[self.time_column] = max(stamps).isoformat() if stamps else utc_now()
        return [row]

    def write(self, node_id: str, value: float) -> None:
        from asyncua import ua

        node = self._connected().get_node(node_id)
        node.write_value(ua.DataValue(ua.Variant(float(value), ua.VariantType.Double)))

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self.client = None


class MqttSource(Source):
    """Keeps the latest value of each subscribed topic and sends a snapshot every cycle (sample and hold)."""

    def __init__(self, cfg: dict, inputs: Dict[str, str], time_column: Optional[str], client=None) -> None:
        topics = cfg.get("topics") or {}
        missing = [name for name in inputs if name not in topics]
        if missing:
            raise ValueError(f"mqtt.topics has no topic for: {missing}")
        self.topics = {topics[name]: name for name in inputs}
        self.value_key = cfg.get("value_key")
        self.time_key = cfg.get("time_key")
        self.max_age = float(cfg.get("max_age_seconds", 0) or 0)
        self.time_column = time_column or "time"
        self.latest: Dict[str, tuple] = {}
        self.lock = threading.Lock()
        self.cfg = cfg
        self.client = client if client is not None else self._default_client()

    def _default_client(self):
        import paho.mqtt.client as mqtt  # pip install paho-mqtt

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.cfg.get("client_id", ""))
        except AttributeError:  # paho-mqtt < 2
            client = mqtt.Client(client_id=self.cfg.get("client_id", ""))
        if self.cfg.get("username"):
            client.username_pw_set(self.cfg["username"], os.environ.get(self.cfg.get("password_env") or "", ""))
        if self.cfg.get("tls"):
            client.tls_set()
        client.on_message = lambda _c, _u, msg: self.on_message(msg.topic, msg.payload)
        client.on_connect = lambda c, *_a, **_k: [c.subscribe(t, qos=int(self.cfg.get("qos", 1))) for t in self.topics]
        client.connect(self.cfg["host"], int(self.cfg.get("port", 1883)))
        client.loop_start()
        return client

    def on_message(self, topic: str, payload: bytes) -> None:
        name = self.topics.get(topic)
        if name is None:
            return
        text = payload.decode("utf-8", errors="replace").strip()
        value: Any = text
        stamp = None
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            value = parsed.get(self.value_key or "value")
            if self.time_key:
                stamp = parsed.get(self.time_key)
        elif parsed is not None:
            value = parsed
        with self.lock:
            self.latest[name] = (to_number(value), stamp, time.time())

    def read(self) -> List[dict]:
        now = time.time()
        with self.lock:
            if not self.latest:
                return []
            row: dict = {}
            for name in self.topics.values():
                value, _, seen = self.latest.get(name, (None, None, 0.0))
                row[name] = None if self.max_age and now - seen > self.max_age else value
            stamps = [s for _, s, _ in self.latest.values() if s]
        row[self.time_column] = max(stamps) if stamps else utc_now()
        return [row]

    def publish(self, topic: str, payload: dict) -> None:
        self.client.publish(topic, json.dumps(payload), qos=int(self.cfg.get("qos", 1)), retain=True)

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass


class PiWebSource(Source):
    """Current values of PI points through the PI Web API (one ad-hoc stream-set request per cycle)."""

    def __init__(self, cfg: dict, inputs: Dict[str, str], time_column: Optional[str], transport=None) -> None:
        web_ids = cfg.get("web_ids") or {}
        missing = [name for name in inputs if name not in web_ids]
        if missing:
            raise ValueError(f"piweb.web_ids has no WebId for: {missing}")
        self.web_ids = {name: web_ids[name] for name in inputs}
        self.base = cfg["base_url"].rstrip("/")
        self.time_column = time_column or "time"
        self.cfg = cfg
        self.transport = transport or self._default_transport

    def _auth(self):
        if self.cfg.get("username"):
            return self.cfg["username"], os.environ.get(self.cfg.get("password_env") or "", "")
        return None

    def _default_transport(self, method: str, url: str, params=None, body=None) -> dict:
        import requests

        resp = requests.request(method, url, params=params, json=body, auth=self._auth(),
                                verify=bool(self.cfg.get("verify_tls", True)), timeout=30,
                                headers={"X-Requested-With": "XMLHttpRequest"})
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def read(self) -> List[dict]:
        params = [("webId", web_id) for web_id in self.web_ids.values()]
        payload = self.transport("GET", f"{self.base}/streamsets/value", params=params)
        by_id = {item.get("WebId"): item.get("Value") or {} for item in payload.get("Items", [])}
        row: dict = {}
        stamps = []
        for name, web_id in self.web_ids.items():
            value = by_id.get(web_id) or {}
            raw = value.get("Value")
            if isinstance(raw, dict):  # a digital state or an error object
                raw = raw.get("Name") if "Name" in raw and not raw.get("IsSystem") else None
            row[name] = to_number(raw) if value.get("Good", True) else None
            if value.get("Timestamp"):
                stamps.append(value["Timestamp"])
        row[self.time_column] = max(stamps) if stamps else utc_now()
        return [row]

    def write(self, web_id: str, stamp: str, value: float) -> None:
        self.transport("POST", f"{self.base}/streams/{web_id}/value", body={"Timestamp": stamp, "Value": value})


def build_source(config: dict) -> Source:
    source = config["source"]
    kind = source["type"]
    inputs: Dict[str, str] = config["inputs"]
    time_column = config.get("time_column")
    if kind == "csv":
        return CsvSource(source["csv"], inputs, time_column)
    if kind == "opcua":
        return OpcUaSource(source["opcua"], inputs, time_column)
    if kind == "mqtt":
        return MqttSource(source["mqtt"], inputs, time_column)
    if kind == "piweb":
        return PiWebSource(source["piweb"], inputs, time_column)
    raise ValueError(f"Unknown source type {kind!r}; use csv, opcua, mqtt or piweb.")


# --------------------------------------------------------------------------- lab results


class CsvLabSource:
    """Lab results appended to a CSV file (sample time, measured value)."""

    def __init__(self, cfg: dict) -> None:
        self.reader = CsvSource(
            {"path": cfg["path"], "follow": True, "time_column": cfg.get("time_column", "sample_time"),
             "start_at_end": cfg.get("start_at_end", False), "delimiter": cfg.get("delimiter", ",")},
            {"value": cfg.get("value_column", "value")}, "time",
        )

    def read(self) -> List[dict]:
        return [{"time": r["time"], "value": r["value"]} for r in self.reader.read() if r.get("value") is not None]


# --------------------------------------------------------------------------- write-back


class WriteBack:
    def __init__(self, config: dict, source: Source) -> None:
        self.cfg = config.get("write_back") or {"type": "none"}
        self.kind = self.cfg.get("type", "none")
        self.source = source
        if self.kind in ("opcua", "mqtt", "piweb") and self.kind != config["source"]["type"]:
            raise ValueError(f"write_back.type {self.kind!r} needs source.type {self.kind!r} (the same connection).")

    def __call__(self, predictions: List[dict]) -> None:
        if self.kind == "none":
            return
        if self.kind == "csv":
            path = Path(self.cfg["csv_path"])
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if new:
                    writer.writerow(["time", "prediction", "corrected", "status", "reasons"])
                for p in predictions:
                    if p.get("status") != "ignored":
                        writer.writerow([p["time"], p.get("prediction"), p.get("corrected"), p["status"],
                                         "; ".join(p.get("reasons") or [])])
            return
        latest = next((p for p in reversed(predictions) if p.get("corrected") is not None), None)
        if latest is None:
            return
        if self.kind == "opcua":
            self.source.write(self.cfg["opcua_node"], latest["corrected"])
        elif self.kind == "mqtt":
            self.source.publish(self.cfg["mqtt_topic"], {
                "time": latest["time"], "value": latest["corrected"], "status": latest["status"],
            })
        elif self.kind == "piweb":
            self.source.write(self.cfg["piweb_web_id"], latest["time"], latest["corrected"])
        else:
            raise ValueError(f"Unknown write_back.type {self.kind!r}")


# --------------------------------------------------------------------------- the loop


class Queue:
    """Rows waiting to be sent, oldest first; optionally mirrored to a JSON-lines spool file."""

    def __init__(self, spool_path: Optional[str]) -> None:
        self.rows: Deque[dict] = deque()
        self.spool = Path(spool_path) if spool_path else None
        if self.spool and self.spool.exists():
            with self.spool.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self.rows.append(json.loads(line))
            log.info("Loaded %d unsent rows from %s", len(self.rows), self.spool)

    def extend(self, rows: List[dict]) -> None:
        self.rows.extend(rows)
        dropped = 0
        while len(self.rows) > MAX_QUEUE_ROWS:
            self.rows.popleft()
            dropped += 1
        if dropped:
            log.error("Queue full: dropped the %d oldest unsent rows", dropped)
        self._save()

    def take(self, n: int) -> List[dict]:
        return [self.rows[i] for i in range(min(n, len(self.rows)))]

    def done(self, n: int) -> None:
        for _ in range(min(n, len(self.rows))):
            self.rows.popleft()
        self._save()

    def _save(self) -> None:
        if not self.spool:
            return
        tmp = self.spool.with_suffix(self.spool.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(self.spool)

    def __len__(self) -> int:
        return len(self.rows)


class Connector:
    def __init__(self, config: dict, client: Client, source: Source, lab_source=None, write_back=None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.client = client
        self.source = source
        self.lab_source = lab_source
        self.write_back = write_back or (lambda predictions: None)
        self.queue = Queue(config.get("spool_path"))
        self.lab_queue: Deque[dict] = deque()
        self.sleep = sleep
        self.backoff = 0.0
        self.stopped = False

    def cycle(self) -> List[dict]:
        """Read, send everything waiting (oldest first), write back; returns the predictions received."""
        try:
            rows = self.source.read()
        except Exception as exc:  # noqa: BLE001 - a source outage must not stop the connector
            log.error("Reading the source failed: %s", exc)
            rows = []
        if rows:
            self.queue.extend(rows)
        received: List[dict] = []
        while len(self.queue):
            batch = self.queue.take(MAX_ROWS_PER_REQUEST)
            try:
                predictions = self.client.predict(batch)
            except ApiError as exc:
                if exc.status in (400, 413, 422):
                    log.error("The server rejected %d rows (%s); skipping them", len(batch), exc.detail)
                    self.queue.done(len(batch))
                    continue
                if exc.status == 401:
                    raise SystemExit(f"Invalid API key: {exc.detail}")
                self._wait(exc.retry_after, f"{exc} - keeping {len(self.queue)} rows to send later")
                break
            except Exception as exc:  # noqa: BLE001 - network errors: keep the rows, try later
                self._wait(None, f"Could not reach the server ({exc}) - keeping {len(self.queue)} rows to send later")
                break
            self.backoff = 0.0
            self.queue.done(len(batch))
            received.extend(predictions)
            for p in predictions:
                if p.get("status") != "ok":
                    log.info("%s %s: %s", p.get("time"), p.get("status"), "; ".join(p.get("reasons") or []))
            try:
                self.write_back(predictions)
            except Exception as exc:  # noqa: BLE001
                log.error("Writing the prediction back failed: %s", exc)
        self._send_labs()
        if received:
            last = next((p for p in reversed(received) if p.get("corrected") is not None), None)
            if last:
                log.info("%s prediction %.6g (status %s)", last["time"], last["corrected"], last["status"])
        return received

    def _send_labs(self) -> None:
        if self.lab_source is None:
            return
        try:
            self.lab_queue.extend(self.lab_source.read())
        except Exception as exc:  # noqa: BLE001
            log.error("Reading lab results failed: %s", exc)
        if not self.lab_queue:
            return
        batch = list(self.lab_queue)[:MAX_ROWS_PER_REQUEST]
        try:
            result = self.client.lab(batch)
        except ApiError as exc:
            if exc.status in (400, 422):
                log.error("The server rejected %d lab results (%s); skipping them", len(batch), exc.detail)
                for _ in batch:
                    self.lab_queue.popleft()
            else:
                log.warning("Lab results not sent (%s); will retry", exc)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("Lab results not sent (%s); will retry", exc)
            return
        for _ in batch:
            self.lab_queue.popleft()
        # A sample newer than every reading sent so far is matched once its reading has been predicted.
        early = [
            sample for sample, outcome in zip(batch, result.get("results", []))
            if outcome.get("status") == "early"
        ] if len(result.get("results", [])) == len(batch) else []
        self.lab_queue.extendleft(reversed(early))
        log.info("Sent %d lab results (%d to resend later); bias is now %.6g",
                 len(batch) - len(early), len(early), result.get("bias", 0.0))

    def _wait(self, retry_after: Optional[float], message: str) -> None:
        self.backoff = min(max(self.backoff * 2, 5.0), 300.0)
        delay = retry_after if retry_after else self.backoff
        log.warning("%s (retrying in %.0f s)", message, delay)
        self.sleep(delay)

    def run(self, interval: float) -> None:
        while not self.stopped:
            started = time.monotonic()
            self.cycle()
            if isinstance(self.source, CsvSource) and self.source.finished and not len(self.queue):
                log.info("Reached the end of %s", self.source.path)
                return
            self.sleep(max(0.0, interval - (time.monotonic() - started)))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    for key in ("server", "deployment_id", "inputs", "source"):
        if key not in config:
            raise ValueError(f"config.json is missing {key!r}")
    return config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Send live plant data to a deployed soft sensor.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--check", action="store_true", help="check the key and the deployment, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    api_key = os.environ.get(config.get("api_key_env") or "SOFTSENSOR_API_KEY")
    if not api_key:
        log.error("Set the API key in the %s environment variable.", config.get("api_key_env") or "SOFTSENSOR_API_KEY")
        return 2
    client = Client(config["server"], config["deployment_id"], api_key)
    info = fetch_info(client, keep_trying=not (args.check or args.once))
    if info is None:
        return 1
    expected = [i["name"] for i in info.get("inputs", []) if not i.get("is_time")]
    unmapped = [name for name in expected if name not in config["inputs"]]
    log.info("Deployment %r (%s); inputs: %s", info.get("name"), info.get("status"), ", ".join(expected))
    if unmapped:
        log.error("config.inputs has no source for: %s", ", ".join(unmapped))
        return 2
    if args.check:
        return 0

    source = build_source(config)
    lab_cfg = config.get("lab") or {}
    lab_source = CsvLabSource(lab_cfg["csv"]) if lab_cfg.get("type") == "csv" else None
    connector = Connector(config, client, source, lab_source, WriteBack(config, source))

    def stop(*_args):
        connector.stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.once:
            connector.cycle()
        else:
            connector.run(float(config.get("interval_seconds", 60)))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
