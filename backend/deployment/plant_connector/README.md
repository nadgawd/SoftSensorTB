# Soft sensor plant connector

This script sends live readings from the plant to your deployed soft sensor and gets the prediction back.

```
plant sensors → PLC / DCS → historian or OPC UA / MQTT
                                   │  (read, every cycle)
                         softsensor_connector.py   ← runs on a plant-side server
                                   │  HTTPS, outbound only
                         Soft Sensor Toolbox deployment → prediction (+ write-back)
```

The connector only makes outbound connections, so you don't need to open any firewall port toward the plant. That is how plant networks are normally allowed to talk to outside services.

## Setup

1. Install Python 3.9 or newer, then run `pip install -r requirements.txt`. Uncomment `asyncua` for OPC UA or `paho-mqtt` for MQTT.
2. Put the deployment's API key in an environment variable. The key is shown once, when you deploy or rotate it.
   - Linux/macOS: `export SOFTSENSOR_API_KEY=ssk_...`
   - Windows: `set SOFTSENSOR_API_KEY=ssk_...`
3. Edit `config.json`:
   - `inputs` maps each model input to its source column or tag. The sources' sections key their node ids, topics and WebIds by the model input name.
   - Set `source.type` to `csv`, `opcua`, `mqtt` or `piweb`, and fill in that section.
   - `interval_seconds` sets how often to read and predict. Match it to the data the model was trained on; 60 s is typical.
4. Check the key and the settings: `python softsensor_connector.py --config config.json --check`
5. Run it: `python softsensor_connector.py --config config.json`. Use a service manager such as systemd or NSSM so it restarts with the server.

## Sources

| `source.type` | What it reads |
|---|---|
| `csv` | A CSV file. With `follow: true` it sends each new line a historian export appends. With `follow: false` it replays the file in batches of `batch_rows`, which is useful for a trial. |
| `opcua` | The current value of each node in `nodes`, for example `ns=2;s=FIC101.PV`. Bad-quality values are sent as missing. Uses `username`, `password_env` and `security_string` if set. |
| `mqtt` | The latest message on each topic in `topics`. A payload can be a bare number or JSON (`value_key`, `time_key`). Values older than `max_age_seconds` are sent as missing. Sparkplug B (protobuf) payloads are not decoded; bridge them to JSON first. |
| `piweb` | The current values of PI points through the AVEVA PI Web API, with one `streamsets/value` request per cycle using the points' `web_ids`. Basic authentication uses `username` and `password_env`. For Kerberos, put a proxy in front. |

Readings must be in the same units and the same clock as the data the model was trained on.

## Lab results

Set `lab.type` to `csv` and point `lab.csv.path` to a file where the lab system (LIMS) appends `sample_time,value` lines. Each new result is matched to the prediction made at its sample time. It then corrects the model's bias, using the bias gain set in the app.

## Writing the prediction back

`write_back.type` can be:

- `none`
- `csv`: appends every prediction to `csv_path`.
- `opcua`: writes the corrected prediction to `opcua_node`. Needs `source.type` set to `opcua`.
- `mqtt`: publishes `{time, value, status}` to `mqtt_topic` as a retained message. Needs `source.type` set to `mqtt`.
- `piweb`: writes to the PI point `piweb_web_id`. Needs `source.type` set to `piweb`.

Write-back is usually pointed at a separate soft-sensor tag that the DCS operators can trend.

## Prediction status

Every prediction comes back with a status:

- `ok`
- `warning`: there is a prediction, but something looks off. The inputs may be outside the training range, a sensor may look stuck, or the plant may be outside the operating region the model was trained on (T² / SPE).
- `suppressed`: no prediction, for example during a shutdown that training also excluded, or with a missing input.
- `ignored`: a row that is not newer than the last one sent.

The reasons are in the log and in the app's Deployment step.

## When the network or server is down

Rows that cannot be sent are kept, up to 100,000, and sent in order once the server is reachable again. Set `spool_path`, for example to `"unsent.jsonl"`, to keep them across restarts of the connector.

If the server cannot be reached when the connector starts, it waits and tries again, up to every 5 minutes, rather than exiting. A refused API key or an unknown deployment stops it straight away. `--check` and `--once` never wait.
