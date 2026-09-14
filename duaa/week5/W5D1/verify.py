#!/usr/bin/env python3
"""Check W5D1 artifacts and execute the dashboard's actual Prometheus queries."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fields(text, names, context):
    result = {}
    for name in names:
        hits = re.findall(r"^" + re.escape(name) + r":[ \t]*(\S[^\n]*)$", text, re.M)
        require(len(hits) == 1, f"{context}: fill '{name}:' once on its own line")
        value = hits[0].strip()
        require(value.lower() not in {"tbd", "todo", "...", "n/a"},
                f"{context}: '{name}' is unfinished")
        result[name] = value
    return result


def read_targets(path):
    text = path.read_text()
    require(not re.search(r"<[A-Za-z][^<>\n]*>", text),
            "my-slo-targets.md still contains placeholders")
    parts = re.split(r"^## SLI\s+\d+\s*$", text, flags=re.M)
    require(2 <= len(parts) - 1 <= 3, "complete two or three '## SLI N' sections")
    fields(parts[0], ["Team", "Use case", "Service measured", "Workload",
                     "Measurement period", "Instrumentation gaps"], "targets header")
    slis = []
    for i, part in enumerate(parts[1:], 1):
        part = re.split(r"^## ", part, flags=re.M)[0]
        item = fields(part, ["Indicator", "Panel", "Unit", "Target", "Window",
                             "Observed", "Evidence", "Why it fits", "Limitations"],
                      f"SLI {i}")
        for key in ("Target", "Observed"):
            require(re.search(r"(?<![\w.])[-+]?\d+(?:\.\d+)?", item[key]),
                    f"SLI {i}: {key} needs a numeric value")
        require(re.search(r"\d+(?:\.\d+)?\s*(?:ms|s|min|h|d|w|second|minute|hour|day|week)",
                          item["Window"], re.I),
                f"SLI {i}: Window needs a duration, such as '5 minutes'")
        slis.append(item)
    require(len({s["Panel"] for s in slis}) == len(slis),
            "each SLI must name its own stat panel")
    return slis


def panels_in(items):
    for panel in items:
        if panel.get("type") != "row":
            yield panel
        yield from panels_in(panel.get("panels", []))


def signature(dashboard):
    keys = ("id", "type", "title", "description", "datasource", "targets",
            "fieldConfig", "options", "transformations")
    return {"title": dashboard.get("title"), "panels": [
        {k: p.get(k) for k in keys} for p in panels_in(dashboard.get("panels", []))]}


class Grafana:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def get(self, path):
        headers = {}
        if os.environ.get("GRAFANA_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["GRAFANA_TOKEN"]
        req = urllib.request.Request(self.url + path, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"Grafana HTTP {exc.code}; check access and the data source") from None
        except (urllib.error.URLError, TimeoutError):
            raise ValueError("Grafana or its data source did not answer; check the connection") from None


def verify(url, directory):
    slis = read_targets(directory / "my-slo-targets.md")
    exported = json.loads((directory / "my-dashboard.json").read_text())
    require(not exported.get("__inputs"), "export JSON with external sharing switched off")
    exported = exported.get("dashboard", exported)
    client = Grafana(url)
    require(client.get("/api/health").get("database") == "ok", "Grafana database is not healthy")
    found = client.get("/api/search?query=Team%20service&type=dash-db")
    found = [d for d in found if d.get("title") == "Team service"]
    require(len(found) == 1, "save exactly one dashboard titled 'Team service'")
    uid = urllib.parse.quote(found[0]["uid"], safe="")
    dashboard = client.get("/api/dashboards/uid/" + uid)["dashboard"]
    require(signature(exported) == signature(dashboard),
            "my-dashboard.json differs from the saved panels; save and export again")
    panels = list(panels_in(dashboard.get("panels", [])))
    data_panels = [p for p in panels if p.get("type") != "text"]
    stats = [p for p in data_panels if p.get("type") == "stat"]
    require(any(p.get("type") == "timeseries" for p in data_panels),
            "add at least one time-series panel")
    for sli in slis:
        require(sum(p.get("title") == sli["Panel"] for p in stats) == 1,
                f"SLI '{sli['Indicator']}': stat panel '{sli['Panel']}' must exist once")
    sources = client.get("/api/datasources")
    queried = set()
    now = time.time()
    for panel in data_panels:
        name = panel.get("title", "")
        require(name.strip(), "a data panel has no title")
        require((panel.get("description") or "").strip(), f"{name}: add a description")
        unit = panel.get("fieldConfig", {}).get("defaults", {}).get("unit")
        require(unit, f"{name}: select a unit; use 'none' explicitly for a count")
        targets = [t for t in panel.get("targets", []) if not t.get("hide")]
        require(targets, f"{name}: no visible query")
        for target in targets:
            expr = (target.get("expr") or "").strip()
            require(expr, f"{name}: use a Prometheus query directly for this lab")
            require(not re.search(r"\$(?:[A-Za-z_]|\{)", expr),
                    f"{name}: use fixed query windows and labels for this lab")
            source = target.get("datasource") or panel.get("datasource")
            if isinstance(source, dict):
                matches = [s for s in sources if s.get("uid") == source.get("uid")]
            elif isinstance(source, str):
                matches = [s for s in sources if source in (s.get("uid"), s.get("name"))]
            else:
                matches = [s for s in sources if s.get("isDefault")]
            require(len(matches) == 1 and matches[0].get("type") == "prometheus",
                    f"{name}: choose your Prometheus data source")
            source_uid = matches[0]["uid"]
            query_kind = "query_range" if panel.get("type") == "timeseries" else "query"
            key = (source_uid, expr, query_kind)
            if key in queried:
                continue
            params = {"query": expr}
            if query_kind == "query_range":
                params.update(start=now - 1800, end=now, step=15)
            else:
                params["time"] = now
            path = ("/api/datasources/proxy/uid/" + urllib.parse.quote(source_uid, safe="")
                    + "/api/v1/" + query_kind + "?" + urllib.parse.urlencode(params))
            response = client.get(path)
            require(response.get("status") == "success", f"{name}: Prometheus rejected its query")
            data = response.get("data", {})
            result = data.get("result", [])
            require(result, f"{name}: query is empty; check spelling, labels and scrape")
            if data.get("resultType") == "scalar":
                values = [[result]]
            elif data.get("resultType") == "matrix":
                values = [s.get("values", []) for s in result]
            else:
                values = [[s["value"]] if "value" in s else [] for s in result]
            require(all(any(math.isfinite(float(v[1])) for v in series) for series in values),
                    f"{name}: query has no finite samples; generate traffic and wait for scrapes")
            queried.add(key)
    print(f"Checked {len(slis)} indicators, {len(data_panels)} data panels and {len(queried)} queries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        verify(args.url, args.directory)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"GREEN CHECK: FAIL ({exc})")
        return 1
    print("GREEN CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
