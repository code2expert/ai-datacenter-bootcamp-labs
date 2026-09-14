#!/usr/bin/env python3
"""Read Grafana and validate the alert lab's saved artifacts."""
import argparse
import base64
from datetime import datetime
import getpass
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_notifications(path):
    starts = {}
    matched = False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        received = datetime.fromisoformat(event['received_at'].replace('Z', '+00:00'))
        require(received.tzinfo is not None, 'notification timestamps need a timezone')
        for alert in event.get('alerts', []):
            labels = alert.get('labels', {})
            if labels.get('alertname') != 'Lab notification test':
                continue
            if labels.get('grafana_folder') != 'AIDC lab':
                continue
            # A datasource failure is not proof of the artificial threshold.
            if (alert.get('annotations') or {}).get('Error'):
                continue
            values = alert.get('values') or {}
            if not values or not all(isinstance(v, (float, int)) and math.isfinite(v)
                                     for v in values.values()):
                continue
            identity = (tuple(sorted(labels.items())), alert.get('startsAt'))
            if alert.get('status') == 'firing':
                starts[identity] = received
            elif alert.get('status') == 'resolved' and identity in starts:
                ended = datetime.fromisoformat(alert['endsAt'].replace('Z', '+00:00'))
                began = datetime.fromisoformat(alert['startsAt'].replace('Z', '+00:00'))
                matched |= received >= starts[identity] and ended > began
    require(matched, 'need matching firing and resolved Lab notification test records')


def check_report(path):
    text = path.read_text()
    fields = [
        'Team', 'Use case', 'Service and model', 'Measured requests or tasks',
        'Indicator and unit', 'SLO target and window', 'Measurement start and end',
        'Workload', 'Observed result and sample count', 'Evidence', 'Conclusion',
        'Limitations', 'Follow-up action', 'Condition and unit', 'Evaluation interval',
        'Pending period', 'Relationship to the SLO', 'First response to a notification',
        'Firing received at', 'Resolved received at', 'What the test establishes']
    for field in fields:
        matches = re.findall(r'^' + re.escape(field) + r':[ \t]*([^\n]*)$', text, re.M)
        require(len(matches) == 1 and matches[0].strip() and
                matches[0].strip().lower() not in {'todo', 'tbd', '...', '<fill in>'},
                'complete report field: ' + field)
    conclusion = re.search(r'^Conclusion:[ \t]*(.*)$', text, re.M).group(1).lower()
    require(conclusion in {'met', 'not met', 'insufficient evidence'},
            'Conclusion must be met, not met or insufficient evidence')
    section = text.split('## Measurement query', 1)
    require(len(section) == 2 and '```' in section[1].split('## Service alert')[0],
            'include the measurement query in a fenced code block with an explanation')
    require('Paste the expression used' not in text, 'replace the measurement-query prompt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    base = args.url.rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    require(parsed.scheme in {'http', 'https'} and parsed.hostname and not parsed.username,
            'use an HTTP(S) Grafana URL without embedded credentials')
    token = os.environ.get('GRAFANA_TOKEN')
    if token:
        auth = 'Bearer ' + token
    else:
        user = os.environ.get('GRAFANA_USER', 'admin')
        password = os.environ.get('GRAFANA_PASSWORD') or getpass.getpass('Grafana password: ')
        auth = 'Basic ' + base64.b64encode((user + ':' + password).encode()).decode()

    def api(path):
        req = urllib.request.Request(base + path, headers={'Authorization': auth})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(f'Grafana returned HTTP {exc.code}; check login and permissions') from None
        except (urllib.error.URLError, TimeoutError):
            raise ValueError('Grafana or its data source could not be reached') from None

    rules = api('/api/v1/provisioning/alert-rules')
    matches = [r for r in rules if r.get('title') == 'Team service alert']
    require(len(matches) == 1, 'save exactly one rule named Team service alert')
    rule = matches[0]
    folder = api('/api/folders/' + urllib.parse.quote(rule['folderUID'], safe=''))
    require(folder.get('title') == 'AIDC lab', 'save the service rule in AIDC lab')
    require(not rule.get('isPaused'), 'Team service alert must remain enabled')
    require(rule.get('notification_settings', {}).get('receiver') == 'Lab inbox',
            'choose Lab inbox directly on the service rule')
    require(rule.get('noDataState') == 'NoData' and rule.get('execErrState') == 'Error',
            'keep No Data and Error as distinct service-rule states')
    for r in rules:
        if r.get('title') == 'Lab notification test':
            require(r.get('isPaused'), 'pause Lab notification test after recovery')
    data = {x['refId']: x for x in rule.get('data', [])}
    require({'A', 'B', 'C'} <= data.keys() and rule.get('condition') == 'C',
            'use query A, Reduce B and Threshold C as documented')
    a, b, c = (data[k]['model'] for k in ('A', 'B', 'C'))
    expr = a.get('expr', '').strip()
    require(expr and '$' not in expr, 'use an explicit PromQL expression without dashboard variables')
    require(not re.search(r'\b(?:time|vector)\s*\(', expr),
            'the service query must use service measurements, not the artificial test')
    require(a.get('instant') is True, 'set service query A to Instant')
    require(b.get('type') == 'reduce' and b.get('expression') == 'A' and
            b.get('reducer') == 'last', 'expression B must reduce A with Last')
    require(c.get('type') == 'threshold' and c.get('expression') == 'B',
            'expression C must apply a threshold to B')
    require(rule.get('annotations', {}).get('summary', '').strip(), 'add a service-rule summary')
    uid = data['A'].get('datasourceUid', '')
    source = api('/api/datasources/uid/' + urllib.parse.quote(uid, safe=''))
    require(source.get('type') == 'prometheus', 'query A must use a Prometheus data source')
    result = api('/api/datasources/proxy/uid/' + urllib.parse.quote(uid, safe='') +
                 '/api/v1/query?' + urllib.parse.urlencode({'query': expr}))
    require(result.get('status') == 'success', 'Prometheus query did not succeed')
    body = result.get('data', {})
    require(body.get('resultType') == 'vector' and body.get('result'),
            'service query returned no series; check labels and generate traffic')
    require(all(math.isfinite(float(row['value'][1])) for row in body['result']),
            'service query returned NaN or infinity; collect usable observations')
    runtime = api('/api/prometheus/grafana/api/v1/rules')
    states = [r for g in runtime.get('data', {}).get('groups', []) for r in g.get('rules', [])
              if r.get('name') == 'Team service alert']
    require(len(states) == 1 and states[0].get('health') == 'ok',
            'wait for a successful scheduled service-rule evaluation')
    require(not any(x.get('state') in {'Error', 'NoData'} for x in states[0].get('alerts', [])),
            'the service rule still has a missing-data or query-error instance')
    check_notifications(Path('notification-evidence.jsonl'))
    check_report(Path('my-service-report.md'))
    Path('my-alert.json').write_text(json.dumps(rule, indent=2) + '\n')
    print('Saved my-alert.json from Grafana.')
    print('GREEN CHECK: PASS (service rule, query, notification evidence and report fields)')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else \
            'missing or invalid lab artifact; check the instructions and file formats'
        print('GREEN CHECK: FAIL (' + message + ')', file=sys.stderr)
        sys.exit(1)
