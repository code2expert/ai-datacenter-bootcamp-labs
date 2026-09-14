# Service report

Team: Team 2

Use case: Content Tagging & Competency Mapping

Service and model: Team vLLM Chat API serving Qwen/Qwen2.5-1.5B-Instruct-AWQ

Measured requests or tasks: Successful chat completion requests to the team vLLM service with finished_reason equal to "stop" or "length".

Indicator and unit: Completed requests per minute (req/min).

SLO target and window: At least 20 completed requests per minute over a rolling 5-minute window.

Measurement start and end: 2026-09-14 13:50:40 to 13:55:40 Asia/Riyadh.

Workload: Chat completion traffic sent to the team vLLM OpenAI-compatible API during the lab workload.

Observed result and sample count: The completed-request rate dropped to 0 req/min, causing the service alert to fire. The rate later recovered to 25.26 req/min, causing the alert to resolve. The exact total request count was not available.

Evidence: Grafana Team service alert in the AIDC lab folder and notification-evidence.jsonl.

Conclusion: not met

Limitations: The observation period was short and does not establish long-term SLO compliance. The exact total request count was not recorded. The measured completion rate depends on the traffic generated during the observation period. No Data events were observed separately and were not treated as successful service measurements. The measurement does not evaluate model-output quality.

Follow-up action: Investigate the request-rate drop, verify sustained traffic generation, inspect the vLLM pod health and logs, and repeat the measurement with a longer representative workload.

## Measurement query

```promql
60 * (
  sum(rate(vllm:request_success_total{job="serving",finished_reason=~"stop|length"}[5m]))
  or
  sum(rate(vllm_request_success_total{job="serving",finished_reason=~"stop|length"}[5m]))
)
```

Evaluation time or range: Instant Prometheus query using a rolling 5-minute rate window.

The measurement is taken from vLLM request-success counters collected by Prometheus and evaluated in Grafana.

It includes successful requests that finish with "stop" or "length".

It excludes model-output quality and does not measure the quality or correctness of the generated response.

## Service alert

Condition and unit: Completed requests per minute is below 20 req/min.

Evaluation interval: 1 minute

Pending period: 1 minute

Relationship to the SLO: The alert directly checks whether the completed-request rate falls below the target of 20 requests/min over the rolling 5-minute measurement window.

First response to a notification: Check the Grafana dashboard for recent error spikes, inspect the vLLM pod logs using `kubectl logs -n team -l app=vllm`, and verify that the traffic generator client is running correctly.

## Notification test

Firing received at: 2026-09-14 13:18:20 Asia/Riyadh

Resolved received at: 2026-09-14 13:19:10 Asia/Riyadh

What the test establishes: The test confirms that Grafana can evaluate the artificial alert and deliver both firing and resolved webhook notifications to the Lab inbox. It proves recovery of the artificial test condition, not recovery of the model service.