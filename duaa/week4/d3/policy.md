# Kubernetes Resource & Scheduling Policy for Model Serving

## Policy

Serving is Guaranteed to protect its p95 latency SLO, Dashboard can use spare resources for short refresh spikes, and Batch is throttled first because it has no immediate deadline.

## Serving

**YAML**

resources:
  requests:
    cpu: "1"
    memory: "4Gi"
  limits:
    cpu: "1"
    memory: "4Gi"

**QoS Class:** Guaranteed

## Dashboard

**YAML**

resources: {}

**QoS Class:** BestEffort

## Batch

**YAML**

resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "250m"
    memory: "512Mi"

**QoS Class:** Burstable

## Evidence

With an unlimited CPU neighbour, serving p95 latency was **4 ms** with zero failures. With the CPU-limited neighbour, p95 was also **4 ms** with zero failures. Therefore, Batch is throttled first to protect the latency-sensitive Serving workload while allowing the Dashboard to use spare resources when available.
