# HPA Failure Analysis

1. Today's HPA works for the echo backend because the workload uses CPU. On the real vLLM engine, the model is GPU-bound, so CPU usage can stay relatively low even when many requests are waiting. In our test, the vLLM pod used about 769m CPU during load, which would not clearly represent the real pressure on the GPU service.

2. I would use the engine queue depth, especially `vllm_num_requests_waiting`, as the scaling signal. This directly shows when requests are waiting for inference and is more useful than CPU for our GPU-bound workload.

3. I would start with a small queue threshold, for example 5 waiting requests. Then I would monitor queue depth, latency, TTFT, GPU utilization, and GPU memory, and adjust the threshold based on how quickly latency increases under load.
