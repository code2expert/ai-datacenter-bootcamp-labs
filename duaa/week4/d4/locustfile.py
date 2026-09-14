# Pre-tuned laptop load for the W4D4 HPA lab, through a port-forward.
#
#   kubectl port-forward svc/team-serving 8000:8000 &
#   locust -f locustfile.py --headless -u 24 -r 6 -t 4m -H http://127.0.0.1:8000
#
# Tuned on a 4-core reference laptop: 24 users at this prompt size saturate the
# chart's 250m-request pods well past the HPA's 50% target without saturating
# the laptop that is generating the load. If your machine wheezes, drop -u
# before you drop the prompt size; the prompt is what makes the echo backend
# spend CPU.
from locust import HttpUser, task, between


class ChatUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def chat(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user",
                              "content": "repeat the word load " * 60}],
                "max_tokens": 200,
            },
            name="/v1/chat/completions",
        )
