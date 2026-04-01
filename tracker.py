import logging
from datetime import datetime, timezone
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)


class AgentOsTracker:
    """Sends lifecycle events to the agent-os ingest API."""

    def __init__(self, api_url: str, api_key: str, agent_name: str):
        self.api_url = api_url.rstrip("/") + "/api/ingest"
        self.api_key = api_key
        self.agent_name = agent_name
        self.sdk_run_id = str(uuid4())

    def _send(self, event_type: str, payload: dict, **extra_fields) -> dict | None:
        body = {
            "event_type": event_type,
            "sdk_run_id": self.sdk_run_id,
            "payload": payload,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            **extra_fields,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = requests.post(self.api_url, json=body, headers=headers, timeout=10)
            resp.raise_for_status()
            logger.info("Sent %s event (run %s)", event_type, self.sdk_run_id)
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Failed to send %s event: %s", event_type, exc)
            return None

    def run_start(self, payload: dict | None = None) -> dict | None:
        return self._send(
            "run_start",
            payload or {},
            agent_name=self.agent_name,
        )

    def step(self, node_name: str, payload: dict | None = None) -> dict | None:
        return self._send("step", {"node": node_name, **(payload or {})})

    def tool_call(self, tool_name: str, payload: dict | None = None) -> dict | None:
        return self._send("tool_call", {"tool": tool_name, **(payload or {})})

    def error(self, error_msg: str, payload: dict | None = None) -> dict | None:
        return self._send("error", {"error": error_msg, **(payload or {})})

    def run_end(self, payload: dict | None = None) -> dict | None:
        return self._send("run_end", payload or {})
