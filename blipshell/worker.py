"""Standalone remote worker daemon.

Polls the main BlipShell instance's HTTP API for pending tasks,
claims them, executes using local Ollama, and reports results back.

Usage:
    python -m blipshell.worker --main-url http://192.168.1.x:8000 --name remote-pc
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

import httpx
from ollama import AsyncClient as OllamaClient

logger = logging.getLogger(__name__)


class Worker:
    """Remote worker that polls for and executes background tasks."""

    def __init__(
        self,
        main_url: str,
        name: str,
        ollama_url: str = "http://localhost:11434",
        poll_interval: int = 10,
        api_key: Optional[str] = None,
        model: str = "gemma3:4b",
    ):
        self.main_url = main_url.rstrip("/")
        self.name = name
        self.ollama_url = ollama_url
        self.poll_interval = poll_interval
        self.api_key = api_key
        self.model = model
        self._running = False
        self._http: Optional[httpx.AsyncClient] = None
        self._ollama: Optional[OllamaClient] = None

    def _headers(self) -> dict:
        """Build HTTP headers with optional auth."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def start(self):
        """Start the worker polling loop."""
        self._running = True
        self._http = httpx.AsyncClient(timeout=30)
        self._ollama = OllamaClient(host=self.ollama_url)

        logger.info(
            "Worker '%s' starting — main: %s, ollama: %s, model: %s, poll: %ds",
            self.name, self.main_url, self.ollama_url, self.model, self.poll_interval,
        )

        try:
            while self._running:
                try:
                    await self._poll_and_execute()
                except Exception as e:
                    logger.error("Poll cycle error: %s", e)

                await asyncio.sleep(self.poll_interval)
        finally:
            if self._http:
                await self._http.aclose()

    async def stop(self):
        """Stop the worker."""
        self._running = False

    async def _poll_and_execute(self):
        """Poll for tasks, claim one, execute it, report result."""
        # Poll for pending tasks
        resp = await self._http.get(
            f"{self.main_url}/api/worker/poll",
            params={"endpoint_name": self.name},
            headers=self._headers(),
        )
        if resp.status_code != 200:
            logger.debug("Poll returned %d", resp.status_code)
            return

        tasks = resp.json()
        if not tasks:
            return

        # Process each available task
        for task_data in tasks:
            task_id = task_data["id"]

            # Claim the task
            claim_resp = await self._http.post(
                f"{self.main_url}/api/worker/claim/{task_id}",
                headers=self._headers(),
            )
            if claim_resp.status_code != 200:
                logger.debug("Could not claim task #%d", task_id)
                continue

            logger.info(
                "Claimed task #%d: %s", task_id, task_data.get("title", ""),
            )

            # Execute
            try:
                await self._report_progress(task_id, 0.1, "Starting execution...")

                result = await self._execute_task(task_data)

                await self._report_progress(task_id, 0.9, "Completing...")

                # Report success
                await self._http.post(
                    f"{self.main_url}/api/worker/complete/{task_id}",
                    json={"status": "completed", "result": result},
                    headers=self._headers(),
                )
                logger.info("Task #%d completed", task_id)

            except Exception as e:
                logger.error("Task #%d failed: %s", task_id, e)
                await self._http.post(
                    f"{self.main_url}/api/worker/complete/{task_id}",
                    json={"status": "failed", "error_message": str(e)},
                    headers=self._headers(),
                )

    async def _execute_task(self, task_data: dict) -> str:
        """Execute a task using local Ollama."""
        prompt = task_data.get("prompt", "")
        if not prompt:
            return "No prompt provided."

        response = await self._ollama.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )

        # Extract response
        msg = getattr(response, "message", None)
        if msg is not None:
            return getattr(msg, "content", "") or ""
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        return str(response)

    async def _report_progress(
        self, task_id: int, pct: float, message: str,
    ):
        """Report progress back to main instance."""
        try:
            await self._http.post(
                f"{self.main_url}/api/worker/progress/{task_id}",
                json={"progress_pct": pct, "progress_message": message},
                headers=self._headers(),
            )
        except Exception as e:
            logger.debug("Progress report failed for task #%d: %s", task_id, e)


def main():
    parser = argparse.ArgumentParser(
        description="BlipShell Remote Worker Daemon",
    )
    parser.add_argument(
        "--main-url", required=True,
        help="URL of the main BlipShell instance (e.g., http://192.168.1.100:8000)",
    )
    parser.add_argument(
        "--name", required=True,
        help="Name of this worker (matches target_endpoint in task routing)",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="URL of the local Ollama instance",
    )
    parser.add_argument(
        "--model", default="gemma3:4b",
        help="Model to use for task execution",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=10,
        help="Seconds between polling (default: 10)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for authentication with main instance",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    worker = Worker(
        main_url=args.main_url,
        name=args.name,
        ollama_url=args.ollama_url,
        model=args.model,
        poll_interval=args.poll_interval,
        api_key=args.api_key,
    )

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        print("\nWorker shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
