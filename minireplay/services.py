"""Host the boundary ledger and the LLM store.

Both run as aiohttp apps on a private event loop in a background thread, bound to
loopback on a kernel-assigned port. HTTP is the transport because the processes that
must reach them are not all Python: a framework worker, a container and a Node
plugin all speak the same protocol.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from .boundary import BoundaryLedger
from .errors import InfrastructureError
from .llm_store import LLMStore


@dataclass
class ServiceEndpoint:
    url: str
    port: int


class ReplayServices:
    def __init__(
        self,
        *,
        mode: str,
        stage_dir: Path,
        auth_token: str,
        adapter: str,
        upstreams: dict[str, str],
        run_root: Path,
        repo: Path,
        bundle: Any | None = None,
        replay_mode: str = "tool-only",
        fast_claim: bool = False,
        bind_host: str = "127.0.0.1",
        force_secret: str | None = None,
        audit_path: Path | None = None,
        audit_namespace: str | None = None,
    ) -> None:
        self.bind_host = bind_host
        self.llm = LLMStore(
            mode=mode,
            stage_dir=stage_dir,
            upstreams=upstreams,
            bundle=bundle,
            replay_mode=replay_mode,
            force_secret=force_secret,
            audit_path=audit_path,
            audit_namespace=audit_namespace,
        )
        self.boundary = BoundaryLedger(
            mode=mode,
            stage_dir=stage_dir,
            auth_token=auth_token,
            adapter=adapter,
            run_root=run_root,
            repo=repo,
            bundle=bundle,
            llm_index=self.llm,
            fast_claim=fast_claim,
        )
        terminal_actors: set[str] = set()
        if bundle is not None:
            for terminal in bundle.terminal.get("task_terminals", []):
                actor_id = terminal.get("actor_id") if isinstance(terminal, dict) else None
                if isinstance(actor_id, str):
                    terminal_actors.add(actor_id)
        tail_actors = {
            str(entry["actor_id"])
            for section in ("operations", "llm_requests")
            for entry in (bundle.cutoff_tails.get(section, []) if bundle is not None else [])
        }
        self._cutoff_actors = (
            (set(bundle.actor_ids()) - terminal_actors) | tail_actors
            if bundle is not None
            else set()
        )
        # Each service can only see its own queues, but "past the end of the
        # recorded window" is a property of the whole run.
        self.llm.run_complete = self.expected_complete
        self.boundary.run_complete = self.expected_complete
        # Source-live actors stop after their last closed slot. Cutoff tails are
        # evidence only, so the first request beyond that prefix is held before it
        # enters either vLLM or a native implementation.
        self.llm.actor_complete = self.cutoff_actor_prefix_consumed
        self.boundary.actor_complete = self.cutoff_actor_prefix_consumed
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._runners: dict[str, web.AppRunner] = {}
        self.boundary_endpoint: ServiceEndpoint | None = None
        self.llm_endpoint: ServiceEndpoint | None = None

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.boundary_endpoint = self._start_service("boundary")
        try:
            self.llm_endpoint = self._start_service("llm")
        except BaseException:
            self.stop()
            raise

    def _start_service(self, name: str) -> ServiceEndpoint:
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        error: list[BaseException] = []
        endpoint: list[ServiceEndpoint] = []

        async def boot() -> None:
            if name == "llm":
                self.llm.client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
                    # Agent tools routinely run longer than vLLM's HTTP keep-alive
                    # window.  Reusing one of those silently expired connections can
                    # fail the next POST with ``ServerDisconnectedError`` even though
                    # vLLM itself stayed healthy.  A loopback TCP handshake is tiny
                    # compared with generation and avoids retrying a non-idempotent
                    # completion request after an ambiguous disconnect.
                    connector=aiohttp.TCPConnector(force_close=True),
                )
                app = self.llm.application()
            else:
                app = self.boundary.application()
            endpoint.append(await self._serve(name, app))

        def run() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(boot())
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                error.append(exc)
                ready.set()
                return
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=run, name=f"minireplay-{name}", daemon=True)
        self._loops[name] = loop
        self._threads[name] = thread
        thread.start()
        ready.wait()
        if error:
            raise InfrastructureError(f"could not start {name} service: {error[0]}") from error[0]
        return endpoint[0]

    async def _serve(self, name: str, app: web.Application) -> ServiceEndpoint:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.bind_host, 0)
        await site.start()
        self._runners[name] = runner
        sockets = list(site._server.sockets or ())  # noqa: SLF001 - the only way to read the port
        if not sockets:
            raise InfrastructureError("replay service bound no socket")
        port = int(sockets[0].getsockname()[1])
        return ServiceEndpoint(url=f"http://{self.bind_host}:{port}", port=port)

    def stop(self) -> None:
        for name in tuple(self._loops):
            self._stop_service(name)

    def _stop_service(self, name: str) -> None:
        loop = self._loops.get(name)
        if loop is None:
            return

        async def shutdown() -> None:
            # Requests held open past the end of the recorded window never return by
            # design, so cancel them explicitly instead of letting the loop close
            # underneath them.
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            runner = self._runners.get(name)
            if runner is not None:
                with suppress(Exception):
                    await runner.cleanup()
            if name == "llm" and self.llm.client is not None:
                await self.llm.client.close()

        future = asyncio.run_coroutine_threadsafe(shutdown(), loop)
        # Teardown must never mask the failure that caused it.
        with suppress(Exception):
            future.result(timeout=30)
        loop.call_soon_threadsafe(loop.stop)
        thread = self._threads.get(name)
        if thread is not None:
            thread.join(timeout=10)
        self._loops.pop(name, None)
        self._threads.pop(name, None)
        self._runners.pop(name, None)

    def call(self, name: str, coroutine) -> Any:
        loop = self._loops.get(name)
        if loop is None:
            raise InfrastructureError(f"replay {name} service is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout=60)

    # ---- run-level state -----------------------------------------------------

    def assert_healthy(self) -> None:
        for name, failure in (
            ("boundary", self.boundary.hard_failure),
            ("llm", self.llm.hard_failure),
        ):
            if failure is not None:
                raise InfrastructureError(f"{name} service failed: {failure}")

    def expected_complete(self) -> bool:
        return self.boundary.expected_complete() and self.llm.expected_complete()

    def cutoff_actor_complete(self, actor_id: str) -> bool:
        """True when a source-live actor has reached its own causal cutoff."""

        return (
            actor_id in self._cutoff_actors
            and self.boundary.actor_expected_complete(actor_id)
            and self.llm.actor_expected_complete(actor_id)
        )

    def cutoff_actor_prefix_consumed(self, actor_id: str) -> bool:
        """True once a source-live actor entered every recorded causal slot."""

        return (
            actor_id in self._cutoff_actors
            and self.boundary.actor_expected_prefix_consumed(actor_id)
            and self.llm.actor_expected_prefix_consumed(actor_id)
        )

    def has_unexpected_active(self) -> bool:
        return self.boundary.has_unexpected_active()

    def assert_consumed(self) -> None:
        self.boundary.assert_consumed()
        self.llm.assert_consumed()

    def freeze_source_cutoff(self, cutoff_at_ns: int) -> dict[str, Any]:
        async def freeze_boundary() -> list[dict[str, Any]]:
            return self.boundary.freeze_source_cutoff(cutoff_at_ns)

        async def freeze_llm() -> list[dict[str, Any]]:
            return self.llm.freeze_source_cutoff(cutoff_at_ns)

        if not self._loops:
            raise InfrastructureError("cannot freeze cutoff before replay services start")
        # Each snapshot runs beside that ledger's request handlers.  A direct
        # supervisor-thread read could classify one request as both closed and
        # truncated.  The shared cutoff timestamp makes the two snapshots one
        # logical boundary even though their transports are intentionally sharded.
        return {
            "operations": self.call("boundary", freeze_boundary()),
            "llm_requests": self.call("llm", freeze_llm()),
        }

    def status(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.outstanding(),
            "llm": self.llm.outstanding(),
        }
