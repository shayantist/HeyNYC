"""FastAPI app factory: build the shared Agent once, mount the configured provider(s),
drain in-flight tasks on shutdown. The Agent is concurrency-safe across users; per-user
state lives only in Session."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from heynyc.core import config, pii_crypto, telemetry
from heynyc.core.agent import Agent
from heynyc.core.drafts import DraftStore
from heynyc.core.registry import Registry
from heynyc.core.session import migrate_plaintext_sessions, purge_expired_sessions

from . import analytics
from .base import drain
from .orchestrator import Deps

_INDEX_PATH = config.HEYNYC_DATA_DIR / "index.lance"
_PURGE_INTERVAL_S = 24 * 60 * 60


def purge_private_data(data_dir: Path) -> None:
    purge_expired_sessions(data_dir / "sessions")
    DraftStore(data_dir / "drafts").purge_expired()


def purge_channel_data(store) -> None:
    store.purge_inbox(before=time.time() - pii_crypto.retention_days() * 24 * 60 * 60)


def migrate_private_data(data_dir: Path) -> None:
    migrate_plaintext_sessions(data_dir / "sessions")
    DraftStore(data_dir / "drafts").migrate_plaintext()
    analytics.migrate_plaintext_feedback(data_dir / "feedback.jsonl")


async def _purge_loop(data_dir: Path, store) -> None:
    while True:
        await asyncio.sleep(_PURGE_INTERVAL_S)
        purge_private_data(data_dir)
        purge_channel_data(store)


def _load_retriever():
    from heynyc.core.index import IndexRetriever, default_embedder, open_store

    if not _INDEX_PATH.exists():
        return None
    embedder = default_embedder()
    return IndexRetriever(
        store=open_store(_INDEX_PATH, model_id=embedder.model_id),
        embedder=embedder,
    )


def build_agent() -> Agent:
    registry = Registry.discover(
        config.MODULES_DIR,
        config.BASE_ALLOWLIST,
        config.NEWS_ALLOWLIST,
    )
    index = _load_retriever()
    if config.HEYNYC_AGENT_RUNTIME != "pydantic":
        raise RuntimeError("Public channels require the Pydantic runtime")
    from heynyc.core.pydantic_runtime import build_configured_runtime

    return build_configured_runtime(
        registry,
        model=config.HEYNYC_MODEL,
        index=index,
    )


def build_deps(agent: Agent) -> Deps:
    from .base import KeyedLocks
    from .store import ChannelStore

    data = config.HEYNYC_DATA_DIR
    store = ChannelStore(
        data / "channels.sqlite3", rate_limit=config.CHANNEL_RATE_LIMIT,
        window_s=config.CHANNEL_RATE_WINDOW_S, dedup_ttl_s=config.CHANNEL_DEDUP_TTL_S,
    )
    if not hasattr(agent, "conversation_from_state"):
        store.clear_pending_approvals()
    return Deps(
        agent=agent, store=store, sessions_dir=data / "sessions", salt=config.HEYNYC_PII_SALT,
        user_daily_spend_cap=config.HEYNYC_USER_DAILY_SPEND_CAP,
        telemetry_path=telemetry.default_path(config.HEYNYC_DATA_DIR), feedback_path=data / "feedback.jsonl",
        locks=KeyedLocks(), semaphore=asyncio.Semaphore(config.CHANNEL_MAX_CONCURRENCY),
        drafts=DraftStore(data / "drafts"),
    )


def create_app(provider: str | None = None):
    from fastapi import FastAPI

    provider = provider or config.WHATSAPP_PROVIDER
    if not config.HEYNYC_PII_SALT:
        raise RuntimeError("HEYNYC_PII_SALT must be set to serve (pseudonymizes senders).")
    if not pii_crypto.is_enabled():
        raise RuntimeError("HEYNYC_PII_KEY must be set to serve (encrypts persisted conversations).")
    try:
        pii_crypto.encrypt("")
    except pii_crypto.PiiCryptoError as exc:
        raise RuntimeError("HEYNYC_PII_KEY must be a valid encryption key.") from exc

    deps = build_deps(build_agent())
    workers = []

    @asynccontextmanager
    async def lifespan(_app):
        migrate_private_data(config.HEYNYC_DATA_DIR)
        purge_private_data(config.HEYNYC_DATA_DIR)
        purge_channel_data(deps.store)
        purge_task = asyncio.create_task(_purge_loop(config.HEYNYC_DATA_DIR, deps.store))
        for worker in workers:
            worker.start()
        try:
            yield
        finally:
            await asyncio.gather(*(worker.stop() for worker in workers))
            purge_task.cancel()
            with suppress(asyncio.CancelledError):
                await purge_task
            await drain()   # graceful shutdown: finish in-flight replies

    app = FastAPI(lifespan=lifespan)
    app.state.deps = deps

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if provider in ("meta", "both"):
        from .meta import attach_meta
        attach_meta(app, deps)
    if provider in ("twilio", "both"):
        from twilio.rest import Client

        from .twilio import TwilioInboxWorker, make_twilio_router

        worker = TwilioInboxWorker(
            deps, Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN),
            concurrency=config.CHANNEL_MAX_CONCURRENCY,
        )
        workers.append(worker)
        app.include_router(make_twilio_router(deps, worker))
    return app
