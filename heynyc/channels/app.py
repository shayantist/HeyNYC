"""FastAPI app factory: build the shared Agent once, mount the configured provider(s),
drain in-flight tasks on shutdown. The Agent is concurrency-safe across users; per-user
state lives only in Session."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from heynyc.core import config, telemetry
from heynyc.core.agent import Agent
from heynyc.core.registry import Registry

from .base import drain
from .orchestrator import Deps

_INDEX_PATH = config.HEYNYC_DATA_DIR / "index.lance"


def _load_retriever():
    from heynyc.core.index import IndexRetriever, default_embedder, open_store

    if not _INDEX_PATH.exists():
        return None
    return IndexRetriever(store=open_store(_INDEX_PATH), embedder=default_embedder())


def build_agent() -> Agent:
    return Agent(Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST), model=config.HEYNYC_MODEL, index=_load_retriever())


def build_deps(agent: Agent) -> Deps:
    from heynyc.core.drafts import DraftStore

    from .base import KeyedLocks
    from .store import ChannelStore

    data = config.HEYNYC_DATA_DIR
    store = ChannelStore(
        data / "channels.sqlite3", rate_limit=config.CHANNEL_RATE_LIMIT,
        window_s=config.CHANNEL_RATE_WINDOW_S, dedup_ttl_s=config.CHANNEL_DEDUP_TTL_S,
    )
    return Deps(
        agent=agent, store=store, sessions_dir=data / "sessions", salt=config.HEYNYC_PII_SALT,
        telemetry_path=telemetry.default_path(config.HEYNYC_DATA_DIR), feedback_path=data / "feedback.jsonl",
        locks=KeyedLocks(), semaphore=asyncio.Semaphore(config.CHANNEL_MAX_CONCURRENCY),
        drafts=DraftStore(data / "drafts"),
    )


def create_app(provider: str | None = None):
    from fastapi import FastAPI

    provider = provider or config.WHATSAPP_PROVIDER
    if not config.HEYNYC_PII_SALT:
        raise RuntimeError("HEYNYC_PII_SALT must be set to serve (pseudonymizes senders).")

    deps = build_deps(build_agent())

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await drain()   # graceful shutdown: finish in-flight replies

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if provider in ("meta", "both"):
        from .meta import attach_meta
        attach_meta(app, deps)
    if provider in ("twilio", "both"):
        from .twilio import make_twilio_router
        app.include_router(make_twilio_router(deps))
    return app
