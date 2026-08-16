"""Twilio adapter (WhatsApp sandbox + SMS). Inbound is form-encoded + X-Twilio-Signature;
reply out-of-band via the sync REST client in a threadpool (create_async hangs on uvloop).

NOTE: no `from __future__ import annotations` here on purpose, the FastAPI route below
annotates `request: Request`, and FastAPI must see the real class (not a deferred string)
to inject it rather than treat it as a query parameter."""
import asyncio
import base64
import json
import logging

from heynyc.core import config

from .base import InboundMessage
from .format import _split
from .identity import user_key
from .orchestrator import Deps, handle
from .store import PENDING_APPROVAL_OUTBOX_KEY, InboxPayloadError

_TYPING_URL = "https://messaging.twilio.com/v3/Indicators/Typing.json"
_TYPING_REFRESH_SECONDS = 20
_TEXT_LIMIT = 1600
_PAGE_PREFIX_RESERVE = 16
logger = logging.getLogger("heynyc.channels.twilio")


def _ordered_chunks(text: str) -> list[str]:
    chunks = _split(text, _TEXT_LIMIT - _PAGE_PREFIX_RESERVE)
    return chunks if len(chunks) < 2 else [
        f"{index}/{len(chunks)} {chunk}" for index, chunk in enumerate(chunks, 1)
    ]


class TwilioReplier:
    def __init__(
        self, client, from_: str, to: str, message_id: str = "", on_sent=None,
    ) -> None:
        self._client, self._from, self._to, self._mid = client, from_, to, message_id
        self._on_sent = on_sent
        self.outbound_ids: list[str] = []

    async def _create(self, **kwargs) -> None:
        created = await asyncio.to_thread(self._client.messages.create, **kwargs)
        sid = getattr(created, "sid", "")
        if sid:
            self.outbound_ids.append(sid)
            if self._on_sent is not None:
                self._on_sent(sid)

    async def send_text(self, text: str) -> None:
        for chunk in _ordered_chunks(text):
            await self._create(from_=self._from, to=self._to, body=chunk)

    async def send_part(self, part: dict) -> None:
        kwargs = {"from_": self._from, "to": self._to, "body": part["body"]}
        if part.get("media_url"):
            kwargs["media_url"] = part["media_url"]
        await self._create(**kwargs)

    async def send_document(self, path: str, caption: str = "") -> None:
        # Twilio fetches media by PUBLIC URL; a local file can't be attached directly. If we're
        # given a hosted URL, send it as media; otherwise degrade to a text note (Meta is the
        # document-capable channel, see the channels README).
        if path.startswith(("http://", "https://")):
            chunks = _ordered_chunks(caption or "")
            for chunk in chunks[:-1]:
                await self._create(
                    from_=self._from,
                    to=self._to,
                    body=chunk,
                )
            await self._create(
                from_=self._from, to=self._to, body=chunks[-1], media_url=[path],
            )
        else:
            await self.send_text(f"{caption or 'Your document is ready'}, I'll send a download link shortly.")

    async def indicate_typing(self) -> None:
        if not self._from.startswith("whatsapp:") or not self._mid.startswith(("SM", "MM")):
            return
        try:
            response = await asyncio.to_thread(
                self._client.request,
                "POST",
                _TYPING_URL,
                data={"channel": "WHATSAPP", "messageId": self._mid},
                headers={"Content-Type": "application/json"},
            )
            if response.status_code >= 400:
                logger.warning("Twilio typing indicator returned HTTP %s", response.status_code)
        except Exception:
            logger.warning("Twilio typing indicator failed", exc_info=True)


class TwilioOutboxReplier:
    """Collect a complete Twilio reply and persist it before delivery starts."""

    def __init__(self, store, client, *, from_: str, to: str, message_id: str) -> None:
        self.store = store
        self.parts: list[dict] = []
        self._typing = TwilioReplier(client, from_=from_, to=to, message_id=message_id)
        self._typing_enabled = from_.startswith("whatsapp:") and message_id.startswith(("SM", "MM"))
        self._typing_task: asyncio.Task | None = None
        self._message_id = message_id
        self._finalized = False

    async def send_text(self, text: str) -> None:
        self.parts.extend({"body": chunk} for chunk in _ordered_chunks(text))

    async def send_document(self, path: str, caption: str = "") -> None:
        if path.startswith(("http://", "https://")):
            chunks = _ordered_chunks(caption or "")
            self.parts.extend({"body": chunk} for chunk in chunks[:-1])
            self.parts.append({"body": chunks[-1], "media_url": [path]})
        else:
            await self.send_text(
                f"{caption or 'Your document is ready'}, I'll send a download link shortly."
            )

    async def indicate_typing(self) -> None:
        await self._typing.indicate_typing()
        if self._typing_enabled and self._typing_task is None:
            self._typing_task = asyncio.create_task(self._refresh_typing())

    async def _refresh_typing(self) -> None:
        while True:
            await asyncio.sleep(_TYPING_REFRESH_SECONDS)
            await self._typing.indicate_typing()

    async def stop_typing(self) -> None:
        if self._typing_task is None:
            return
        task, self._typing_task = self._typing_task, None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def finalize(self) -> None:
        await self.stop_typing()
        if not self._finalized:
            self.store.stage_outbox(self._message_id, self.parts)
            self._finalized = True

    def stage_pending_approval(self, user_key: str, state: bytes, *, ttl_s: float) -> None:
        """Keep approval inert inside the encrypted outbox until every review part is delivered."""
        if not self.parts:
            raise RuntimeError("cannot stage approval without a review")
        self.parts[-1][PENDING_APPROVAL_OUTBOX_KEY] = {
            "user_key": user_key,
            "state": base64.b64encode(state).decode("ascii"),
            "ttl_s": ttl_s,
        }
        self.store.stage_outbox(self._message_id, self.parts)


class TwilioInboxWorker:
    """Process Twilio envelopes already durably accepted by the webhook."""

    def __init__(
        self, deps: Deps, client, *, lease_s: float = 300,
        retry_after_s: float = 5, max_attempts: int = 3, concurrency: int = 1,
    ) -> None:
        self.deps = deps
        self.client = client
        self.lease_s = lease_s
        self.retry_after_s = retry_after_s
        self.max_attempts = max_attempts
        self.concurrency = max(1, concurrency)
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    def wake(self) -> None:
        self._wake.set()

    def start(self) -> None:
        if not self._tasks:
            self._stopping = False
            self._tasks = [asyncio.create_task(self._run()) for _ in range(self.concurrency)]

    async def stop(self) -> None:
        if not self._tasks:
            return
        self._stopping = True
        self.wake()
        await asyncio.gather(*self._tasks)
        self._tasks = []

    async def _run(self) -> None:
        while not self._stopping:
            self._wake.clear()
            while not self._stopping and await self.process_one():
                pass
            if self._stopping or self._wake.is_set():
                continue
            try:
                await asyncio.wait_for(
                    self._wake.wait(), timeout=max(0.1, min(self.retry_after_s, 5)),
                )
            except asyncio.TimeoutError:
                pass

    async def process_one(self) -> bool:
        try:
            item = self.deps.store.claim_next(lease_s=self.lease_s)
        except InboxPayloadError:
            logger.exception("Twilio inbox row quarantined")
            return True
        if item is None:
            return False
        try:
            envelope = json.loads(item["payload"])
            recipient = envelope.pop("recipient")
            message = InboundMessage(**envelope)
        except Exception:
            self.deps.store.fail(item["message_id"])
            logger.exception("Twilio inbox envelope quarantined")
            return True
        parts = item["outbox"]
        delivered_parts = item["delivered_parts"]
        if parts is None:
            buffered = TwilioOutboxReplier(
                self.deps.store, self.client, from_=recipient, to=message.sender,
                message_id=message.message_id,
            )
            try:
                await handle(message, buffered, self.deps, deduplicate=False)
                await buffered.finalize()
            except Exception:
                retry_after_s = (
                    self.retry_after_s if item["attempts"] < self.max_attempts else None
                )
                self.deps.store.fail(message.message_id, retry_after_s=retry_after_s)
                logger.exception("Twilio inbox message generation failed")
                return True
            finally:
                await buffered.stop_typing()
            parts = buffered.parts
            delivered_parts = 0
        replier = TwilioReplier(
            self.client, from_=recipient, to=message.sender, message_id=message.message_id,
            on_sent=lambda sid: self.deps.store.record_outbound(message.message_id, sid),
        )
        try:
            pending_approval = (
                parts[-1].get(PENDING_APPROVAL_OUTBOX_KEY) if parts else None
            )
            if (
                pending_approval is not None
                and pending_approval.get("user_key") != item["user_key"]
            ):
                raise ValueError("approval outbox resident mismatch")
            for part in parts[delivered_parts:]:
                await replier.send_part(part)
            if pending_approval is not None:
                self.deps.store.set_pending_approval(
                    item["user_key"],
                    base64.b64decode(pending_approval["state"], validate=True),
                    ttl_s=float(pending_approval["ttl_s"]),
                )
        except Exception:
            retry_after_s = self.retry_after_s if item["attempts"] < self.max_attempts else None
            self.deps.store.fail(message.message_id, retry_after_s=retry_after_s)
            logger.exception("Twilio inbox message delivery failed")
            return True
        self.deps.store.complete(message.message_id)
        return True


def to_inbound(params: dict) -> InboundMessage:
    try:
        count = max(0, int(params.get("NumMedia", 0) or 0))
    except (TypeError, ValueError):
        count = 0
    media = [
        {
            "url": params[f"MediaUrl{index}"],
            "content_type": params.get(f"MediaContentType{index}", ""),
        }
        for index in range(count)
        if params.get(f"MediaUrl{index}")
    ]
    sender = params.get("From", "")
    return InboundMessage(
        channel="whatsapp_twilio" if sender.startswith("whatsapp:") else "sms_twilio",
        sender=sender,
        text=params.get("Body", "") or "",
        message_id=params.get("MessageSid", ""),
        profile_name=params.get("ProfileName", "") or "",
        media=media,
        raw=dict(params),
    )


def public_url(request) -> str:
    """Reconstruct the URL Twilio signed (it sees the public host, not ngrok's internal one)."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    url = f"{proto}://{host}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
    return url


def make_twilio_router(deps: Deps, worker):
    from fastapi import APIRouter, Request, Response
    from twilio.request_validator import RequestValidator

    router = APIRouter()
    validator = RequestValidator(config.TWILIO_AUTH_TOKEN)

    @router.post("/webhook/twilio")
    async def _twilio_webhook(request: Request) -> Response:
        form = await request.form()
        params = dict(form)
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(public_url(request), params, signature):
            return Response(status_code=403)
        sender = params.get("From")
        recipient = params.get("To")
        if not sender or not recipient:
            return Response(status_code=400)
        inbound = to_inbound(params)
        payload = json.dumps({
            "channel": inbound.channel,
            "sender": inbound.sender,
            "recipient": recipient,
            "text": inbound.text,
            "message_id": inbound.message_id,
            "profile_name": inbound.profile_name,
            "media": inbound.media,
        })
        if deps.store.enqueue(
            inbound.message_id,
            user_key(inbound.channel, inbound.sender, deps.salt),
            payload,
        ):
            worker.wake()
        return Response(status_code=200)

    return router
