"""Twilio adapter (WhatsApp sandbox + SMS). Inbound is form-encoded + X-Twilio-Signature;
reply out-of-band via the sync REST client in a threadpool (create_async hangs on uvloop).

NOTE: no `from __future__ import annotations` here on purpose, the FastAPI route below
annotates `request: Request`, and FastAPI must see the real class (not a deferred string)
to inject it rather than treat it as a query parameter."""
import asyncio
import logging

from heynyc.core import config

from .base import InboundMessage, dispatch
from .format import _split
from .orchestrator import Deps, handle

_TYPING_URL = "https://messaging.twilio.com/v3/Indicators/Typing.json"
_TEXT_LIMIT = 1600
_PAGE_PREFIX_RESERVE = 16
logger = logging.getLogger("heynyc.channels.twilio")


def _ordered_chunks(text: str) -> list[str]:
    chunks = _split(text, _TEXT_LIMIT - _PAGE_PREFIX_RESERVE)
    return chunks if len(chunks) < 2 else [
        f"{index}/{len(chunks)} {chunk}" for index, chunk in enumerate(chunks, 1)
    ]


class TwilioReplier:
    def __init__(self, client, from_: str, to: str, message_id: str = "") -> None:
        self._client, self._from, self._to, self._mid = client, from_, to, message_id

    async def send_text(self, text: str) -> None:
        for chunk in _ordered_chunks(text):
            await asyncio.to_thread(
                self._client.messages.create, from_=self._from, to=self._to, body=chunk
            )

    async def send_document(self, path: str, caption: str = "") -> None:
        # Twilio fetches media by PUBLIC URL; a local file can't be attached directly. If we're
        # given a hosted URL, send it as media; otherwise degrade to a text note (Meta is the
        # document-capable channel, see the channels README).
        if path.startswith(("http://", "https://")):
            chunks = _ordered_chunks(caption or "")
            for chunk in chunks[:-1]:
                await asyncio.to_thread(
                    self._client.messages.create,
                    from_=self._from,
                    to=self._to,
                    body=chunk,
                )
            await asyncio.to_thread(
                self._client.messages.create,
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


def make_twilio_router(deps: Deps):
    from fastapi import APIRouter, Request, Response
    from twilio.request_validator import RequestValidator
    from twilio.rest import Client

    router = APIRouter()
    validator = RequestValidator(config.TWILIO_AUTH_TOKEN)
    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

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
        replier = TwilioReplier(
            client,
            from_=recipient,
            to=sender,
            message_id=inbound.message_id,
        )
        dispatch(handle(inbound, replier, deps))   # 200 returns fast; agent runs out-of-band
        return Response(status_code=200)

    return router
