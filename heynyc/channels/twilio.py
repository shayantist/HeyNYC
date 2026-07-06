"""Twilio adapter (WhatsApp sandbox + SMS). Inbound is form-encoded + X-Twilio-Signature;
reply out-of-band via the sync REST client in a threadpool (create_async hangs on uvloop).

NOTE: no `from __future__ import annotations` here on purpose — the FastAPI route below
annotates `request: Request`, and FastAPI must see the real class (not a deferred string)
to inject it rather than treat it as a query parameter."""
import asyncio

from heynyc.core import config

from .base import InboundMessage, dispatch
from .orchestrator import Deps, handle

CHANNEL = "whatsapp_twilio"


class TwilioReplier:
    def __init__(self, client, from_: str, to: str) -> None:
        self._client, self._from, self._to = client, from_, to

    async def send_text(self, text: str) -> None:
        await asyncio.to_thread(
            self._client.messages.create, from_=self._from, to=self._to, body=text
        )

    async def send_document(self, path: str, caption: str = "") -> None:
        # Twilio fetches media by PUBLIC URL; a local file can't be attached directly. If we're
        # given a hosted URL, send it as media; otherwise degrade to a text note (Meta is the
        # document-capable channel — see the channels README).
        if path.startswith(("http://", "https://")):
            await asyncio.to_thread(
                self._client.messages.create,
                from_=self._from, to=self._to, body=caption or "", media_url=[path],
            )
        else:
            await self.send_text(f"{caption or 'Your document is ready'} — I'll send a download link shortly.")

    async def indicate_typing(self) -> None:
        return  # Twilio has no typing indicator


def to_inbound(params: dict) -> InboundMessage:
    return InboundMessage(
        channel=CHANNEL,
        sender=params.get("From", ""),
        text=params.get("Body", "") or "",
        message_id=params.get("MessageSid", ""),
        profile_name=params.get("ProfileName", "") or "",
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
        inbound = to_inbound(params)
        replier = TwilioReplier(client, from_=config.TWILIO_FROM, to=inbound.sender)
        dispatch(handle(inbound, replier, deps))   # 200 returns fast; agent runs out-of-band
        return Response(status_code=200)

    return router
