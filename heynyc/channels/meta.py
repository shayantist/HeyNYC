"""Meta WhatsApp Cloud API adapter. pywa owns the webhook route, the GET handshake,
and the X-Hub-Signature-256 HMAC; we own normalization + the out-of-band reply."""
from __future__ import annotations

from heynyc.core import config

from .base import InboundMessage, dispatch
from .orchestrator import Deps, handle

CHANNEL = "whatsapp_meta"


class MetaReplier:
    def __init__(self, wa, to: str, message_id: str) -> None:
        self._wa, self._to, self._mid = wa, to, message_id

    async def send_text(self, text: str) -> None:
        await self._wa.send_message(to=self._to, text=text, preview_url=True)

    async def send_document(self, path: str, caption: str = "") -> None:
        await self._wa.send_document(to=self._to, document=path, caption=caption)

    async def indicate_typing(self) -> None:
        await self._wa.indicate_typing(message_id=self._mid)   # also marks read


def to_inbound(msg) -> InboundMessage:
    return InboundMessage(
        channel=CHANNEL,
        sender=msg.from_user.wa_id,
        text=msg.text or "",
        message_id=msg.id,
        profile_name=getattr(msg.from_user, "name", "") or "",
    )


def attach_meta(app, deps: Deps) -> None:
    from pywa_async import WhatsApp, filters, types

    wa = WhatsApp(
        phone_id=config.WHATSAPP_PHONE_NUMBER_ID,
        token=config.WHATSAPP_TOKEN,
        server=app,
        webhook_endpoint="/webhook/meta",
        verify_token=config.WHATSAPP_VERIFY_TOKEN,
        app_secret=config.WHATSAPP_APP_SECRET,
        validate_updates=True,
    )

    @wa.on_message(filters.text)
    async def _on_text(client: WhatsApp, msg: types.Message) -> None:
        inbound = to_inbound(msg)
        replier = MetaReplier(client, to=inbound.sender, message_id=inbound.message_id)
        dispatch(handle(inbound, replier, deps))   # 200 returns fast; agent runs out-of-band
