import pytest

pytest.importorskip("pywa_async")

from heynyc.channels.meta import MetaReplier, to_inbound


class FakeWa:
    def __init__(self):
        self.sent, self.typed = [], []

    async def send_message(self, to, text, preview_url=False):
        self.sent.append((to, text, preview_url))

    async def indicate_typing(self, message_id):
        self.typed.append(message_id)


class FakeUser:
    wa_id = "+15551234567"
    name = "Ada"


class FakeMsg:
    id = "wamid.X"
    text = "hi"
    from_user = FakeUser()


async def test_meta_replier_sends_with_preview():
    wa = FakeWa()
    r = MetaReplier(wa, to="+15551234567", message_id="wamid.X")
    await r.send_text("hello")
    await r.indicate_typing()
    assert wa.sent == [("+15551234567", "hello", True)]
    assert wa.typed == ["wamid.X"]


def test_to_inbound_maps_fields():
    m = to_inbound(FakeMsg())
    assert m.channel == "whatsapp_meta"
    assert m.sender == "+15551234567" and m.text == "hi"
    assert m.message_id == "wamid.X" and m.profile_name == "Ada"
