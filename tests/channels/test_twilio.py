import pytest

pytest.importorskip("twilio")
pytest.importorskip("fastapi")

from heynyc.channels.twilio import TwilioReplier, to_inbound, public_url


class FakeMessages:
    def __init__(self):
        self.created = []

    def create(self, from_, to, body):
        self.created.append((from_, to, body))


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


class FakeURL:
    scheme = "http"
    netloc = "internal:8000"
    path = "/webhook/twilio"
    query = ""


class FakeRequest:
    def __init__(self, headers):
        self.headers = headers
        self.url = FakeURL()


async def test_twilio_replier_uses_threadpool_create():
    client = FakeClient()
    r = TwilioReplier(client, from_="whatsapp:+14155238886", to="whatsapp:+1555")
    await r.send_text("hi")
    await r.indicate_typing()   # no-op, must not raise
    assert client.messages.created == [("whatsapp:+14155238886", "whatsapp:+1555", "hi")]


def test_public_url_honors_forwarded_headers():
    req = FakeRequest({"x-forwarded-proto": "https", "x-forwarded-host": "bot.ngrok.app"})
    assert public_url(req) == "https://bot.ngrok.app/webhook/twilio"


def test_to_inbound_maps_twilio_params():
    m = to_inbound({"From": "whatsapp:+1555", "Body": "hi", "MessageSid": "SM1", "ProfileName": "Ada"})
    assert m.channel == "whatsapp_twilio" and m.sender == "whatsapp:+1555"
    assert m.text == "hi" and m.message_id == "SM1" and m.profile_name == "Ada"


def test_signature_validation_accepts_genuine_and_rejects_tampered():
    from twilio.request_validator import RequestValidator
    token = "test_auth_token"
    url = "https://bot.ngrok.app/webhook/twilio"
    params = {"From": "whatsapp:+1555", "Body": "hi", "MessageSid": "SM1"}
    sig = RequestValidator(token).compute_signature(url, params)
    v = RequestValidator(token)
    assert v.validate(url, params, sig) is True
    assert v.validate(url, {**params, "Body": "tampered"}, sig) is False
