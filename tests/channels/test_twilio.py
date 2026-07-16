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
        self.requests = []

    def request(self, method, url, data, headers):
        self.requests.append((method, url, data, headers))

        class Response:
            status_code = 200

        return Response()


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
    r = TwilioReplier(
        client, from_="whatsapp:+14155238886", to="whatsapp:+1555", message_id="SM1"
    )
    await r.send_text("hi")
    await r.indicate_typing()
    assert client.messages.created == [("whatsapp:+14155238886", "whatsapp:+1555", "hi")]
    assert client.requests == [(
        "POST",
        "https://messaging.twilio.com/v3/Indicators/Typing.json",
        {"channel": "WHATSAPP", "messageId": "SM1"},
        {"Content-Type": "application/json"},
    )]


async def test_twilio_typing_indicator_failure_does_not_block_the_reply():
    class FailingClient(FakeClient):
        def request(self, method, url, data, headers):
            raise RuntimeError("beta unavailable")

    client = FailingClient()
    replier = TwilioReplier(
        client, from_="whatsapp:+14155238886", to="whatsapp:+1555", message_id="SM1"
    )

    await replier.indicate_typing()
    await replier.send_text("final answer")

    assert client.messages.created == [
        ("whatsapp:+14155238886", "whatsapp:+1555", "final answer")
    ]


async def test_twilio_typing_indicator_is_a_noop_for_sms():
    client = FakeClient()
    replier = TwilioReplier(client, from_="+14155550100", to="+1555", message_id="SM1")

    await replier.indicate_typing()

    assert client.requests == []


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
