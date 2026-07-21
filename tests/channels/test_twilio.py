import pytest

pytest.importorskip("twilio")
pytest.importorskip("fastapi")

from heynyc.channels.twilio import TwilioReplier, public_url, to_inbound


class FakeMessages:
    def __init__(self):
        self.created = []

    def create(self, from_, to, body, media_url=None):
        self.created.append((from_, to, body, media_url))


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
    assert client.messages.created == [("whatsapp:+14155238886", "whatsapp:+1555", "hi", None)]
    assert client.requests == [(
        "POST",
        "https://messaging.twilio.com/v3/Indicators/Typing.json",
        {"channel": "WHATSAPP", "messageId": "SM1"},
        {"Content-Type": "application/json"},
    )]


async def test_twilio_replier_splits_bodies_at_twilio_limit():
    client = FakeClient()
    replier = TwilioReplier(
        client, from_="whatsapp:+14155238886", to="whatsapp:+1555", message_id="SM1"
    )

    await replier.send_text(("a" * 1200) + "\n\n" + ("b" * 1200))

    bodies = [body for _, _, body, _ in client.messages.created]
    assert bodies == [f"1/2 {'a' * 1200}", f"2/2 {'b' * 1200}"]
    assert all(len(body) <= 1600 for body in bodies)


async def test_twilio_document_caption_uses_the_same_limit():
    client = FakeClient()
    replier = TwilioReplier(client, from_="whatsapp:+1", to="whatsapp:+2")

    await replier.send_document("https://example.com/file.pdf", ("a" * 1200) + "\n\n" + ("b" * 1200))

    assert client.messages.created == [
        ("whatsapp:+1", "whatsapp:+2", f"1/2 {'a' * 1200}", None),
        ("whatsapp:+1", "whatsapp:+2", f"2/2 {'b' * 1200}", ["https://example.com/file.pdf"]),
    ]


async def test_twilio_document_caption_does_not_renumber_bounded_parts():
    client = FakeClient()
    replier = TwilioReplier(client, from_="whatsapp:+1", to="whatsapp:+2")
    caption = "\n\n".join(character * 1584 for character in "abc")

    await replier.send_document("https://example.com/file.pdf", caption)

    bodies = [body for _, _, body, _ in client.messages.created]
    assert len(bodies) == 3
    assert [body[:4] for body in bodies] == ["1/3 ", "2/3 ", "3/3 "]
    assert all(len(body) <= 1600 for body in bodies)


async def test_twilio_document_allows_an_empty_caption():
    client = FakeClient()
    replier = TwilioReplier(client, from_="whatsapp:+1", to="whatsapp:+2")

    await replier.send_document("https://example.com/file.pdf")

    assert client.messages.created == [
        ("whatsapp:+1", "whatsapp:+2", "", ["https://example.com/file.pdf"]),
    ]


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
        ("whatsapp:+14155238886", "whatsapp:+1555", "final answer", None)
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


def test_to_inbound_distinguishes_sms_from_whatsapp():
    m = to_inbound({
        "From": "+15551234567",
        "To": "+18882120042",
        "Body": "hi",
        "MessageSid": "SM2",
    })

    assert m.channel == "sms_twilio"


def test_twilio_router_uses_recipient_as_reply_sender_and_rejects_missing_addresses(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from heynyc.channels import twilio

    routed = []

    class Validator:
        def validate(self, url, params, signature):
            return True

    class Replier:
        def __init__(self, client, from_, to, message_id):
            routed.append((from_, to, message_id))

    monkeypatch.setattr("twilio.request_validator.RequestValidator", lambda token: Validator())
    monkeypatch.setattr("twilio.rest.Client", lambda sid, token: object())
    monkeypatch.setattr(twilio, "TwilioReplier", Replier)
    monkeypatch.setattr(twilio, "dispatch", lambda coro: coro.close())

    app = FastAPI()
    app.include_router(twilio.make_twilio_router(object()))
    with TestClient(app) as client:
        accepted = client.post("/webhook/twilio", data={
            "From": "+15551234567",
            "To": "+18882120042",
            "Body": "hi",
            "MessageSid": "SM2",
        })
        missing_to = client.post("/webhook/twilio", data={
            "From": "+15551234567",
            "Body": "hi",
            "MessageSid": "SM3",
        })

    assert accepted.status_code == 200
    assert missing_to.status_code == 400
    assert routed == [("+18882120042", "+15551234567", "SM2")]


def test_to_inbound_maps_twilio_media_without_downloading_it():
    m = to_inbound({
        "From": "whatsapp:+1555",
        "Body": "What does this notice say?",
        "MessageSid": "MM1",
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/media/ME1",
        "MediaContentType0": "image/jpeg",
    })

    assert m.media == [{
        "url": "https://api.twilio.com/media/ME1",
        "content_type": "image/jpeg",
    }]


def test_signature_validation_accepts_genuine_and_rejects_tampered():
    from twilio.request_validator import RequestValidator
    token = "test_auth_token"
    url = "https://bot.ngrok.app/webhook/twilio"
    params = {"From": "whatsapp:+1555", "Body": "hi", "MessageSid": "SM1"}
    sig = RequestValidator(token).compute_signature(url, params)
    v = RequestValidator(token)
    assert v.validate(url, params, sig) is True
    assert v.validate(url, {**params, "Body": "tampered"}, sig) is False
