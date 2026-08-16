import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("twilio")
pytest.importorskip("fastapi")

from heynyc.channels.store import ChannelStore
from heynyc.channels.twilio import (
    TwilioOutboxReplier,
    TwilioReplier,
    public_url,
    to_inbound,
)
from heynyc.core import pii_crypto


class FakeMessages:
    def __init__(self):
        self.created = []

    def create(self, from_, to, body, media_url=None):
        self.created.append((from_, to, body, media_url))
        return SimpleNamespace(sid=f"SM-out-{len(self.created)}")


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


async def test_twilio_outbox_refreshes_typing_until_reply_is_ready(monkeypatch):
    from heynyc.channels import twilio

    class Store:
        def stage_outbox(self, message_id, parts):
            return None

    monkeypatch.setattr(twilio, "_TYPING_REFRESH_SECONDS", 0, raising=False)
    client = FakeClient()
    replier = TwilioOutboxReplier(
        Store(),
        client,
        from_="whatsapp:+14155238886",
        to="whatsapp:+1555",
        message_id="SM1",
    )

    await replier.indicate_typing()
    for _ in range(20):
        await asyncio.sleep(0)

    assert len(client.requests) > 1
    await replier.finalize()
    request_count = len(client.requests)
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(client.requests) == request_count


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


def test_twilio_router_persists_before_ack_and_rejects_missing_addresses(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from heynyc.channels import twilio

    enqueued = []

    class Validator:
        def validate(self, url, params, signature):
            return True

    class Store:
        def enqueue(self, message_id, user_key, payload):
            enqueued.append((message_id, user_key, json.loads(payload)))
            return True

    class Worker:
        def __init__(self):
            self.wakes = 0

        def wake(self):
            self.wakes += 1

    monkeypatch.setattr("twilio.request_validator.RequestValidator", lambda token: Validator())
    worker = Worker()
    deps = SimpleNamespace(store=Store(), salt="test-salt")

    app = FastAPI()
    app.include_router(twilio.make_twilio_router(deps, worker))
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
    assert worker.wakes == 1
    assert len(enqueued) == 1
    message_id, stored_user_key, payload = enqueued[0]
    assert message_id == "SM2"
    assert stored_user_key and "+15551234567" not in stored_user_key
    assert payload == {
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "hi",
        "message_id": "SM2",
        "profile_name": "",
        "media": [],
    }


def test_twilio_router_acknowledges_duplicate_without_waking_worker(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from heynyc.channels import twilio

    class Validator:
        def validate(self, url, params, signature):
            return True

    class Store:
        def enqueue(self, message_id, user_key, payload):
            return False

    class Worker:
        def __init__(self):
            self.wakes = 0

        def wake(self):
            self.wakes += 1

    monkeypatch.setattr("twilio.request_validator.RequestValidator", lambda token: Validator())
    worker = Worker()
    app = FastAPI()
    app.include_router(twilio.make_twilio_router(
        SimpleNamespace(store=Store(), salt="test-salt"), worker,
    ))

    with TestClient(app) as client:
        response = client.post("/webhook/twilio", data={
            "From": "+15551234567",
            "To": "+18882120042",
            "Body": "same delivery",
            "MessageSid": "SM-duplicate",
        })

    assert response.status_code == 200
    assert worker.wakes == 0


async def test_twilio_worker_processes_envelope_and_records_outbound_sids(monkeypatch, tmp_path):
    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    payload = json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "hello",
        "message_id": "SM-in-1",
        "profile_name": "",
        "media": [],
    })
    store.enqueue("SM-in-1", "u1", payload)
    calls = []

    async def fake_handle(message, replier, deps, *, deduplicate=True):
        calls.append((message, deduplicate))
        await replier.send_text("reply")

    monkeypatch.setattr(twilio, "handle", fake_handle)
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=FakeClient(), lease_s=30,
    )

    assert await worker.process_one() is True

    message, deduplicate = calls[0]
    assert message.message_id == "SM-in-1"
    assert message.sender == "+15551234567"
    assert deduplicate is False
    assert store._db.execute(
        "SELECT state, payload, outbound_ids FROM inbox WHERE message_id = ?", ("SM-in-1",)
    ).fetchone() == ("sent", None, '["SM-out-1"]')


async def test_twilio_worker_stops_retrying_after_bounded_failures(monkeypatch, tmp_path):
    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-in-fail", "u1", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "hello",
        "message_id": "SM-in-fail",
        "profile_name": "",
        "media": [],
    }))

    async def fail_handle(message, replier, deps, *, deduplicate=True):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(twilio, "handle", fail_handle)
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=FakeClient(), lease_s=30,
        retry_after_s=0, max_attempts=3,
    )

    assert [await worker.process_one() for _ in range(3)] == [True, True, True]
    assert await worker.process_one() is False
    state, attempts, payload = store._db.execute(
        "SELECT state, attempts, payload FROM inbox WHERE message_id = ?", ("SM-in-fail",)
    ).fetchone()
    assert state == "failed"
    assert attempts == 3
    assert payload is not None


async def test_twilio_worker_resumes_only_unsent_reply_parts(monkeypatch, tmp_path):
    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-partial", "u1", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "hello",
        "message_id": "SM-partial",
        "profile_name": "",
        "media": [],
    }))

    async def partial_handle(message, replier, deps, *, deduplicate=True):
        await replier.send_text("first accepted part")
        await replier.send_text("second accepted part")

    class FlakyMessages(FakeMessages):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def create(self, from_, to, body, media_url=None):
            if body == "second accepted part" and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("temporary provider failure")
            return super().create(from_, to, body, media_url)

    client = FakeClient()
    client.messages = FlakyMessages()

    monkeypatch.setattr(twilio, "handle", partial_handle)
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=client, retry_after_s=0,
    )

    assert await worker.process_one() is True
    state, outbox, delivered_parts = store._db.execute(
        "SELECT state, outbox, delivered_parts FROM inbox WHERE message_id = ?", ("SM-partial",)
    ).fetchone()
    assert state == "retrying" and delivered_parts == 1
    assert isinstance(outbox, bytes)
    assert b"first accepted part" not in outbox and b"second accepted part" not in outbox
    assert await worker.process_one() is True
    assert await worker.process_one() is False
    assert store._db.execute(
        "SELECT state, payload, outbound_ids FROM inbox WHERE message_id = ?", ("SM-partial",)
    ).fetchone() == ("sent", None, '["SM-out-1", "SM-out-2"]')
    assert [body for _, _, body, _ in client.messages.created] == [
        "first accepted part", "second accepted part",
    ]


async def test_twilio_activates_approval_only_after_every_review_part_is_delivered(
    monkeypatch,
    tmp_path,
):
    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-approval", "u1", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "submit it",
        "message_id": "SM-approval",
        "profile_name": "",
        "media": [],
    }))

    async def approval_handle(message, replier, deps, *, deduplicate=True):
        await replier.send_text("review part one")
        await replier.send_text("review part two")
        await replier.finalize()
        replier.stage_pending_approval("u1", b'{"pending":true}', ttl_s=60)

    class FlakyMessages(FakeMessages):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def create(self, from_, to, body, media_url=None):
            if body == "review part two" and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("temporary provider failure")
            return super().create(from_, to, body, media_url)

    client = FakeClient()
    client.messages = FlakyMessages()
    monkeypatch.setattr(twilio, "handle", approval_handle)
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=client, retry_after_s=0,
    )

    assert await worker.process_one() is True
    assert store.has_pending_approval("u1") is False

    assert await worker.process_one() is True
    assert store.get_pending_approval("u1") == b'{"pending":true}'


async def test_twilio_rejects_an_approval_outbox_bound_to_another_resident(
    monkeypatch,
    tmp_path,
):
    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-wrong-resident", "resident-b", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "submit it",
        "message_id": "SM-wrong-resident",
        "profile_name": "",
        "media": [],
    }))

    async def mismatched_handle(message, replier, deps, *, deduplicate=True):
        await replier.send_text("review")
        await replier.finalize()
        replier.stage_pending_approval("resident-a", b'{"pending":true}', ttl_s=60)

    client = FakeClient()
    monkeypatch.setattr(twilio, "handle", mismatched_handle)
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=client, retry_after_s=0,
    )

    assert await worker.process_one() is True
    assert client.messages.created == []
    assert store.has_pending_approval("resident-a") is False
    assert store.has_pending_approval("resident-b") is False


async def test_legacy_restart_scrubs_approval_from_an_unfinished_twilio_outbox(
    monkeypatch,
    tmp_path,
):
    import base64

    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    path = tmp_path / "channels.sqlite3"
    store = ChannelStore(
        path, rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-rollback", "resident", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "submit it",
        "message_id": "SM-rollback",
        "profile_name": "",
        "media": [],
    }))
    store.stage_outbox("SM-rollback", [
        {"body": "Already delivered"},
        {
            "body": "Review the proposed action",
            twilio.PENDING_APPROVAL_OUTBOX_KEY: {
                "user_key": "resident",
                "state": base64.b64encode(b'{"pending":true}').decode("ascii"),
                "ttl_s": 60,
            },
        },
    ])
    store.record_outbound("SM-rollback", "SM-out-already-delivered")

    store.clear_pending_approvals()
    outbox, delivered_parts, outbound_ids = store._db.execute(
        "SELECT outbox, delivered_parts, outbound_ids FROM inbox WHERE message_id = ?",
        ("SM-rollback",),
    ).fetchone()
    scrubbed = json.loads(pii_crypto.decrypt(outbox))
    assert [part["body"] for part in scrubbed] == [
        "Already delivered",
        "Review the proposed action",
    ]
    assert twilio.PENDING_APPROVAL_OUTBOX_KEY not in scrubbed[-1]
    assert delivered_parts == 1
    assert json.loads(outbound_ids) == ["SM-out-already-delivered"]
    client = FakeClient()
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=client, retry_after_s=0,
    )
    assert await worker.process_one() is True
    assert [message[2] for message in client.messages.created] == [
        "Review the proposed action",
    ]

    later_pydantic_store = ChannelStore(
        path, rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    assert later_pydantic_store.has_pending_approval("resident") is False


async def test_twilio_worker_recovers_queued_work_on_startup(monkeypatch, tmp_path):
    import asyncio

    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-before-start", "u1", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15551234567",
        "recipient": "+18882120042",
        "text": "survive restart",
        "message_id": "SM-before-start",
        "profile_name": "",
        "media": [],
    }))
    processed = asyncio.Event()

    async def fake_handle(message, replier, deps, *, deduplicate=True):
        await replier.send_text("recovered")
        processed.set()

    monkeypatch.setattr(twilio, "handle", fake_handle)
    worker = twilio.TwilioInboxWorker(SimpleNamespace(store=store), client=FakeClient())

    worker.start()
    await asyncio.wait_for(processed.wait(), timeout=1)
    await worker.stop()

    assert store._db.execute(
        "SELECT state FROM inbox WHERE message_id = ?", ("SM-before-start",)
    ).fetchone() == ("sent",)


async def test_twilio_worker_processes_different_senders_concurrently(monkeypatch, tmp_path):
    import asyncio

    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    for index in (1, 2):
        store.enqueue(f"SM-{index}", f"u{index}", json.dumps({
            "channel": "sms_twilio",
            "sender": f"+1555000000{index}",
            "recipient": "+18882120042",
            "text": "hello",
            "message_id": f"SM-{index}",
            "profile_name": "",
            "media": [],
        }))
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def fake_handle(message, replier, deps, *, deduplicate=True):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()

    monkeypatch.setattr(twilio, "handle", fake_handle)
    worker = twilio.TwilioInboxWorker(
        SimpleNamespace(store=store), client=FakeClient(), concurrency=2,
    )

    worker.start()
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await worker.stop()

    assert started == 2


async def test_twilio_worker_quarantines_unreadable_row_and_continues(monkeypatch, tmp_path):
    from heynyc.channels import twilio

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600,
    )
    store.enqueue("SM-bad", "u1", '{"text":"bad"}')
    store._db.execute("UPDATE inbox SET payload = ? WHERE message_id = ?", (b"corrupt", "SM-bad"))
    store._db.commit()
    store.enqueue("SM-good", "u2", json.dumps({
        "channel": "sms_twilio",
        "sender": "+15550000002",
        "recipient": "+18882120042",
        "text": "hello",
        "message_id": "SM-good",
        "profile_name": "",
        "media": [],
    }))

    async def fake_handle(message, replier, deps, *, deduplicate=True):
        await replier.send_text("ok")

    monkeypatch.setattr(twilio, "handle", fake_handle)
    worker = twilio.TwilioInboxWorker(SimpleNamespace(store=store), client=FakeClient())

    assert await worker.process_one() is True
    assert await worker.process_one() is True
    assert store._db.execute(
        "SELECT message_id, state FROM inbox ORDER BY message_id"
    ).fetchall() == [("SM-bad", "failed"), ("SM-good", "sent")]


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
