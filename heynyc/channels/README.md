# HeyNYC channels — the messaging on-ramp

This package puts the grounded agent in front of people on the platforms they already use.
A New Yorker texts our number; the existing `Agent` answers — cited, multilingual, abstaining
when it should — and the reply comes back as platform-native messages.

## How it's shaped

One **channel-agnostic core** with thin per-provider **adapters**:

```
base.py          InboundMessage + Replier port, dispatch() fire-and-forget seam, KeyedLocks
identity.py      user_key(channel, sender, salt) — the PII boundary
store.py         sqlite dedup + per-user rate-limit (crash-aware, no Redis)
format.py        render(result) -> WhatsApp chunks (strip {cite:Sn}, Sources footer, 4096 split)
orchestrator.py  handle(msg, replier, deps) — dedup → rate → lock → run agent → reply → record
analytics.py     pseudonymous interaction log + user-flag feedback log
meta.py          Meta WhatsApp Cloud API adapter (pywa)
twilio.py        Twilio adapter (WhatsApp sandbox + SMS)
app.py           FastAPI factory: build the Agent once, mount providers, drain on shutdown
```

The orchestrator is universal; only the adapters know a provider. Adding SMS, Instagram DMs, or a
web channel is another adapter — the core doesn't change. The webhook handler verifies → `dispatch`
(fire-and-forget) → returns 200 fast; the agent runs out-of-band (pywa awaits the handler, so this is
mandatory, not optional). Per-user `asyncio.Lock` keeps one person's messages in order; a global
`Semaphore` bounds concurrency and LLM spend.

## Privacy

Senders are reduced to a salted-HMAC `user_key` at the door. **Raw phone numbers are never
persisted** — not session filenames, not telemetry, not the feedback log. The raw address lives only
in memory during a request, held by the `Replier` to send the reply. Set `HEYNYC_PII_SALT` to serve.

## Inherited WhatsApp constraints (documented, not built around)

- A user message opens/resets a **24-hour service window** where free-form replies are free and
  unlimited — an inbound assistant lives here, so WhatsApp messaging cost ≈ $0; the LLM call is the
  real cost.
- Re-initiating after 24h needs a pre-approved **template** (out of scope for v1).
- Opt-in is required; quality rating can throttle; **Business Verification** (a registered entity)
  is only needed to scale past the free test number / sandbox.

## Setup (ranked, do top-to-bottom)

1. **Generate the salt:** `HEYNYC_PII_SALT=$(openssl rand -hex 32)` → put it in `.env`.
2. **Fastest demo — Twilio sandbox** (no Meta app, no money): create a Twilio account → Messaging →
   *Try WhatsApp* → send `join <code>` from your phone → copy `TWILIO_ACCOUNT_SID`,
   `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` into `.env` → point the sandbox's "when a message comes
   in" at `https://<tunnel>/webhook/twilio`.
3. **Production path — Meta Cloud API:** create a Business-type Meta App + add the WhatsApp product →
   copy `WHATSAPP_PHONE_NUMBER_ID` and `WHATSAPP_APP_SECRET` → invent a `WHATSAPP_VERIFY_TOKEN` (the
   same value in the dashboard and `.env`) → mint a system-user **Admin** token for `WHATSAPP_TOKEN`
   (the dashboard's temporary token dies in <24h) → set the callback URL to
   `https://<tunnel>/webhook/meta` and subscribe the `messages` field.
4. **Build the index** (if not already): `uv run python -m heynyc index-build`.
5. **Expose localhost:** `ngrok http 8000` (or `cloudflared tunnel --url http://localhost:8000`).
6. **Run it:** `uv run python -m heynyc serve --provider both` (or `meta` / `twilio`).

Install the deps with `uv sync --extra whatsapp` (the messaging deps live in their own extra; the base `--extra dev` install does not pull them).

## Flagging a bad answer

A user can reply with `wrong`, `report`, `incorrect`, `bad answer`, or 👎 to flag the previous reply.
That turn is written to `.data/feedback.jsonl` (keyed by `user_key`) for human review — shaped to
feed the agent-as-judge rubric later.

## v1 scope

Built: WhatsApp (Meta + Twilio) + SMS-ready Twilio, dedup, rate-limit, per-user ordering, analytics,
user flagging. Deferred (seams in place): inbound voice notes, outbound templates, a Celery/Dramatiq queue,
dashboards, the public-reply governance layer, and the web channel. See the spec:
`docs/superpowers/specs/2026-06-29-multichannel-onramp-design.md`.
