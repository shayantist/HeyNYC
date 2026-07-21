# HeyNYC channels, the messaging on-ramp

This package puts the grounded agent in front of people on the platforms they already use.
A New Yorker texts our number; the existing `Agent` answers, cited, multilingual, abstaining
when it should, and the reply comes back as platform-native messages.

## How it's shaped

One **channel-agnostic core** with thin per-provider **adapters**:

```
base.py          InboundMessage + Replier port, dispatch() fire-and-forget seam, KeyedLocks
identity.py      user_key(channel, sender, salt), the PII boundary
store.py         sqlite dedup + per-user rate-limit (crash-aware, no Redis)
format.py        render(result, channel) -> channel-appropriate chunks (SMS plain / WhatsApp native, strip {cite:Sn}, Sources footer, 4096 split)
orchestrator.py  handle(msg, replier, deps), dedup → rate → lock → run agent → reply → record
analytics.py     pseudonymous interaction log + user-flag feedback log
meta.py          Meta WhatsApp Cloud API adapter (pywa)
twilio.py        Twilio adapter (WhatsApp sandbox + SMS)
app.py           FastAPI factory: build the Agent once, mount providers, drain on shutdown
```

The orchestrator is universal; only the adapters know a provider. Adding SMS, Instagram DMs, or a
web channel is another adapter, the core doesn't change. The webhook handler verifies → `dispatch`
(fire-and-forget) → returns 200 fast; the agent runs out-of-band (pywa awaits the handler, so this is
mandatory, not optional). Per-user `asyncio.Lock` keeps one person's messages in order; a global
`Semaphore` bounds concurrency and LLM spend.

## Privacy

Senders are reduced to a salted-HMAC `user_key` at the door. **Raw phone numbers are never
persisted**, not session filenames, not telemetry, not the feedback log. The raw address lives only
in memory during a request, held by the `Replier` to send the reply. Serving requires both
`HEYNYC_PII_SALT` and `HEYNYC_PII_KEY`; conversation and draft files are encrypted, and the service
deletes expired files on startup and once per day using `HEYNYC_PII_RETENTION_DAYS` (30 by default).
Resident-authored queries, notes, and assistant text are PII-redacted before feedback is stored.

## Inherited WhatsApp constraints (documented, not built around)

- WhatsApp uses a smaller text-formatting dialect than Markdown. `format.py` converts common
  headings, bold, links, lists, and strikethrough at the channel boundary while preserving code
  and URLs, following [Twilio's WhatsApp formatting reference](https://help.twilio.com/articles/360037743094).
- Civic citations never use a third-party shortener. `format.py` groups exact row citations under
  their canonical official dataset or ArcGIS layer for display while retaining row-addressed
  evidence internally.
- Twilio WhatsApp sends a native typing indicator before model work using Twilio's
  [public-beta indicator API](https://www.twilio.com/docs/whatsapp/api/typing-indicators-resource).
  It fails open so a beta outage cannot block the final reply.
- A user message opens/resets a **24-hour service window** where free-form replies are free and
  unlimited, an inbound assistant lives here, so WhatsApp messaging cost ≈ $0; the LLM call is the
  real cost.
- Re-initiating after 24h needs a pre-approved **template** (out of scope for v1).
- Opt-in is required; quality rating can throttle; **Business Verification** (a registered entity)
  is only needed to scale past the free test number / sandbox.

## Setup (ranked, do top-to-bottom)

1. **Generate the privacy secrets:** create `HEYNYC_PII_SALT` with `openssl rand -hex 32`, create
   `HEYNYC_PII_KEY` with `uv run python -c "from heynyc.core.pii_crypto import generate_key; print(generate_key())"`,
   and put both in `.env`.
2. **Fastest demo, Twilio sandbox** (no Meta app, no money): create a Twilio account → Messaging →
   *Try WhatsApp* → send `join <code>` from your phone → copy `TWILIO_ACCOUNT_SID`,
   `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` into `.env` → point the sandbox's "when a message comes
   in" at `https://<tunnel>/webhook/twilio`.
3. **Production path, Meta Cloud API:** create a Business-type Meta App + add the WhatsApp product →
   copy `WHATSAPP_PHONE_NUMBER_ID` and `WHATSAPP_APP_SECRET` → invent a `WHATSAPP_VERIFY_TOKEN` (the
   same value in the dashboard and `.env`) → mint a system-user **Admin** token for `WHATSAPP_TOKEN`
   (the dashboard's temporary token dies in <24h) → set the callback URL to
   `https://<tunnel>/webhook/meta` and subscribe the `messages` field.
4. **Build the index** (if not already): `uv run python -m heynyc index-build`.
5. **Expose localhost:** `ngrok http 8000` (or `cloudflared tunnel --url http://localhost:8000`).
6. **Run it:** `uv run python -m heynyc serve --provider both` (or `meta` / `twilio`).

For the configured HeyNYC production WhatsApp sender and assigned ngrok development domain, run
the local demo stack with one command: `./scripts/serve_demo.sh`. It starts the Twilio provider and
the tunnel together on dedicated local port 8791 and stops both on Ctrl-C. This remains a laptop
demo, not durable hosting.

Install the deps with `uv sync --extra whatsapp` (the messaging deps live in their own extra; the base `--extra dev` install does not pull them).

## Flagging a bad answer

A user can reply with `wrong`, `report`, `incorrect`, `bad answer`, or 👎 to flag the previous
reply. Nothing is shared until they confirm: the command stages a pointer and replies with consent
copy naming exactly what a human will see (that one exchange, nothing else); only YES writes the
flag, anything else cancels and is handled as a normal message. Confirmed flags are pointers only
(no message content) joined to the encrypted session by `heynyc feedback` for local triage; the
redacted aggregate in `.data/feedback.jsonl` still feeds the systematic-error view.

## v1 scope

Built: WhatsApp (Meta + Twilio) + SMS-ready Twilio, dedup, rate-limit, per-user ordering, analytics,
user flagging. Deferred (seams in place): inbound voice notes, outbound templates, a Celery/Dramatiq queue,
dashboards, the public-reply governance layer, and the web channel. The full v1 scope and the
deferred seams are described in the project's internal multichannel-onramp design spec.
