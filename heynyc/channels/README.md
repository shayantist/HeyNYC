# HeyNYC channels, the messaging on-ramp

This package puts the grounded agent in front of people on the platforms they already use. A New Yorker texts our number; the selected runtime answers, cited, multilingual, abstaining when it should, and the reply comes back as platform-native messages.

## How it's shaped

One **channel-agnostic core** with thin per-provider **adapters**:

```
base.py          InboundMessage + Replier port, Meta dispatch seam, KeyedLocks
identity.py      user_key(channel, sender, salt), the PII boundary
store.py         encrypted SQLite inbox + dedup + per-user rate-limit
format.py        render(result, channel) -> text-channel inline cited links, 4096 split
orchestrator.py  handle(msg, replier, deps), dedup → rate → lock → run agent → reply → record
analytics.py     pseudonymous interaction log + user-flag feedback log
meta.py          Meta WhatsApp Cloud API adapter (pywa)
twilio.py        Twilio adapter + restart-safe inbox worker (WhatsApp + SMS)
app.py           FastAPI factory: build the selected runtime once, mount providers, start/stop workers
```

The orchestrator is universal; only the adapters know a provider. Adding Instagram DMs or a web channel is another adapter, the core doesn't change. The Twilio webhook verifies, encrypts and persists the inbound envelope by `MessageSid`, then returns 200. A bounded worker resumes unfinished work after restart. Meta still uses the in-memory dispatch seam. Per-user ordering keeps one person's messages sequential while the global concurrency limit lets different residents proceed together and bounds LLM spend.

## Privacy

Senders are reduced to a salted-HMAC `user_key` at the door. Raw phone numbers and message text in the Twilio inbox are encrypted at rest; session filenames, telemetry, and the feedback index use only the pseudonymous key. Serving requires both `HEYNYC_PII_SALT` and `HEYNYC_PII_KEY`. The service deletes expired inbox rows, conversations, and drafts on startup and once per day using `HEYNYC_PII_RETENTION_DAYS` (30 by default). Successful inbox work is scrubbed immediately after Twilio accepts every reply part.

## Inherited WhatsApp constraints (documented, not built around)

- WhatsApp uses a smaller text-formatting dialect than Markdown. `format.py` converts common
  headings, bold, links, lists, and strikethrough at the channel boundary while preserving code and URLs, following [Twilio's WhatsApp formatting reference](https://help.twilio.com/articles/360037743094).
- Civic citations never use a third-party shortener. `format.py` places each exact cited link
  beside its supported claim on SMS and WhatsApp while retaining row-addressed evidence internally.
- Twilio WhatsApp sends a native typing indicator before model work using Twilio's
  [public-beta indicator API](https://www.twilio.com/docs/whatsapp/api/typing-indicators-resource). It fails open so a beta outage cannot block the final reply.
- A user message opens/resets a **24-hour service window** where free-form replies are free and
  unlimited, an inbound assistant lives here, so WhatsApp messaging cost ≈ $0; the LLM call is the real cost.
- Re-initiating after 24h needs a pre-approved **template** (out of scope for v1).
- Opt-in is required; quality rating can throttle; **Business Verification** (a registered entity)
  is only needed to scale past the free test number / sandbox.

## Setup (ranked, do top-to-bottom)

1. **Generate the privacy secrets:** create `HEYNYC_PII_SALT` with `openssl rand -hex 32`, create
   `HEYNYC_PII_KEY` with `uv run python -c "from heynyc.core.pii_crypto import generate_key; print(generate_key())"`, and put both in `.env`.
2. **Fastest demo, Twilio sandbox** (no Meta app, no money): create a Twilio account → Messaging →
   *Try WhatsApp* → send `join <code>` from your phone → copy `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` into `.env` → point the sandbox's "when a message comes in" at `https://<tunnel>/webhook/twilio`.
3. **Production path, Meta Cloud API:** create a Business-type Meta App + add the WhatsApp product →
   copy `WHATSAPP_PHONE_NUMBER_ID` and `WHATSAPP_APP_SECRET` → invent a `WHATSAPP_VERIFY_TOKEN` (the same value in the dashboard and `.env`) → mint a system-user **Admin** token for `WHATSAPP_TOKEN` (the dashboard's temporary token dies in <24h) → set the callback URL to `https://<tunnel>/webhook/meta` and subscribe the `messages` field.
4. **Build the index** (if not already): `uv run python -m heynyc index-build`.
5. **Expose localhost:** `ngrok http 8000` (or `cloudflared tunnel --url http://localhost:8000`).
6. **Run it:** `uv run python -m heynyc serve --provider both` (or `meta` / `twilio`).

For the configured HeyNYC production WhatsApp sender and assigned ngrok development domain, run the local demo stack with one command: `./scripts/serve_demo.sh`. It starts the Twilio provider and the tunnel together on dedicated local port 8791 and stops both on Ctrl-C. This remains a laptop demo, not durable hosting.

Install the deps with `uv sync --extra whatsapp` (the messaging deps live in their own extra; the base `--extra dev` install does not pull them).

## Agent runtime selection

[`HEYNYC_AGENT_RUNTIME`](../core/config.py) selects one runtime for the entire process. `pydantic`
is the default and requires
`uv sync --extra pydantic-ai`. Both use the same channel orchestrator, encrypted transcript and
channel store, tools, grounding policy, and SMS and WhatsApp renderers. Exact native snapshots are
runtime-specific; the shared transcript preserves model-visible turns across a rollback. This is
an operator switch, not per-resident traffic splitting.

Set `HEYNYC_AGENT_RUNTIME=legacy` for the retained rollback path. Do not change the runtime during
an ordinary unattended restart. A production switch needs a
supervised health check, real SMS and WhatsApp smoke, continuity and approval-resume checks, and
an immediate rollback path to `legacy`. Starting the [`legacy` runtime](app.py) cancels
[pending Pydantic tool proposals](store.py) so an old approval cannot reappear after intervening
conversation. The resident must request and review that action again. The internal WSL runbook
owns the full procedure.

An approval becomes actionable only after the complete review is accepted for delivery and its
proposal turn is committed. Twilio keeps the encrypted native approval state inert with the
durable outbox until every review part is provider-accepted; a partial or failed review cannot be
approved.

## Single-host deployment and recovery

The WSL pilot keeps `.env` and resident state outside versioned release directories. Each release links the shared `.env`, and `HEYNYC_DATA_DIR` points at the shared data directory. This keeps one encryption identity and one resident-data store across exact-SHA releases.

[`scripts/deploy_via_ssh.sh`](../../scripts/deploy_via_ssh.sh) is the command to run on the Mac. It connects through the operator's local `heynyc-wsl` SSH alias, uses WSL's native [`--cd ~` home-directory option](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#change-directory-to-home), fast-forwards the WSL checkout, and opens an interactive terminal for the target deployment. The repository contains no private address or connection details; those stay in the operator's local [OpenSSH configuration](https://man.openbsd.org/ssh_config).

[`scripts/deploy.sh`](../../scripts/deploy.sh) runs inside the target Linux host, which is WSL for the current pilot. It fetches and pins the head of the configured pushed `origin/*` ref; an optional full commit selects an earlier exact release. It takes a deployment lock, rebuilds that release from a fresh detached worktree while the old release serves, then hands control to the target release's `deploy.sh`. The target builds an isolated retrieval index, requires successful Notify NYC source probes, and preserves the prior index with an inventory and SHA-256 checksums under `to-delete`. It configures native [systemd temporary-file cleanup](https://www.freedesktop.org/software/systemd/man/systemd-tmpfiles.html) for expired snapshots, then stops writes once for a resident-state snapshot, application-level verification, and the index and release-pointer swaps. It rejects modified or extra tracked release content, requires local and public health, and compares recent provider-side inbound SIDs with the local inbox using Twilio's [Messages resource](https://www.twilio.com/docs/messaging/api/message-resource). Reconciliation reports counts and missing SIDs, never message bodies, and never replays a resident message automatically. A provider record without either Twilio timestamp is included conservatively and counted separately for operator review.

Configure the local alias with private values outside this repository, then run only after CI passes:

```bash
./scripts/deploy_via_ssh.sh

# Optional exact rollback or supervised pin:
./scripts/deploy_via_ssh.sh <40-character-sha>

# Directly on the WSL host, including a supervised candidate:
HEYNYC_DEPLOY_REF=origin/codex/pydantic-ai-refactor \
  ./scripts/deploy.sh
```

[`scripts/state_snapshot.py`](../../scripts/state_snapshot.py) copies SQLite through Python's [online backup API](https://docs.python.org/3.11/library/sqlite3.html#sqlite3.Connection.backup), copies the rest of the data directory while writes are stopped, and records file sizes, SHA-256 hashes, the application commit, SQLite schema version, and the current deletion generation. The generation marker is not copied into snapshot data. A confirmed deletion advances the live marker first, so verification and restore reject any older snapshot. Verification also rejects snapshots past `HEYNYC_PII_RETENTION_DAYS`. Application verification opens the schema read-only and authenticates encrypted inbox, session, draft, and feedback records with the configured key without printing their content. `restore` refuses a nonempty destination. Recovery therefore restores into a new directory, verifies it against the live generation marker, and uses an operator-controlled directory switch instead of overwriting live state:

```bash
uv run python scripts/state_snapshot.py verify <snapshot-directory> --deletion-generation <live-data-directory>/.deletion-generation
uv run python scripts/state_snapshot.py restore <snapshot-directory> --target <new-empty-data-directory> --deletion-generation <live-data-directory>/.deletion-generation
```

After the new process starts, the deploy script never restores an older snapshot automatically. The process may already have accepted new work, so restoring old state could discard a resident message. A failed health check attempts to stop the service and prints the snapshot and previous release for supervised recovery, including a separate warning if systemd could not confirm the stop.

Inbox inspection stays metadata-only. Operators may inspect `message_id`, `state`, `attempts`, `delivered_parts`, and timestamps, but not `payload` or `outbox`. A generation failure before the outbox is staged can run the model again and produce different wording or additional spend; the session turn is still uncommitted at that point. Delivery resumes from the staged outbox after it exists.

```sql
SELECT state, COUNT(*), MIN(updated_at), MAX(updated_at)
FROM inbox GROUP BY state ORDER BY state;

SELECT message_id, state, attempts, delivered_parts, updated_at
FROM inbox WHERE state = 'failed' ORDER BY updated_at;
```

## Flagging a bad answer

A user can reply with `wrong`, `report`, `incorrect`, `bad answer`, 👎, or Apple's SMS `Disliked` tapback to flag the previous reply. Nothing is shared until they confirm: the command stages a pointer and replies with consent copy naming exactly what a human will see (that one exchange, nothing else); only YES writes the flag, anything else cancels and is handled as a normal message. Confirmed flags are pointers only (no message content) joined to the encrypted session by `heynyc feedback` for local triage; the redacted aggregate in `.data/feedback.jsonl` still feeds the systematic-error view.

## v1 scope

Built: WhatsApp (Meta + Twilio), Twilio SMS, encrypted durable Twilio intake, dedup, bounded retries, per-user ordering, rate limits, analytics, and user flagging. Deferred: durable Meta intake, inbound voice notes, outbound templates, dashboards, the public-reply governance layer, and the web channel. The full v1 scope and deferred seams are described in the project's internal multichannel-onramp design spec.
