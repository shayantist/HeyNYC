# HeyNYC scripts

Updated 2026-08-18.

## Normal commands

Run [`deploy_via_ssh.sh`](deploy_via_ssh.sh) on the Mac to deploy the latest pushed `main` to the private WSL host:

```bash
./scripts/deploy_via_ssh.sh
```

It connects to Windows through the operator's private SSH configuration, enters WSL, fast-forwards the checkout, and invokes [`deploy.sh`](deploy.sh) there. Run `deploy.sh` directly only from the target Linux host.

The WSL pilot needs one interactive privilege bootstrap:

```bash
./scripts/install_deploy_privileges.sh
```

The installer copies a four-operation root-owned helper and its narrow sudo policy. After that, `deploy_via_ssh.sh` can stop and start only `heynyc.service`, run the existing snapshot-retention cleanup, and verify that permission without a password. It cannot run repository code as root or invoke arbitrary system commands. A new host must complete one ordinary interactive deployment first so its browser dependencies and retention policy already exist.

There is no `serve.sh`. `serve` is an application command implemented by HeyNYC itself:

```bash
uv run python -m heynyc serve --provider twilio --port 8791
```

[`serve_demo.sh`](serve_demo.sh) wraps that command with the configured ngrok tunnel, local and public health gates, and process cleanup for a Mac-hosted demo:

```bash
./scripts/serve_demo.sh
```

The npm-style analogy is:

| Familiar command | HeyNYC equivalent | Meaning |
|---|---|---|
| `npm run dev` | `uv run python -m heynyc repl` | Interactive local conversation |
| `npm run serve` | `uv run python -m heynyc serve` | Run the request-handling process |
| `npm run demo` | `./scripts/serve_demo.sh` | Serve locally and expose the configured demo tunnel |
| `npm run deploy` | `./scripts/deploy_via_ssh.sh` | Update and activate the persistent remote service |

## Operator scripts

| Script | Run from | What it does | Live or paid effects |
|---|---|---|---|
| [`deploy_via_ssh.sh`](deploy_via_ssh.sh) | Mac | Connects to the private Windows host, updates the WSL checkout, and invokes the target deploy script | Starts a real deployment |
| [`deploy.sh`](deploy.sh) | Target Linux or WSL host | Builds an exact release and fresh index, snapshots resident state, switches the service, checks health, and reconciles Twilio metadata | Contacts configured external services |
| [`install_deploy_privileges.sh`](install_deploy_privileges.sh) | Target Linux or WSL host, once | Installs the root-owned fixed-operation helper and validates its sudo policy | Changes local administrative policy after an interactive sudo approval |
| [`serve_demo.sh`](serve_demo.sh) | Mac | Runs the Twilio server and ngrok tunnel until interrupted | Resident requests can spend model budget |
| [`health_watch.sh`](health_watch.sh) | Mac cron or operator shell | Checks the public health endpoint at a bounded cadence and notifies on outage transitions | Public HTTP checks only |
| [`state_snapshot.py`](state_snapshot.py) | Target host | Creates, verifies, or restores resident-state snapshots | Local resident-state operation |
| [`reconcile_twilio.py`](reconcile_twilio.py) | Target host or deployment | Compares provider-side inbound message IDs with the durable local inbox | Reads Twilio metadata, never message bodies |

## Evaluation and development scripts

| Script | Purpose | Live or paid effects |
|---|---|---|
| [`persona_turn.py`](persona_turn.py) | Runs one persistent simulated-resident turn through the real channel orchestrator | Can spend model budget |
| [`pydantic_ai_ab.py`](pydantic_ai_ab.py) | Compares runtime configurations through the same evaluation route | Can spend model budget |
| [`pydantic_ai_repl.py`](pydantic_ai_repl.py) | Opens an interactive Pydantic AI runtime session | Can spend model budget |
| [`live_verification_fallback_probe.py`](live_verification_fallback_probe.py) | Runs the live F179 fallback probe | Spends model budget and requires owner approval |
| [`demo_multilingual.py`](demo_multilingual.py) | Demonstrates translation masking and entity preservation | Deterministic by default; optional real backends may spend or load a model |
| [`demo_tier2.py`](demo_tier2.py) | Demonstrates the Tier-2 faithfulness check | Deterministic by default; do not run its local-model backends on the owner's Mac |
| [`demo_snap_fill.py`](demo_snap_fill.py) | Demonstrates persistent SNAP draft and PDF filling | Deterministic and offline |

## Data and documentation scripts

| Script | Purpose | Live or paid effects |
|---|---|---|
| [`build_nta_gazetteer.py`](build_nta_gazetteer.py) | Rebuilds the bundled NYC neighborhood gazetteer from NYC Open Data | Network read, no model call |
| [`export_service_field_inventory.py`](export_service_field_inventory.py) | Regenerates the typed service-location source coverage comparison | Offline |
| [`export_testing_docs.py`](export_testing_docs.py) | Regenerates the public testing records from the internal sources | Offline |
| [`unwrap_docs.py`](unwrap_docs.py) | Removes hard wrapping from Markdown without changing rendered content | Offline |
