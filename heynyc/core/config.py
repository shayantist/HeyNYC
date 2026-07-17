"""Environment-driven configuration. Secrets via env only, never hardcoded."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # the `heynyc` package
PROJECT_ROOT = PACKAGE_DIR.parent  # repo root (holds pyproject, .env, docs)
# NOTE: this module no longer loads .env on import, the app entrypoint (heynyc/__main__.py)
# does, so the reusable engine never auto-reads a dotenv and tests stay hermetic.

# LLM
HEYNYC_MODEL = os.getenv("HEYNYC_MODEL", "anthropic/claude-sonnet-4-6")
# Small semantic preflight that blocks unrelated questions before retrieval. Keep this cheaper than
# the resident-answer model; deployments can override it independently.
HEYNYC_SCOPE_MODEL = os.getenv("HEYNYC_SCOPE_MODEL", "openai/gpt-5.4-nano")
# Structured continuity compaction runs only when measured context pressure requires it.
HEYNYC_MEMORY_MODEL = os.getenv("HEYNYC_MEMORY_MODEL", "openai/gpt-5.4-nano")
# Context window for self-hosted Ollama models. Ollama defaults to ~2-4K tokens, which silently
# truncates HeyNYC's ~7.5K-token system prompt and breaks tool-calling; size it to fit the prompt.
OLLAMA_NUM_CTX = int(os.getenv("HEYNYC_OLLAMA_NUM_CTX", "16384"))

# Optional hard USD spend ceiling for one agent session (security-audit F2b, OWASP LLM10). Default
# OFF: unset, blank, 0, or a non-positive/garbage value all mean disabled (None) and behavior is
# unchanged. When set, the spend guard (heynyc/core/spend.py) halts further model calls once the
# cumulative LiteLLM-priced cost meets or exceeds this ceiling, rather than silently spending past it.
def _parse_spend_cap(raw: str) -> float | None:
    try:
        cap = float(raw.strip())
    except ValueError:
        return None
    return cap if cap > 0 else None


HEYNYC_SPEND_CAP = _parse_spend_cap(os.getenv("HEYNYC_SPEND_CAP", ""))

# Multilingual translate-at-edge pipeline (heynyc/core/multilingual.py), reason in English, translate
# only the verified answer at the edge (OTI Gap 2, Local Law 30 language access). OFF by default and
# DEFINED-BUT-NOT-CONSUMED here: the agent loop does not read this yet, mirroring the Tier-2 NLI guard.
# Wiring it into agent.py is a deliberate follow-on.
HEYNYC_MULTILINGUAL = os.getenv("HEYNYC_MULTILINGUAL", "").strip().lower() in ("1", "true", "yes", "on")

# Judge model, deliberately a DIFFERENT family than the agent to avoid
# self-enhancement bias in LLM-as-judge (judges prefer their own outputs).
# Falls back to the agent model if no alternate provider key is configured.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HEYNYC_JUDGE_MODEL = os.getenv("HEYNYC_JUDGE_MODEL") or (
    "openai/gpt-4o-mini" if OPENAI_API_KEY else HEYNYC_MODEL
)

# Paths
HEYNYC_DATA_DIR = Path(os.getenv("HEYNYC_DATA_DIR") or (PROJECT_ROOT / ".data"))
MODULES_DIR = PACKAGE_DIR / "modules"  # modules ship inside the package

# External services
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY", "")  # Discovery API (events backbone)
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")  # forgiving fallback geocoder (intersections/POIs)
GEOSEARCH_BASE = "https://geosearch.planninglabs.nyc/v2"
MAPBOX_GEOCODE_BASE = "https://api.mapbox.com/geocoding/v5/mapbox.places"

# Forgiving geocoder (intersections/POIs/fuzzy), swappable provider via geopy.
# NYC GeoSearch stays the authoritative address path; this is the fallback.
# Prefer Mapbox when a token is present (free ≤100k/mo, NYC-biased, reliable);
# otherwise fall back to the keyless public Nominatim (dev/demo-grade, slow).
HEYNYC_GEOCODER = os.getenv("HEYNYC_GEOCODER") or ("mapbox" if MAPBOX_TOKEN else "nominatim")
HEYNYC_USER_AGENT = os.getenv("HEYNYC_USER_AGENT", "heynyc/0.1 (civic assistant)")
# Below this provider confidence, an intersection result is flagged low-confidence
# so the agent clarifies (which borough?) instead of answering for the wrong place.
HEYNYC_GEOCODE_MIN_CONFIDENCE = float(os.getenv("HEYNYC_GEOCODE_MIN_CONFIDENCE", "0.8"))
SOCRATA_BASE = "https://data.cityofnewyork.us/resource"
OSRM_BASE = os.getenv("HEYNYC_OSRM_BASE", "https://router.project-osrm.org")

# NYC bounding box (W,S,E,N) to keep Mapbox results within the city.
NYC_BBOX = "-74.2591,40.4774,-73.7004,40.9176"
NYC_PROXIMITY = "-73.9857,40.7484"  # Midtown, biases POI matches toward NYC

# Base domains every module may cite via web search; modules extend this.
BASE_ALLOWLIST = [
    "nyc.gov",
    "nyctourism.com",
    "cityofnewyork.us",
    "mta.info",
]

# The "currency layer", a small, curated set of reputable news / legal-news domains used
# ONLY by the recency check (the recent_developments tool), never by the default web_search.
# This is a deliberate, SUBORDINATE tier: news ranks BELOW gov/authoritative sources and its
# results are labeled developing/contested, so the official grounded answer always stays
# primary. Kept engine-independent (injected like BASE_ALLOWLIST) so it never pollutes the
# per-module allowlists. Intentionally short and reputable-only, this is not "the open web."
NEWS_ALLOWLIST = [
    # NYC local newsrooms
    "thecity.nyc",
    "gothamist.com",
    "citylimits.org",
    "amny.com",
    "nydailynews.com",
    # National wire services / paper of record (for state + federal rulings that hit NYC)
    "nytimes.com",
    "apnews.com",
    "reuters.com",
    # Legal news (court rulings, new/amended law)
    "nylj.com",            # New York Law Journal
    "courthousenews.com",
]

# --- Messaging on-ramp (channels). Secrets via env only. ---
HEYNYC_PII_SALT = os.getenv("HEYNYC_PII_SALT", "")  # required when serving; pseudonymizes senders
WHATSAPP_PROVIDER = os.getenv("WHATSAPP_PROVIDER", "meta")  # meta | twilio | both

# Meta WhatsApp Cloud API
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")          # system-user Admin token (does not expire)
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

# Twilio (WhatsApp sandbox / SMS)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")  # e.g. whatsapp:+14155238886
# Generic Twilio sender. A plain +E.164 number (e.g. +12125550123) sends SMS; a whatsapp:+… value
# sends WhatsApp. Falls back to TWILIO_WHATSAPP_FROM so existing WhatsApp setups keep working.
TWILIO_FROM = os.getenv("TWILIO_FROM", "") or TWILIO_WHATSAPP_FROM

# Channel runtime knobs
CHANNEL_RATE_LIMIT = int(os.getenv("HEYNYC_CHANNEL_RATE_LIMIT", "20"))          # msgs / window / user
CHANNEL_RATE_WINDOW_S = int(os.getenv("HEYNYC_CHANNEL_RATE_WINDOW_S", "60"))
CHANNEL_MAX_CONCURRENCY = int(os.getenv("HEYNYC_CHANNEL_MAX_CONCURRENCY", "8"))
CHANNEL_DEDUP_TTL_S = int(os.getenv("HEYNYC_CHANNEL_DEDUP_TTL_S", str(7 * 24 * 3600)))

# --- NYC Benefits Screening API (Module B) ---
SCREENING_ENV = os.getenv("SCREENING_ENV", "sandbox")  # sandbox | prod
SCREENING_BASES = {
    "sandbox": "https://sandbox.screeningapi.cityofnewyork.us",
    "prod": "https://screeningapi.cityofnewyork.us",
}


def screening_creds() -> tuple[str, str, str]:
    """(base_url, username, password) for the active SCREENING_ENV; empty strings if unset."""
    env = SCREENING_ENV if SCREENING_ENV in SCREENING_BASES else "sandbox"
    user = os.getenv(f"SCREENING_{env.upper()}_USERNAME", "")
    pw = os.getenv(f"SCREENING_{env.upper()}_PASSWORD", "")
    return SCREENING_BASES[env], user, pw
