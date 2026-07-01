"""LDSS-4826 SNAP application: slot schema, validation, PDF fill, copy-summary.

PII discipline: slot values are written onto the PDF locally and returned; they are
NEVER logged, never registered in citations, never sent to a third party. The tool
fills only the slots it is given — it does not fabricate missing fields.

The 4826 is a FLAT scan (no AcroForm fields), so filling is a text overlay anchored to
the form's own labels (see forms/ldss-4826.map.yaml). Provenance + a drift-guard keep us
from printing onto a form that has changed under us (see ldss-4826.meta.yaml).
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

FORM_DIR = Path(__file__).parent / "forms"
TEMPLATE = FORM_DIR / "ldss-4826.pdf"
META = FORM_DIR / "ldss-4826.meta.yaml"


class FormDriftError(RuntimeError):
    """The vendored form no longer matches its recorded provenance — we must NOT print onto
    a form that may have changed under us; the caller degrades to the blank official form."""

DISCLAIMER = (
    "This is an unofficial draft prepared by HeyNYC — not affiliated with the City or "
    "State of New York. Review every field, sign it, and submit it yourself. Nothing "
    "here is a guarantee of benefits."
)

# Two-tier attestation (research §11 / USCIS N-400 model): HeyNYC certifies only that it
# faithfully recorded the user's answers; the USER certifies the facts under penalty of perjury.
SCRIBE_CERT = (
    "Prepared by HeyNYC from what you told me, at your request. HeyNYC recorded your answers "
    "— it did not verify them, and it is not a lawyer or a government caseworker."
)
APPLICANT_ATTESTATION = (
    "Before you sign and submit: you're certifying the information is true and complete to the "
    "best of your knowledge, under penalty of perjury. You — not HeyNYC — are responsible for "
    "what you submit."
)


@dataclass(frozen=True)
class Slot:
    key: str
    label: str           # human label for the readback/summary
    kind: str            # "text" | "date" | "int" | "money" | "enum"
    required: bool = False
    enum: tuple[str, ...] = ()
    pii: bool = False     # values for pii slots are never logged
    high_stakes: bool = False   # gets a per-field read-back at attestation (research §11)


# Curated demo subset of the LDSS-4826 (Rev. 12/23). Required = name + residence only,
# matching the form's own stated minimum to file ("name, address, and signature").
# Household members beyond the applicant are out of v1 scope (single-applicant fill).
SLOTS: tuple[Slot, ...] = (
    Slot("legal_name", "Legal name", "text", required=True, pii=True, high_stakes=True),
    Slot("residence_street", "Home street address", "text", required=True, pii=True),
    Slot("residence_city", "City", "text", required=True, pii=True),
    Slot("residence_zip", "ZIP code", "text", required=True, pii=True),
    Slot("dob", "Date of birth", "date", pii=True, high_stakes=True),
    Slot("ssn", "Social Security Number", "text", pii=True, high_stakes=True),
    Slot("phone", "Phone number", "text", pii=True),
    Slot("monthly_income", "Total monthly household income", "money", high_stakes=True),
    Slot("monthly_rent", "Monthly rent or mortgage", "money"),
)
_BY_KEY = {s.key: s for s in SLOTS}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce(slot: Slot, value):
    """Return (coerced_value, error_or_None). Never raises."""
    if slot.kind in ("text", "date", "enum"):
        v = str(value).strip()
        if slot.kind == "date" and not _DATE_RE.match(v):
            return None, f"{slot.key}: expected a date as YYYY-MM-DD"
        if slot.kind == "enum" and v not in slot.enum:
            return None, f"{slot.key}: must be one of {slot.enum}"
        return v, None
    if slot.kind in ("int", "money"):
        try:
            return (int(value) if slot.kind == "int" else round(float(value), 2)), None
        except (TypeError, ValueError):
            return None, f"{slot.key}: expected a number"
    return None, f"{slot.key}: unknown slot kind {slot.kind}"


def validate_slots(raw: dict) -> tuple[dict, list[str], list[str]]:
    """(clean, missing_required, errors). Drops unknown keys (errors), coerces types,
    flags missing required slots. Never fabricates a value for a field it wasn't given."""
    clean: dict = {}
    errors: list[str] = []
    for key, value in raw.items():
        slot = _BY_KEY.get(key)
        if slot is None:
            errors.append(f"unknown field: {key}")
            continue
        if value in (None, ""):
            continue
        coerced, err = _coerce(slot, value)
        if err:
            errors.append(err)
        else:
            clean[key] = coerced
    missing = [s.key for s in SLOTS if s.required and s.key not in clean]
    return clean, missing, errors


def _fmt(slot: Slot, value) -> str:
    if slot.kind == "money":
        return f"${value:,.2f}"
    return str(value)


def application_summary(clean: dict, missing: list[str]) -> str:
    """The editable readback ('check your answers'). Regenerated on every edit; the PDF is
    rendered from the same `clean` only at draft/final milestones."""
    lines = ["Here's your draft SNAP application (LDSS-4826):", ""]
    for slot in SLOTS:
        if slot.key in clean:
            lines.append(f"- {slot.label}: {_fmt(slot, clean[slot.key])}")
    if missing:
        labels = ", ".join(_BY_KEY[k].label for k in missing)
        lines += ["", f"Still needed before you submit: {labels}."]
    lines += ["", DISCLAIMER, "", APPLICANT_ATTESTATION]
    try:
        lines.append(provenance_stamp())
    except Exception:
        pass                       # summary still works if the meta file is absent
    return "\n".join(lines)


def review_request(clean: dict) -> str:
    """The meaningful-attestation step (research §11): read back each filled value — high-stakes
    fields flagged for the user to re-confirm in their own words — then the two-tier scribe
    certification + the user's penalty-of-perjury attestation, and ask them to confirm or edit
    BEFORE any PDF is produced. Targeted friction on the load-bearing fields, not a 12-page wall."""
    lines = ["Let's check your answers before I prepare the form — tell me if any are wrong:", ""]
    for slot in SLOTS:
        if slot.key in clean:
            tag = "   ← please double-check this one" if slot.high_stakes else ""
            lines.append(f"- {slot.label}: {_fmt(slot, clean[slot.key])}{tag}")
    lines += ["", SCRIBE_CERT, "", APPLICANT_ATTESTATION, "",
              "Reply 'yes, that's correct' and I'll prepare your draft, or tell me what to change."]
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_provenance(meta_path: Path | None = None) -> dict:
    return yaml.safe_load((meta_path or META).read_text())


def verify_template_integrity(template: Path | None = None, meta: dict | None = None) -> bool:
    """True iff the vendored template still hashes to the recorded sha256 (catches a
    corrupted/swapped/silently-updated template). Never touches the network."""
    template = template or TEMPLATE
    meta = meta or template_provenance()
    return _sha256(template) == meta["sha256"]


def provenance_stamp(meta: dict | None = None) -> str:
    meta = meta or template_provenance()
    return (f"Based on {meta['form']} ({meta['revision']}), verified {meta['verified_on']} — "
            f"confirm it's current at otda.ny.gov.")


# --- PDF fill: the 4826 is a FLAT scan, so we overlay each value next to its label anchor ---
MAP = FORM_DIR / "ldss-4826.map.yaml"


def load_map(path: Path | None = None) -> dict:
    data = yaml.safe_load((path or MAP).read_text())
    if data.get("mode") != "overlay-anchor":
        raise ValueError(f"unexpected form-map mode: {data.get('mode')!r}")
    return data


def _find_anchor(words: list[dict], text: str, occurrence: int = 0):
    matches = [w for w in words if w.get("text") == text]
    return matches[occurrence] if occurrence < len(matches) else None


def fill_application(clean: dict, *, template: Path | None = None,
                     fmap: dict | None = None) -> bytes:
    """Overlay each provided value next to its label anchor on the flat LDSS-4826. Fills only the
    slots in `clean` that the map knows; raises FormDriftError if a mapped anchor can't be located
    (the form drifted — caller degrades rather than printing into the wrong place)."""
    import pdfplumber
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas

    template = template or TEMPLATE
    fmap = fmap or load_map()
    by_page: dict[int, list[tuple[dict, str]]] = {}
    for key, value in clean.items():
        spec = fmap["fields"].get(key)
        if spec:
            by_page.setdefault(int(spec["page"]), []).append((spec, _fmt(_BY_KEY[key], value)))

    writer = PdfWriter(clone_from=str(template))   # pages attached to the writer → reliable merge
    with pdfplumber.open(str(template)) as plumb:
        for i, specs in by_page.items():
            page = writer.pages[i]
            words = plumb.pages[i].extract_words()
            ph, pw = float(page.mediabox.height), float(page.mediabox.width)
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=(pw, ph))
            for spec, text in specs:
                anchor = _find_anchor(words, spec["anchor"], int(spec.get("occurrence", 0)))
                if anchor is None:
                    raise FormDriftError(
                        f"anchor {spec['anchor']!r} not found on page {i} — form may have changed")
                x = float(anchor["x1"]) + float(spec.get("dx", 4))
                y = ph - float(anchor["bottom"]) + float(spec.get("dy", 0))
                c.setFont("Helvetica", int(spec.get("size", 9)))
                c.drawString(x, y, text)
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
