"""
Guardrails layer for the agentic RAG API.

Demonstrates two validators from the Guardrails AI Hub (https://hub.guardrailsai.com):

  1. PII redaction    — GuardrailsPII   applied to the ANSWER  (output guard)
  2. Toxicity checker — ToxicLanguage   applied to the QUERY   (input guard)

Install the validators once before running the app:

    pip install guardrails-ai
    guardrails configure                                  # paste your hub token
    guardrails hub install hub://guardrails/guardrails_pii
    guardrails hub install hub://guardrails/toxic_language

GuardrailsPII runs a local GLiNER model (+ Presidio), so its model is downloaded
on first use. ToxicLanguage can run locally or on Guardrails' hosted endpoint
(remote inferencing only skips the model *download*, not the validator install).

Setting GUARDRAILS_API_KEY in .env only configures the Hub *token* on first use
(see _ensure_guardrails_configured below) — it does NOT install the validators,
so the `guardrails hub install` commands above are still required.

See references/guardrails-demo-guide.md for the full walkthrough.
"""

import os
import re
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# The ValidationError import path has shifted across guardrails versions — be
# defensive so this module imports cleanly regardless of the installed version.
try:
    from guardrails.errors import ValidationError
except Exception:  # pragma: no cover - import path varies by version
    ValidationError = Exception


# ── Configuration ────────────────────────────────────────────────────────────

# Presidio entity labels that the PII validator will redact from answers.
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
]

TOXICITY_THRESHOLD = float(os.getenv("GUARDRAIL_TOXICITY_THRESHOLD", "0.5"))

# Bare numeric customer/account ids are not a recognized PII entity, so the PII
# validator leaves them in the answer (an 11-digit id may only get masked when it
# happens to look like a phone number; shorter ids slip through entirely). We
# mask them ourselves: any run of 6+ digits → <CUSTOMER_ID>. The 6-digit floor
# keeps 4-digit years (2026) and short section numbers intact.
CUSTOMER_ID_RE = re.compile(r"\b\d{6,}\b")


class GuardrailViolation(Exception):
    """Raised when an input guardrail blocks a request.

    `guard` is the short name of the guard that fired; `message` is a
    user-facing explanation suitable for returning in an HTTP 400 response.
    """

    def __init__(self, guard: str, message: str):
        self.guard = guard
        self.message = message
        super().__init__(f"[{guard}] {message}")


# ── Lazy guard construction ──────────────────────────────────────────────────
# Building a Guard imports the hub validators (and downloads their models on
# first use). We build lazily and cache, so importing this module never fails
# just because the validators aren't installed yet — the clear error only
# surfaces when a guard is actually used.

_guards = None


def _ensure_guardrails_configured() -> None:
    """Configure the Guardrails Hub token from the GUARDRAILS_API_KEY env var.

    This lets you set the token in `.env` instead of running `guardrails
    configure` interactively. If `~/.guardrailsrc` already exists (e.g. you ran
    `guardrails configure`), it is left untouched.

    Set `GUARDRAILS_USE_REMOTE_INFERENCING=true` to run the validators on
    Guardrails' hosted endpoint (no local model downloads) — this is the path
    that actually needs the token at runtime.
    """
    api_key = os.getenv("GUARDRAILS_API_KEY")
    if not api_key:
        return

    # Expose the token to any guardrails code path that reads it from the env.
    os.environ.setdefault("GUARDRAILS_TOKEN", api_key)

    rc_path = os.path.expanduser("~/.guardrailsrc")
    if os.path.exists(rc_path):
        return

    use_remote = os.getenv("GUARDRAILS_USE_REMOTE_INFERENCING", "false")
    try:
        with open(rc_path, "w") as rc_file:
            rc_file.write(
                f"id={uuid.uuid4()}\n"
                f"token={api_key}\n"
                "enable_metrics=false\n"
                f"use_remote_inferencing={use_remote}\n"
            )
    except OSError:
        # Non-fatal: fall back to any existing guardrails configuration.
        pass


def _build_guards() -> dict:
    _ensure_guardrails_configured()
    try:
        from guardrails import Guard
        from guardrails.hub import GuardrailsPII, ToxicLanguage
    except ImportError as exc:
        raise RuntimeError(
            "Guardrails validators are not installed. Run:\n"
            "  pip install guardrails-ai\n"
            "  guardrails configure\n"
            "  guardrails hub install hub://guardrails/guardrails_pii\n"
            "  guardrails hub install hub://guardrails/toxic_language"
        ) from exc

    return {
        # Output guard — rewrite the answer, replacing PII with <ENTITY> tags.
        # Covers GuardrailsPII's built-in entities only; domain customer ids are
        # masked separately in guard_output (CUSTOMER_ID_RE).
        "pii": Guard().use(GuardrailsPII(entities=PII_ENTITIES, on_fail="fix")),
        # Input guard — raise if the query is toxic.
        "toxicity": Guard().use(
            ToxicLanguage(
                threshold=TOXICITY_THRESHOLD,
                validation_method="sentence",
                on_fail="exception",
            )
        ),
    }


def _get_guards() -> dict:
    global _guards
    if _guards is None:
        _guards = _build_guards()
    return _guards


# ── Public API ───────────────────────────────────────────────────────────────


def guard_input(query: str) -> None:
    """Run input guardrails on the user's query.

    Raises GuardrailViolation if the query is toxic.
    """
    guards = _get_guards()

    try:
        guards["toxicity"].validate(query)
    except ValidationError as exc:
        raise GuardrailViolation(
            "toxic_language",
            "Your message was flagged as abusive or toxic and cannot be processed.",
        ) from exc


def guard_output(answer: str) -> str:
    """Redact PII from the model's answer. Returns the cleaned text.

    Two passes: mask domain customer ids ourselves (CUSTOMER_ID_RE — the PII
    validator has no recognizer for them), then run GuardrailsPII for standard
    PII (emails, names, formatted phones, ...).
    """
    if not answer:
        return answer
    answer = CUSTOMER_ID_RE.sub("<CUSTOMER_ID>", answer)
    guards = _get_guards()
    outcome = guards["pii"].validate(answer)
    return getattr(outcome, "validated_output", None) or answer
