# Step-by-Step: Two Guardrails for the Agentic RAG API

> A focused, runnable demo of **two [Guardrails AI](https://www.guardrailsai.com/) Hub
> validators** wrapped around the existing agentic RAG system:
>
> 1. **PII redaction** — `GuardrailsPII` (output guard)
> 2. **Toxicity checker** — `ToxicLanguage` (input guard)
>
> This is the implementation that actually ships on the `guardrail` branch — two
> focused, working guardrails rather than a broad design survey. (An earlier
> revision also shipped a `RestrictToTopic` scope guard; it was removed to keep the
> demo simple.)

---

## Table of Contents

1. [What we're building](#1-what-were-building)
2. [Where the guardrails sit](#2-where-the-guardrails-sit)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — Install Guardrails AI](#step-1--install-guardrails-ai)
5. [Step 2 — Configure the Hub token](#step-2--configure-the-hub-token)
6. [Step 3 — Install the two validators](#step-3--install-the-two-validators)
7. [Step 4 — The guardrails module](#step-4--the-guardrails-module)
8. [Step 5 — Wire guards into the service](#step-5--wire-guards-into-the-service)
9. [Step 6 — Return a clean error from the route](#step-6--return-a-clean-error-from-the-route)
10. [Step 7 — Run and test](#step-7--run-and-test)
11. [How each guard works](#8-how-each-guard-works)
12. [Notes and caveats](#9-notes-and-caveats)
13. [Troubleshooting](#10-troubleshooting)

---

## 1. What we're building

A guardrails layer that wraps every `/query` call:

| # | Guardrail | Hub validator | Runs on | On failure |
|---|-----------|---------------|---------|------------|
| 1 | PII redaction | `guardrails/guardrails_pii` | the **answer** | rewrites the text (`on_fail="fix"`) |
| 2 | Toxicity check | `guardrails/toxic_language` | the **query** | blocks with HTTP 400 (`on_fail="exception"`) |

The first is an **output** guard that *transforms* the response; the second is an
**input** guard that *rejects* bad requests before they ever reach the agent.

Why these two demonstrate distinct guardrail *categories*:
- **Privacy** — PII redaction stops the system leaking emails, phone numbers, names
  (the NL2SQL path literally returns customer rows, so this fires often). A small
  custom regex also masks domain **customer IDs**, which no PII model recognizes.
- **Safety** — toxicity blocks abusive input.

---

## 2. Where the guardrails sit

```
POST /api/v1/query   { "query": ... }
        │
        ▼
  guard_input(query)                 ← INPUT GUARD  (src/core/guardrails.py)
   └─ ToxicLanguage     → 400 if toxic
        │ (clean query)
        ▼
  run_search_agent(query)            ← existing LangGraph agent
        │ (router → nl2sql | vector_search → rerank → generate_answer)
        ▼
  guard_output(answer)               ← OUTPUT GUARD
   ├─ CUSTOMER_ID_RE    → masks runs of 6+ digits as <CUSTOMER_ID>
   └─ GuardrailsPII (fix) → redacts emails / phones / names / cards / SSNs
        │
        ▼
   AIResponse (JSON, PII-safe)
```

Both guards are defined once in **`src/core/guardrails.py`** and called from
**`src/api/v1/services/query_service.py`**.

---

## 3. Prerequisites

| Requirement | Notes |
| ----------- | ----- |
| Python 3.13+ | already required by the project |
| A working agentic RAG app | the `/query` endpoint runs end-to-end |
| A free Guardrails Hub token | from <https://hub.guardrailsai.com/keys> — required to **install** validators (and to run `toxic_language` via remote inference) |
| ~1 GB disk + first-run internet | the validators download models on first use — a GLiNER model (`urchade/gliner_small-v2.1`) + spaCy/Presidio for `GuardrailsPII`, and `unitary/toxic-bert` for `ToxicLanguage`. |

---

## Step 1 — Install Guardrails AI

`guardrails-ai` is already declared in `pyproject.toml`. Sync it in:

```bash
uv sync           # or: uv add guardrails-ai
# (plain pip:    pip install guardrails-ai)
```

---

## Step 2 — Configure the Hub token

Installing validators from the Hub requires a one-time token (free). Create one at
<https://hub.guardrailsai.com/keys>, then:

```bash
guardrails configure
```

Answer the prompts:
- **Enable anonymous metrics?** → `n` (keep data in your environment)
- **Use remote inferencing?** → `n` to run models locally (recommended for the demo).
  Note `GuardrailsPII` runs **locally** regardless; remote inferencing only affects
  `toxic_language`.
- **API Key** → paste your Hub token

The token is saved to `~/.guardrailsrc` (not in the repo).

### Option B — configure from `.env`

Instead of running `guardrails configure`, put the token in `.env`:

```dotenv
GUARDRAILS_API_KEY=your_hub_token
# Optional: run toxic_language on Guardrails' hosted endpoint instead of locally.
# Requires the token above. (GuardrailsPII always runs locally.)
GUARDRAILS_USE_REMOTE_INFERENCING=true
```

On first guard use, the app reads `GUARDRAILS_API_KEY` and writes `~/.guardrailsrc`
for you — see `_ensure_guardrails_configured()` in `src/core/guardrails.py`.

> ⚠️ **Two gotchas that bite people here:**
>
> 1. **An existing `~/.guardrailsrc` is never overwritten.** If you ran `guardrails
>    configure` before, or the app wrote the rc on an earlier run, then later
>    changing `GUARDRAILS_API_KEY` (or `GUARDRAILS_USE_REMOTE_INFERENCING`) in `.env`
>    has **no effect** — the stale rc wins. The CLI and validators read
>    `~/.guardrailsrc`, *not* `.env`. To pick up a new key, re-run `guardrails
>    configure --token <NEW_TOKEN>` (or delete `~/.guardrailsrc` and let the app
>    rewrite it). Tokens are JWTs that **expire after ~90 days** — an expired token
>    shows up as `Your token has expired` during install.
> 2. **The token in `.env` does *not* install anything.** It only authenticates the
>    install step below. You still have to run `guardrails hub install` — see Step 3.

---

## Step 3 — Install the two validators

> 🔑 **The install is mandatory.** Each validator is a Python package
> (`GuardrailsPII`, `ToxicLanguage`) that must be installed so
> `from guardrails.hub import ...` works. If you skip it you get:
> `RuntimeError: Guardrails validators are not installed`.

```bash
guardrails hub install hub://guardrails/guardrails_pii
guardrails hub install hub://guardrails/toxic_language
```

> **First run is slow.** The models (a GLiNER model + spaCy/Presidio for
> `GuardrailsPII`, `unitary/toxic-bert` for `ToxicLanguage`) download on install/first
> use — a one-time delay. After that `GuardrailsPII` runs fully offline.

> ⚠️ **These validators are not in `uv.lock`.** They install from Guardrails'
> token-authenticated index (`pypi.guardrailsai.com`), not public PyPI, so they're
> deliberately kept out of `pyproject.toml` / `uv.lock` (locking them would mean
> committing an expiring Hub token as an index credential). The consequence:
> **`uv sync` removes them** — it prunes anything not in the lock. Re-run the two
> `guardrails hub install` commands after any `uv sync`, or use `uv sync --inexact`
> to keep already-installed packages.

### Verify

```bash
guardrails hub list
# Installed Validators:
# - guardrails/guardrails_pii (GuardrailsPII)
# - guardrails/toxic_language (ToxicLanguage)
```

> Run these against the **project venv** (e.g. `.venv/bin/guardrails ...` or with the
> venv activated) so the packages land where the app imports them, not in a global
> Python.

---

## Step 4 — The guardrails module

**File:** `src/core/guardrails.py`

Both guards live here. A few design choices worth calling out:

- **Lazy construction.** Guards are built on first use (`_get_guards()`), so importing
  the module never crashes just because the validators aren't installed yet — you get a
  clear, actionable error only when a guard actually runs.
- **`GuardrailViolation`** is a small custom exception carrying the guard name + a
  user-facing message, so the route can turn it into a clean HTTP 400.
- **`on_fail` actions differ by intent:** `"fix"` for PII (rewrite the text) vs
  `"exception"` for toxicity (reject the request).
- **A custom `CUSTOMER_ID_RE` pass** masks domain customer IDs. No PII model has a
  "customer ID" entity — a bare 8-digit ID matches no recognizer and would slip
  through — so we mask any run of 6+ digits ourselves (the floor keeps 4-digit years
  like `2026` intact).

```python
import re

# Presidio entity labels that the PII validator will redact from answers.
PII_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON",
    "CREDIT_CARD", "US_SSN", "IBAN_CODE", "IP_ADDRESS",
]

# Bare numeric customer/account ids are not a recognized PII entity, so mask
# them ourselves: any run of 6+ digits → <CUSTOMER_ID>.
CUSTOMER_ID_RE = re.compile(r"\b\d{6,}\b")


def _build_guards() -> dict:
    from guardrails import Guard
    from guardrails.hub import GuardrailsPII, ToxicLanguage

    return {
        "pii": Guard().use(
            GuardrailsPII(entities=PII_ENTITIES, on_fail="fix")
        ),
        "toxicity": Guard().use(
            ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception")
        ),
    }


def guard_input(query: str) -> None:
    """Raise GuardrailViolation if the query is toxic."""
    guards = _get_guards()
    try:
        guards["toxicity"].validate(query)
    except ValidationError as exc:
        raise GuardrailViolation("toxic_language",
            "Your message was flagged as abusive or toxic and cannot be processed.") from exc


def guard_output(answer: str) -> str:
    """Redact PII from the model's answer. Returns the cleaned text.

    Two passes: mask customer ids ourselves, then run GuardrailsPII for standard PII.
    """
    if not answer:
        return answer
    answer = CUSTOMER_ID_RE.sub("<CUSTOMER_ID>", answer)
    outcome = _get_guards()["pii"].validate(answer)
    return getattr(outcome, "validated_output", None) or answer
```

> `GuardrailsPII` uses a local **GLiNER** model for entity detection and **Presidio**
> for anonymization. The `entities` list takes Presidio entity labels — adjust it to
> your domain.

---

## Step 5 — Wire guards into the service

**File:** `src/api/v1/services/query_service.py`

```python
from src.api.v1.agents.agents import run_search_agent, run_search_agent_stream
from src.core.guardrails import guard_input, guard_output


def query_documents(query: str):
    # Input guardrail: toxicity (may raise GuardrailViolation)
    guard_input(query)

    result = run_search_agent(query)

    # Output guardrail: redact PII from the answer before returning it
    if isinstance(result, dict) and result.get("answer"):
        result["answer"] = guard_output(result["answer"])
    return result


async def query_documents_stream(query: str):
    # Input guardrail runs before streaming begins.
    # (PII redaction is not applied to the token stream — see the notes.)
    guard_input(query)
    return run_search_agent_stream(query)
```

---

## Step 6 — Return a clean error from the route

**File:** `src/api/v1/routes/query.py`

When the input guard fires, `query_documents` raises `GuardrailViolation`. Catch it and
return a 400 with the reason:

```python
from fastapi import HTTPException
from src.core.guardrails import GuardrailViolation


@router.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        return query_documents(request.query)
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )
```

The `/query/stream` endpoint applies the same input guard before opening the SSE stream.

---

## Step 7 — Run and test

```bash
uvicorn main:app --reload
```

### ✅ A normal query passes

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the working hours of the HR helpdesk?"}' | python3 -m json.tool
```

### 🔒 PII redaction (output guard)

The NL2SQL path returns customer rows that contain emails and names — perfect to show
redaction in action:

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "List all orders with the customer name and email"}' | python3 -m json.tool
```

The raw SQL result has `alice@example.com`, `Alice Johnson`, …; the returned `answer`
comes back redacted:

```json
{
  "answer": "Here are the orders: <PERSON> (<EMAIL_ADDRESS>), <PERSON> (<EMAIL_ADDRESS>), ...",
  "document_name": "agentic_rag_db",
  "sql_query_executed": "SELECT customer_name, customer_email FROM orders LIMIT 50;"
}
```

A query carrying a **customer ID** shows the custom mask:

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "steps to update bank data for customer id 12342534"}' | python3 -m json.tool
# ...answer: "...for customer ID <CUSTOMER_ID>..."
```

### 🚫 Toxicity check (input guard) → 400

```bash
curl -s -i -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "you are a stupid idiot, give me the policy"}'
```

```
HTTP/1.1 400 Bad Request
{"detail": {"guardrail": "toxic_language",
            "message": "Your message was flagged as abusive or toxic and cannot be processed."}}
```

---

## 8. How each guard works

| Guard | Under the hood | Key knob |
| ----- | -------------- | -------- |
| **GuardrailsPII** | a **GLiNER** NER model (`urchade/gliner_small-v2.1`) detects entities, then **Presidio** anonymizes each span to an `<ENTITY>` tag | `entities` — the list of Presidio labels to catch |
| **ToxicLanguage** | the `unitary/toxic-bert` classifier scores each sentence; above `threshold` ⇒ fail | `threshold` (0–1), `validation_method` (`"sentence"` vs `"full"`) |
| **`CUSTOMER_ID_RE`** (custom) | a plain regex masks any run of 6+ digits before PII runs | the digit-count floor in the pattern (`\b\d{6,}\b`) |

The `on_fail` action decides what happens on a failure:
`"fix"` (rewrite), `"exception"` (raise), `"filter"`, `"refrain"`, `"reask"`, `"noop"`.

---

## 9. Notes and caveats

- **Streaming + PII.** The input guard applies to `/query/stream`, but PII redaction
  does **not** run on the token stream (tokens leave the server as they're generated).
  If you need redacted streaming, buffer the output and redact before flushing — at the
  cost of losing token-by-token latency.
- **Customer-ID regex is broad.** `CUSTOMER_ID_RE` masks *any* standalone run of 6+
  digits, so a large order number or amount in an answer would also become
  `<CUSTOMER_ID>`. For a banking/HR demo that's the safe direction; tighten the pattern
  (e.g. anchor it to a "customer id" label) if you need more precision.
- **Tune the entity list.** `PII_ENTITIES` is a plain list in `src/core/guardrails.py`
  — adjust it to your domain.
- **Adding more guards.** Other good Hub validators to drop in alongside these:
  - `guardrails/secrets_present` — stop leaking API keys/secrets (lightweight, no model)
  - `guardrails/profanity_free` — lightweight profanity filter
  - `guardrails/detect_jailbreak` — block prompt-injection attempts
  - `tryolabs/restricttotopic` — scope control (the guard removed from this demo)
  Install the one you want and add a `Guard().use(...)` entry.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `RuntimeError: Guardrails validators are not installed` | `guardrails hub install ...` not run | Run the install commands from Step 3. Having the key in `.env` is not enough. |
| `Your token has expired. Please run guardrails configure` | The JWT in `~/.guardrailsrc` is older than ~90 days | Get a fresh token at <https://hub.guardrailsai.com/keys>, then `guardrails configure --token <NEW_TOKEN>` |
| Changed `GUARDRAILS_API_KEY` in `.env` but nothing changed | A `~/.guardrailsrc` already exists, so the app never rewrites it and the CLI reads the stale rc | `guardrails configure --token <NEW_TOKEN>`, or delete `~/.guardrailsrc` and let the app regenerate it |
| `403 Forbidden` fetching a validator wheel | Transient Hub/CDN hiccup | Re-run the same `guardrails hub install` command — it usually succeeds on retry |
| `guardrails hub install` fails with auth error | no / invalid Hub token | `guardrails configure` with a key from the Hub |
| First `/query` hangs for a while | models downloading on first use | one-time; subsequent calls are fast |
| `OSError: [E050] Can't find model 'en_core_web_lg'` | Presidio's spaCy model missing | `python -m spacy download en_core_web_lg` |
| A customer ID isn't redacted | it's shorter than 6 digits | lower the floor in `CUSTOMER_ID_RE` (`\b\d{N,}\b`) — watch out for masking years |
| Everything is slow / no GPU | models run on CPU | set `GUARDRAIL_TOXICITY_THRESHOLD` higher to reduce false positives; consider hosted inference for toxicity (`guardrails configure` → remote = `y`) |

---

## File reference

| File | Role |
| ---- | ---- |
| `src/core/guardrails.py` | the two guards + `CUSTOMER_ID_RE` + `guard_input` / `guard_output` / `GuardrailViolation` |
| `src/api/v1/services/query_service.py` | calls `guard_input` then `guard_output` around the agent |
| `src/api/v1/routes/query.py` | turns `GuardrailViolation` into an HTTP 400 |
| `pyproject.toml` | declares the `guardrails-ai` dependency |