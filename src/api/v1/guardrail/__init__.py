"""Package shim for guardrail helpers.

This module re-exports the primary guardrail helpers so callers can
import from `src.api.v1.guardrail` whether the implementation is a
single module or a package directory.
"""

from .guardrail import (
    GuardrailViolation,
    validate_user_input,
    validate_sql,
    protect_response,
    safe_guardrail_error,
    mask_record,
    mask_records,
    mask_pii_in_text,
)

__all__ = [
    "GuardrailViolation",
    "validate_user_input",
    "validate_sql",
    "protect_response",
    "safe_guardrail_error",
    "mask_record",
    "mask_records",
    "mask_pii_in_text",
]
