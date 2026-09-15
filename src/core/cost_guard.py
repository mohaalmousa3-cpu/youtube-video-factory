"""Phase 1E: minimal runtime cost-governance guard.

docs/spec-v4/IMPLEMENTATION-PLAN.md's Phase 1E requires "a runtime guard
that refuses to call a provider marked as paid unless a corresponding
paid-proposal record exists with status: 'approved'". This module is that
guard, and only that guard — it is not wired into any existing provider
call site (src/core/scene_generator.py's Groq/TokenRouter calls, or any
future Qwen-Image/Kokoro call site) in this slice, and it does not add a
proposal-approval CLI. docs/spec-v4/TECHNICAL-SPEC-EN.md section 5's
"a paid-proposal with status != 'approved' must never result in a paid
call being made" is enforced here by raising before the caller ever
proceeds to a provider call, not by any post-hoc check.

Storage, per explicit approval: a channel-wide JSON file (array of records
conforming to docs/spec-v4/schemas/paid-proposal.schema.json), not SQLite —
the implementation plan explicitly allows a manual JSON review/edit
workflow ("even a manual one — e.g. a human reads and edits the JSON
file's status field"), and this avoids a database migration for a first
slice. The path is always supplied explicitly by the caller; this module
never assumes or invents a default location.

Fails closed on every ambiguity: an unrecognized service_name, a missing/
unreadable/malformed proposals file, or any record that does not fully
validate against PaidProposalRecord all raise the same
PaidApprovalRequiredError rather than silently permitting a paid call. No
network, no file write, no SQLite, no manifest access anywhere in this
module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from src.models.common import FrozenStrictModel

# The complete, fixed vocabulary of service names this guard recognizes —
# matching src/providers/*.py's actual provider set (llm_groq.py,
# llm_tokenrouter.py, tts_kokoro.py, image_qwen.py). A service_name outside
# this set is always treated as unknown and fails closed, regardless of
# is_paid, rather than silently passing through on a typo.
CANONICAL_SERVICE_NAMES: frozenset[str] = frozenset({"groq", "tokenrouter", "kokoro", "qwen-image"})


class PaidApprovalRequiredError(Exception):
    """Raised whenever a paid service call is not covered by an approved
    paid-proposal record, or service_name is not one of
    CANONICAL_SERVICE_NAMES. Message text is always a short, fixed,
    sanitized string — never a raw filesystem path, raw JSON error, or raw
    validation error."""


class PaidProposalRecord(FrozenStrictModel):
    """One entry in the JSON array at a proposals_path, matching
    docs/spec-v4/schemas/paid-proposal.schema.json field-for-field
    (additionalProperties: false is FrozenStrictModel's extra='forbid')."""

    proposal_id: str = Field(min_length=1)
    service_name: str = Field(min_length=1)
    estimated_monthly_cost_usd: float = Field(ge=0, le=25)
    justification: str = Field(min_length=1)
    status: Literal["proposed", "approved", "rejected", "expired"]
    requires_explicit_user_approval: Literal[True]
    approved_by: str | None = None
    approved_at: str | None = None


def require_paid_approval(service_name: str, proposals_path: Path, *, is_paid: bool) -> None:
    """Raise PaidApprovalRequiredError unless this call is allowed to
    proceed; return None (no other side effect) otherwise.

    - service_name not in CANONICAL_SERVICE_NAMES: always raises, whether
      or not is_paid is True.
    - is_paid=False for a recognized service_name: returns immediately;
      proposals_path is never opened.
    - is_paid=True: proposals_path is read as a JSON array of
      PaidProposalRecord. Any problem loading it (missing file,
      unreadable, invalid JSON, not a list, any element failing
      PaidProposalRecord validation) raises the same sanitized error as a
      missing approval — one malformed record invalidates the whole file
      rather than being skipped. Allowed only if at least one record has
      service_name exactly equal to the requested name and
      status == 'approved'."""
    if service_name not in CANONICAL_SERVICE_NAMES:
        raise PaidApprovalRequiredError(f"unknown service_name {service_name!r}")

    if not is_paid:
        return

    try:
        raw_text = proposals_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PaidApprovalRequiredError(
            "paid-proposal data could not be loaded (missing, unreadable, or invalid)"
        ) from exc

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PaidApprovalRequiredError(
            "paid-proposal data could not be loaded (missing, unreadable, or invalid)"
        ) from exc

    if not isinstance(raw_data, list):
        raise PaidApprovalRequiredError(
            "paid-proposal data could not be loaded (missing, unreadable, or invalid)"
        )

    try:
        records = tuple(PaidProposalRecord.model_validate(item) for item in raw_data)
    except Exception as exc:
        raise PaidApprovalRequiredError(
            "paid-proposal data could not be loaded (missing, unreadable, or invalid)"
        ) from exc

    for record in records:
        if record.service_name == service_name and record.status == "approved":
            return

    raise PaidApprovalRequiredError(f"no approved paid-proposal record found for service_name {service_name!r}")
