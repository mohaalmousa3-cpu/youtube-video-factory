"""Tests for src/core/cost_guard.py — the Phase 1E runtime paid-service
guard. No provider, no network, no SQLite, no file write anywhere in this
file; proposals_path is only ever read."""
from __future__ import annotations

import json

import pytest

from src.core.cost_guard import (
    CANONICAL_SERVICE_NAMES,
    PaidApprovalRequiredError,
    PaidProposalRecord,
    require_paid_approval,
)


def _record(
    service_name: str = "qwen-image",
    status: str = "approved",
    proposal_id: str = "prop-1",
    estimated_monthly_cost_usd: float = 5.0,
    justification: str = "Needed for character-consistent scene art.",
    requires_explicit_user_approval: bool = True,
    approved_by: str | None = "the-user",
    approved_at: str | None = "2026-09-15T00:00:00Z",
) -> dict:
    return dict(
        proposal_id=proposal_id,
        service_name=service_name,
        estimated_monthly_cost_usd=estimated_monthly_cost_usd,
        justification=justification,
        status=status,
        requires_explicit_user_approval=requires_explicit_user_approval,
        approved_by=approved_by,
        approved_at=approved_at,
    )


def _write(path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


# ---------------------------------------------------------------------
# CANONICAL_SERVICE_NAMES / unknown service_name — fails closed regardless
# of is_paid
# ---------------------------------------------------------------------


def test_canonical_service_names_is_the_expected_fixed_set():
    assert CANONICAL_SERVICE_NAMES == frozenset({"groq", "tokenrouter", "kokoro", "qwen-image"})


@pytest.mark.parametrize("is_paid", [True, False])
def test_unknown_service_name_rejects_regardless_of_is_paid(tmp_path, is_paid):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(service_name="qwen-image", status="approved")])

    with pytest.raises(PaidApprovalRequiredError, match="unknown service_name"):
        require_paid_approval("not-a-real-service", proposals_path, is_paid=is_paid)


def test_unknown_service_name_rejects_even_with_no_proposals_file_at_all(tmp_path):
    proposals_path = tmp_path / "does-not-exist.json"

    with pytest.raises(PaidApprovalRequiredError, match="unknown service_name"):
        require_paid_approval("bogus", proposals_path, is_paid=False)


# ---------------------------------------------------------------------
# is_paid=False: always allowed for a canonical name, file never opened
# ---------------------------------------------------------------------


@pytest.mark.parametrize("service_name", sorted(CANONICAL_SERVICE_NAMES))
def test_free_call_allowed_without_reading_proposals_file(tmp_path, service_name):
    proposals_path = tmp_path / "does-not-exist.json"

    require_paid_approval(service_name, proposals_path, is_paid=False)  # must not raise

    assert not proposals_path.exists()


# ---------------------------------------------------------------------
# is_paid=True: missing/malformed/unreadable/invalid data all fail closed
# ---------------------------------------------------------------------


def test_paid_call_with_missing_file_rejects(tmp_path):
    proposals_path = tmp_path / "does-not-exist.json"

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_directory_as_proposals_path_rejects(tmp_path):
    proposals_path = tmp_path / "a-directory"
    proposals_path.mkdir()

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_invalid_json_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    proposals_path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_json_object_instead_of_array_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    proposals_path.write_text(json.dumps(_record()), encoding="utf-8")

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_record_missing_required_field_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    bad_record = _record()
    del bad_record["justification"]
    _write(proposals_path, [bad_record])

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_cost_above_ceiling_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(estimated_monthly_cost_usd=30.0)])

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_requires_explicit_user_approval_false_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(requires_explicit_user_approval=False)])

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_one_malformed_record_poisons_the_whole_file(tmp_path):
    """A second, otherwise-valid, approved record for the requested
    service does not rescue a file containing one malformed record."""
    proposals_path = tmp_path / "proposals.json"
    bad_record = _record(proposal_id="bad")
    del bad_record["status"]
    good_record = _record(proposal_id="good", service_name="qwen-image", status="approved")
    _write(proposals_path, [bad_record, good_record])

    with pytest.raises(
        PaidApprovalRequiredError, match="paid-proposal data could not be loaded"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


# ---------------------------------------------------------------------
# is_paid=True: well-formed data, approval matching logic
# ---------------------------------------------------------------------


def test_paid_call_with_empty_array_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [])

    with pytest.raises(
        PaidApprovalRequiredError, match="no approved paid-proposal record found"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


@pytest.mark.parametrize("status", ["proposed", "rejected", "expired"])
def test_paid_call_with_non_approved_status_rejects(tmp_path, status):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(service_name="qwen-image", status=status)])

    with pytest.raises(
        PaidApprovalRequiredError, match="no approved paid-proposal record found"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_approved_record_for_different_service_rejects(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(service_name="groq", status="approved")])

    with pytest.raises(
        PaidApprovalRequiredError, match="no approved paid-proposal record found"
    ):
        require_paid_approval("qwen-image", proposals_path, is_paid=True)


def test_paid_call_with_matching_approved_record_allows(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(service_name="qwen-image", status="approved")])

    require_paid_approval("qwen-image", proposals_path, is_paid=True)  # must not raise


def test_paid_call_scans_every_record_not_just_the_first(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(
        proposals_path,
        [
            _record(proposal_id="p1", service_name="groq", status="approved"),
            _record(proposal_id="p2", service_name="qwen-image", status="rejected"),
            _record(proposal_id="p3", service_name="qwen-image", status="approved"),
        ],
    )

    require_paid_approval("qwen-image", proposals_path, is_paid=True)  # must not raise


@pytest.mark.parametrize("service_name", sorted(CANONICAL_SERVICE_NAMES))
def test_each_canonical_service_name_can_be_approved(tmp_path, service_name):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(service_name=service_name, status="approved")])

    require_paid_approval(service_name, proposals_path, is_paid=True)  # must not raise


# ---------------------------------------------------------------------
# is_paid is required, keyword-only, no default
# ---------------------------------------------------------------------


def test_is_paid_is_required_keyword_argument():
    with pytest.raises(TypeError):
        require_paid_approval("qwen-image", None)  # type: ignore[call-arg]


def test_is_paid_cannot_be_passed_positionally(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    with pytest.raises(TypeError):
        require_paid_approval("groq", proposals_path, False)  # type: ignore[misc]


# ---------------------------------------------------------------------
# read-only guarantee
# ---------------------------------------------------------------------


def test_never_writes_to_the_proposals_file(tmp_path):
    proposals_path = tmp_path / "proposals.json"
    _write(proposals_path, [_record(service_name="qwen-image", status="approved")])
    bytes_before = proposals_path.read_bytes()

    require_paid_approval("qwen-image", proposals_path, is_paid=True)

    assert proposals_path.read_bytes() == bytes_before


def test_never_creates_a_proposals_file_for_a_free_call(tmp_path):
    proposals_path = tmp_path / "does-not-exist.json"

    require_paid_approval("groq", proposals_path, is_paid=False)

    assert not proposals_path.exists()


# ---------------------------------------------------------------------
# PaidProposalRecord itself — matches the JSON schema's constraints
# ---------------------------------------------------------------------


def test_paid_proposal_record_rejects_unknown_fields():
    data = _record()
    data["extra_field"] = "not allowed"
    with pytest.raises(Exception):
        PaidProposalRecord.model_validate(data)


def test_paid_proposal_record_rejects_cost_above_ceiling():
    with pytest.raises(Exception):
        PaidProposalRecord.model_validate(_record(estimated_monthly_cost_usd=25.01))


def test_paid_proposal_record_accepts_cost_exactly_at_ceiling():
    PaidProposalRecord.model_validate(_record(estimated_monthly_cost_usd=25.0))


def test_paid_proposal_record_rejects_empty_justification():
    with pytest.raises(Exception):
        PaidProposalRecord.model_validate(_record(justification=""))


def test_paid_proposal_record_rejects_invalid_status():
    with pytest.raises(Exception):
        PaidProposalRecord.model_validate(_record(status="approved-ish"))
