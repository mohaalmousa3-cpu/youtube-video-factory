"""Phase 2B: typed, immutable local artifact records.

An ArtifactRecord is a durable, local claim that one specific file exists
for a project (or one of its scenes) — its kind, identity, and integrity
facts (path, size, checksum) — registered in SQLite by
src/database/artifact_repository.py. It is NOT proof the file is valid;
that verification is src/core/artifact_verifier.py's job, run on demand
against these recorded facts. Nothing in this module reads a file, writes
to disk, or touches SQLite — pure data, same convention as
src/models/project_state.py and src/models/manifest.py.

Phase 2B never produces an artifact itself — no TTS, image generation,
animation, rendering, or QC runs anywhere in this phase."""
from __future__ import annotations

import re
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from pydantic import Field, field_serializer, field_validator, model_validator

from src.models.common import FrozenStrictModel
from src.models.enums import ArtifactKind

# Scene-level kinds describe one specific scene's produced asset and
# therefore require scene_id. Project-level kinds describe a whole-video
# artifact (the assembled render, its QC report) and therefore must NOT
# carry a scene_id — there is no such thing as "half a render".
SCENE_LEVEL_ARTIFACT_KINDS: frozenset[str] = frozenset({"audio", "visual", "animation"})
PROJECT_LEVEL_ARTIFACT_KINDS: frozenset[str] = frozenset({"render", "qc_report", "overlay_render"})

# Same scene_id shape as src/models/scene.py's ScenePlanItem.scene_id —
# duplicated here (rather than imported) because this is an independent
# identity check on an already-built string, not a structural scene model.
_SCENE_ID_PATTERN = re.compile(r"^scene-[0-9]{2,3}$")


def _validate_relative_path(value: str) -> str:
    """A normalized, safe, forward-slash relative path: not empty, not
    absolute (POSIX '/' or a Windows drive letter), no backslashes, and no
    '.', '..', or empty path segments. This is a string-only check — it
    cannot see a symlink, so it does not by itself prove the resolved path
    stays under a project directory at runtime; src/core/artifact_verifier.py
    does that filesystem-aware check separately."""
    if not value or not value.strip():
        raise ValueError("relative_path must not be empty")
    if "\\" in value:
        raise ValueError("relative_path must use forward slashes only ('/'), not backslashes")
    if value.startswith("/"):
        raise ValueError("relative_path must be relative, not absolute (starts with '/')")
    if re.match(r"^[A-Za-z]:", value):
        raise ValueError("relative_path must be relative, not an absolute Windows path (drive letter)")
    segments = value.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(
            f"relative_path {value!r} must be a normalized relative path "
            "(no empty, '.', or '..' segments)"
        )
    return value


class ArtifactRecord(FrozenStrictModel):
    """One produced (or to-be-produced) artifact's identity and integrity
    facts."""

    artifact_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    project_id: str = Field(min_length=1)
    kind: ArtifactKind

    # Required for scene-level kinds, forbidden for project-level kinds —
    # see _scene_association_matches_kind below.
    scene_id: str | None = None

    # Always forward-slash, always relative to the project directory (the
    # directory containing that project's manifest.json) — never an
    # absolute path, drive letter, or traversal segment.
    relative_path: str

    byte_size: int = Field(ge=0)
    sha256_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    created_at: datetime

    # Minimal, JSON-safe only — no nested containers, no arbitrary
    # objects. A small bag of production notes (e.g. {"voice": "kokoro-af"}),
    # never a place to stash something structural that belongs in its own
    # typed field. Deeply immutable, matching this model's frozen=True:
    # pydantic validates the input as a normal mapping first (so the
    # scalar-only / no-nested-container checks below still apply exactly
    # as before), then _metadata_is_immutable freezes it into a
    # types.MappingProxyType — item assignment, update(), pop(), clear(),
    # and del all fail with AttributeError, not just attribute
    # reassignment (which frozen=True already blocked on its own).
    # _serialize_metadata converts it back to a plain dict on model_dump,
    # since pydantic-core's serializer does not know how to encode a
    # mappingproxy directly; artifact_repository.py's own
    # json.dumps(dict(record.metadata), ...) does the same for the SQLite
    # row.
    metadata: Mapping[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("relative_path")
    @classmethod
    def _relative_path_is_safe(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("scene_id")
    @classmethod
    def _scene_id_format(cls, value: str | None) -> str | None:
        if value is not None and not _SCENE_ID_PATTERN.fullmatch(value):
            raise ValueError(f"scene_id {value!r} must match {_SCENE_ID_PATTERN.pattern!r}")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_is_immutable(
        cls, value: Mapping[str, str | int | float | bool | None]
    ) -> MappingProxyType:
        return MappingProxyType(dict(value))

    @field_serializer("metadata")
    def _serialize_metadata(
        self, value: Mapping[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        return dict(value)

    @model_validator(mode="after")
    def _scene_association_matches_kind(self) -> "ArtifactRecord":
        if self.kind in SCENE_LEVEL_ARTIFACT_KINDS and self.scene_id is None:
            raise ValueError(f"artifact kind {self.kind!r} is scene-level and requires scene_id")
        if self.kind in PROJECT_LEVEL_ARTIFACT_KINDS and self.scene_id is not None:
            raise ValueError(f"artifact kind {self.kind!r} is project-level and must not set scene_id")
        return self


class ArtifactVerificationResult(FrozenStrictModel):
    """One artifact's read-only verification outcome, as produced by
    src/core/artifact_verifier.py. Never itself mutates the artifact
    registry, a project's lifecycle state, or any file."""

    artifact_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    kind: ArtifactKind
    scene_id: str | None = None
    relative_path: str = Field(min_length=1)
    passed: bool
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reasons_consistent_with_passed(self) -> "ArtifactVerificationResult":
        if self.passed and self.reasons:
            raise ValueError("reasons must be empty when passed is True")
        if not self.passed and not self.reasons:
            raise ValueError("reasons must be non-empty when passed is False")
        return self
