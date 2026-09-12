"""Kept deliberately tiny — only what the real providers need right now.
Don't add estimate_cost()/cancel()/retry() etc. until a provider that needs
them actually exists; the full Provider contract from the plan is for later,
once there's more than one provider per role competing for the same job."""
from __future__ import annotations


class ProviderError(Exception):
    """Raised when a provider call fails — network, auth, or the service
    itself erroring. Callers catch this to decide retry/fallback."""
