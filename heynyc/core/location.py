"""Shared typed request fields for location-based tools."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LocationRequest(BaseModel):
    """Common spatial anchor for a location-aware lookup."""

    model_config = ConfigDict(extra="forbid")

    near: str | None = Field(
        default=None,
        description=(
            "NYC address, neighborhood, landmark, or current conversation location to use as the "
            "spatial anchor. Omit only when the tool permits a citywide search."
        ),
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum choices requested by the resident. Set only when the resident explicitly "
            "asks for a number; otherwise omit it and use the tool default. Each tool sets its "
            "own maximum when it returns a list."
        ),
    )
