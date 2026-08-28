"""Shared typed request fields for location-based tools."""
from __future__ import annotations

from pydantic import Field

from .tools.base import ToolInput


class LocationRequest(ToolInput):
    """Common spatial anchor for a location-aware lookup."""

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
            "asks for a number; otherwise omit it."
        ),
    )
