from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools import geo
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint


def test_current_resolved_location_reuses_typed_state():
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="yes im still right there",
        user_turns=(
            "im at 82nd St and Roosevelt Ave in Queens",
            "yes im still right there",
        ),
        current_location=GeoPoint(
            40.74723,
            -73.88396,
            "82nd Street & Roosevelt Avenue, Queens, NY 11373",
            resident_query="82nd St and Roosevelt Ave in Queens",
        ),
    )

    assert geo.current_resolved_location(
        "82nd Street & Roosevelt Avenue, Queens, NY 11373",
        ctx,
    ) == ctx.current_location


def test_current_resolved_location_ignores_a_model_rephrasing_on_a_follow_up():
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="is there one open later today?",
        user_turns=(
            "im at 82nd St and Roosevelt Ave in Queens",
            "is there one open later today?",
        ),
        current_location=GeoPoint(
            40.74723,
            -73.88396,
            "82nd Street & Roosevelt Avenue, Queens, NY 11373",
            resident_query="82nd St and Roosevelt Ave in Queens",
        ),
    )

    assert geo.current_resolved_location(
        "82nd Street and Roosevelt Avenue, Elmhurst, New York 11372",
        ctx,
    ) == ctx.current_location


def test_current_resolved_location_rejects_an_unstored_model_location():
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="yes im still right there",
        user_turns=("im at 82nd St and Roosevelt Ave in Queens",),
    )

    assert geo.current_resolved_location("Times Square, Manhattan", ctx) is None


def test_current_resolved_location_does_not_override_a_new_current_address():
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="actually im at 100 Centre St now",
        user_turns=("I was near Times Square", "actually im at 100 Centre St now"),
        current_location=GeoPoint(
            40.7580,
            -73.9855,
            "Times Square, Manhattan, NY 10036",
            resident_query="Times Square",
        ),
    )

    assert geo.current_resolved_location("Times Square, Manhattan, NY 10036", ctx) is None
