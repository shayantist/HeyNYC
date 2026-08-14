from types import SimpleNamespace

from heynyc.core.tools.geo import GeoPoint, rank_nearby


def test_rank_nearby_sorts_and_deduplicates_normalized_locations() -> None:
    origin = GeoPoint(40.75, -73.99, "origin")
    farther = SimpleNamespace(name="Farther", lat=40.77, lon=-73.99)
    nearest = SimpleNamespace(name="Nearest", lat=40.751, lon=-73.99)
    duplicate = SimpleNamespace(name="Nearest", lat=40.751, lon=-73.99)

    ranked = rank_nearby(
        origin,
        [farther, duplicate, nearest],
        key=lambda place: (place.name.casefold(), place.lat, place.lon),
        limit=2,
    )

    assert [place.name for place, _distance in ranked] == ["Nearest", "Farther"]
    assert ranked[0][1] < ranked[1][1]
