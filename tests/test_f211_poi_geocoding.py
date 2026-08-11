import pytest

from heynyc.core.tools.geo import GeoPoint, geocode


@pytest.mark.xfail(
    strict=True,
    reason="F211: transit station identity needs an authoritative station resolver",
)
async def test_f211_station_query_uses_the_poi_geocoder_first() -> None:
    expected = GeoPoint(
        lat=40.77504,
        lon=-73.91203,
        label="Astoria-Ditmars Boulevard Station, Queens, New York",
        confidence=1.0,
        match_type="nominatim",
    )

    async def forgiving(_text: str) -> GeoPoint:
        return expected

    class AddressClient:
        async def get(self, *_args, **_kwargs):
            raise AssertionError("station queries must try the POI geocoder before address search")

    result = await geocode(
        "Ditmars Blvd subway station, Astoria, NY",
        client=AddressClient(),
        forgiving=forgiving,
    )

    assert result == expected
