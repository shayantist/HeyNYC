import importlib.util

from heynyc.core.tools import geo
from heynyc.modules.childcare import tools as childcare
from heynyc.modules.clinics import tools as clinics
from heynyc.modules.cooling_centers import tools as cooling
from heynyc.modules.events import tools as events
from heynyc.modules.food_pantries import tools as food
from heynyc.modules.libraries import tools as libraries
from heynyc.modules.nyc311_status import tools as nyc311
from heynyc.modules.public_restrooms import tools as restrooms
from heynyc.modules.street_closures import tools as street_closures
from heynyc.modules.wic import tools as wic


def test_location_tools_share_request_vocabulary():
    schemas = {
        "cooling": cooling.get_tools()[0]._input_schema(),
        "childcare": childcare.get_tools()[0]._input_schema(),
        "clinics": clinics.get_tools()[0]._input_schema(),
        "events": events.get_tools()[0]._input_schema(),
        "food": food.get_tools()[0]._input_schema(),
        "libraries": libraries.get_tools()[0]._input_schema(),
        "restrooms": restrooms.get_tools()[0]._input_schema(),
        "street_closures": street_closures.get_tools()[0]._input_schema(),
        "wic": wic.get_tools()[0]._input_schema(),
    }

    for schema in schemas.values():
        properties = schema["properties"]
        assert {"near", "max_results"} <= properties.keys()
        assert not {"k", "limit", "on", "window_start", "window_start_time"} & properties.keys()

    for name in ("cooling", "events", "food", "restrooms"):
        assert "active_at" in schemas[name]["properties"]
        assert not {"visit_date", "visit_time"} & schemas[name]["properties"].keys()
    for name in ("events", "food"):
        assert {"starts_after", "starts_before"} <= schemas[name]["properties"].keys()
    assert "visit_date" in schemas["street_closures"]["properties"]
    assert "visit_time" not in schemas["street_closures"]["properties"]
    for name in ("childcare", "clinics", "libraries", "wic"):
        assert not {"visit_date", "visit_time"} & schemas[name]["properties"].keys()

    for name in ("childcare", "clinics", "cooling", "food", "libraries", "restrooms", "street_closures", "wic"):
        assert "near" in schemas[name]["required"]
    assert "near" not in schemas["events"].get("required", [])


def test_shared_location_fields_have_one_definition():
    assert importlib.util.find_spec("heynyc.core.location") is not None
    from heynyc.core import location
    from heynyc.core.location import LocationRequest

    assert not hasattr(location, "NearbyRequest")
    assert not hasattr(location, "DatedNearbyRequest")
    assert issubclass(geo.NearestQuery, LocationRequest)
    assert issubclass(nyc311.ComplaintSearchQuery, LocationRequest)
    assert issubclass(cooling.CoolingQuery, LocationRequest)
    assert issubclass(childcare.ChildCareQuery, LocationRequest)
    assert issubclass(clinics.ClinicQuery, LocationRequest)
    assert issubclass(events.EventQuery, LocationRequest)
    assert issubclass(food.FoodHelpQuery, LocationRequest)
    assert issubclass(libraries.LibraryQuery, LocationRequest)
    assert issubclass(restrooms.PublicRestroomQuery, LocationRequest)
    assert issubclass(street_closures.StreetClosureQuery, LocationRequest)
    assert issubclass(wic.WicQuery, LocationRequest)


def test_location_tools_do_not_reject_a_resident_requested_count():
    tools = (
        childcare.get_tools()[0],
        clinics.get_tools()[0],
        cooling.get_tools()[0],
        events.get_tools()[0],
        food.get_tools()[0],
        libraries.get_tools()[0],
        restrooms.get_tools()[0],
        street_closures.get_tools()[0],
        wic.get_tools()[0],
    )

    for tool in tools:
        field = tool._input_schema()["properties"]["max_results"]
        assert all("maximum" not in choice for choice in field["anyOf"])
