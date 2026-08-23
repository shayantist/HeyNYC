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
        "cooling": cooling.get_tools()[0].parameters,
        "childcare": childcare.get_tools()[0].parameters,
        "clinics": clinics.get_tools()[0].parameters,
        "events": events.get_tools()[0].parameters,
        "food": food.get_tools()[0].parameters,
        "libraries": libraries.get_tools()[0].parameters,
        "restrooms": restrooms.get_tools()[0].parameters,
        "street_closures": street_closures.get_tools()[0].parameters,
        "wic": wic.get_tools()[0].parameters,
    }

    for schema in schemas.values():
        properties = schema["properties"]
        assert {"near", "max_results"} <= properties.keys()
        assert not {"k", "limit", "on", "window_start", "window_start_time"} & properties.keys()

    for name in ("cooling", "events", "food", "restrooms"):
        assert {"visit_date", "visit_time"} <= schemas[name]["properties"].keys()
    assert "visit_date" in schemas["street_closures"]["properties"]
    assert "visit_time" not in schemas["street_closures"]["properties"]
    for name in ("childcare", "clinics", "libraries", "wic"):
        assert not {"visit_date", "visit_time"} & schemas[name]["properties"].keys()

    for name in ("childcare", "clinics", "cooling", "food", "libraries", "restrooms", "street_closures", "wic"):
        assert "near" in schemas[name]["required"]
    assert "near" not in schemas["events"].get("required", [])
    assert "only when" in schemas["events"]["properties"]["visit_time"]["description"].lower()


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


def test_each_location_tool_owns_its_result_ceiling():
    def maximum(tool) -> int:
        field = tool.parameters["properties"]["max_results"]
        return next(choice["maximum"] for choice in field["anyOf"] if "maximum" in choice)

    maxima = {
        "childcare": maximum(childcare.get_tools()[0]),
        "clinics": maximum(clinics.get_tools()[0]),
        "cooling": maximum(cooling.get_tools()[0]),
        "events": maximum(events.get_tools()[0]),
        "food": maximum(food.get_tools()[0]),
        "libraries": maximum(libraries.get_tools()[0]),
        "restrooms": maximum(restrooms.get_tools()[0]),
        "street_closures": maximum(street_closures.get_tools()[0]),
        "wic": maximum(wic.get_tools()[0]),
    }

    assert maxima == {
        "childcare": 10,
        "clinics": 10,
        "cooling": 10,
        "events": 10,
        "food": 10,
        "libraries": 8,
        "restrooms": 10,
        "street_closures": 10,
        "wic": 10,
    }
