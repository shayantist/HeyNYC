"""Model-facing event extraction without the structured catalog connector."""

from heynyc.modules.events.tools import extract_events_tool


def get_tools():
    return [extract_events_tool()]
