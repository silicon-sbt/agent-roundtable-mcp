"""Built-in role library (optional, caller opt-in) compiled from agency-agents-zh.

320 roles across 21 departments, extracted deterministically into compact cards
(name / role / worldview / strengths / style). Source and license: see NOTICE /
config/agency_personas.json _meta (agency-agents-zh, MIT).

The roundtable does NOT use this library unless the caller asks for it: a council is
always exactly the personas configured in config/councils + config/domain_experts.
Callers opt in per run by passing add_personas entries of the form
{"agency_id": "<id>"} to run_roundtable, and can browse ids with
list_agents(source="agency[:<department>]").
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "config" / "agency_personas.json"


@lru_cache(maxsize=1)
def _payload() -> dict:
    """Load the compiled library once; any failure means no library."""
    try:
        data = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"_meta": {}, "personas": []}
    if not isinstance(data, dict) or not isinstance(data.get("personas"), list):
        return {"_meta": {}, "personas": []}
    return data


@lru_cache(maxsize=1)
def _cards() -> dict[str, dict]:
    cards: dict[str, dict] = {}
    for card in _payload()["personas"]:
        if isinstance(card, dict) and str(card.get("id", "")).strip():
            cards[str(card["id"])] = card
    return cards


def library_meta() -> dict:
    """Provenance of the built-in library (empty when it is not shipped)."""
    return dict(_payload().get("_meta") or {})


def list_cards(department: str = "", query: str = "", limit: int = 0) -> list[dict]:
    """Library cards, optionally filtered by department and/or a free-text query."""
    wanted_department = (department or "").strip().lower()
    needle = (query or "").strip().lower()
    cards = list(_cards().values())
    if wanted_department:
        cards = [c for c in cards if str(c.get("department", "")).lower() == wanted_department]
    if needle:
        cards = [
            c for c in cards
            if needle in (str(c.get("id", "")) + str(c.get("name", "")) + str(c.get("role", ""))).lower()
        ]
    if limit and limit > 0:
        cards = cards[:limit]
    return cards


def get_card(agency_id: str) -> dict | None:
    """One library card by id."""
    return _cards().get(str(agency_id or "").strip())


def card_to_persona_spec(card: dict) -> dict:
    """Convert a library card into an add_personas spec for run_roundtable."""
    return {
        "agency_id": str(card.get("id", "")),
        "name": str(card.get("name", "")),
        "role": str(card.get("role", "")),
        "worldview": str(card.get("worldview", "")),
        "speaking_style": "；".join(str(s) for s in (card.get("style") or [])),
        "strengths": [str(s) for s in (card.get("strengths") or [])],
        "agent_type": "agency_library",
    }


def agency_persona_spec(agency_id: str) -> dict:
    """add_personas spec for a library id; ValueError with a hint when unknown."""
    card = get_card(agency_id)
    if card is None:
        raise ValueError(
            "add_personas: unknown agency_id '" + str(agency_id) + "'. "
            "Browse ids with list_agents(source='agency') or list_agents(source='agency:<department>').",
        )
    return card_to_persona_spec(card)


__all__ = [
    "LIBRARY_PATH",
    "agency_persona_spec",
    "card_to_persona_spec",
    "get_card",
    "library_meta",
    "list_cards",
]