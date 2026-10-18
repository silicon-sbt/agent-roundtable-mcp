"""Built-in role library tests (opt-in cards from agency-agents-zh, MIT)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server.core import apply_persona_changes, list_agents_impl, plan_council_impl
from roundtable import agency_library
from roundtable.agency_library import agency_persona_spec, get_card, library_meta, list_cards
from roundtable.loader import load_council_personas

ROOT = Path(__file__).resolve().parent.parent


def test_library_ships_and_is_attributed():
    meta = library_meta()
    cards = list_cards()
    assert len(cards) >= 100
    assert meta.get("license") == "MIT"
    assert "agency-agents-zh" in str(meta.get("source", ""))
    assert get_card("academic_historian") is not None


def test_list_agents_default_is_unchanged():
    agents = list_agents_impl(ROOT)
    assert agents
    assert all("agency_id" not in agent for agent in agents)


def test_list_agents_can_browse_the_library():
    cards = list_agents_impl(ROOT, "agency")
    assert len(cards) >= 100
    assert {"agency_id", "name", "role", "department"} <= set(cards[0])
    engineering = list_agents_impl(ROOT, "agency:engineering")
    assert engineering
    assert all(card["department"] == "engineering" for card in engineering)


def test_list_agents_rejects_an_unknown_source():
    with pytest.raises(ValueError):
        list_agents_impl(ROOT, "personas")


def test_unknown_agency_id_raises_with_a_hint():
    with pytest.raises(ValueError) as excinfo:
        agency_persona_spec("no-such-agency-role")
    assert "list_agents" in str(excinfo.value)


def test_add_personas_accepts_an_agency_id():
    _council, personas = load_council_personas("experts", ROOT)
    before = len(personas)
    updated = apply_persona_changes(personas, add_personas='[{"agency_id": "academic_historian"}]')
    assert len(updated) == before + 1
    added = updated[-1]
    assert added.name == "历史学家"
    assert added.role
    assert added.strengths
    assert added.agent_type == "agency_library"


def test_add_personas_agency_id_allows_explicit_overrides():
    _council, personas = load_council_personas("experts", ROOT)
    updated = apply_persona_changes(
        personas,
        add_personas='[{"agency_id": "academic_historian", "name": "历史顾问", "role": "只谈史料与年代" }]',
    )
    added = updated[-1]
    assert added.name == "历史顾问"
    assert added.role == "只谈史料与年代"
    assert added.strengths  # other card fields still come from the library


def test_plan_council_mentions_the_library():
    plan = plan_council_impl(topic="测试话题", council="experts", root_dir=ROOT)
    assert any("agency" in line for line in plan["suggestions"])


def test_missing_library_degrades_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr(agency_library, "LIBRARY_PATH", tmp_path / "missing.json")
    agency_library._payload.cache_clear()
    agency_library._cards.cache_clear()
    try:
        assert list_cards() == []
        assert library_meta() == {}
        with pytest.raises(ValueError):
            agency_persona_spec("academic_historian")
    finally:
        agency_library._payload.cache_clear()
        agency_library._cards.cache_clear()