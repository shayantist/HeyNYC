"""The per-user structured draft store — real state, not LLM reconstruction."""
from __future__ import annotations

from heynyc.core.drafts import DraftStore


def test_merge_accumulates_losslessly_across_turns(tmp_path):
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"legal_name": "Ana Diaz"})
    merged = d.merge("snap", {"monthly_income": 1500})   # next turn passes only the new field
    assert merged == {"legal_name": "Ana Diaz", "monthly_income": 1500}   # turn-1 name retained


def test_edits_overwrite_but_empty_does_not_clobber(tmp_path):
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"monthly_income": 1500})
    assert d.merge("snap", {"monthly_income": 1800})["monthly_income"] == 1800   # an edit
    assert d.merge("snap", {"monthly_income": ""})["monthly_income"] == 1800       # empty ignored


def test_persists_across_store_instances(tmp_path):
    DraftStore(tmp_path).for_user("u1").merge("snap", {"legal_name": "Ana"})
    # a fresh store (a later request / restart) sees the persisted draft
    assert DraftStore(tmp_path).for_user("u1").load("snap") == {"legal_name": "Ana"}


def test_users_are_isolated(tmp_path):
    store = DraftStore(tmp_path)
    store.for_user("u1").merge("snap", {"legal_name": "Ana"})
    assert store.for_user("u2").load("snap") == {}


def test_clear_removes_a_program_draft(tmp_path):
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"legal_name": "Ana"})
    d.clear("snap")
    assert d.load("snap") == {}
