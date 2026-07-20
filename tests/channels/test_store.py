import time
from heynyc.channels.store import ChannelStore


def _store(tmp_path, **kw):
    kw.setdefault("rate_limit", 3)
    kw.setdefault("window_s", 60)
    kw.setdefault("dedup_ttl_s", 3600)
    return ChannelStore(tmp_path / "ch.sqlite3", **kw)


def test_seen_dedups_and_persists(tmp_path):
    s = _store(tmp_path)
    assert s.seen("wamid.A") is False   # first time
    assert s.seen("wamid.A") is True    # repeat
    reopened = _store(tmp_path)
    assert reopened.seen("wamid.A") is True   # survives a restart


def test_allow_trips_after_limit(tmp_path):
    s = _store(tmp_path, rate_limit=3, window_s=60)
    assert [s.allow("u1") for _ in range(4)] == [True, True, True, False]
    assert s.allow("u2") is True        # other users unaffected


def test_allow_window_resets(tmp_path):
    s = _store(tmp_path, rate_limit=1, window_s=1)
    assert s.allow("u1") is True
    assert s.allow("u1") is False
    time.sleep(1.1)
    assert s.allow("u1") is True


def test_pending_delete_stages_and_pops_once(tmp_path):
    s = _store(tmp_path)
    assert s.pop_pending_delete("u1") is None
    s.set_pending_delete("u1")
    s.set_pending_delete("u1")            # a fresh DELETE replaces the un-confirmed one
    staged = s.pop_pending_delete("u1")
    assert staged is not None
    assert s.pop_pending_delete("u1") is None   # consumed once


def test_delete_user_removes_flags_but_keeps_spend(tmp_path):
    """DELETE MY DATA wipes the resident's own flag rows (pending + confirmed) but leaves the
    anonymized daily spend record standing for abuse control (the survivor promised in the copy)."""
    s = _store(tmp_path)
    s.set_pending_flag("u1", 3, "report")
    s.add_flag("u1", 5, "report")
    s.set_pending_delete("u1")
    s.add_spend("u1", "2026-07-20", 0.07)
    s.add_flag("u2", 1, "report")         # another user is untouched

    s.delete_user("u1")

    remaining = s.flags()
    assert [f["user_key"] for f in remaining] == ["u2"]   # only the other user's flag is left
    assert s.pop_pending_flag("u1") is None
    assert s.pop_pending_delete("u1") is None
    assert abs(s.daily_spend("u1", "2026-07-20") - 0.07) < 1e-9   # spend survives


def test_first_contact_is_true_once_then_false_forever(tmp_path):
    s = _store(tmp_path)
    assert s.first_contact("u1") is True
    assert s.first_contact("u1") is False
    assert s.first_contact("u2") is True          # per user
    reopened = _store(tmp_path)
    assert reopened.first_contact("u1") is False  # survives a restart (once EVER)
