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
