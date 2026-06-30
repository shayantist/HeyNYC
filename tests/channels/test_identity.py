from heynyc.channels.identity import user_key


def test_user_key_is_deterministic_and_channel_scoped():
    a = user_key("whatsapp_meta", "+15551234567", "salt")
    assert a == user_key("whatsapp_meta", "+15551234567", "salt")          # deterministic
    assert a != user_key("whatsapp_twilio", "+15551234567", "salt")        # channel-scoped
    assert a != user_key("whatsapp_meta", "+15551234567", "other-salt")    # salt matters
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)        # opaque, short
    assert "+15551234567" not in a                                         # not reversible by eye
