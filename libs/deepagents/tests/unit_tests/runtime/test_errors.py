from deepagents.runtime.errors import public_error_message


def test_public_error_message_normalizes_quota_errors():
    message = public_error_message("Error code: 429 - insufficient_quota")

    assert "模型额度可能已用尽" in message
    assert "210825684@qq.com" in message


def test_public_error_message_keeps_unknown_errors():
    assert public_error_message("custom parser failed") == "custom parser failed"
