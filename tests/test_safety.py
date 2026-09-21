from src.safety import ConfirmationGate, Permission, SafetyPolicy


def test_unknown_actions_are_denied_and_logged():
    policy = SafetyPolicy()
    decision = policy.check("open_app")
    assert decision.allowed is False
    assert decision.reason == "no permission"
    assert len(policy.audit_log()) == 1


def test_explicit_permission_can_allow_action():
    policy = SafetyPolicy([Permission("read_file", True)])
    decision = policy.check("read_file")
    assert decision.allowed is True
    assert decision.reason == "explicit permission"


def test_permission_can_be_revoked():
    policy = SafetyPolicy([Permission("read_file", True)])
    policy.set_permission("read_file", False)
    assert policy.check("read_file").allowed is False


def test_confirmation_gate_blocks_unconfirmed_destructive_action():
    gate = ConfirmationGate(["delete_file"])
    assert gate.requires_confirmation("delete_file") is True
    assert gate.authorize("delete_file") is False
    assert gate.authorize("delete_file", confirmed=True) is True
    assert gate.authorize("read_file") is True


def test_audit_log_can_be_cleared():
    policy = SafetyPolicy([Permission("read_file", True)])
    policy.check("read_file")
    policy.clear_audit_log()
    assert policy.audit_log() == []
