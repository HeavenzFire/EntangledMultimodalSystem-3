from unittest.mock import MagicMock

from src.core.system_validator import SystemValidator


def test_validator_uses_injected_manager_and_monitor():
    manager = MagicMock()
    monitor = MagicMock()

    validator = SystemValidator(
        system_manager=manager,
        system_monitor=monitor,
    )

    assert validator.system_manager is manager
    assert validator.system_monitor is monitor
