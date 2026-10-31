from src.utils.errors import EntangledSystemError, ManagementError


def test_management_error_is_a_system_error():
    error = ManagementError("system management failed")

    assert isinstance(error, EntangledSystemError)
    assert str(error) == "system management failed"
