from elf.main import greet


def test_greet_default() -> None:
    assert greet() == "Hello, World!"


def test_greet_custom_name() -> None:
    assert greet("Copilot") == "Hello, Copilot!"
