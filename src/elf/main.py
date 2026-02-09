from __future__ import annotations


def greet(name: str = "World") -> str:
    return f"Hello, {name}!"


def main() -> int:
    message = greet()
    print(message)
    return 0
