# src/midlearn/utils.py

def match_arg(arg: str, choices: list[str]) -> str:
    matches = [c for c in choices if c.startswith(arg)]
    if not matches:
        raise ValueError(f"'{arg}' is not a valid choice. Available choices: {choices}")
    if len(matches) > 1:
        raise ValueError(f"'{arg}' is ambiguous. It could be any of: {matches}")
    return matches[0]
