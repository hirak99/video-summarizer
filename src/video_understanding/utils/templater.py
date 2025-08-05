import re


class UnusedArgs(ValueError):
    pass


class LeftoverArgs(ValueError):
    pass


def fill(template: list[str], prompt_args: dict[str, str]) -> list[str]:
    lines: list[str] = []
    seen_keys: set[str] = set()
    for line in template:
        for key, val in prompt_args.items():
            if f"{{{key}}}" in line:
                seen_keys.add(key)
            line = line.replace(f"{{{key}}}", val)
            # Check that nothing that looks like {...} is left in the line.
        remaining_args: list[str] = re.findall(r"{(.*?)}", line)
        if remaining_args:
            raise LeftoverArgs(
                f"Args still remain after replacement: {remaining_args=}, {line=}."
            )
        lines.append(line)

    # Check that all keys were used.
    unused_keys = set(prompt_args.keys()) - seen_keys
    if unused_keys:
        raise UnusedArgs(f"Unused keys: {unused_keys}")

    return lines
