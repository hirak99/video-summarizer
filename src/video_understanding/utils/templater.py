import enum
import re


class UnusedArgs(ValueError):
    pass


class LeftoverArgs(ValueError):
    pass


class _DoubleBrace(enum.Enum):
    OPEN = "{{"
    CLOSE = "}}"


def _split_double_brace(s: str) -> list[str | _DoubleBrace]:
    result: list[str | _DoubleBrace] = []
    for open_split in s.split(_DoubleBrace.OPEN.value):
        if result:
            result.append(_DoubleBrace.OPEN)
        first_close = True
        for close_split in open_split.split(_DoubleBrace.CLOSE.value):
            if not first_close:
                result.append(_DoubleBrace.CLOSE)
            result.append(close_split)
            first_close = False
    return result


def _join_double_brace(splitted: list[str | _DoubleBrace]) -> str:
    result: list[str] = []
    for s in splitted:
        if isinstance(s, _DoubleBrace):
            result.append(s.value[0])
        else:
            result.append(s)
    return "".join(result)


def fill(template: list[str], prompt_args: dict[str, str]) -> list[str]:
    lines: list[str] = []
    seen_keys: set[str] = set()
    for full_line in template:
        line_splitted = _split_double_brace(full_line)
        for index in range(0, len(line_splitted), 2):
            line = line_splitted[index]
            if isinstance(line, _DoubleBrace):
                continue

            required_args: list[str] = re.findall(r"{(.*?)}", line)

            for key, val in prompt_args.items():
                if f"{{{key}}}" in line:
                    seen_keys.add(key)
                line = line.replace(f"{{{key}}}", val)
            # Process leftovers to {{ -> { and }} -> }.
            line = re.sub(r"{{(.*?)}}", r"{\1}", line)

            # Check that nothing that looks like {...} is left in the line.
            remaining_args: list[str] = re.findall(r"{(.*?)}", line)
            # Do not count anything that was newly introduced by replacement.
            remaining_args = [arg for arg in remaining_args if arg in required_args]

            if remaining_args:
                raise LeftoverArgs(
                    f"Args still remain after replacement: {remaining_args=}, {line=}."
                )
            line_splitted[index] = line
        lines.append(_join_double_brace(line_splitted))

    # Check that all keys were used.
    unused_keys = set(prompt_args.keys()) - seen_keys
    if unused_keys:
        raise UnusedArgs(f"Unused keys: {unused_keys}")

    return lines
