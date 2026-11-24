import re
import subprocess

_IDEA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_idea_name(name):
    if not isinstance(name, str) or not _IDEA_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Idea 'Name' must contain only ASCII letters, digits, underscores, and hyphens."
        )
    return name


def run_chktex(writeup_file):
    result = subprocess.run(
        ["chktex", writeup_file, "-q", "-n2", "-n24", "-n13", "-n1"],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout
