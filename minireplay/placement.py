from __future__ import annotations

import os
import re

from .errors import ValidationError

_CPUSET = re.compile(r"^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$")


def declared_docker_cpuset() -> str | None:
    value = os.environ.get("NATIVE_REPLAY_DOCKER_CPUSET")
    if value is None:
        return None
    value = value.strip()
    if not value or _CPUSET.fullmatch(value) is None:
        raise ValidationError("NATIVE_REPLAY_DOCKER_CPUSET is invalid")
    return value
