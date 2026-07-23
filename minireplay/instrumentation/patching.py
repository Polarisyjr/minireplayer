from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def patch_method(
    owner: Any, name: str, factory: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> None:
    original = getattr(owner, name)
    if getattr(original, "__minireplay_wrapped__", False):
        return
    wrapped = factory(original)
    if wrapped is None:
        # A factory that forgets to return its wrapper otherwise fails several
        # frames later inside functools, as `'NoneType' has no attribute
        # '__module__'`, naming neither the adapter nor the patch point.
        raise RuntimeError(
            f"native replay patch factory returned nothing for {owner}.{name}: "
            f"{getattr(factory, '__qualname__', factory)} must return the wrapper"
        )
    functools.update_wrapper(wrapped, original)
    wrapped.__minireplay_wrapped__ = True
    wrapped.__minireplay_original__ = original
    setattr(owner, name, wrapped)


def method_identity(method: Callable[..., Any]) -> str:
    target = getattr(method, "__minireplay_original__", method)
    return f"{target.__module__}.{target.__qualname__}"
