"""ReCoRe package and lightweight process-local environment compatibility."""

import os as _os


def _configure_huggingface_cache(environ=None):
    """Replace the deprecated Transformers override before optional imports.

    Keep an explicit modern HF_HUB_CACHE. If only TRANSFORMERS_CACHE was set,
    reuse its directory through HF_HUB_CACHE so existing downloads stay usable.
    HF_HOME and the caller's shell configuration are left intact.
    """
    environ = _os.environ if environ is None else environ
    legacy = environ.pop("TRANSFORMERS_CACHE", None)
    if legacy:
        environ.setdefault("HF_HUB_CACHE", legacy)


_configure_huggingface_cache()
