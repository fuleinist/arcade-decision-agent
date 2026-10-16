from .stub import StubGalaga

__all__ = ["StubGalaga", "make_env"]


def make_env(kind: str, **kw):
    """Env factory. Kept in one place so the CLI can't drift from the server.

    Unknown kwargs are dropped rather than raising: `seed` is meaningful to the stub and
    meaningless to a live page, and the eval runner passes the same set to both.
    """
    import inspect

    if kind in ("stub", "fake"):
        cls = StubGalaga
    elif kind in ("page", "emularity", "classicgamesarcade"):
        from .emularity import EmularityEnv
        cls = EmularityEnv
    else:
        raise SystemExit(f"unknown env {kind!r} (try: stub, page)")
    params = inspect.signature(cls.__init__).parameters
    allowed = {k: v for k, v in kw.items() if k in params}
    return cls(**allowed)