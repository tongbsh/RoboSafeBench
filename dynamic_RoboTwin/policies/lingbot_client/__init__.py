"""Vendored copy of LingBot-VA's websocket client utilities.

These two modules (``msgpack_numpy.py`` and ``websocket_client_policy.py``)
are tiny, self-contained, and live under ``lingbot-va/evaluation/robotwin/``
which is *not* an importable Python package (no ``__init__.py`` files in
the parent chain) and uses relative imports. Vendoring them here keeps the
``lingbot_va_policy`` adapter free from sys.path / package-init hacks.

If LingBot-VA upstream changes the wire protocol, sync the two files in
this directory.
"""
