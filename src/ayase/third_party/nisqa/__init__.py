"""Vendored NISQA — Non-Intrusive Speech Quality Assessment.

Source: https://github.com/gabrielmittag/NISQA (MIT licensed, vendored to
avoid the upstream pip package's cascading dependency downgrade — see
the parent module docstring in ``ayase/modules/audio_nisqa.py`` for the
full story).

Public entrypoint: :class:`nisqaModel` from ``NISQA_model``.
"""

from .NISQA_model import nisqaModel

__all__ = ["nisqaModel"]
