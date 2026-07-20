"""Visualization helpers for the SmartPixels Phase-3 digiRefit study.

Public entry point: :func:`ngtagger.viz.refit_replay.build_refit_viz`.
"""


def build_refit_viz(*args, **kwargs):
    """Lazy proxy to :func:`ngtagger.viz.refit_replay.build_refit_viz`."""
    from ngtagger.viz.refit_replay import build_refit_viz as _impl

    return _impl(*args, **kwargs)
