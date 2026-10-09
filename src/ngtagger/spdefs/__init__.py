"""Shared SmartPixels angle definitions (plan D.3, P0.3; definition sheet v1).

    from ngtagger.spdefs import angles as SA

One module for every consumer of the cluster angles (census, replay, studies): the angle
representation and its residuals, the incidence keys, the module-local conversion, class T,
the improved-angle knob k with its two uncertainty modes, the angle-uncertainty table (I/O,
lookup, application) and the cutting-consumer quantile. New files only: nothing that runs
today imports this package (ngtagger/__init__.py imports no subpackage), so it can be built
while census and tagger jobs run. Known-answer checks: ``python -m ngtagger.spdefs.checks``.
"""
