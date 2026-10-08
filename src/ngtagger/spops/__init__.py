"""SmartPixels study operations (working rules 4-6): the machine-wide heavy-job lane,
frozen-copy launching with an import guard, and the results registry.

  lane.py      one heavy-job lock (flock) + legacy-pattern / Docker / resource checks
  frozen.py    frozen copies of whole trees, their manifest and the launch-time patch
  guard/       the import guard installed in every frozen-copy Python process
  launch.py    the launcher every heavy job goes through (CLI: ops/spheavy)
  registry.py  one file per result, with status, bias line and launch-time provenance

Nothing here is imported by ngtagger/__init__.py; every module is stdlib-only except
where noted, so the launcher runs under any Python >= 3.9.
"""
