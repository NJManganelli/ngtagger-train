"""spcheck: the invariant suite; the fast gate is its < 15 min tier (plan App. C.4, D.0, D.1).

    python -m ngtagger.spcheck.gate list
    python -m ngtagger.spcheck.gate run [--items ...] [--json OUT]            judge the code
    python -m ngtagger.spcheck.gate known-bad [--items ...] [--json OUT]      each must be RED
    python -m ngtagger.spcheck.gate derive-refs [--items ...]                 references from the known-good

Every item computes a statistic on the known-good input (canon_v2 TTbar AAAA file 3, first 5
events) with the code it is run from (the frozen copy, under the launcher), judges it by a
rule fixed in advance (identity with the registered reference, an invariant, or agreement
within the family-wise threshold of event-bootstrap errors), and has a named known-bad on
which the same rule must fail.
"""
