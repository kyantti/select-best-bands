# 12 – A half-started search makes `train_final.py` accuse its own number

Status: needs-triage
Blocked by:

## Description

`train_final.refuse_a_number_an_earlier_protocol_owns` decides a number belongs
to *this* protocol by looking for two markers:

```python
ours = (
    config.TABLES_DIR / f"{prefix}_candidates.csv",
    config.TABLES_DIR / f"{prefix}_ga_summary.json",
)
```

It omits `{prefix}_ga_config.json`, which `CandidateCache.check_sidecar` writes
the first time the cache is constructed — **before any candidate has trained**,
and therefore before `candidates.csv` exists.

So: `ga.py 22 --no-evaluate` against an empty cache writes
`exp_22_ga_config.json` and then dies on the first `LookupError`. Now
`train_final.py 22 --bands R G B` falls through to the `glob(f"{prefix}_*")`
branch and refuses with

> `exp_22_ga_config.json` belongs to an earlier run that this script cannot
> reproduce, so experiment 22 is not free. Use a new experiment number.

— a false accusation about a number this protocol created thirty seconds
earlier, and one that burns an experiment number unless the file is deleted by
hand. The same happens to any search killed early.

## Done when

- [ ] `ga.py NN --no-evaluate` against an empty cache, followed by
      `train_final.py NN --bands R G B`, is not refused as an earlier
      protocol's number.
- [ ] A real experiment 1–20 number is still refused.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Encontrado por `/code-review high` al cerrar `protocolo-80-20/17`. Parece una
línea (`ga_config.json` a `ours`), pero conviene mirar si `bootstrap.py` y
`analyze_test_errors.py` tienen la misma lista y si alguno de los 1–20 tiene un
`ga_config.json` que haría que el guard dejara de protegerlos.
