# 13 – `analyze_test_errors.py` reads a CSV without closing it

Status: needs-triage
Blocked by:

## Description

`analyze_test_errors.py:300`:

```python
rows = list(csv.DictReader(path.open(newline="")))
```

No context manager, so the handle stays open until the garbage collector gets
to it. Every other reader in the repo uses `with`, and the function raises
`ValueError` a few lines later when the columns are not a predictions file's —
on that path the descriptor is left dangling.

Harmless in a script that exits, untidy in a module the tests import.

## Done when

- [ ] The file is read inside a `with`.
- [ ] `exp_21_test_error_analysis.csv` and `exp_21_test_errors.png` are redrawn
      and unchanged.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Encontrado por `/code-review high` al cerrar `protocolo-80-20/17`. Trivial, y
por eso mismo va a triaje en vez de hacerse dentro de un ticket que es
documentación: el ticket 17 no toca código.
