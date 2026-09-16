# 07 – One crop grid, drawn in one place

Status: needs-triage
Blocked by:

## Description

`check_data.draw_grid` and `analyze_test_errors.draw_errors` draw the same kind
of figure and share the same body: prepare each crop through
`prepare_model_input`, concatenate the foreground pixels of the whole grid, fit
one linear scale to them, clip each panel into it, paint the masked background
flat grey, blank the unused axes, and put the bands in nm in the suptitle.
`BACKGROUND_GREY = 0.72` is declared twice, the second time with a comment
saying where the first one is.

The two differ only in what each panel is (train crops per class against
misread test crops), in the panel titles, and in whether the reflectance is
normalized on the train foreground or left raw. That is a small scale-and-grey
helper wanting to live in one place — `cnn/data_setup.py` beside
`prepare_model_input`, or a `cnn/figures.py`.

Found by `/code-review` while closing `protocolo-80-20/11`, which was not
allowed to touch another module ("Nothing in `config.py` changes and no other
module is touched"), so the duplication was left in on purpose rather than
widening that ticket's scope.

## Done when

- [ ] The scale-and-grey step and `BACKGROUND_GREY` are declared once.
- [ ] `sanity-check/data_sanity_check.png` and `out/figures/exp_21_test_errors.png`
      are redrawn and are the same figures they are now.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Cosmético y sin efecto sobre ningún resultado: las dos figuras tienen que salir
iguales. Por eso va a triaje y no a `ready-for-agent` directamente.
