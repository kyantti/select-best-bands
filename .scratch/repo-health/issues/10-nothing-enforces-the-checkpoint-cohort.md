# 10 – "Two checkpoints, never one" is enforced by documentation alone

Status: needs-triage
Blocked by:

## Description

`pretrain.py` writes two backbones and a JSON sidecar for each, and its
docstring states the invariant that makes them two:

> `fit` sees the 22 train-fit Acquisitions and is the only backbone a search may
> start from … a search initialized from a backbone that had seen the validation
> crops would optimize a fitness contaminated by the very set it is scored on.

**Nothing checks it.** `cnn.model.checkpoint_path` only asks `is_file()`, and
`grep -rn "cohort" ga.py train_final.py cnn/` finds one comment and no reader:
the `pretrain_*.json` sidecar, which already carries `cohort` and
`acquisition_ids`, is never opened by anything that consumes a checkpoint.

So `ga.py 22 --checkpoint out/models/pretrain_train.pt` — one character away
from the right filename, and both files sit side by side in `out/models/` — is
accepted silently, and every cached fitness of a 22-hour search is scored on 6
validation acquisitions the backbone was adapted to. The run looks fine; the
number is meaningless; the cache makes it permanent.

The same hole has a second mouth. Under `VALIDATION_BALANCE = "crop_count_balanced"`
the inner boundary moves, so a checkpoint written under the 10 Sep boundary has
seen acquisitions that are now validation. Measured against the checkpoints on
disk today: **5 of the 6** validation acquisitions the balanced policy chooses
are inside the 22 that `pretrain_fit.pt` saw.

Related, and probably the same fix: `pretrain.py`'s sidecar records
`"validation_balance": config.VALIDATION_BALANCE` — the *constant* — not
anything derived from the partition file it actually read. `split_dataset.py
--validation-balance crop_count_balanced` leaves the constant at its default, so
a checkpoint pretrained against a balanced split writes `"acquisition_stratified"`
into the field the docstring calls the leakage record. The `acquisition_ids`
stay correct; the field a reader checks first does not.

README's "🚀 The phase-2 run" works around all of this with a leakage check the
reader is told to run by hand before spending the 22 hours. That is a stopgap,
not the fix.

## Done when

- [ ] `ga.py --checkpoint` refuses a backbone whose sidecar `acquisition_ids`
      intersect the validation acquisitions of the partition on disk, and
      `train_final.py --checkpoint` refuses one that intersects test.
- [ ] A checkpoint with no sidecar, or a sidecar that does not match the
      partition, is refused rather than assumed clean.
- [ ] The sidecar's `validation_balance` describes the partition the cohort was
      actually drawn from, not the constant at the time.
- [ ] The by-hand leakage check can come out of README's step 3.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Encontrado por `/code-review high` al cerrar `protocolo-80-20/17`. Verificado a
mano: el solapamiento de 5 sobre 6 es real, y `grep` confirma que nada lee el
sidecar. **Es el hallazgo más caro de los abiertos**: no rompe nada hoy porque
`BACKBONE_CHECKPOINT = None` por defecto y la 21 no usó ninguno, pero la corrida
de fase 2 es precisamente la que pasa `--checkpoint`, y el fallo no se ve — sale
un número bonito.

Va a triaje y no a `ready-for-agent` porque hay que decidir qué hace el guard
cuando no hay sidecar (¿rechazar, o avisar?) y si `train_final.py` debe exigir
la cohorte `train` o solo comprobar que no hay test.
