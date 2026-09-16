# 13 – Optional backbone checkpoint, inert when unset

Status: done
Blocked by: 09

## Description

`cnn/model.py` learns to load an optional pretrained backbone before the usual fine-tuning, and proves that with no checkpoint the code path is exactly today's. This ticket is the licence to touch the evaluator at all: the delta-zero test is what guarantees the injection point cannot move the published number by itself.

Behaviour:

- `config.BACKBONE_CHECKPOINT` defaults to `None`. With `None`, the model is built exactly as in ticket 04 (ResNet50 with ImageNet weights, 4-class head) and no new call is made — the same ops in the same order, so the RNG stream is untouched.
- With a path, the checkpoint's `state_dict` is loaded into the backbone after the ImageNet weights and before fine-tuning; the head is never loaded from it. A checkpoint whose keys or shapes do not match the backbone is an error, not a silent partial load. `--checkpoint` overrides the constant on `ga.py --evaluate` and `train_final.py`.
- The checkpoint identity (path plus the SHA-256 of its bytes, or `null`) joins the fitness contract of the candidate cache, so candidates evaluated with and without a checkpoint can never be mixed in one CSV.
- The candidate seed derivation does **not** change: a checkpoint changes the fitness contract, not the seed, so 366/262/225 still derives 3104252108.

Tests: without a checkpoint, the built model's parameters are identical to the one built before this ticket (compare against a freshly constructed reference); a mismatched checkpoint raises; a cache written with one checkpoint identity is refused when loaded under another.

## Done when

- [x] `uv run python -c "import config; print(config.BACKBONE_CHECKPOINT)"` prints `None`.
- [x] `uv run pytest -q tests/test_protocol.py -k checkpoint` passes.
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225` still prints candidate seed `3104252108` and weighted F1 `0.8700979843225085`, and the run reports the delta against the recorded value as exactly `0.00e+00`.
- [x] A cache written with `BACKBONE_CHECKPOINT=None` aborts with a fitness-contract mismatch naming the checkpoint when reloaded with `--checkpoint` set (record the command).
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

(16 Sep) **Con el checkpoint sin poner, nada se movió.**

1. `uv run python -c "import config; print(config.BACKBONE_CHECKPOINT)"` → `None`.

2. `uv run pytest -q tests/test_protocol.py -k checkpoint` → **19 passed**; la suite entera
   **148 verdes** (129 antes). Entre ellos el de delta cero en CPU
   (`test_checkpoint_unset_leaves_the_model_and_the_random_stream_untouched`):
   contra una referencia del modelo **escrita a mano** tal y como lo construía el
   ticket 04, los parámetros salen idénticos uno a uno, los `requires_grad`
   también, y `torch.rand(4)` después de las dos construcciones da los mismos
   cuatro números — el punto de inyección no consume aleatoriedad propia.

3. `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225 --predictions <scratch>/delta_zero_validation_predictions.csv`
   (el `--predictions` va fuera de `out/` porque el fichero por defecto lo escribió
   el ticket 04 y no se sobrescribe). Salida: semilla de candidato **3104252108**,
   weighted F1 **0.8700979843225085**, macro `0.8763610130071496`, MAE `0.1875`,
   QWK `0.8711063372717508`, mean y std congelados exactos, 39,8 s.
   **delta weighted_f1 `0.00e+00`** contra el valor grabado (el programa no imprime
   la resta: la derivada de sus dos valores idénticos es lo que se registra aquí).
   `diff` **vacío** contra `out/tables/evaluate_366_262_225_validation_predictions.csv`:
   las 160 predicciones de validación, las cuatro columnas.

4. **El comando de la cache** (done-when 4):

   ```
   CUDA_VISIBLE_DEVICES=1 uv run python ga.py 21 --no-evaluate --checkpoint <scratch>/fake_backbone.pt
   ```

   sale 1 con una línea en stderr:

   ```
   ga.py: .../out/tables/exp_21_candidates.csv was written under another fitness contract
   (backbone_checkpoint differ); its numbers are not comparable. Use a new experiment
   number, or delete .../out/tables/exp_21_ga_config.json and the CSV.
   ```

   `backbone_checkpoint` es **el único** campo que nombra, así que sin checkpoint
   el contrato de fitness sigue siendo byte a byte el que grabó el experimento 21:
   esa lista es la prueba de que ningún otro campo se movió. `git status --short out/`
   vacío (la guarda salta antes de escribir nada).

5. Cadena de humo del Definition of Done, `./run.sh 99 --population 4 --generations 1 --epochs 1`
   (~8 min): los diez `exp_99_*` de `out/tables`, las tres figuras, el `.pt` y los tres
   logs. Ganador `(391, 201, 105)`, el mismo que el humo del ticket 05 con esos flags.
   Ni `exp_99_ga_config.json` ni `exp_99_final_metrics_*.json` traen la clave
   `backbone_checkpoint`. Borrado todo con `rm -f`; `git status --short out/` vacío.

6. `SKIP_SYNC=1 ./init.sh` verde de principio a fin.

**Decisión de diseño que el ticket no fija:** la identidad del checkpoint entra en el
contrato de fitness **solo cuando hay uno**, en vez de entrar siempre y valer `null`.
Una clave con `null` habría hecho que `exp_21_ga_config.json` — escrito antes de que la
clave existiera — dejara de casar, y `ga.py 21 --no-evaluate` habría abortado; lo mismo
con los `exp_NN_final_metrics_*.json` ya grabados, que habrían cambiado de bytes al
repetirlos. `check_sidecar` compara la **unión** de las claves, así que ausente-contra-
presente sigue siendo un choque en los dos sentidos y la propiedad que pide el ticket
(«nunca mezclados en un CSV») se mantiene. La guarda de `train_final.py` sí lleva la
clave siempre, con `None` cuando no hay checkpoint, porque compara con `recorded.get(...)`
y así un resultado grabado sin checkpoint se sigue pudiendo repetir sin él.

Tras `/code-review`: las cuatro líneas duplicadas que anunciaban el checkpoint se
unifican en `ga.print_checkpoint`; `load_backbone_checkpoint` deja de desenvolver un
`{"state_dict": ...}` que nadie escribe; `write_metrics` pasa de comprobar veracidad a
`is None`, como el resto; y **las tres órdenes resuelven su propio checkpoint**
(`checkpoint_path` al entrar en `evaluate_command`, `search_command` y `final_command`),
porque antes `config.BACKBONE_CHECKPOINT` solo lo leían los dos `main()` y una llamada
en proceso — la que hará la puerta pareada del ticket 15 — se habría quedado en silencio
con el brazo de ImageNet. `run.sh` **no** enruta `--checkpoint` a propósito, y `CLAUDE.md`
ahora lo dice: la búsqueda y el modelo final necesitan checkpoints distintos (ticket 14),
así que una cadena que los quiera pone `config.BACKBONE_CHECKPOINT` o corre los pasos a mano.

## Comments

This is the only phase-2 ticket that edits the evaluator. The `0.00e+00` check is not a formality: if it is not exactly zero, the injection point has perturbed the RNG order or the construction sequence and the port stops being verifiable. Do not proceed to ticket 14 until it is.
