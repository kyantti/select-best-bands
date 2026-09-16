# 14 – `pretrain.py`: SimCLR over band-triplet views, two checkpoints

Status: done
Blocked by: 13

## Description

`uv run python pretrain.py` pretrains a ResNet50 backbone with a SimCLR loop whose two views of a crop are two random band triplets of that same crop, and writes a checkpoint that ticket 13's injection point can load. Contrastive initialization is the only intervention in the fig-aflatoxin record that beat its own proposed criterion (+0.023 pooled, 8 of 8 candidates), rewritten here in this repo's shape — the original experiment-13 scripts are not ported and its checkpoints were deleted on 14 Sep.

Behaviour:

- Two views of a crop are two random band triplets drawn from the same crop, put through the same model-input preparation the network sees (normalize → resize → flips → rotation, image and mask together, background re-zeroed). NT-Xent loss, backbone only, no classification head.
- **Two checkpoints, never one**: `--cohort fit` pretrains on the 22 train-fit acquisitions and is the one the search uses; `--cohort train` pretrains on the 28 train acquisitions and is the one the final model uses. Neither may see a test acquisition.
- Each checkpoint is saved to `out/models/` (gitignored) next to a sidecar recording the acquisition identities it saw, the cohort name, the seed, the epochs and the library versions.
- Only train rows are ever loaded, the same way the search loads them, so a test crop cannot enter through the unsupervised path.

Tests (no GPU where possible): the saved cohort of a `fit` checkpoint contains no test and no validation acquisition; the cohort of a `train` checkpoint contains no test acquisition; the two view-drawing calls on one crop return two distinct in-range triplets of three distinct bands.

## Done when

- [x] `CUDA_VISIBLE_DEVICES=1 uv run python pretrain.py --cohort fit` writes `out/models/pretrain_fit.pt` and its sidecar listing exactly 22 acquisition identities (~3 min).
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python pretrain.py --cohort train` writes `out/models/pretrain_train.pt` and its sidecar listing exactly 28 acquisition identities (~3 min).
- [x] `uv run pytest -q tests/test_protocol.py -k pretrain` passes (cohort exclusion for both checkpoints, distinct band-triplet views).
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225 --checkpoint out/models/pretrain_fit.pt --epochs 1` runs to completion and prints metrics (correctness of the number is ticket 15's job, not this one's).
- [x] Re-running the same cohort with the same seed produces a checkpoint with identical parameters.
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

(16 Sep, sesión 13) **Dos checkpoints, y la trayectoria NT-Xent del experimento 13 replicada.**

1. **`--cohort fit`** (`out/logs/pretrain_fit.log`): 708 recortes en **22 capturas**, 200 épocas,
   NT-Xent **5.4670 → 1.2901** en **179,3 s** (~3 min, como pedía el ticket; los ~2,5 min
   extra de reloj son la carga de los 708 NPZ). sha `c40561a7…`. El sidecar
   `pretrain_fit.json` cruzado contra `data/evaluation_partitions.csv`:

   ```
   acquisitions         22 (expected 22) -> True
   crops                708
   no test acquisition  True | shared: 0
   no validation acq.   True | shared: 0
   is exactly the train-fit cohort of the manifest: True
   ```

2. **`--cohort train`** (`out/logs/pretrain_train.log`): 868 recortes en **28 capturas**,
   NT-Xent **5.5154 → 1.2776** en **196,0 s**. sha `b123ab04…`, distinto del anterior.
   `no test acquisition True | shared: 0`, `is exactly the train cohort: True`, y **contiene
   estrictamente** la cohorte `fit` más las 6 capturas de validación. Los dos checkpoints
   existen a la vez y ninguno vio una captura de test.

   La referencia de fig-aflatoxin (`docs/13-contrastive-init.md`) dice «NT-Xent falls from
   5.49 to 1.28 in about three minutes per fold». Las dos cohortes caen a 1,29 y 1,28 en
   179 y 196 s: la reescritura reproduce el **comportamiento** del original, no solo su forma.

3. `uv run pytest -q tests/test_protocol.py -k pretrain` → **21 passed**; la suite entera
   **169** (148 antes). Los tres que pide el ticket, más: la cohorte `train` no trae test,
   `draw_two_views` da siempre dos tripletes distintos de tres bandas distintas y en rango
   (64 sorteos) y se repite bajo la misma semilla de torch, las estadísticas por banda
   **coinciden con `fit_foreground_normalization`** del evaluador dentro de 1e-6 (es el
   `--verify-stats` del original, convertido en test), el checkpoint escrito **carga en
   `build_resnet50`** y la cabeza sigue siendo de 4 clases, y cada refusal tiene su test.

4. **Determinismo (done-when 5).** Segunda corrida de `fit` con la misma semilla a un
   fichero de scratch, comparada tensor a tensor con la de `out/models/`:

   ```
   tensors        318 vs 318  | same keys: True
   bit-identical  318/318
   worst delta    0.00e+00  (every parameter equal)
   ```

   El **sha del fichero sí difiere** (`c40561a7…` vs `0e37ff80…`): `torch.save` escribe un
   zip y su metadato no es reproducible byte a byte. Lo que el ticket pide son los
   parámetros, y esos son idénticos uno a uno. La identidad sha del contrato de fitness
   sigue siendo conservadora en el sentido correcto: dos ficheros de pesos iguales con sha
   distinto hacen que una cache se niegue, nunca que acepte de más.

   Esa misma corrida prueba una segunda cosa: `out/models/pretrain_fit.pt` lo escribió el
   código **anterior** a la refactorización del `/code-review` (mover los hiperparámetros al
   contexto) y el de scratch el **posterior**. 318/318 iguales ⇒ la refactorización no movió
   un solo peso, así que los dos checkpoints publicados son los que produce el código de
   este commit.

5. **`ga.py --evaluate` con el checkpoint (done-when 4).** Corre entero e imprime métricas;
   semilla de candidato **3104252108** sin cambio (el checkpoint mueve el contrato de
   fitness, no la semilla, que es el contrato del ticket 13), `checkpoint sha c40561a7…`,
   `mean`/`std` congelados idénticos a los de siempre.

   | brazo, **1 época** | weighted F1 | macro F1 | MAE | QWK |
   | --- | --- | --- | --- | --- |
   | ImageNet (control) | 0.3640 | 0.3742 | 1.0625 | 0.2104 |
   | `pretrain_fit.pt`  | 0.2528 | 0.2746 | 1.1188 | 0.0933 |

   **Esto no dice nada sobre la intervención y no hay que leerlo como si lo dijera.** Una
   época no entrena nada (el azar en 4 clases ya está en ~0,25) y el ticket dice
   explícitamente que la corrección del número es trabajo del 15. Se apunta el control
   porque sin él un 0,25 suelto parece una avería, y no lo es: el checkpoint carga bien
   (`load_backbone_checkpoint` aborta ante cualquier clave o forma que no sea la del
   backbone) y lo que se mide es un backbone que se ha alejado de ImageNet. Con solo
   `layer4` descongelado en el fine-tuning, las capas 1–3 se quedan **en los valores
   contrastivos**, que es justo lo que la puerta pareada del 15 tiene que medir a 50 épocas,
   con 5 semillas y sobre la validación equilibrada del 12.

6. `SKIP_SYNC=1 ./init.sh` **verde** de principio a fin (169 tests).

**Decisiones que el ticket no fija.**

- **La pérdida NT-Xent se calcula fuera del bloque `autocast`**, que es el único sitio donde
  esta reescritura se aparta del script original: allí estaba dentro, así que la matriz de
  similitud corría en float16 pese al `.float()`. NT-Xent exponencia una similitud dividida
  por 0,2 y float16 satura mucho antes. Queda anotado en el código; el 15 mide este bucle
  con sus propios números, así que lo que se mide es la versión correcta.
- **El sorteo de vistas usa el stream de torch** (`randperm` para las bandas, `rand` para el
  aumento), como `SelectedBandDataset`, en vez del `random.Random(initial_seed + index)` del
  original. Es reproducible **al mismo número de workers**, que por eso el sidecar registra.
- `--force` y `--output` no están en el ticket: sin uno de los dos no hay forma de comprobar
  el done-when 5 sin destruir el checkpoint publicado. Por defecto `pretrain.py` **se niega**
  a reescribir un checkpoint, porque es parte del contrato de fitness de todo candidato
  entrenado desde él.
- **`out/models/` sigue gitignored** (91 MB por fichero), así que los dos `.pt` y sus
  sidecars no entran en git; los dos logs de `out/logs/` sí, que son el registro de cómo se
  hicieron. Quien clone el repo los regenera con los dos comandos de arriba.
- La duplicación de los dos filtros de fila con `ga.load_selection_data` se deja a propósito:
  extraerlos obligaría a tocar `cnn/data_setup.py`, que es el camino compartido que la
  verificación bit a bit protege, para ahorrar dos líneas.

## Comments

Two checkpoints is a protocol requirement, not an optimization: the search must never be initialized from a backbone that saw validation crops, or the fitness it optimizes is contaminated by the set it is scored on.
