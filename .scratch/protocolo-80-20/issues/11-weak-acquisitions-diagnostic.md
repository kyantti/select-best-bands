# 11 – Diagnose the two weak test acquisitions before touching the model

Status: done
Blocked by: 08

## Description

First ticket of **phase 2**, and the one that goes first of all of them: a short script that answers whether the two weakest test acquisitions are a data problem before any improvement tunes the model against one. Zero GPU.

Of the eight test acquisitions, two sit at 0.689 weighted F1 against 0.857–0.951 for the other six: one is entirely C0, the other entirely C1. Seventeen of the errors are C0↔C3 confusions, which are the extreme classes and should be the easy ones to tell apart.

The script crosses the per-crop test predictions of an experiment with the partition manifest and the cropped-hypercube manifest, and reports:

- Errors per acquisition and per (actual, predicted) class pair, so the C0↔C3 block is visible and it is clear whether those 17 confusions are concentrated in the two weak acquisitions or spread across all eight.
- Whether the two weak acquisitions differ from the other six in anything measurable from the crops themselves: mean and spread of foreground reflectance per band range (exposure/illumination), foreground pixel fraction and crop area (segmentation quality), crop count.
- A crop grid of the misclassified crops from the two weak acquisitions so their images can be looked at.

The output is a table plus a figure; the conclusion (data problem or not) goes in this ticket's `## Evidence` in prose. Nothing in `config.py` changes and no other module is touched.

## Done when

- [x] `uv run python analyze_test_errors.py 21` prints the per-acquisition weighted F1 (matching the per-acquisition table written by `bootstrap.py`), the confusion breakdown per acquisition, and the exposure / foreground-fraction comparison of the two weak acquisitions against the other six.
- [x] It writes `out/tables/exp_21_test_error_analysis.csv` and `out/figures/exp_21_test_errors.png` (misclassified crops of the two weak acquisitions, actual and predicted class in each panel title, bands in nm).
- [x] Running it twice produces identical CSV.
- [x] `SKIP_SYNC=1 ./init.sh` is green.
- [x] `## Evidence` states, in one paragraph, whether the two acquisitions look broken (exposure, illumination, segmentation or labels) or simply hard, and therefore whether phase 2 proceeds as planned.

## Evidence

**No están rotas: son difíciles, y la fase 2 sigue como estaba planeada.**

`uv run python analyze_test_errors.py 21` (63 s, sin GPU, 256 recortes de test
leídos y solo medidos) escribe `out/tables/exp_21_test_error_analysis.csv` y
`out/figures/exp_21_test_errors.png`. Los ocho weighted F1 por captura salen
float a float iguales a los del `exp_21_bootstrap_366_262_225.json` — el script
no los recalcula, reutiliza `bootstrap.per_acquisition_scores` — y la matriz de
confusión del test entero que imprime, `[[50,3,6,13],[6,43,9,10],[4,7,46,3],
[4,3,3,46]]`, es exactamente la que registró el ticket 07. Dos corridas
seguidas dejan el CSV byte a byte idéntico (`diff` vacío), también después de
los cambios del `/code-review`.

**Nada que se pueda medir en los recortes distingue a las dos capturas flojas.**
Reflectancia media del primer plano: visible 0,1690 frente a 0,1804; red-edge
0,5877 frente a 0,5776; NIR 0,6305 frente a 0,6237. Las diferencias son de
±0,011 como mucho, contra una dispersión de 0,05 (visible) y 0,11 (NIR)
*dentro* de cada captura, y cada valor de las flojas cae dentro del rango que
abarcan las otras seis (la única excepción es el NIR de 042ef12d, 0,6347 contra
un máximo de 0,6285 en el resto: un 1 %). Segmentación igual: fracción de
primer plano 0,536 frente a 0,537 (rango del resto 0,518–0,550) y área media
9,9–10,1 kpx frente a 9,0–12,4 kpx. Exposición, iluminación y máscaras están
descartadas. La rejilla de los 38 recortes mal leídos lo confirma a ojo: son
higos normales, bien recortados, ninguno negro, quemado, vacío ni doble, y
ninguna captura tiene aspecto de estar mal etiquetada.

**Lo único que las separa es que son las dos capturas más grandes**: 40
recortes cada una frente a 24–36 (media 29,3) de las demás. Eso les da más peso
en el weighted F1 del test entero, pero no explica una exactitud por recorte de
0,53. Y conviene tener presente que **cada captura de test es de una sola
clase**, así que su weighted F1 es `2a/(1+a)` con `a` la exactitud: «captura
floja» no dice nada más que «poca exactitud en esa captura».

**Las 17 confusiones extremas no están concentradas en las dos flojas.** Son 13
C0→C3 y 4 C3→C0, repartidas entre cuatro capturas: 11 de las C0→C3 en la floja
042ef12d, 2 en la captura C0 fuerte (0,951), y las 4 C3→C0 enteras en las dos
capturas C3, ambas del grupo de referencia. Lo llamativo no es el reparto sino
el orden: dentro de 042ef12d los errores van C3 11 > C2 6 > C1 2, es decir, un
higo sano se lee como el grado más severo más veces que como su vecino. Eso es
un fallo ordinal del modelo con estas tres bandas, no un defecto de los datos.

**Conclusión:** no hay problema de datos que arreglar antes de la fase 2. Los
tickets 12–17 se implementan como estaban previstos.

### Decisiones y alcance

- El CSV y la figura se llaman por el número de experimento, sin sufijo de
  bandas, como pide el ticket; por eso el script lleva una columna
  `selected_bands` y `refuse_an_analysis_of_another_triplet`, que impide que un
  segundo tripletas del mismo experimento reescriba en silencio la tabla del
  primero (invariante de `out/`). `--bands` existe por lo mismo y por simetría
  con `bootstrap.py`; sin él, `21` resuelve su único modelo final.
- `config.py` no cambia y ningún otro módulo se toca, como pide el ticket. La
  duplicación que eso deja (el escalado + gris de la rejilla, compartido con
  `check_data.draw_grid`) queda anotada en `.scratch/repo-health/issues/07`.
- `run.sh` no llama a este script: la cadena son cuatro pasos y esto es un
  diagnóstico, no un resultado.

### Comandos

```
uv run python analyze_test_errors.py 21        # 63 s; la tabla, la figura y el informe
uv run python analyze_test_errors.py 21        # otra vez: diff vacío sobre el CSV
uv run python analyze_test_errors.py 21 --bands 1 2 3   # rechaza, exit 1
SKIP_SYNC=1 ./init.sh                          # verde, 111 tests (97 antes)
```

## Comments

Phase-2 ordering (spec, 15 Sep): this ticket is 0 h of GPU and gates nothing technically, but it goes first deliberately — if the labels or the crops of those two acquisitions are wrong, every later improvement is tuning a model against a data problem.
