# Spec: estructura del TFG original con el protocolo fiable de fig-aflatoxin

Status: ready-for-agent

Plan de implementación detallado (pasos, orígenes de cada bloque de código, verificación): `~/.claude/plans/quiero-reorganizar-el-proyecto-parsed-llama.md`. Los tickets de esta feature se crean con `/to-tickets` a partir de este spec.

## Problem Statement

Pablo tiene dos versiones de su TFG de selección de bandas hiperespectrales para detectar aflatoxina en higos.

La versión original (este repositorio) es plana y fácil de seguir: un `ga.py` con las constantes en cabecera, una carpeta `cnn/` con la carga de datos y el bucle de entrenamiento, un `run.sh` y las salidas en `out/tables/exp_NN_*`. Pero su protocolo daba resultados que no se pueden defender: repartía train y test por higo, de modo que las 36 capturas (Acquisitions) aparecían en los dos lados; el algoritmo genético puntuaba cada candidato sobre el test; normalizaba con las constantes de ImageNet; sembraba el generador aleatorio después de crear la población; y la mutación era tan pequeña que la población colapsaba a un solo candidato.

La versión actual (`~/Documents/fig-aflatoxin`) corrigió todo eso y produjo el primer resultado fiable el 10 de septiembre de 2026: weighted F1 0,722 en un test que nunca intervino en la selección, con intervalo bootstrap por captura [0,645, 0,848] y bandas 366/262/225 (891,0 / 747,5 / 697,1 nm). Pero para conseguirlo creció hasta un paquete de tres capas con configuración pydantic en YAML, MLflow, artefactos inmutables direccionados por huella SHA-256, un DAG de etapas y más de 300 tests. Pablo ya no entiende su propio proyecto y no puede explicarlo ni modificarlo con soltura.

## Solution

Una rama nueva de este repositorio que conserva su forma (un script por paso, constantes en un único módulo, `cnn/` con tres ficheros, `run.sh`, salidas `exp_NN_*`) y contiene, copiado literalmente, el comportamiento que hace fiable el resultado actual:

- Un único reparto 80/20 por captura y estratificado por SeverityClass, y dentro del train otro 80/20 que separa una partición de validación. La búsqueda puntúa en validación; el test se lee una sola vez, al final.
- Normalización por canal calculada solo sobre los píxeles de higo del train; imagen y máscara se transforman juntas en el aumento.
- Fitness = weighted F1 en validación; semilla de entrenamiento derivada de cada candidato y de los datos, así que el valor no depende del orden en que la búsqueda lo encuentre.
- Operadores genéticos que garantizan tres bandas distintas, mutación reparada (σ 20, probabilidad 0,3), siembra antes de crear la población y aislamiento del estado aleatorio de la evolución respecto del entrenamiento.
- Modelo final entrenado con todo el train sin ningún cargador de evaluación, una única inferencia sobre el test, e intervalo bootstrap por captura.

Todo lo que era infraestructura (MLflow, huellas, pydantic, YAML, DAG, contenedores) desaparece. Los datos de entrada son los 1.124 recortes ya generados (CroppedHypercubes en NPZ) con su manifiesto y su SpectralAxis; el preprocesado desde los ENVI crudos no se porta.

La versión nueva debe reproducir bit a bit el candidato ganador, el modelo final y el intervalo del 10 de septiembre, y replicar la búsqueda genética entera sin GPU a partir de los 383 candidatos ya evaluados.

## User Stories

1. Como estudiante, quiero ver toda la configuración del experimento en un único módulo de constantes, para entender de un vistazo qué semillas, proporciones y parámetros produjeron un resultado.
2. Como estudiante, quiero lanzar cada paso con un script de nombre evidente (`split_dataset`, `ga`, `train_final`, `bootstrap`), para seguir el flujo sin leer un orquestador.
3. Como estudiante, quiero un `run.sh` que encadene los cuatro pasos con un número de experimento, para reproducir la cadena completa con un solo comando.
4. Como estudiante, quiero que el reparto train/test agrupe por captura (Acquisition), para que ningún recorte de una captura de test haya sido visto durante la búsqueda ni el entrenamiento.
5. Como estudiante, quiero que el reparto esté estratificado por SeverityClass en las dos fronteras (train/test y train/validation), para que cada clase esté representada a ambos lados.
6. Como estudiante, quiero que el reparto falle con un mensaje claro si alguna clase se queda sin capturas a un lado, en vez de degradar la estratificación en silencio.
7. Como estudiante, quiero que el reparto con la semilla 20230717 reproduzca exactamente la partición del 10 de septiembre (22 / 6 / 8 capturas, 708 / 160 / 256 recortes), para que los resultados nuevos sean comparables con el registrado.
8. Como estudiante, quiero que al cargar el manifiesto de partición se compruebe que ninguna captura ni recorte cruza fronteras, para que un fichero editado a mano no introduzca una fuga.
9. Como estudiante, quiero que la búsqueda genética solo cargue en memoria los recortes de train (ajuste y validación), para que sea estructuralmente imposible que lea el test.
10. Como estudiante, quiero que la normalización se calcule solo con píxeles de primer plano del train, para que ni el fondo ni el test desplacen la distribución de reflectancia.
11. Como estudiante, quiero que un canal con varianza cero haga fallar la evaluación, para no dividir por un valor arbitrario sin enterarme.
12. Como estudiante, quiero que los volteos y la rotación se apliquen a la imagen y a la máscara a la vez y que el fondo vuelva a cero después, para que el aumento no invente reflectancia donde no hay higo.
13. Como estudiante, quiero que cada candidato entrene con una semilla derivada de sus bandas y de la identidad de los datos, para que su fitness sea el mismo lo encuentre quien lo encuentre y cuando lo encuentre.
14. Como estudiante, quiero que la semilla derivada coincida con la de la corrida real (3104252108 para 366/262/225), para poder verificar el porte contra los números ya publicados.
15. Como estudiante, quiero que el fitness sea weighted F1 en validación y que el desempate sea MAE ordinal menor y luego índices menores, para que el ganador esté definido sin ambigüedad.
16. Como estudiante, quiero que el ganador reportado se re-desempate sobre todos los candidatos evaluados que igualen al mejor, para que no dependa del orden interno de DEAP.
17. Como estudiante, quiero que las tripletas tengan siempre tres bandas distintas, tras la inicialización, el cruce y la mutación, para no entrenar una ResNet con dos canales iguales.
18. Como estudiante, quiero que la reparación de duplicados conserve el orden de las bandas, porque el orden de canal es semántico para la red.
19. Como estudiante, quiero que el rango de bandas se lea del SpectralAxis y no de una constante, para que un eje distinto no rompa la búsqueda en silencio.
20. Como estudiante, quiero que la mutación use σ 20 y probabilidad por gen 0,3, para que la población no colapse a un solo candidato como en los 20 estudios originales.
21. Como estudiante, quiero que el generador aleatorio se siembre antes de crear la población, para que dos ejecuciones con la misma semilla partan de la misma población.
22. Como estudiante, quiero que el estado aleatorio de la evolución se guarde y restaure alrededor de cada entrenamiento, para que la trayectoria del GA no dependa de cuántos candidatos vinieron de caché.
23. Como estudiante, quiero que cada candidato evaluado se añada a un CSV en disco nada más terminar, para que una caída a las 15 horas no pierda el trabajo hecho.
24. Como estudiante, quiero que relanzar el mismo número de experimento reutilice ese CSV y continúe la búsqueda donde se quedó, sin reentrenar candidatos ya evaluados.
25. Como estudiante, quiero que la caché se niegue a cargar si el contrato de fitness ha cambiado (semilla de candidato, épocas, datos, versiones), para no mezclar fitness incomparables.
26. Como estudiante, quiero un registro por generación (media, desviación, mínimo, máximo, mejor, candidatos únicos, evaluaciones), para dibujar la evolución como en el TFG original.
27. Como estudiante, quiero un registro por candidato único (bandas, nm, generación en que apareció, métricas, segundos), para inspeccionar qué exploró la búsqueda.
28. Como estudiante, quiero ver cada tripleta traducida a longitudes de onda en nm en todas las tablas, resúmenes y figuras, para poder leer los resultados sin consultar el eje espectral.
29. Como estudiante, quiero poder evaluar una tripleta suelta desde la línea de comandos, para verificar el candidato ganador o probar una hipótesis sin lanzar la búsqueda.
30. Como estudiante, quiero poder correr la búsqueda en modo "solo caché" que falle ante cualquier candidato no evaluado, para demostrar que el GA portado recorre exactamente la trayectoria del 10 de septiembre.
31. Como estudiante, quiero que el entrenamiento final use todo el train, validación incluida, y no tenga ningún cargador de evaluación, para que el test no pueda influir en ninguna decisión.
32. Como estudiante, quiero que el modelo final se guarde antes de abrir el test, para que el orden de operaciones sea la garantía y no una promesa.
33. Como estudiante, quiero que el test se infiera una sola vez y con la semilla 2718 reproduzca el weighted F1 0,7221 y la matriz de confusión del 10 de septiembre.
34. Como estudiante, quiero predicciones por recorte con su captura, para poder remuestrear por captura sin volver a ejecutar el modelo.
35. Como estudiante, quiero un intervalo bootstrap del 95 % remuestreando capturas enteras, para citar el resultado como exige la profesora, con su anchura y no solo el punto.
36. Como estudiante, quiero la matriz de confusión y las curvas de entrenamiento como PNG, para pegarlas en la memoria y en el informe.
37. Como estudiante, quiero que las salidas sigan el patrón `exp_NN_<cosa>_R_G_B` y que la numeración continúe en 21, para que los 20 experimentos originales queden como registro sin mezclarse.
38. Como estudiante, quiero que las dependencias estén fijadas a las versiones exactas que produjeron el resultado y que torch se instale con CUDA desde el índice correcto, para que la verificación bit a bit sea posible.
39. Como estudiante, quiero unos pocos tests que vigilen solo lo que hace fiable el resultado, para poder cambiar código sin miedo a reintroducir una fuga.
40. Como estudiante, quiero un README que explique el protocolo en llano y cómo lanzar, reanudar y verificar, para poder contárselo a la profesora y al tribunal.
41. Como estudiante, quiero poder comprobar una vez los checksums de los 1.124 NPZ y ver una rejilla de recortes por clase, para asegurarme de que los datos enlazados son los correctos.
42. Como agente, quiero que `./init.sh` siga siendo la puerta de verificación tras el cambio (importa los módulos nuevos, comprueba el manifiesto de recortes y la partición, corre los tests), para que el flujo de trabajo de `CLAUDE.md` no se rompa.
43. Como agente, quiero que los CSV de referencia de `tests/data/` estén versionados, para que el test de reproducción de la partición funcione en un clon limpio.

## Implementation Decisions

**Punto de partida.** Rama `feature/protocolo-80-20` creada de nuevo desde `feature/experiments` en `b7f1fc9` (el harness de agentes ya está ahí). Existió una rama con ese nombre anoche; otra sesión commiteó en ella los borrados de ficheros obsoletos (`f1e4759`, `3a19e48`), la abandonó y la borró. Esos dos commits cuelgan en el reflog y no hacen falta: los borrados se repiten en la rama nueva.

**Forma del proyecto.** Se conserva la del TFG original: un módulo de configuración con todas las constantes; scripts de nivel raíz `split_dataset`, `ga`, `train_final`, `bootstrap`; paquete `cnn` con `data_setup` (datos), `engine` (bucles de entrenamiento) y `model` (red, métricas, evaluación de candidato, entrenamiento final, predicción); `run.sh`; `out/` con logs, tablas, figuras y modelos. Se eliminan los ficheros del original que ya no tienen sentido (entrenamiento suelto, resumen de arquitectura, utilidades de curso, manifiestos por higo, `run.log`). `cnn/engine2.py` ya fue borrado en `7357da7`.

**Harness.** `init.sh` se actualiza en la misma feature: importa `ga`, `cnn.data_setup`, `cnn.engine`, `cnn.model`; comprueba `data/cropped_hypercubes.csv`, `data/spectral_axes.csv` y `data/evaluation_partitions.csv` (avisa, no falla, si `data/` no existe); corre pytest. `CLAUDE.md` se retoca en la sección Layout e invariantes para nombrar los scripts nuevos y las constantes en `config.py`. `.gitignore`: `data/` pasa a `/data/` para que `tests/data/` quede versionado; se añade `out/models/`; se deja de ignorar `.python-version`.

**Datos de entrada.** Los CroppedHypercubes en NPZ de fig-aflatoxin (reflectancia float32 en [0,1] con fondo a cero y máscara booleana), enlazados con un symlink en `data/`, junto con copias del manifiesto de recortes y del SpectralAxis. El manifiesto conserva el `spectral_axis_id` por fila. Los hipercubos se cargan en RAM en el orden del manifiesto, filtrando por identidad; nunca ordenados ni iterados desde un conjunto. Comprobaciones baratas en cada carga (claves del NPZ, forma, tipo, linaje con la partición); la verificación de checksums pasa a `check_data.py`, que se ejecuta una vez. Las carpetas `data/interim` y `data/processed` del protocolo antiguo no se tocan.

**Partición.** Función pura: deduplica a capturas con su clase, ordena las identidades, hace un `train_test_split` estratificado con la semilla, y otro sobre el train con la misma semilla; comprueba al menos una captura por clase a cada lado; cada recorte hereda la partición de su captura. Manifiesto con `partition` (train/test) y `selection_partition` (train/validation/vacío). El lector revalida fronteras y exige los tres grupos.

**Traducción de bandas a longitud de onda.** Una única función que, dado el SpectralAxis, convierte tres índices distintos y en rango a nm. Es la única fuente de nm para todas las salidas.

**Evaluación de candidato.** Semilla derivada como `int(sha256(json({candidate, input_checksums, seed, seed_policy}))[:8], 16)`, con la política `sha256-candidate-split-v1` y los `input_checksums` calculados igual que en fig-aflatoxin (identidad de arrays de train y validación, filas de partición de train, eje espectral). Con ello la semilla del ganador es 3104252108. Se siembran `random`, NumPy, torch y CUDA, con algoritmos deterministas y un generador propio para el cargador de train; el cargador de validación no lleva generador (se copia tal cual). Cargadores antes que el modelo. 50 épocas, batch 32, Adam 0,001, 64×128, 8 workers persistentes, AMP en CUDA, sin early stopping. Métricas con etiquetas explícitas 0..3 y `zero_division=0`.

**Búsqueda genética.** DEAP `eaSimple` con individuos lista sin `creator`, torneo de 3, cruce blend α 0,5 con truncamiento entero y reparación, mutación gaussiana μ 0 σ 20 probabilidad 0,3 con reparación, población 20, 25 generaciones, cxpb 0,8, mutpb 0,15, hall of fame de uno que solo observa. Clave de orden `(weighted_f1, -ordinal_mae, -índices)`. Semilla de estudio 23, semilla de candidato 1729. Overrides de smoke por línea de comandos (`--population`, `--generations`, `--epochs`), no por variables de entorno.

**Caché de candidatos.** Un CSV por experimento (bandas, semilla de candidato, métricas, generación, segundos, nm), escrito con `flush` y `fsync` tras cada evaluación, y un JSON compañero con el contrato de fitness (semilla de candidato, hiperparámetros de entrenamiento, tipo de dispositivo, semilla y proporciones de partición, número de bandas, identidad de datos, versiones) y los parámetros del GA. Al cargar: aborta si el contrato de fitness difiere; comprueba la semilla de cada fila; avisa si solo cambian los parámetros del GA. La reanudación es un replay: la misma semilla de estudio regenera la misma trayectoria y los candidatos ya evaluados son hits sin coste. No se serializa el estado de DEAP. Modo "solo caché" para verificación.

**Modelo final.** Carga solo train (28 capturas), ajusta normalización sobre ellos, siembra con 2718, entrena 50 épocas llamando solo al paso de entrenamiento, guarda el estado en `out/models/`, y solo entonces carga el test y hace una inferencia ordenada. Salidas: métricas en JSON (con bandas en índices y nm, normalización congelada, semilla), matriz de confusión CSV y PNG, predicciones por recorte con captura, historial de entrenamiento CSV y PNG.

**Bootstrap.** 5.000 remuestreos de capturas enteras con `default_rng(1729)`, intervalo percentil 2,5/97,5 del weighted F1, más weighted F1 y accuracy por captura.

**Entorno.** Python 3.13; dependencias fijadas a deap 1.4.4, numpy 2.5.1, scikit-learn 1.9.0, torch 2.13.0, torchvision 0.28.0, pandas 2.3.3, matplotlib 3.11.1, seaborn 0.13.2, pytest 9.1.1; índice uv explícito de PyTorch cu126. El `.venv` actual tiene torch 2.7.1 por el `uv.lock` viejo y hay que resincronizarlo. GPU 1 por defecto en `run.sh` (la 0 está ocupada).

**Numeración.** Los experimentos nuevos empiezan en 21; los 20 originales quedan intactos y el README marca la frontera de protocolo. Los smoke usan 90–99 y se borran.

**Relación con `repo-health`.** Esta feature reescribe `ga.py`, así que deja sin objeto `repo-health/02` (importar `engine2`), `03` (overrides por entorno: aquí son flags) y `04` (tests sobre los operadores antiguos). `05` (scripts de análisis para los 20 experimentos) sigue siendo válido e independiente. `06` (README) queda absorbido por la reescritura del README. La decisión de cerrarlos como `wontfix` o `done` es de Pablo al triar.

## Testing Decisions

Un buen test aquí comprueba el comportamiento observable del protocolo (qué recortes ve cada paso, qué valores salen, qué se rechaza) y no cómo está escrito. Nada de tests sobre nombres internos ni sobre el formato exacto de los mensajes de log.

Dos costuras, ambas heredadas de fig-aflatoxin, más una prueba de sistema (confirmado con Pablo el 15 Sep):

1. **La búsqueda con evaluador inyectado.** `run_search` recibe una función de evaluación y una caché; con un evaluador falso (sin torch) se prueban operadores, tres bandas distintas, siembra reproducible, hits de caché, reanudación desde CSV con cero llamadas, aislamiento del estado aleatorio (un evaluador que resiembra `random` no cambia la secuencia de candidatos) y rechazo de una caché con otro contrato. Prior art: los tests de `run_ga_study` con `candidate_evaluator` inyectado de fig-aflatoxin.
2. **La partición y la carga como funciones puras.** `assign_partitions` sobre datos sintéticos (agrupación, estratificación, fallo por clase sin capturas) y sobre el manifiesto real (reproduce la partición de referencia del 10 de septiembre, guardada en `tests/data/`); `load_partitions` rechaza fugas y manifiestos incompletos; `load_hypercubes` no devuelve identidades de test y rechaza linaje discordante, con NPZ sintéticos diminutos. Prior art: los tests de particionado y de evaluación de candidato de fig-aflatoxin.

Como unidades puras: normalización (solo primer plano de train, fondo a cero, varianza cero falla) y preparación de entrada (máscara alineada tras volteos y rotación, fondo a cero). Prior art: los tests de unidades de evaluación de bandas de fig-aflatoxin.

**Prueba de sistema, en GPU, contra la corrida real:** la partición generada coincide con la de referencia; el candidato 366/262/225 da la semilla 3104252108, la normalización y el weighted F1 0,8700979843225085 registrados; el modelo final da 0,722142952443074 y la misma matriz de confusión y predicciones; el bootstrap da [0,645, 0,848]; y la búsqueda en modo "solo caché", precargada con los 383 candidatos reales, termina con el mismo ganador y la misma historia por generación. Tolerancia: exacta con las mismas versiones, la misma GPU (A100), 8 workers persistentes y el orden del manifiesto; cualquier desviación de esas condiciones explica diferencias de ±0,01–0,02 sin que sea un fallo del porte.

## Out of Scope

- Portar el catálogo de ENVI, la segmentación con Grounded-SAM-2 y el recorte con corrección radiométrica. Los recortes ya existen y se enlazan.
- MLflow, huellas SHA-256 de artefactos, publicación inmutable, configuración pydantic/YAML, DAG de etapas, contenedores, mise, lefthook.
- El vigilante de búsqueda por artefactos; basta `tail -f` del log y el CSV de candidatos.
- Los scripts de los experimentos 4 a 13 de fig-aflatoxin y la inicialización contrastiva.
- Los documentos y ADRs de fig-aflatoxin; quedan allí como registro.
- Borrar nada de `fig-aflatoxin`, ni `data/interim` y `data/processed` de este repo (Pablo decide).
- Lanzar la corrida real de 22 h: la lanza Pablo con el comando que se le dé.
- Commits: solo cuando Pablo lo pida.

## Further Notes

- Mantener la derivación exacta de la semilla por candidato (unas 40 líneas) es lo que permite verificar el porte del GA entero sin gastar 22 horas de GPU y reutilizar los 383 candidatos ya evaluados en una búsqueda nueva con la misma semilla de estudio.
- El resultado de validación (0,870) y el de test (0,722) no son comparables; la caída es la anchura esperada con 6 capturas de validación y 8 de test, no un sesgo. Citar siempre el test con su intervalo.
- Los 20 experimentos originales no son comparables con los nuevos: medían accuracy, repartían por higo y dejaban que el test eligiera las bandas.
- Estado al publicar (15 Sep, 12:40): en el árbol de trabajo ya están el symlink `data/cropped_hypercubes`, `data/cropped_hypercubes.csv`, `data/spectral_axes.csv`, `tests/data/*.csv` (ignorados por el `.gitignore` actual) y `out/models/`. Nada más está hecho.
