#============================================================================

Tutorial_Fundamentos_ML.jl — FUNDAMENTOS DE ML, PASO A PASO CON JULIA

Este fichero es el complemento práctico del tema de teoría "Fundamentos de ML".
Sigue el mismo orden que el tema y convierte cada idea de ese tema en algo que 
puedes explorar, ejecutar, y observar. Necesita ML.jl en la misma carpeta.

CÓMO USARLO EN VS CODE (extensión Julia):
  - Cada celda empieza con una línea que comienza por `##`.
  - Alt+Enter     → ejecuta la celda completa donde está el cursor.
  - Shift+Enter   → ejecuta la línea (o selección) y avanza.
  - Ejecuta las celdas EN ORDEN: algunas reutilizan variables anteriores.
  - Lee el comentario ANTES de ejecutar cada celda e intenta prever la salida.
    Luego compara con la salida real.
  - Las líneas marcadas con `# EJERCICIO` son preguntas para pensar o para 
    ampliar el código.

CORRESPONDENCIA ENTRE LA NOTACIÓN DEL TEMA Y EL CÓDIGO

  Teoría                          Código
  ──────────────────────────────  ───────────────────────────────────────────
  dato (x, y)                     una fila de X (matriz m×n) y su y[i] asociado
  m (tamaño de la muestra)        m, n = size(X)
  n (nº de características)       m, n = size(X)
  etiquetas, espacio Y            y, a_pm1, codificar_etiquetas
  hipótesis h_W(x)                predecir(X, W)
  clasificador sign∘h_W           clasificar(X, W)
  función de pérdida L(y, h)      perdida_cuadratica, perdida_absoluta, perdida_01, ...
  pérdida media (riesgo)          riesgo_empirico(perdida, y, predicciones)
  matriz de confusión             matriz_confusion, metricas_por_clase
  búsqueda de la mejor h          regresion_lineal, bgd_mse, regresion_logistica_gd, ...

  Nota: en el tema se escribe h_W(x) = W·x. En ML.jl el primer peso W[1] es el
  término independiente (bias), equivalente a añadir una característica
  constante igual a 1.

ÍNDICE
  1. Qué es aprender: de ejemplos a una regla general
  2. Tipos de problemas: regresión, clasificación, supervisado y no supervisado
  3. Los tres componentes de un problema de ML
  4. Datos: ejemplos, características y etiquetas
  5. Modelos: hipótesis y espacio de hipótesis
  6. Función de pérdida: medir la calidad de una hipótesis
  7. Cierre: un problema completo de principio a fin
============================================================================#


## ═══ 0. PREPARACIÓN ═══════════════════════════════════════════════════════

include("ML.jl")
Random.seed!(2026)

#============================================================================
1. QUÉ ES APRENDER: DE EJEMPLOS A UNA REGLA GENERAL
============================================================================#

## ─── 1.1 Aprender a partir de la experiencia ──────────────────────────────

# Idea del tema: aprender = obtener un enunciado general a partir de casos
# particulares (INDUCCIÓN). No programamos la regla: damos EJEMPLOS y dejamos
# que el algoritmo la descubra.
#
# Experimento: ¿cómo se pasa de grados Celsius a Fahrenheit? Fingimos que no
# sabemos la fórmula y solo tenemos 15 mediciones con un termómetro imperfecto.

celsius = rand(-10.0:0.5:40.0, 15)
fahrenheit = 32 .+ 1.8 .* celsius .+ 0.5 .* randn(15)

# El algoritmo de regresión lineal recibe los ejemplos y devuelve los pesos 
# W = [w₀, w₁]. En este caso, la regla aprendida es F ≈ w₀ + w₁·C. Observa que 
# la regla aprendida coincide en formato con la regla real, ambas son lineales.

W_cf = regresion_lineal(celsius, fahrenheit)
println("Regla aprendida: F ≈ ", round(W_cf[1], digits=2), " + ", round(W_cf[2], digits=3), " · C")
println("Regla real:      F  = 32 + 1.8 · C")

# La regla aprendida se aplica a datos que el algoritmo NO ha visto:

println("100 °C → ", round(predecir([100.0], W_cf)[1], digits=1), " °F  (real: 212)")
println("-40 °C → ", round(predecir([-40.0], W_cf)[1], digits=1), " °F  (real: -40)")

display(graficar_ajuste(celsius, fahrenheit, t -> predecir(t, W_cf);
        nombres=["Regla aprendida"], titulo="Aprender una regla a partir de 15 ejemplos"))

## ─── 1.2 Memorizar NO es aprender ─────────────────────────────────────────

# Un "modelo" que solo memoriza los ejemplos tiene error 0 sobre ellos...

tabla_memoria = Dict(zip(celsius, fahrenheit))
println("Ejemplo conocido:   ", get(tabla_memoria, celsius[1], missing))

# ...pero no sabe decir nada de un dato nuevo:

println("Ejemplo desconocido: ", get(tabla_memoria, 100.0, missing))

# Aprender es GENERALIZAR: la regla lineal sí responde para cualquier valor.

## ─── 1.3 La inducción es incompleta: siempre hay incertidumbre ────────────

# Como nunca vemos TODOS los casos, la regla aprendida depende de la muestra.
# Repetimos el experimento 200 veces (cada una con 15 ejemplos distintos), y 
# volvemos a calcular la regla aprendida.
# (Los coeficientes de la fórmula real son 32 / 1.8).

function estimar_pendiente(m)
    c = rand(-10.0:0.5:40.0, m)
    f = 32 .+ 1.8 .* c .+ 0.5 .* randn(m)
    return regresion_lineal(c, f)[2]
end

pendientes = [estimar_pendiente(15) for _ in 1:200]
display(histogram(pendientes, bins=25, legend=false, xlabel="Pendiente aprendida",
                  ylabel="Nº de muestras", title="200 muestras de 15 ejemplos cada una"))
println("Media = ", round(mean(pendientes), digits=4), "   desviación = ", round(std(pendientes), digits=4))

# Más ejemplos → menos incertidumbre (la regla aprendida se parece más a la real)
# Comparamos las desviaciones de la pendiente aprendida para distintos tamaños 
# de muestra m (m = 5, 15, 50, 500).
display(DataFrame(
    m = [5, 15, 50, 500],
    desviacion_pendiente = [round(std([estimar_pendiente(m) for _ in 1:200]), digits=5) for m in (5, 15, 50, 500)]))

# EJERCICIO 1.3: sube el ruido del termómetro de 0.5 a 5. ¿Cómo cambia la tabla?

#============================================================================
2. TIPOS DE PROBLEMAS
============================================================================#

## ─── 2.1 Regresión y clasificación sobre los MISMOS datos ─────────────────

# Regresión: predecir un valor REAL.  
# Clasificación: predecir una CLASE.
# Datos: superficie (m²) y precio (miles de €) de 120 viviendas.

# Creamos los datos sintéticos con una relación lineal + ruido. La semilla fija 
# la aleatoriedad para que los resultados sean reproducibles. Cambiar la semilla 
# genera otros datos.

Random.seed!(2)
superficie = rand(40.0:0.5:160.0, 120)
precio = 30 .+ 2.2 .* superficie .+ 25 .* randn(120)

# (a) Regresión: ¿cuánto vale una vivienda de 100 m²?

W_reg = regresion_lineal(superficie, precio)
println("Precio previsto para 100 m²: ", round(predecir([100.0], W_reg)[1], digits=1), " mil €")
display(graficar_ajuste(superficie, precio, t -> predecir(t, W_reg);
        nombres=["Regresión"], titulo="Regresión: predecir un valor real"))

# (b) Clasificación: la discretizamos con un umbral → "cara" (+1) o "no cara" (-1)

cara = [p > 200 ? 1 : -1 for p in precio]
println("Viviendas caras: ", count(==(1), cara), " de ", length(cara))

# Normalizamos (el descenso de gradiente lo necesita) y entrenamos un clasificador
# En este caso, normalizamos usando z-score (media 0, desviación 1) para que la 
# superficie tenga media 0 y desviación 1.

X_sup = reshape(superficie, :, 1)
X_sup_n, nz_sup = normalizar(X_sup, :z_score)

# La regresión logística devuelve un vector de pesos W y un historial de 
# pérdidas. La idea de usar regresión logística es que la salida del modelo es 
# un número real que se puede interpretar como la confianza de que la vivienda 
# sea cara.

W_cls, _ = regresion_logistica_gd(X_sup_n, cara; lr=0.5, max_iter=500)

# Ten en cuenta que, esta vez, la salida del modelo no es un valor real (precio) 
# sino una clase (-1 o +1). La regresión logística es una aproximación continua
# y derivable de la función de salto (la función signo) y, por tanto, adecuada
# para optimización por descenso de gradiente. 

# Y, de nuevo, podemos volver de clasificación a regresión: la CONFIANZA del 
# clasificador es un número real que podemos aproximar.

prob_cara(t) = probabilidad(normalizar(reshape(t, :, 1), nz_sup), W_cls)
display(graficar_ajuste(superficie, (cara .+ 1) ./ 2, prob_cara;
        nombres=["P(vivienda cara)"], titulo="Clasificación: de la clase a la probabilidad"))

# Observa que la frontera entre ambos mundos es difusa: se puede pasar de uno a
# otro con:
#      - umbrales (regresión → clasificación) o 
#      - confianzas (clasificación → regresión).

## ─── 2.2 Supervisado, no supervisado y semisupervisado ────────────────────

X_2d, y_2d = generar_clasificacion_2d(200; separacion=4.0, seed=3)

# SUPERVISADO: tenemos ejemplos ETIQUETADOS (x, y). El objetivo es generalizar
# la correspondencia entrada → salida a ejemplos no vistos.

display(graficar_datos(X_2d, y_2d; titulo="Supervisado: todos los datos con etiqueta"))

# NO SUPERVISADO: solo tenemos las características. No hay "respuesta correcta";
# se buscan patrones. ¿Ves grupos a simple vista? (K-medias lo formalizará.)

display(scatter(X_2d[:, 1], X_2d[:, 2], label="", color=:gray, alpha=0.7,
                xlabel="x₁", ylabel="x₂", title="No supervisado: sin etiquetas"))

# SEMISUPERVISADO: unos pocos datos etiquetados y muchos sin etiquetar.

p_semi = scatter(X_2d[:, 1], X_2d[:, 2], label="Sin etiqueta", color=:lightgray,
                 xlabel="x₁", ylabel="x₂", title="Semisupervisado: 8 etiquetados, 192 sin etiquetar")
idx_etiq = randperm(200)[1:8]
for c in (-1, 1)
    sel = idx_etiq[y_2d[idx_etiq] .== c]
    scatter!(p_semi, X_2d[sel, 1], X_2d[sel, 2], label="Clase $c", markersize=8,
             markerstrokecolor=:black)
end
display(p_semi)

# Otros dos tipos que se nombran en el tema y que no veremos aquí porque no 
# profundizaremos en ellos en el curso:
#   - RANKING: ordenar objetos por relevancia (p. ej. resultados de un buscador).
#   - REFUERZO: un agente aprende por ensayo y error en un entorno dinámico.

## ─── 2.3 Ejercicio del tema ───────────────────────────────────────────────

# EJERCICIO 1 (del tema). Clasifica cada problema según el TIPO DE TAREA
# (regresión, clasificación binaria, multiclase, ranking) y el TIPO DE APRENDIZAJE
# (supervisado, no supervisado, semisupervisado):
#   1. Predecir si una imagen contiene un gato o no.
#   2. Agrupar documentos similares sin etiquetar previamente.
#   3. Predecir el precio de una casa a partir de sus características.
#   4. Clasificar emails como spam / no-spam / importante.
#   5. Ordenar resultados de búsqueda por relevancia.
#   6. Detectar anomalías en un servidor sin datos de entrenamiento.
#   7. Predecir si un cliente comprará (pocos datos etiquetados y muchos sin etiquetar).
#   8. Estimar la temperatura máxima de mañana.
#
# Para cada uno, piensa qué serían X e y, y qué cambiaría en el código.

#============================================================================
3. LOS TRES COMPONENTES DE UN PROBLEMA DE ML
============================================================================#

## ---- 3.1 Datos + Modelo (Hipótesis) + Función de pérdida (Error) -----------
#=
Un problema de ML necesita ELEGIR tres componentes principales. Todo lo demás 
son detalles:

  - DATOS:               los ejemplos (la "experiencia")
  - MODELO / HIPÓTESIS:  el conjunto de funciones h entre las que buscamos 
  - PÉRDIDA:             cómo medimos si una h es buena o mala

Estas componentes se combinan en un objetivo:

"Encontrar una hipótesis que produzca la mejor calidad (menor pérdida) posible 
 sobre los datos conocidos y que generalice correctamente a los desconocidos."

Las herramientas/funciones que se han definido en ML.jl se pueden agrupar en 
estas tres componentes como se muestra en la tabla siguiente:

Componente          herramienta              
─────────────────────────────────────────────────────
Datos               preparar_datos / train_test_split
Datos               a_pm1, codificar_etiquetas
Modelo (espacio H)  predecir (lineal), clasificar (sign∘h)
Modelo (espacio H)  ajustar_polinomio (H más grande)
Pérdida             perdida_cuadratica, perdida_absoluta
Búsqueda de h       regresion_lineal, bgd_mse, regresion_logistica_gd
Evaluación          matriz_confusion, roc_auc, mse

Observa que la tabla anterior no es exhaustiva: hay muchas más funciones en 
ML.jl, pero estas son las más representativas de cada componente. Además, se
especifican algunas funciones que pueden hacer labores de comunicación entre 
componentes, como `preparar_datos`, que prepara los datos para el modelo y la 
pérdida, o los apartados de `Búsqueda de h`o `Evaluación`, que realizan tareas 
de búsqueda y evaluación de los modelos por medio de la Pérdida.

En las secciones siguientes recorremos los tres componentes uno a uno.
=#

#============================================================================
4. DATOS
============================================================================#

## ─── 4.1 Datos = ejemplos (x, y) ──────────────────────────────────────────

# Cada fila de un DataFrame es un dato. Separamos CARACTERÍSTICAS (entrada) de
# ETIQUETA (salida) y obtenemos una matriz de datos, X (m individuos × n 
# características), y un vector de etiquetas, y (m individuos).

Random.seed!(4)
casas = DataFrame(
    superficie   = rand(40:150, 8),
    habitaciones = rand(1:5, 8),
    antiguedad   = rand(0:50, 8) )

    # Rellenamos la etiqueta con una relación lineal + ruido. Estos datos sintéticos 
# pueden simular un problema de regresión: predecir el precio de una vivienda a 
# partir de sus características.

casas.precio = round.(30 .+ 2.0 .* casas.superficie .+ 8 .* casas.habitaciones .- 0.8 .* casas.antiguedad .+ 10 .* randn(8))
display(casas)

# Usamos matrices y vectores porque son mucho más eficientes que DataFrames para 
# cálculos numéricos. La conversión es sencilla:

X_casas = Matrix{Float64}(select(casas, Not(:precio)))     # características
y_casas = casas.precio                                     # etiqueta

# m ejemplos, con n características cada uno. La relación m/n es importante: si 
# hay muchas más características que ejemplos, el modelo puede sobreajustarse. 

m, n = size(X_casas)
println("m = $m ejemplos, n = $n características, m/n = ", m / n)

## ─── 4.2 El tamaño de la muestra frente al número de características ──────

# Regla del tema: conviene que m/n ≫ 1. Lo comprobamos. Generamos datos con
# n = 20 características (solo 5 influyen en la etiqueta) y variamos m.

# La función `experimento_mn` genera un problema sintético con m ejemplos y n 
# características, entrena un modelo de regresión lineal y devuelve el error 
# cuadrático medio (MSE) en los datos de entrenamiento y en datos nuevos.

function experimento_mn(m; n=20, informativas=5, ruido=1.0, n_test=1000)
    w = zeros(n)
    w[1:informativas] .= 1.0    # solo las informativas influyen en la etiqueta
    # Datos para entrenar (calcular) la regresión 
    X = randn(m, n);        
    y = X * w .+ ruido .* randn(m)
    # X_t son datos nuevos para evaluar la regresión (no vistos en el entrenamiento)
    Xt = randn(n_test, n);  
    yt = Xt * w .+ ruido .* randn(n_test)
    W = regresion_lineal(X, y)
    return mse(y, predecir(X, W)), mse(yt, predecir(Xt, W))
end

# Vamos a probar distintos tamaños de muestra m y ver cómo afecta al error de 
# entrenamiento (train) y al error en datos nuevos (test). En todos ellos se 
# considera que n = 20 y que solo 5 características son informativas.

ms = [10, 15, 20, 25, 40, 60, 100, 200, 500, 2000]
err_train = Float64[]
err_test = Float64[]
for m_i in ms
    r = [experimento_mn(m_i) for _ in 1:50]       # promediamos 20 repeticiones
    push!(err_train, mean(first.(r)))
    push!(err_test, mean(last.(r)))
end

display(DataFrame(m = ms, m_sobre_n = ms ./ 20,
                  error_train = round.(err_train, digits=3), 
                  error_test = round.(err_test, digits=3)))

p_mn = plot(ms ./ 20, err_test, marker=:circle, linewidth=2, xscale=:log10, yscale=:log10,
            label="Error en datos nuevos", xlabel="m / n", ylabel="MSE",
            title="El error en datos nuevos baja al aumentar m/n")
hline!(p_mn, [1.0], linestyle=:dash, color=:gray, label="Ruido irreducible (σ² = 1)")
display(p_mn)

# Interpretación: con pocos datos por característica el modelo "se inventa" la 
# regla (error de entrenamiento ≈ 0, pero error en datos nuevos enorme). Con m/n 
# grande, el error en datos nuevos se acerca al mínimo posible: el ruido de los 
# datos. 

## ─── 4.3 Más características NO siempre es mejor ──────────────────────────

# Fijamos m = 40 y añadimos características IRRELEVANTES (ruido puro) a un
# problema con 3 características útiles. ¿Mejora o empeora el modelo?

# La función `experimento_irrelevantes` genera un problema sintético con 3 + k 
# características, donde 3 son útiles y k son irrelevantes. Dependiendo del 
# parámetro λ, entrena un modelo de regresión lineal o de regresión ridge y 
# devuelve el error cuadrático medio (MSE) en datos nuevos.

# La regresion_ridge añade regularización L2 (λ > 0) para controlar la 
# complejidad del modelo. La forma en que funciona es que penaliza los coeficientes 
# grandes, lo que ayuda a evitar el sobreajuste cuando hay muchas características 
# irrelevantes.

function experimento_irrelevantes(k; m=40, n_test=2000, ruido=1.0, λ=0.0)
    w = vcat([2.0, -1.0, 1.0], zeros(k))
    X = randn(m, 3 + k);       y = X * w .+ ruido .* randn(m)
    Xt = randn(n_test, 3 + k); yt = Xt * w .+ ruido .* randn(n_test)
    W = λ == 0 ? regresion_lineal(X, y) : regresion_ridge(X, y, λ)
    return mse(yt, predecir(Xt, W))
end

# Con la función anterior lanzamos experimentos para distintos valores de k (nº 
# de características irrelevantes) y promediamos 30 repeticiones para cada k.

ks = 0:3:30
err_sin_reg = [mean([experimento_irrelevantes(k) for _ in 1:30]) for k in ks]
err_con_reg = [mean([experimento_irrelevantes(k; λ=10.0) for _ in 1:30]) for k in ks]

p_irr = plot(ks, err_sin_reg, marker=:circle, linewidth=2, label="Mínimos cuadrados",
             xlabel="Nº de características irrelevantes añadidas", ylabel="MSE en datos nuevos",
             title="Cada característica irrelevante añade ruido")
plot!(p_irr, ks, err_con_reg, marker=:square, linewidth=2, label="Ridge (λ = 10)")
hline!(p_irr, [1.0], linestyle=:dash, color=:gray, label="Mínimo posible")
display(p_irr)

# Interpretación: añadir características irrelevantes empeora el modelo, y lo 
# hace más cuanto menos datos hay.
# Controlar la complejidad (Ridge, o elegir mejor las características) ayuda,
# pero no elimina el problema, por lo que la mejor solución es tener 
# características relevantes.

# EJERCICIO 4.3: repite con m = 400. ¿Sigue importando tanto?

## ─── 4.4 Características no numéricas → numéricas ──────────────────────────

# La mayoría de los métodos de ML trabajan con vectores numéricos. Para texto 
# limitado o categorías hay procedimientos estándar de conversión; el más 
# habitual es el one-hot, que añade independencia entre las categorías.

ropa = DataFrame(talla = ["S", "M", "L", "M"], color = ["rojo", "azul", "rojo", "verde"],
                 precio = [10, 12, 14, 12])
display(ropa)
display(codificar_todas(ropa, :one_hot; excluir=[:precio]))

# Para datos como texto o imágenes se usan "embeddings" (representaciones
# numéricas aprendidas): veremos algo sobre esto en temas posteriores.

## ─── 4.5 Etiquetas: el espacio Y ──────────────────────────────────────────

# Regresión, Y = ℝ: la etiqueta ya es un número (precio, temperatura...).

println("Regresión → ", y_casas[1:3])

# Clasificación BINARIA: Y = {-1, 1}, {0, 1} o {Sí, No}. La mayoría de los 
# modelos usan -1/+1.

println("De {0,1}:       ", a_pm1([0, 1, 1, 0]))
println("De {Sí, No}:    ", a_pm1(["Sí", "No", "Sí"]; positivo="Sí"))

# Clasificación MULTICLASE: Y = {c₁,…,c_k}. Se codifican como enteros 1..k.

codigos, clases = codificar_etiquetas(["perro", "gato", "pájaro", "gato"])
println("Multiclase → ", codigos, " con clases ", clases)

# Etiquetas ORDINALES: hay orden entre las clases. ¡Cuidado!: el orden
# alfabético NO coincide con el orden real.

codigos_ord, clases_ord = codificar_etiquetas(["bajo", "medio", "alto", "bajo"])
println("Orden alfabético (incorrecto): ", clases_ord, " → ", codigos_ord)
orden_real = Dict("bajo" => 1, "medio" => 2, "alto" => 3)
println("Orden real:                    ", [orden_real[v] for v in ["bajo", "medio", "alto", "bajo"]])

# Clasificación MULTIETIQUETA: cada dato puede tener VARIAS clases a la vez.
# Se representa con una columna 0/1 por etiqueta (puede haber varias activas a 
# la vez).

display(DataFrame(película = ["A", "B", "C"], acción = [1, 0, 1], comedia = [0, 1, 1], drama = [1, 0, 0]))

## ─── 4.6 ¿Característica o etiqueta? Es una decisión de diseño ────────────

# La misma propiedad puede ser entrada en un problema y salida en otro.

# Datos ambientales: (presión, temperatura, viento) ¿qué podemos predecir a 
# partir de qué?

Random.seed!(5)
n_dias = 300
presion = 1013 .+ 8 .* randn(n_dias)
temperatura = 20 .- 0.4 .* (presion .- 1013) .+ 2 .* randn(n_dias)
viento = 12 .- 0.5 .* (presion .- 1013) .+ 1.5 .* randn(n_dias)
ambiente = DataFrame(presion = presion, temperatura = temperatura, viento = viento)

# Vamos a analizar tres problemas distintos, cada uno con un objetivo distinto. 
# Para cada uno, preparamos los datos, entrenamos un modelo de regresión lineal 
# y evaluamos su rendimiento en datos nuevos (R² sobre el test).

resultado_objetivos = DataFrame(objetivo = Symbol[], R2_test = Float64[])
for obj in (:temperatura, :viento, :presion)
    dd = preparar_datos(ambiente, obj; normalizacion=:z_score)
    W_obj = regresion_lineal(dd.Xtrain, dd.ytrain)
    push!(resultado_objetivos, (obj, r2_score(dd.ytest, predecir(dd.Xtest, W_obj))))
end
display(resultado_objetivos)

# Una sola función (`preparar_datos`) cambia el problema entero con la elección
# del símbolo del objetivo.

# EJERCICIO 2 (del tema). Para cada dominio propón 5 características adecuadas
# y 2-3 inadecuadas: (1) predicción de diabetes, (2) recomendación de películas,
# (3) detección de fraude bancario, (4) clasificación de noticias por tema.

## ─── 4.7 Aproximación probabilística: datos i.i.d. ────────────────────────

# Supuesto de modelización: los datos son muestras independientes de la misma
#                           distribución p(x) (i.i.d.). 

# Principio de máxima verosimilitud: elegimos la distribución que más 
#                                    probablemente genera los datos observados.

# Para una gaussiana, los estimadores de máxima verosimilitud son la media
# muestral y la varianza muestral (con 1/m). La dificultad es que, en la 
# práctica, no sabemos cuál es la distribución real de los datos.

μ_real, σ_real = 5.0, 2.0

tabla_mv = DataFrame(m = Int[], media = Float64[], varianza = Float64[])
for m_i in (10, 100, 1_000, 100_000)
    x = μ_real .+ σ_real .* randn(m_i)
    μ_est = sum(x) / m_i
    var_est = sum((x .- μ_est) .^ 2) / m_i
    push!(tabla_mv, (m_i, μ_est, var_est))
end
display(tabla_mv)
println("Valores reales: media = ", μ_real, "  varianza = ", σ_real^2)

# La distribución estimada frente a los datos:
x_mv = μ_real .+ σ_real .* randn(1000)
μ_est, σ_est = mean(x_mv), std(x_mv, corrected=false)
pdf_gauss(t) = exp(-(t - μ_est)^2 / (2σ_est^2)) / (σ_est * sqrt(2π))

p_mv = histogram(x_mv, normalize=:pdf, bins=30, alpha=0.5, label="Datos (m=1000)",
                 xlabel="x", ylabel="Densidad", title="Gaussiana de máxima verosimilitud")
plot!(p_mv, range(minimum(x_mv), maximum(x_mv), length=200), pdf_gauss, linewidth=3, label="p(x) estimada")
display(p_mv)

## ─── 4.8 Cuando NO se cumple i.i.d. ───────────────────────────────────────

# i.i.d. es una SUPOSICIÓN de modelización, no se puede comprobar de verdad.
# Veamos qué pasa cuando es claramente falsa: tomaremos puntos de un camino 
# aleatorio (donde cada valor depende claramente del anterior). Para ver que 
# no se cumple i.i.d., observamos cómo se comportan las medias acumuladas.
# La media acumulada de datos i.i.d. converge; la de un paseo aleatorio no.

n_pasos = 2000
media_acumulada(x) = cumsum(x) ./ (1:length(x))

p_iid = plot(title="i.i.d.: las medias convergen", legend=false, xlabel="m")
p_rw = plot(title="Camino aleatorio: NO convergen", legend=false, xlabel="m")
for _ in 1:5
    plot!(p_iid, media_acumulada(randn(n_pasos)))
    plot!(p_rw, media_acumulada(cumsum(randn(n_pasos))))
end
display(plot(p_iid, p_rw, layout=(1, 2), size=(900, 350)))

# EJERCICIO 3 (del tema). ¿Se puede asumir i.i.d. en cada caso? Justifícalo.
#   · Lanzamientos de una moneda.
#   · Temperaturas diarias en una ciudad.
#   · Precios de acciones en la bolsa.
#   · Notas de exámenes de 100 estudiantes.
#   · Imágenes de perros de una base de datos.
# Sugerencia: simula cada uno con código y comprueba si se comporta como el
# panel de la izquierda o el de la derecha.

#============================================================================
5. MODELOS: HIPÓTESIS Y ESPACIO DE HIPÓTESIS
============================================================================#

## ─── 5.1 El principio de ML: y ≈ h(x) ────────────────────────────────────

# Una HIPÓTESIS es una función h que a cada x le asigna una predicción de y.
# Observa que aquí estamos dando una visión desde el punto de vista del 
# aprendizaje supervisado: tenemos ejemplos (x, y) y queremos aprender una regla 
# general que nos permita predecir la salida a partir de x. La hipótesis es esa 
# regla. 

# Para facilitar la "búsqueda" de la mejor hipótesis, se suele parametrizar h 
# con un conjunto de parámetros θ, h_Θ(x). En el caso de regresión lineal, por 
# ejemplo, θ contiene los coeficientes en forma de vector/matriz para que 
# h_Θ(x) = θ₀ + θ₁·x₁ + θ₂·x₂ + ... + θ_n·x_n. En ML.jl, para los casos de 
# regresión usamos la notación W en lugar de Θ.

# Por ejemplo, con hipótesis lineales 1D h_W(x) = w₀ + w₁·x:

x_m, y_m = generar_datos(t -> 1.5t + 0.5, 40; ruido=0.3, seed=5)
W_m = regresion_lineal(x_m, y_m)

h(t) = W_m[1] + W_m[2] * t                  # la hipótesis, escrita a mano
println("Máxima diferencia entre h a mano y `predecir`: ", maximum(abs.(h.(x_m) .- predecir(x_m, W_m))))

# `predecir(X, W)` hace lo mismo, pero para todas las filas de X de una vez
# (producto matricial) y con n características.

# Una hipótesis también se llama PREDICTOR: dado x sin etiqueta, estima y.

println("h(0.8) = ", round(h(0.8), digits=3), "   (real sin ruido: ", 1.5 * 0.8 + 0.5, ")")

## ─── 5.2 Espacio de hipótesis: el conjunto donde buscamos ─────────────────

# Un modelo NO es una única función: es un CONJUNTO H de funciones posibles.
# Para H lineal, cada hipótesis queda determinada por sus pesos W. Dibujamos
# seis hipótesis elegidas al azar frente a la mejor de todas:

Random.seed!(6)
Ws_azar = [[randn(), 2 * randn()] for _ in 1:6]
modelos_azar = [t -> predecir(t, w) for w in Ws_azar]

display(graficar_ajuste(x_m, y_m, modelos_azar..., t -> predecir(t, W_m);
        nombres=vcat(["h$i" for i in 1:6], "Mejor h"), titulo="Seis hipótesis de H frente a la mejor"))

# Cada hipótesis tiene una "calidad" (pérdida media) sobre los datos:

display(DataFrame(hipótesis = vcat(["h$i" for i in 1:6], "Mejor h"),
                  w0 = round.(vcat([w[1] for w in Ws_azar], W_m[1]), digits=2),
                  w1 = round.(vcat([w[2] for w in Ws_azar], W_m[2]), digits=2),
                  MSE = round.(vcat([mse(y_m, predecir(x_m, w)) for w in Ws_azar], mse(y_m, predecir(x_m, W_m))), digits=3)))

## ─── 5.3 Aprender = buscar en H optimizando la pérdida ────────────────────

# Buscar entre TODAS las funciones posibles es inabordable. Incluso buscar entre
# todas las funciones de un espacio de funciones determinado. Sin embargo, si 
# usamos un espacio de funciones específico, como el de las lineales, convertimos 
# el problema en buscar entre un conjunto de vectores W (los pesos de cada 
# hipótesis). Y eso es un problema de OPTIMIZACIÓN para el que ya hemos 
# desarrollado métodos: si no pudiéramos sacar provecho de las propiedades de H, 
# podemos usar métodos agnósticos, como Templado Simulado, pero si H tiene 
# propiedades especiales, podemos usar métodos específicos, como el descenso de gradiente.

X_m = reshape(x_m, :, 1)
W_gd, hist_gd = bgd_mse(X_m, y_m; lr=0.1, max_iter=200)

println("Solución analítica: ", round.(W_m, digits=3))
println("Descenso de grad.:  ", round.(W_gd, digits=3))
display(graficar_superficie_perdida(X_m, y_m, hist_gd))

# Cada punto del plano representado (w₀, w₁) es UNA hipótesis; el color es su 
# pérdida (siguiendo la escala lateral). El descenso de gradiente empieza en un 
# punto aleatorio y va bajando hasta llegar a la zona más oscura (mínimo de 
# pérdida). La línea blanca es el camino del algoritmo hacia la mejor.

# En este caso, se sabe que la pérdida es convexa, por lo que el mínimo es único 
# y el descenso de gradiente lo encuentra siempre. En otros casos, la pérdida 
# puede tener varios mínimos locales y el descenso de gradiente puede quedarse 
# atrapado en uno de ellos. En esos casos, se pueden usar técnicas como el 
# reinicio aleatorio del descenso de gradiente, o métodos más sofisticados 
# (aunque generalmente más lentos y complejos) como el Templado Simulado, que 
# permiten escapar de los mínimos locales.

## ─── 5.4 Clasificadores y regiones de decisión ────────────────────────────

# En clasificación binaria con Y = {-1, 1}, el clasificador más habitual es 
#     h = sign ∘ h_W.
# En general, binaria o no, se puede dividir el espacio en REGIONES DE DECISIÓN: 
#     R(c) = {x : h(x) = c}.

# Para probarlo, generamos un problema de clasificación 2D y entrenamos un 
# clasificador lineal con regresión logística. La frontera de decisión es la 
# línea donde h(x) = 0 que, en el caso binario, separa las regiones de decisión 
# de cada clase.

X_cl, y_cl = generar_clasificacion_2d(200; separacion=3.0, seed=7)
W_cl, _ = regresion_logistica_gd(X_cl, y_cl; lr=0.1, max_iter=500)

println("Frontera: ", round(W_cl[1], digits=2), " + ", round(W_cl[2], digits=2),
        "·x₁ + ", round(W_cl[3], digits=2), "·x₂ = 0")
display(graficar_frontera(X_cl, y_cl, Xm -> clasificar(Xm, W_cl);
        titulo="Regiones de decisión de un clasificador lineal"))

# Para H lineal la frontera es una recta (un hiperplano si n > 2).

## ─── 5.5 Infraajuste: el espacio H es demasiado pequeño ───────────────────

# Cuando H es demasiado pequeño, ninguna hipótesis puede ajustarse bien a los 
# datos. No es un problema de búsqueda: por muchos datos que tengamos, ninguna 
# hipótesis de H es buena. Es un problema de capacidad de representación. 
# Esto se llama INFRAAJUSTE. 

# En el ejemplo siguiente, los datos siguen una relación cuadrática. Si usamos  
# un H que solo contiene rectas, no hay ninguna hipótesis suficientemente buena 
# dentro de H, por muchos datos que tengamos, y el error (incluso el de 
# entrenamiento) será elevado.

f_cuad(t) = 1 - 2t^2
x_q, y_q = generar_datos(f_cuad, 80; ruido=0.15, seed=8)
xq_tr, yq_tr, xq_va, yq_va = train_test_split(x_q, y_q, 0.7)

# Vamos a usar polinomios de grado 1 y 2 para ver cómo se ajustan a los datos. 
# Veremos que con polinomios de grado 1 el resultado es insuficiente 
# (infraajuste), mientras que el polinomio de grado 2 puede ajustarse 
# perfectamente a la relación cuadrática subyacente.

W_q1 = ajustar_polinomio(xq_tr, yq_tr, 1)     # H₁: polinomios de grado ≤ 1 (rectas)
W_q2 = ajustar_polinomio(xq_tr, yq_tr, 2)     # H₂: grado ≤ 2

display(graficar_ajuste(xq_tr, yq_tr,
        t -> predecir_polinomio(t, W_q1), t -> predecir_polinomio(t, W_q2), t -> f_cuad.(t);
        nombres=["H₁ (rectas)", "H₂ (parábolas)", "Función real"], titulo="Infraajuste"))
println("Error de la mejor recta:     ", round(mse(yq_va, predecir_polinomio(xq_va, W_q1)), digits=3))
println("Error de la mejor parábola:  ", round(mse(yq_va, predecir_polinomio(xq_va, W_q2)), digits=3))

## ─── 5.6 Sobreajuste: el espacio H es demasiado grande ────────────────────

# Tambien puede ocurrir lo contrario: H es demasiado grande y contiene hipótesis que
# que se ajustan demasiado a los datos de entrenamiento, incluyendo el ruido. 
# Esto se llama SOBREAJUSTE. 

# En el ejemplo siguiente, los datos siguen una relación cuadrática, pero 
# añadimos ruido. Si usamos un H que contiene polinomios de grado 15, el modelo 
# puede ajustarse perfectamente a los datos de entrenamiento, pero el error en 
# datos nuevos será elevado, porque la hipótesis ha "memorizado" el ruido y no 
# generaliza bien.

# Cuando los espacios son ANIDADOS: H₁ ⊂ H₂ ⊂ H₃ ⊂ ... (por ejemplo, el caso de 
# los polinomios, donde cada grado incluye a los anteriores). Cuanto mayor es H, 
# mejor encaja en los datos de entrenamiento (siempre ≤), pero puede perder la 
# capacidad de generalizar.

Ws_q = [ajustar_polinomio(xq_tr, yq_tr, d) for d in (1, 2, 15)]
display(graficar_ajuste(xq_tr, yq_tr, [t -> predecir_polinomio(t, W) for W in Ws_q]..., t -> f_cuad.(t);
        nombres=["grado 1", "grado 2", "grado 15", "Función real"], titulo="Infraajuste, buen ajuste y sobreajuste"))

# El error de entrenamiento y el de datos nuevos cuentan la historia completa:

cc_q = curva_complejidad(xq_tr, yq_tr, xq_va, yq_va; grados=1:15)
display(graficar_curva_complejidad(cc_q))
display(DataFrame(grado = cc_q.grados,
                  error_train = round.(cc_q.errores_train, digits=4),
                  error_nuevo = round.(cc_q.errores_val, digits=4)))

# Datos de entrenamiento: baja siempre (modelos más complejos tienen más 
# parámetros y, en consecuencia, mayor capacidad de ajuste). 
# Datos nuevos: baja y luego sube (un comportamiento en "U").
# Eso es el sobreajuste: la hipótesis "memoriza" ruido y no generaliza.
# La elección de H es un compromiso entre ambos extremos.

## ─── 5.7 Cuando H lineal no basta ─────────────────────────────────────────

# Si la relación entre características y etiqueta es muy poco lineal, ninguna
# hipótesis lineal va a funcionar. Ejemplo clásico: XOR.

X_xor, y_xor = generar_xor(400; seed=9)
Xx_tr, yx_tr, Xx_te, yx_te = train_test_split(X_xor, y_xor, 0.75)

W_xor, _ = regresion_logistica_gd(Xx_tr, yx_tr; lr=0.1, max_iter=500)
println("Accuracy en test con H lineal: ", round(accuracy(yx_te, clasificar(Xx_te, W_xor)), digits=3))
display(graficar_frontera(Xx_te, yx_te, Xm -> clasificar(Xm, W_xor); titulo="H lineal sobre XOR"))

# Solución 1: cambiar la representación (añadir nuevas características) y seguir 
# con H lineal.

aumentar(X) = hcat(X, X[:, 1] .* X[:, 2])
W_xor2, _ = regresion_logistica_gd(aumentar(Xx_tr), yx_tr; lr=0.5, max_iter=1000)
println("Accuracy con la característica x₁·x₂: ",
        round(accuracy(yx_te, clasificar(aumentar(Xx_te), W_xor2)), digits=3))

# Solución 2: cambiar H por un espacio no lineal (redes neuronales, más adelante).

# EJERCICIO 4 (del tema). Quieres predecir el precio de una casa.
#   1. Propón un espacio de hipótesis que produzca INFRAAJUSTE. ¿Por qué?
#   2. Propón uno que produzca SOBREAJUSTE. ¿Por qué?
#   3. Propón uno EQUILIBRADO y justifícalo.
# Pruébalo: genera datos de precios con `generar_datos` y usa `curva_complejidad`.

#============================================================================
6. FUNCIÓN DE PÉRDIDA
============================================================================#

## ─── 6.1 Una función de pérdida convierte una idea intuitiva de calidad en un 
#          problema de optimización objetivo ───────────────────────────────

# L(y, h(x)) mide cuánto se equivoca una hipótesis en UN ejemplo.
# Pero un error puntual no da toda la información. La pérdida media sobre los 
# datos (riesgo empírico) mide la calidad de h.

# De esta forma, formalizamos matemáticamente la idea intuitiva de "aprender":
# "Aprender" = encontrar la h de H con menor pérdida media.

println("Un ejemplo: y=3, h(x)=2.5 → ", perdida_cuadratica(3.0, 2.5))
println("Pérdida media de la mejor recta: ", round(riesgo_empirico(perdida_cuadratica, y_m, predecir(x_m, W_m)), digits=4))

# Para hacer un análisis de la complejidad de la hipótesis, fijaremos el término 
# independiente w₀ y variemos la pendiente w₁. La pérdida media forma una 
# parábola cuyo mínimo es la solución de regresion_lineal.

w1s = range(W_m[2] - 2, W_m[2] + 2, length=100)
riesgos = [riesgo_empirico(perdida_cuadratica, y_m, predecir(x_m, [W_m[1], w1])) for w1 in w1s]

p_par = plot(w1s, riesgos, linewidth=2, legend=false, xlabel="w₁", ylabel="Pérdida media",
             title="La pérdida cuadrática es convexa en W")
vline!(p_par, [W_m[2]], linestyle=:dash, color=:red)
display(p_par)

# Convexa y derivable → un solo mínimo y el gradiente nos lleva a él.

## ─── 6.2 Pérdida 0/1 frente a pérdida logística ───────────────────────────

# La pérdida 0/1 es la más interpretable (tasa de error), pero ni es convexa ni
# derivable. 

# Para verlo: giramos la dirección w = (cos θ, sin θ) de un clasificador
# lineal y calculamos ambas pérdidas medias.

X_g, y_g = generar_clasificacion_2d(200; separacion=3.0, seed=10)
θs = range(0, 2π, length=361)

r01 = [riesgo_empirico(perdida_01, y_g, X_g * [cos(t), sin(t)]) for t in θs]
rlog = [riesgo_empirico(perdida_logistica, y_g, X_g * [cos(t), sin(t)]) for t in θs]

display(plot(θs, [r01 rlog], label=["Pérdida 0/1" "Pérdida logística"], linewidth=2,
             xlabel="Ángulo θ de la dirección de decisión", ylabel="Pérdida media",
             title="Pérdida 0/1: escalones; logística: suave"))

# La 0/1 es plana a trozos: su gradiente es 0 casi siempre, no indica hacia dónde
# ir. La logística es suave y la derivada indica la dirección de descenso. El
# descenso de gradiente funciona. Por eso se ENTRENA con la logística, aunque se 
# EVALÚE con la 0/1 para interepretar qué está pasando (ver 6.5).

## ─── 6.3 Robustez: error cuadrático frente a error absoluto ───────────────

# El error absoluto es más robusto frente a valores atípicos, pero tiene el 
# mismo problema que la pérdida 0/1 que hemos comentado antes: no es derivable.

function ajustar_mae(x, y; lr=0.5, iter=3000)
    Xa = añadir_bias(reshape(x, :, 1))
    W = zeros(2)
    for k in 1:iter
        g = Xa' * sign.(Xa * W .- y) ./ length(y)
        W .-= (lr / sqrt(k)) .* g
    end
    return W
end

x_o, y_o = generar_datos(t -> 2t + 1, 40; ruido=0.2, seed=11)
y_atipico = copy(y_o)
y_atipico[end-2:end] .= 15.0   # Añadimos tres medidas erróneas, que son valores atípicos.

W_limpio = regresion_lineal(x_o, y_o)
W_mse = regresion_lineal(x_o, y_atipico)
W_mae = ajustar_mae(x_o, y_atipico)

display(DataFrame(ajuste = ["Real", "MSE sin atípicos", "MSE con atípicos", "MAE con atípicos"],
                  w0 = round.([1.0, W_limpio[1], W_mse[1], W_mae[1]], digits=3),
                  w1 = round.([2.0, W_limpio[2], W_mse[2], W_mae[2]], digits=3)))

display(graficar_ajuste(x_o, y_atipico,
        t -> predecir(t, W_mse), t -> predecir(t, W_mae);
        nombres=["MSE (cuadrática)", "MAE (absoluta)"], titulo="Efecto de los valores atípicos"))

# La recta MSE se deja arrastrar por los atípicos; la MAE casi no.
# Pero fíjate en el precio: MSE se resolvió con una línea; MAE necesita iterar.

## ─── 6.4 Por qué la pérdida cuadrática no vale para clasificar ────────────

# Ejemplo del tema: y = +1 y el clasificador devuelve h(x) = 1000 (¡acierta, con
# mucha confianza!). La pérdida cuadrática lo castiga con 998001; en cambio, si
# se equivoca con h(x) = -1, solo vale 4.

hs = [1000.0, 3.0, 0.5, -0.5, -1.0]
display(DataFrame(
    h_de_x = hs,
    clase_predicha = [h >= 0 ? 1 : -1 for h in hs],
    cuadrática = perdida_cuadratica.(1, hs),
    pérdida_01 = perdida_01.(1, hs),
    logística = round.(perdida_logistica.(1, hs), digits=4),
    bisagra = perdida_bisagra.(1, hs)))

# La cuadrática castiga la CONFIANZA, no el error. 0/1, logística y bisagra
# castigan solo cuando se falla (y más cuanto más seguro estaba el clasificador).

## ─── 6.5 Entrenar con una pérdida, evaluar con otra ───────────────────────

# Entrenamos con la pérdida logística (eficiente) y evaluamos con la 0/1
# (interpretable). Datos con solapamiento entre clases:

X_e, y_e = generar_clasificacion_2d(400; separacion=2.0, seed=12)
Xtr, ytr, Xte, yte = train_test_split(X_e, y_e, 0.7)

tabla_ent = DataFrame(iteraciones = Int[], pérdida_logística_train = Float64[],
                      error_01_train = Float64[], error_01_test = Float64[])
for it in (1, 5, 20, 100, 500)
    W_it, _ = regresion_logistica_gd(Xtr, ytr; lr=0.1, max_iter=it)
    push!(tabla_ent, (it,
        riesgo_empirico(perdida_logistica, ytr, predecir(Xtr, W_it)),
        riesgo_empirico(perdida_01, ytr, predecir(Xtr, W_it)),
        riesgo_empirico(perdida_01, yte, predecir(Xte, W_it))))
end
display(tabla_ent)

# La pérdida media 0/1 es exactamente la tasa de error = 1 - accuracy
W_f, _ = regresion_logistica_gd(Xtr, ytr; lr=0.1, max_iter=500)
println("Error 0/1 en test: ", round(riesgo_empirico(perdida_01, yte, predecir(Xte, W_f)), digits=4),
        "   1 - accuracy: ", round(1 - accuracy(yte, clasificar(Xte, W_f)), digits=4))

## ─── 6.6 Matriz de confusión: la pérdida 0/1 desglosada ───────────────────

# La matriz de confusión generaliza la pérdida 0/1: usa una pérdida por cada par
# de clases (p → q) y calcula su media. Por definición:
#     L^(p→q)((x,y), h) = 1 si y = p y h(x) = q;  0 en otro caso.

# La matriz de confusión se puede interpretar como una forma de visualizar la pérdida 0/1:
# cada celda (p, q) representa la proporción de ejemplos donde la clase real es p y
# la predicha es q.

pred_f = clasificar(Xte, W_f)
L_pq(y, y_hat, p, q) = (y == p && y_hat == q) ? 1.0 : 0.0

M_def = [mean(L_pq.(yte, pred_f, p, q)) for p in (-1, 1), q in (-1, 1)]
println("Por la definición del tema (filas: real -1, +1; columnas: predicho -1, +1):")
display(M_def)

# La función de ML.jl cuenta ejemplos; dividiendo por m ambas coinciden:
mc = matriz_confusion(yte, pred_f; clases=[-1, 1])
println(mc)
println("¿Coinciden? ", mc.matriz ./ length(yte) ≈ M_def)
display(graficar_matriz_confusion(mc))

## ─── 6.7 Métricas de rendimiento, a mano y con la librería ────────────────

# Con la clase positiva = +1 (filas = real, columnas = predicho):

tn, fp, fn, tp = mc.matriz[1, 1], mc.matriz[1, 2], mc.matriz[2, 1], mc.matriz[2, 2]

accuracy_h  = (tp + tn) / (tp + tn + fp + fn)   # proporción de aciertos
precision_h = tp / (tp + fp)                    # de lo que digo positivo, cuánto lo es
recall_h    = tp / (tp + fn)                    # TPR / sensibilidad: de los positivos, cuántos detecto
especif_h   = tn / (tn + fp)                    # de los negativos, cuántos detecto
fpr_h       = fp / (tn + fp)                    # = 1 - especificidad
f1_h        = 2 * precision_h * recall_h / (precision_h + recall_h)   # media armónica

display(DataFrame(métrica = ["Accuracy", "Precision", "Recall (TPR)", "Specificity", "FPR", "F1"],
                  valor = round.([accuracy_h, precision_h, recall_h, especif_h, fpr_h, f1_h], digits=4)))

# La librería da lo mismo para cada clase (fila de la clase +1):

display(metricas_por_clase(mc))

# EJERCICIO 6.7: comprueba que Specificity + FPR = 1 y que F1 está siempre
# entre Precision y Recall.

## ─── 6.8 Clases desbalanceadas: cuidado con la accuracy ───────────────────

# 95 negativos y 5 positivos. Un "modelo" que siempre responde -1:

y_desb = vcat(fill(-1, 95), fill(1, 5))
mc_desb = matriz_confusion(y_desb, fill(-1, 100); clases=[-1, 1])
println(mc_desb)
display(metricas_por_clase(mc_desb))

# Accuracy 95 %, pero recall 0: no detecta ningún positivo. Con clases
# desequilibradas hay que mirar Precision y Recall, no la accuracy.

# ¿Cuál priorizar? Depende del problema:
#   - Falsos negativos caros (cribado de una enfermedad)   → priorizar RECALL.
#   - Falsos positivos caros (marcar spam un email válido) → priorizar PRECISION.
#   - Ambos igual de importantes                           → F1.

## ─── 6.9 El umbral de decisión: Precision frente a Recall ─────────────────

# Lo normal es que en un modelo probabilístico se clasifique como positivo si 
# p ≥ umbral. 
# Tomar el umbral 0.5 es solo una convención: moverlo cambia sensibilidad y 
# especificidad SIN reentrenar el modelo.

probs = probabilidad(Xte, W_f)
umbrales = 0.05:0.05:0.95

prec_u = Float64[]; rec_u = Float64[]; f1_u = Float64[]
for u in umbrales
    p_u = [p >= u ? 1 : -1 for p in probs]
    m_u = metricas_por_clase(matriz_confusion(yte, p_u; clases=[-1, 1]))
    push!(prec_u, m_u.Precision[2]); push!(rec_u, m_u.Recall[2]); push!(f1_u, m_u.F1[2])
end

display(plot(umbrales, [prec_u rec_u f1_u], label=["Precision" "Recall" "F1"], linewidth=2,
             marker=:circle, xlabel="Umbral", ylabel="Valor",
             title="Compromiso Precision / Recall según el umbral"))

# Subir el umbral → menos falsos positivos (más precision) pero más falsos
# negativos (menos recall). Es imposible mejorar las dos a la vez moviendo solo 
# el umbral.

# La curva ROC resume TODOS los umbrales a la vez en una función.
# El AUC es su área: literalmente "Area Under the Curve". 
#     Cuanto más cerca de 1, mejor; 0.5 = azar.

r = roc_auc(yte, predecir(Xte, W_f))
println("AUC = ", round(r.auc, digits=3), "   (0.5 = azar, 1 = perfecto)")
display(graficar_roc(r; nombres=["Regresión logística"]))

# EJERCICIO 6.9: elige el umbral que maximiza F1. Después, el menor umbral con
# precision ≥ 0.9. ¿Cuánto recall pierdes?

#============================================================================
7. CIERRE: UN PROBLEMA COMPLETO DE PRINCIPIO A FIN
============================================================================#

## ─── 7.1 Datos + Modelo + Pérdida, de forma explícita ─────────────────────

Random.seed!(21)

# Cada decisión de diseño queda a la vista en el código:

# 1. DATOS: 500 ejemplos con dos características y etiqueta -1/+1

X_f, y_f = generar_clasificacion_2d(500; separacion=2.5)
Xtr, ytr, Xte, yte = train_test_split(X_f, y_f, 0.7)
println("m = ", size(Xtr, 1), " ejemplos de entrenamiento, n = ", size(Xtr, 2), " características")

# 2. MODELO: hipótesis lineales h_W(x) = w₀ + w₁x₁ + w₂x₂, clasificador sign∘h_W
# 3. PÉRDIDA para ENTRENAR: la logística (suave y convexa)

W_final, hist_final = regresion_logistica_gd(Xtr, ytr; lr=0.1, max_iter=500)
display(graficar_convergencia(hist_final; nombres=["Pérdida logística"], escala_log=false,
                              titulo="Entrenamiento: la pérdida baja"))

# EVALUACIÓN con otra pérdida, la 0/1, en datos de entrenamiento Y de test

err_tr = riesgo_empirico(perdida_01, ytr, predecir(Xtr, W_final))
err_te = riesgo_empirico(perdida_01, yte, predecir(Xte, W_final))
println("Error 0/1 → entrenamiento: ", round(err_tr, digits=3), "   test: ", round(err_te, digits=3))

# ¿GENERALIZA? Si el error en test es parecido al de entrenamiento, sí.
# Si fuera mucho peor, tendríamos sobreajuste.

display(graficar_frontera(Xte, yte, Xm -> clasificar(Xm, W_final); titulo="Hipótesis aprendida (test)"))

mc_final = matriz_confusion(yte, clasificar(Xte, W_final); clases=[-1, 1])
println(mc_final)
display(metricas_por_clase(mc_final))

## ─── 7.2 Resumen ──────────────────────────────────────────────────────────

#   - Aprender = generalizar a partir de ejemplos; siempre hay incertidumbre.
#   - Un problema de ML = DATOS + MODELO (espacio de hipótesis) + PÉRDIDA.
#   - DATOS: m ejemplos con n características; conviene m/n grande; más
#     características no es siempre mejor. Etiquetas: valores reales o clases.
#   - MODELO: H demasiado pequeño → infraajuste; demasiado grande → sobreajuste.
#   - PÉRDIDA: convierte "predecir bien" en un problema de optimización. Se puede
#     entrenar con una y evaluar con otra.
#   - EVALUAR clasificadores: matriz de confusión, Precision, Recall, F1, umbral,
#     ROC; nunca solo la accuracy si las clases están desbalanceadas.
#
# Próximo paso: los métodos concretos (KNN, K-medias, ID3, redes neuronales) son
# otras elecciones de H y de pérdida, pero hciendo uso de este mismo esquema.

## ─── 7.3 Ejercicios propuestos ────────────────────────────────────────────

# A. Carga un CSV (p. ej. iris.csv) con CSV.read y prepáralo con `preparar_datos`.
#    ¿Qué es X, qué es y, cuáles son m y n? ¿Es regresión o clasificación?
#
# B. Para un problema de regresión con atípicos, compara MSE y MAE (ajustar_mae
#    de 6.3). ¿Con qué porcentaje de atípicos deja de ser robusta la MAE?
#
# C. Con la matriz de 6.8, calcula a mano cada métrica para la clase +1 y
#    comprueba que coincide con `metricas_por_clase`.
#
# D. Reproduce 5.6 con ruido 0.5 y con 15 datos de entrenamiento. ¿Cambia el
#    grado óptimo? ¿Por qué?
#
# E. Repite 4.2 con informativas=20 (todas relevantes). ¿Cambia la conclusión
#    sobre m/