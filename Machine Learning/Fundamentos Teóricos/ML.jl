#============================================================================

ML.jl — FUNDAMENTOS DE ML EN JULIA

Uso:   include("ML.jl")
Paquetes necesarios:   ] add DataFrames CSV Plots

Convenciones:
  · X es una matriz m×n (m muestras, n características) 
  . y un vector de m etiquetas.
  · Un modelo lineal es simplemente un vector de pesos W. W[1] es SIEMPRE el
    término independiente (bias) y W[2:end] los pesos de las características.
  · En clasificación binaria las etiquetas se usan como -1 / +1 (las funciones
    aceptan también 0 / 1 y las convierten con `a_pm1`).
  · Las funciones `graficar_*` devuelven un gráfico: usa `display(p)` si las
    llamas dentro de una función o de un script.

CONTENIDO

  1. DATOS
     1.1 Exploración y estudio de un DataFrame       resumen, graficar_histogramas, graficar_correlaciones
     1.2 Limpieza y transformación                   tratar_faltantes, codificar_categorica, normalizar_df
     1.3 Normalización con estadísticos de train     ajustar_normalizador, normalizar
     1.4 Partición train/test y pipeline completo    train_test_split, preparar_datos
     1.5 Etiquetas                                   codificar_etiquetas, a_pm1
     1.6 Datasets sintéticos                         generar_datos, generar_clasificacion_2d, generar_xor
     1.7 Visualización de datos                      graficar_datos

  2. MODELOS LINEALES
     2.1 Predicción                                  predecir, clasificar, probabilidad
     2.2 Regresión lineal y Ridge (analítica)        regresion_lineal, regresion_ridge
     2.3 Perceptrón                                  perceptron
     2.4 Métricas de regresión                       mse, mae, r2_score, accuracy
     2.5 Visualización de modelos                    graficar_ajuste, graficar_predicciones, graficar_frontera

  3. PÉRDIDAS Y OPTIMIZACIÓN
     3.1 Funciones de pérdida y riesgo empírico      perdida_*, riesgo_empirico
     3.2 Descenso de gradiente                       bgd_mse, sgd_mse
     3.3 Regresión logística                         regresion_logistica_gd
     3.4 Visualización                               graficar_perdidas_*, graficar_convergencia, graficar_superficie_perdida

  4. EVALUACIÓN
     4.1 Matriz de confusión y métricas por clase    matriz_confusion, metricas_por_clase, graficar_matriz_confusion
     4.2 Curva ROC y AUC                             roc_auc, graficar_roc
     4.3 Validación cruzada K-fold                   validacion_cruzada_kfold

  5. SOBREAJUSTE E INFRAAJUSTE
     5.1 Polinomios                                  ajustar_polinomio, predecir_polinomio
     5.2 Curvas de aprendizaje, complejidad, λ       curva_aprendizaje, curva_complejidad, curva_regularizacion
     5.3 Diagnóstico y selección de modelo           diagnosticar_ajuste, seleccionar_mejor_grado, seleccionar_mejor_lambda
     5.4 Sesgo-varianza                              analisis_sesgo_varianza

  6. EJEMPLOS (funciones que puedes ejecutar y usar como plantilla)
     ejemplo_datos(), ejemplo_regresion(), ejemplo_clasificacion(), ejemplo_sobreajuste()
============================================================================#

using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Random
using Plots


#============================================================================
1. DATOS
============================================================================#

# ----------------------------------------------------------------------------
# 1.1 EXPLORACIÓN Y ESTUDIO DE UN DATAFRAME
# ----------------------------------------------------------------------------
# Para cargar un CSV:   df = CSV.read("ruta/iris.csv", DataFrame)

# Nombres (Symbol) de las columnas numéricas (se excluyen Bool y texto)
function _columnas_numericas(df::DataFrame)
    return [c for c in propertynames(df)
            if nonmissingtype(eltype(df[!, c])) <: Real && !(nonmissingtype(eltype(df[!, c])) <: Bool)]
end

"""
Muestra dimensiones, tipo y nº de valores faltantes de cada columna, y
estadísticas descriptivas.
"""
function resumen(df::DataFrame)
    println("Dimensiones: ", nrow(df), " filas × ", ncol(df), " columnas\n")
    info = DataFrame(columna = names(df),
                     tipo = string.(eltype.(eachcol(df))),
                     faltantes = [count(ismissing, c) for c in eachcol(df)])
    println(info)
    println("\nEstadísticas descriptivas:")
    println(describe(df, :mean, :std, :min, :median, :max))
end

"""
Histogramas de todas las columnas numéricas de un DataFrame.
"""
function graficar_histogramas(df::DataFrame; bins=20)
    cols = _columnas_numericas(df)
    ps = [histogram(collect(skipmissing(df[!, c])), bins=bins, title=string(c),
                    legend=false, titlefontsize=9, color=:skyblue, alpha=0.8)
          for c in cols]
    return plot(ps..., size=(900, 260 * ceil(Int, length(ps) / 3)))
end

"""
Matriz de correlaciones (Pearson) entre las columnas numéricas, como mapa de calor.
"""
function graficar_correlaciones(df::DataFrame)
    cols = _columnas_numericas(df)
    d = dropmissing(select(df, cols))
    C = cor(Matrix{Float64}(d))
    n = length(cols)
    nombres = string.(cols)
    p = heatmap(1:n, 1:n, C, yflip=true, c=:RdBu, clims=(-1, 1),
                xticks=(1:n, nombres), yticks=(1:n, nombres),
                title="Correlaciones", xrotation=30)
    for i in 1:n, j in 1:n
        annotate!(p, j, i, text(string(round(C[i, j], digits=2)), 8, :black))
    end
    return p
end

# ----------------------------------------------------------------------------
# 1.2 LIMPIEZA Y TRANSFORMACIÓN DE DATAFRAMES
# ----------------------------------------------------------------------------

"""
    tratar_faltantes(df, metodo, cols=todas)

Devuelve un nuevo DataFrame donde los valores faltantes de `cols` (un símbolo o
una lista de símbolos; por defecto todas las columnas) se tratan con uno de los
siguientes métodos (`metodo`):
  * `:remove` → se eliminan las filas con algún valor faltante en `cols`
  * `:mean`   → se sustituyen por la media de la columna (solo columnas numéricas)
"""
function tratar_faltantes(df::DataFrame, metodo::Symbol, cols=propertynames(df))
    metodo in (:remove, :mean) || error("metodo debe ser :remove o :mean")
    cols = cols isa Symbol ? [cols] : Symbol.(cols)
    out = copy(df)

    if metodo == :remove
        dropmissing!(out, cols)
    else
        for c in cols
            v = out[!, c]
            any(ismissing, v) || continue
            sin_faltantes = collect(skipmissing(v))
            eltype(sin_faltantes) <: Real ||
                error("La columna $c no es numérica: no se puede usar :mean (usa :remove)")
            m = mean(sin_faltantes)
            out[!, c] = [ismissing(x) ? m : float(x) for x in v]
        end
    end
    return out
end

"""
    codificar_categorica(df, metodo, col)

Convierte la columna categórica `col` en numérica según el `metodo`:
  * `:label`   → cada categoría se sustituye por un entero (0, 1, 2...) por orden alfabético
  * `:one_hot` → se crea una columna 0/1 por categoría (`col_valor`) y se elimina la original
"""
function codificar_categorica(df::DataFrame, metodo::Symbol, col::Symbol)
    metodo in (:label, :one_hot) || error("metodo debe ser :label o :one_hot")
    col in propertynames(df) || error("La columna $col no existe en el dataframe")

    out = copy(df)
    valores = sort(unique(df[!, col]))

    if metodo == :label
        mapa = Dict(v => i - 1 for (i, v) in enumerate(valores))
        out[!, col] = [mapa[v] for v in df[!, col]]
    else
        select!(out, Not(col))
        for v in valores
            out[!, Symbol("$(col)_$(v)")] = Int.(df[!, col] .== v)
        end
    end
    return out
end

"""
Aplica `codificar_categorica` a todas las columnas de texto (excepto las de `excluir`).
"""
function codificar_todas(df::DataFrame, metodo::Symbol; excluir=Symbol[])
    out = df
    for c in propertynames(df)
        c in excluir && continue
        T = nonmissingtype(eltype(df[!, c]))
        if T <: AbstractString || T <: Symbol
            out = codificar_categorica(out, metodo, c)
        end
    end
    return out
end

"""
    normalizar_df(df, metodo=:z_score; excluir=Symbol[])

Devuelve un nuevo DataFrame con las columnas numéricas normalizadas
(`:z_score` → media 0 y desviación 1; `:min_max` → rango [0, 1]).

Útil para EXPLORAR datos. Para entrenar modelos usa `preparar_datos` (o
`ajustar_normalizador`), que calcula los estadísticos solo con el conjunto de
entrenamiento y evita filtrar información del test.
"""
function normalizar_df(df::DataFrame, metodo::Symbol=:z_score; excluir=Symbol[])
    metodo in (:min_max, :z_score) || error("metodo debe ser :min_max o :z_score")
    out = copy(df)
    for c in _columnas_numericas(df)
        c in excluir && continue
        v = df[!, c]
        any(ismissing, v) && error("La columna $c tiene valores faltantes: usa tratar_faltantes antes")
        v = Float64.(v)
        a, b = metodo == :z_score ? (mean(v), std(v)) : (minimum(v), maximum(v) - minimum(v))
        b == 0 && (b = 1.0)
        out[!, c] = (v .- a) ./ b
    end
    return out
end

# ----------------------------------------------------------------------------
# 1.3 NORMALIZACIÓN CON ESTADÍSTICOS DE ENTRENAMIENTO (sobre matrices)
# ----------------------------------------------------------------------------

"""
Guarda los parámetros de una normalización: (x - a) / b, columna a columna.
"""
struct Normalizador
    metodo::Symbol
    a::Vector{Float64}   # media (z_score) o mínimo (min_max)
    b::Vector{Float64}   # desviación típica (z_score) o rango (min_max)
end

"""
Calcula los parámetros de normalización de las columnas de `X`.
Se ajusta con X_train y se aplica con `normalizar` a train Y test.
"""
function ajustar_normalizador(X::AbstractMatrix, metodo::Symbol=:z_score)
    if metodo == :z_score
        a = vec(mean(X, dims=1))
        b = vec(std(X, dims=1))
    elseif metodo == :min_max
        a = vec(minimum(X, dims=1))
        b = vec(maximum(X, dims=1)) .- a
    else
        error("metodo debe ser :z_score o :min_max")
    end
    b = [x == 0 ? 1.0 : x for x in b]     # columnas constantes: evitar dividir por 0
    return Normalizador(metodo, Float64.(a), Float64.(b))
end

normalizar(X::AbstractMatrix, n::Normalizador) = (X .- n.a') ./ n.b'

"Atajo: ajusta y aplica la normalización. Devuelve (X_normalizada, normalizador)."
function normalizar(X::AbstractMatrix, metodo::Symbol)
    n = ajustar_normalizador(X, metodo)
    return normalizar(X, n), n
end

# ----------------------------------------------------------------------------
# 1.4 PARTICIÓN TRAIN/TEST Y PIPELINE COMPLETO
# ----------------------------------------------------------------------------

function _indices_train_test(n::Int, train_ratio::Real)
    0 < train_ratio < 1 || error("train_ratio debe estar entre 0 y 1")
    n_train = clamp(round(Int, n * train_ratio), 1, n - 1)
    idx = randperm(n)
    return idx[1:n_train], idx[n_train+1:end]
end

_filas(X::AbstractMatrix, idx) = X[idx, :]
_filas(X::AbstractVector, idx) = X[idx]

"""
    train_test_split(X, y, train_ratio=0.8)
    train_test_split(df, columna_objetivo, train_ratio=0.8)

División aleatoria en entrenamiento y test. Devuelve `Xtrain, ytrain, Xtest, ytest`.
(Para resultados reproducibles, llama antes a `Random.seed!(n)`.)
"""
function train_test_split(X::AbstractVecOrMat, y::AbstractVector, train_ratio::Real=0.8)
    size(X, 1) == length(y) || error("X e y deben tener el mismo número de muestras")
    tr, te = _indices_train_test(length(y), train_ratio)
    return _filas(X, tr), y[tr], _filas(X, te), y[te]
end

function train_test_split(df::DataFrame, objetivo::Symbol, train_ratio::Real=0.8)
    objetivo in propertynames(df) || error("La columna $objetivo no existe en el dataframe")
    tr, te = _indices_train_test(nrow(df), train_ratio)
    X = select(df, Not(objetivo))
    return X[tr, :], df[tr, objetivo], X[te, :], df[te, objetivo]
end

"""
    preparar_datos(df, objetivo; faltantes=:remove, categoricas=:label,
                   normalizacion=:z_score, train_ratio=0.8)

Pipeline completo. A partir de un DataFrame devuelve matrices listas para entrenar:
  1. Elimina las filas sin valor en la columna objetivo.
  2. Trata los faltantes del resto de columnas (`:remove` o `:mean`).
  3. Codifica las columnas de texto (`:label` o `:one_hot`).
  4. Divide en train/test.
  5. Normaliza las características (`:z_score` o `:min_max`) usando SOLO los
     estadísticos de train.

Devuelve una tupla con nombre; se puede desempaquetar directamente:

    Xtrain, ytrain, Xtest, ytest = preparar_datos(df, :Especie)

y además contiene `.columnas` (nombres de las características) y `.normalizador`.
La variable objetivo NO se modifica (usa `codificar_etiquetas` o `a_pm1` si hace falta).
"""
function preparar_datos(df::DataFrame, objetivo::Symbol;
                        faltantes::Symbol=:remove, categoricas::Symbol=:label,
                        normalizacion::Symbol=:z_score, train_ratio::Real=0.8)
    objetivo in propertynames(df) || error("La columna $objetivo no existe en el dataframe")

    d = dropmissing(df, objetivo)
    d = tratar_faltantes(d, faltantes, filter(!=(objetivo), propertynames(d)))
    d = codificar_todas(d, categoricas; excluir=[objetivo])

    Xtr_df, ytr, Xte_df, yte = train_test_split(d, objetivo, train_ratio)
    Xtr = Matrix{Float64}(Xtr_df)
    Xte = Matrix{Float64}(Xte_df)

    nrm = ajustar_normalizador(Xtr, normalizacion)
    return (Xtrain = normalizar(Xtr, nrm), ytrain = ytr,
            Xtest = normalizar(Xte, nrm), ytest = yte,
            columnas = names(Xtr_df), normalizador = nrm)
end

# ----------------------------------------------------------------------------
# 1.5 ETIQUETAS
# ----------------------------------------------------------------------------

"""
Convierte etiquetas de cualquier tipo en enteros 1..k. Devuelve (y_numerica, clases).
"""
function codificar_etiquetas(y::AbstractVector)
    clases = sort(unique(y))
    mapa = Dict(c => i for (i, c) in enumerate(clases))
    return [mapa[v] for v in y], clases
end

"""
Convierte etiquetas binarias a -1/+1. Si `y` es numérica, el positivo es 1
(sirve para {0,1} y {-1,1}); si es texto, indica la clase positiva con `positivo`.
"""
function a_pm1(y::AbstractVector; positivo = eltype(y) <: Real ? 1 : maximum(y))
    return [v == positivo ? 1 : -1 for v in y]
end

# ----------------------------------------------------------------------------
# 1.6 DATASETS SINTÉTICOS
# ----------------------------------------------------------------------------

"""
Genera `n` puntos (x, y) con y = f(x) + ruido gaussiano, con x uniforme en `intervalo`.
Devuelve (x, y) con x ordenado.
"""
function generar_datos(f::Function, n::Int=100; ruido::Real=0.3, intervalo=(-1.0, 1.0), seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    a, b = intervalo
    x = sort(a .+ (b - a) .* rand(n))
    y = f.(x) .+ ruido .* randn(n)
    return x, y
end

"""
Dos nubes gaussianas en 2D (clases -1 y +1). Cuanto menor `separacion`, más solapadas.
Devuelve (X, y).
"""
function generar_clasificacion_2d(m::Int=100; separacion::Real=2.0, seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    m1 = m ÷ 2
    m2 = m - m1
    X = vcat(randn(m1, 2) .+ [-separacion/2 0.0], randn(m2, 2) .+ [separacion/2 0.0])
    y = vcat(fill(-1, m1), fill(1, m2))
    p = randperm(m)
    return X[p, :], y[p]
end

"""
Problema XOR: puntos en [-1,1]², clase +1 si x₁·x₂ ≥ 0. NO es linealmente separable.
"""
function generar_xor(m::Int=200; seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    X = 2 .* rand(m, 2) .- 1
    y = [X[i, 1] * X[i, 2] >= 0 ? 1 : -1 for i in 1:m]
    return X, y
end

# ----------------------------------------------------------------------------
# 1.7 VISUALIZACIÓN DE DATOS
# ----------------------------------------------------------------------------

function _scatter_clases!(p, X, y)
    colores = (:royalblue, :crimson, :seagreen, :darkorange, :purple, :goldenrod)
    marcas = (:circle, :square, :diamond, :utriangle, :dtriangle, :star5)
    for (i, c) in enumerate(sort(unique(y)))
        idx = y .== c
        scatter!(p, X[idx, 1], X[idx, 2], label="Clase $c",
                 color=colores[mod1(i, length(colores))],
                 markershape=marcas[mod1(i, length(marcas))],
                 markersize=5, alpha=0.8, markerstrokecolor=:black)
    end
    return p
end

# ¿y parece una variable de clase (clasificación) o continua (regresión)?
_es_clasificacion(y) = nonmissingtype(eltype(y)) <: Union{AbstractString, Symbol, Bool} || length(unique(y)) <= 10

"""
Dibuja un dataset con 2 características: colorea por clase (clasificación) o por
valor de y (regresión). Detecta el tipo de problema automáticamente.
"""
function graficar_datos(X::AbstractMatrix, y::AbstractVector;
                        nombres=["x₁", "x₂"], titulo="Datos")
    size(X, 2) == 2 || error("graficar_datos solo funciona con 2 características")
    if _es_clasificacion(y)
        p = plot(xlabel=nombres[1], ylabel=nombres[2], title=titulo, legend=:topright)
        return _scatter_clases!(p, X, y)
    else
        return scatter(X[:, 1], X[:, 2], marker_z=y, color=:viridis, markersize=5,
                       alpha=0.8, xlabel=nombres[1], ylabel=nombres[2], title=titulo,
                       colorbar_title="y", label="")
    end
end


#============================================================================
2. MODELOS LINEALES
============================================================================#

# ----------------------------------------------------------------------------
# 2.1 PREDICCIÓN
# ----------------------------------------------------------------------------

"Añade una columna de unos (término independiente) a la izquierda de X."
añadir_bias(X::AbstractMatrix) = hcat(ones(size(X, 1)), X)

"""
    predecir(X, W)

Hipótesis lineal h(x) = W₁ + W₂x₁ + ... + Wₙ₊₁xₙ para todas las filas de X.
Si `X` es un VECTOR se interpreta como una única característica con m muestras.
"""
predecir(X::AbstractMatrix, W::AbstractVector) = añadir_bias(X) * W
predecir(x::AbstractVector, W::AbstractVector) = predecir(reshape(x, :, 1), W)

"Clasificación binaria: devuelve +1 si h(x) ≥ 0 y -1 en caso contrario."
clasificar(X, W::AbstractVector) = [s >= 0 ? 1 : -1 for s in predecir(X, W)]

"Probabilidad de la clase +1 según el modelo logístico: σ(h(x))."
probabilidad(X, W::AbstractVector) = 1 ./ (1 .+ exp.(-predecir(X, W)))

# ----------------------------------------------------------------------------
# 2.2 REGRESIÓN LINEAL Y RIDGE (SOLUCIÓN ANALÍTICA)
# ----------------------------------------------------------------------------

# Mínimos cuadrados (con regularización L2 opcional). La primera columna de Φ
# es el bias y NO se penaliza.
function _minimos_cuadrados(Φ::AbstractMatrix, y::AbstractVector, λ::Real=0.0)
    if λ == 0
        return Φ \ y                       # equivale a W = (ΦᵀΦ)⁻¹Φᵀy, pero más estable
    end
    n = size(Φ, 2)
    Λ = λ * Matrix(1.0I, n, n)
    Λ[1, 1] = 0.0
    return (Φ' * Φ + Λ) \ (Φ' * y)
end

"""
Regresión lineal por mínimos cuadrados: W = argmin Σ (yⁱ - W·xⁱ)².
Devuelve el vector de pesos W (con el bias en W[1]).
"""
regresion_lineal(X::AbstractMatrix, y::AbstractVector) = _minimos_cuadrados(añadir_bias(X), y)
regresion_lineal(x::AbstractVector, y::AbstractVector) = regresion_lineal(reshape(x, :, 1), y)

"""
Regresión Ridge: W = argmin Σ (yⁱ - W·xⁱ)² + λ‖W‖²  (el bias no se penaliza).
"""
regresion_ridge(X::AbstractMatrix, y::AbstractVector, λ::Real=1.0) = _minimos_cuadrados(añadir_bias(X), y, λ)
regresion_ridge(x::AbstractVector, y::AbstractVector, λ::Real=1.0) = regresion_ridge(reshape(x, :, 1), y, λ)

# ----------------------------------------------------------------------------
# 2.3 PERCEPTRÓN
# ----------------------------------------------------------------------------

"""
    perceptron(X, y; max_iter=100, lr=1.0)

Algoritmo del perceptrón. Devuelve `(W, errores)` donde `errores[k]` es el número
de muestras mal clasificadas durante la época k. Converge solo si los datos son
linealmente separables.
"""
function perceptron(X::AbstractMatrix, y::AbstractVector; max_iter::Int=100, lr::Real=1.0)
    ypm = a_pm1(y)
    Xa = añadir_bias(X)
    m, n = size(Xa)
    W = zeros(n)
    errores = Int[]

    for _ in 1:max_iter
        fallos = 0
        for i in 1:m
            if ypm[i] * dot(W, Xa[i, :]) <= 0
                W .+= lr * ypm[i] .* Xa[i, :]
                fallos += 1
            end
        end
        push!(errores, fallos)
        fallos == 0 && break
    end
    return W, errores
end

# ----------------------------------------------------------------------------
# 2.4 MÉTRICAS DE REGRESIÓN Y CLASIFICACIÓN
# ----------------------------------------------------------------------------

"Error cuadrático medio."
mse(y_true, y_pred) = mean((y_true .- y_pred) .^ 2)

"Error absoluto medio."
mae(y_true, y_pred) = mean(abs.(y_true .- y_pred))

"Coeficiente de determinación R² = 1 - SS_res / SS_tot."
r2_score(y_true, y_pred) = 1 - sum((y_true .- y_pred) .^ 2) / sum((y_true .- mean(y_true)) .^ 2)

"Proporción de predicciones correctas."
accuracy(y_true, y_pred) = mean(y_true .== y_pred)

# ----------------------------------------------------------------------------
# 2.5 VISUALIZACIÓN DE MODELOS Y PREDICCIONES
# ----------------------------------------------------------------------------

"""
    graficar_ajuste(x, y, modelos...; nombres=nothing, titulo="Ajuste")

Dibuja los datos (x, y) y una curva por cada modelo. Cada modelo es una función
que recibe un VECTOR de x y devuelve el vector de predicciones, por ejemplo:

    graficar_ajuste(x, y, t -> predecir(t, W); nombres=["Lineal"])
"""
function graficar_ajuste(x::AbstractVector, y::AbstractVector, modelos...;
                         nombres=nothing, titulo="Ajuste")
    xs = collect(range(minimum(x), maximum(x), length=300))
    p = scatter(x, y, label="Datos", xlabel="x", ylabel="y", title=titulo,
                markersize=4, alpha=0.6, legend=:best)
    for (i, f) in enumerate(modelos)
        nombre = isnothing(nombres) ? "Modelo $i" : nombres[i]
        plot!(p, xs, f(xs), label=nombre, linewidth=2)
    end
    return p
end

"Predicciones frente a valores reales (los puntos deberían estar sobre la diagonal)."
function graficar_predicciones(y_true, y_pred; titulo="Predicciones vs valores reales")
    lo = min(minimum(y_true), minimum(y_pred))
    hi = max(maximum(y_true), maximum(y_pred))
    p = scatter(y_true, y_pred, xlabel="Valores reales", ylabel="Predicciones",
                title=titulo, label="Predicciones", markersize=4, alpha=0.6,
                legend=:bottomright)
    plot!(p, [lo, hi], [lo, hi], label="Predicción perfecta", linestyle=:dash,
          linewidth=2, color=:red)
    return p
end

"""
    graficar_frontera(X, y, f; titulo, resolucion=150)

Dibuja un clasificador en 2D: el fondo muestra lo que predice `f` en cada punto
y los puntos muestran los datos. `f` recibe una matriz (k×2) y devuelve un vector
NUMÉRICO (clases -1/+1, o probabilidades). Por ejemplo:

    graficar_frontera(X, y, Xm -> clasificar(Xm, W))
    graficar_frontera(X, y, Xm -> probabilidad(Xm, W))
"""
function graficar_frontera(X::AbstractMatrix, y::AbstractVector, f::Function;
                           titulo="Frontera de decisión", resolucion::Int=150)
    size(X, 2) == 2 || error("graficar_frontera solo funciona con 2 características")
    r = resolucion
    x1 = range(minimum(X[:, 1]) - 0.5, maximum(X[:, 1]) + 0.5, length=r)
    x2 = range(minimum(X[:, 2]) - 0.5, maximum(X[:, 2]) + 0.5, length=r)

    # Rejilla: x1 varía más rápido que x2
    Xg = Matrix{Float64}(undef, r * r, 2)
    k = 1
    for b in x2, a in x1
        Xg[k, 1] = a
        Xg[k, 2] = b
        k += 1
    end
    Z = permutedims(reshape(Float64.(f(Xg)), r, r))      # Z[i, j] ↔ (x2[i], x1[j])

    p = heatmap(x1, x2, Z, c=:RdBu, alpha=0.3, colorbar=false,
                xlabel="x₁", ylabel="x₂", title=titulo, legend=:topright)
    _scatter_clases!(p, X, y)
    return p
end


#============================================================================
3. FUNCIONES DE PÉRDIDA Y OPTIMIZACIÓN
============================================================================#

# ----------------------------------------------------------------------------
# 3.1 FUNCIONES DE PÉRDIDA Y RIESGO EMPÍRICO
# ----------------------------------------------------------------------------
# Están definidas para UN ejemplo; para un vector se usa el punto (broadcast):
#     perdida_cuadratica.(y, y_pred)

# --- Regresión: (y, y_pred) ---
perdida_cuadratica(y, y_pred) = (y - y_pred)^2
perdida_absoluta(y, y_pred) = abs(y - y_pred)

# --- Clasificación binaria con y ∈ {-1,+1} y score = h(x) ---
# (el "margen" es y·score: positivo = bien clasificado)
perdida_01(y, score) = sign(score) == y ? 0.0 : 1.0
perdida_bisagra(y, score) = max(0.0, 1 - y * score)
perdida_logistica(y, score) = (mg = y * score; mg > 0 ? log1p(exp(-mg)) : -mg + log1p(exp(mg)))

# --- Entropía cruzada binaria con y ∈ {0,1} y p = probabilidad de la clase 1 ---
function perdida_entropia_cruzada(y, p)
    q = clamp(p, 1e-12, 1 - 1e-12)
    return -y * log(q) - (1 - y) * log(1 - q)
end

"""
Riesgo empírico: promedio de la pérdida sobre los datos.

    riesgo_empirico(perdida_cuadratica, y, y_pred)      # = MSE
    riesgo_empirico(perdida_01, y, scores)              # = tasa de error
"""
riesgo_empirico(perdida::Function, y, pred) = mean(perdida.(y, pred))

# ----------------------------------------------------------------------------
# 3.2 DESCENSO DE GRADIENTE
# ----------------------------------------------------------------------------
# Todas devuelven (W, historial) con historial = (perdidas = ..., pesos = ...):
# la pérdida y los pesos al inicio de cada iteración (o época).

"""
    bgd_mse(X, y; lr=0.01, max_iter=1000, tol=1e-8)

Descenso de gradiente por lotes (batch) para regresión lineal con pérdida MSE.
Gradiente: ∇L(W) = (2/m)·Xᵀ(XW - y).  Actualización: W ← W - lr·∇L(W).
Para cuando el cambio en W es menor que `tol` (o si el descenso diverge).
"""
function bgd_mse(X::AbstractMatrix, y::AbstractVector; lr::Real=0.01, max_iter::Int=1000, tol::Real=1e-8)
    Xa = añadir_bias(X)
    m, n = size(Xa)
    W = zeros(n)
    perdidas = Float64[]
    pesos = Vector{Float64}[]

    for _ in 1:max_iter
        err = Xa * W .- y
        push!(perdidas, mean(err .^ 2))
        push!(pesos, copy(W))
        if !isfinite(perdidas[end])
            @warn "El descenso de gradiente diverge: prueba con un lr más pequeño o normaliza los datos"
            break
        end
        W_nuevo = W .- lr .* ((2 / m) .* (Xa' * err))
        parar = norm(W_nuevo - W) < tol
        W = W_nuevo
        parar && break
    end
    return W, (perdidas = perdidas, pesos = pesos)
end

"""
    sgd_mse(X, y; lr=0.01, epochs=100, batch_size=1)

Descenso de gradiente estocástico para regresión lineal con pérdida MSE.
Con `batch_size=1` actualiza con una muestra cada vez (SGD puro); con un valor
intermedio es mini-batch; con `batch_size=m` coincide con el descenso por lotes.
El historial guarda un punto por época.
"""
function sgd_mse(X::AbstractMatrix, y::AbstractVector; lr::Real=0.01, epochs::Int=100, batch_size::Int=1)
    Xa = añadir_bias(X)
    m, n = size(Xa)
    W = zeros(n)
    perdidas = Float64[]
    pesos = Vector{Float64}[]

    for _ in 1:epochs
        idx = shuffle(1:m)
        for inicio in 1:batch_size:m
            lote = idx[inicio:min(inicio + batch_size - 1, m)]
            Xl = Xa[lote, :]
            err = Xl * W .- y[lote]
            W = W .- lr .* ((2 / length(lote)) .* (Xl' * err))
        end
        push!(perdidas, mean((Xa * W .- y) .^ 2))
        push!(pesos, copy(W))
    end
    return W, (perdidas = perdidas, pesos = pesos)
end

# ----------------------------------------------------------------------------
# 3.3 REGRESIÓN LOGÍSTICA
# ----------------------------------------------------------------------------

"""
    regresion_logistica_gd(X, y; lr=0.1, max_iter=1000)

Clasificación binaria minimizando la pérdida logística con descenso de gradiente.
`y` puede ser {0,1} o {-1,+1}. Devuelve (W, historial). Para clasificar:
`clasificar(X, W)`; para obtener probabilidades: `probabilidad(X, W)`.
"""
function regresion_logistica_gd(X::AbstractMatrix, y::AbstractVector; lr::Real=0.1, max_iter::Int=1000)
    ypm = a_pm1(y)
    Xa = añadir_bias(X)
    m, n = size(Xa)
    W = zeros(n)
    perdidas = Float64[]
    pesos = Vector{Float64}[]

    for _ in 1:max_iter
        scores = Xa * W
        push!(perdidas, mean(perdida_logistica.(ypm, scores)))
        push!(pesos, copy(W))
        σ = 1 ./ (1 .+ exp.(ypm .* scores))              # σ(-y·score)
        W = W .+ lr .* ((1 / m) .* (Xa' * (ypm .* σ)))   # W ← W - lr·∇L
    end
    return W, (perdidas = perdidas, pesos = pesos)
end

# ----------------------------------------------------------------------------
# 3.4 VISUALIZACIÓN DE PÉRDIDAS Y CONVERGENCIA
# ----------------------------------------------------------------------------

"Compara las pérdidas cuadrática y absoluta en función del error (y - ŷ)."
function graficar_perdidas_regresion()
    e = range(-3, 3, length=300)
    p = plot(e, perdida_cuadratica.(0.0, e), label="Cuadrática", linewidth=2,
             xlabel="Error (y - ŷ)", ylabel="Pérdida",
             title="Pérdidas para regresión", legend=:top)
    plot!(p, e, perdida_absoluta.(0.0, e), label="Absoluta", linewidth=2)
    return p
end

"Compara las pérdidas 0/1, bisagra y logística en función del margen y·h(x)."
function graficar_perdidas_clasificacion()
    mg = range(-2, 3, length=300)
    p = plot(mg, perdida_01.(1, mg), label="0/1", linewidth=2,
             xlabel="Margen y·h(x)", ylabel="Pérdida",
             title="Pérdidas para clasificación", ylims=(0, 3.5))
    plot!(p, mg, perdida_bisagra.(1, mg), label="Bisagra", linewidth=2)
    plot!(p, mg, perdida_logistica.(1, mg), label="Logística", linewidth=2, linestyle=:dash)
    vline!(p, [0], label="", color=:gray, linestyle=:dot)
    return p
end

"""
Curvas de pérdida de uno o varios historiales (los que devuelven bgd_mse, sgd_mse
y regresion_logistica_gd):

    graficar_convergencia(h_bgd, h_sgd; nombres=["Batch", "SGD"])
"""
function graficar_convergencia(historiales...; nombres=nothing, escala_log::Bool=true,
                               titulo="Convergencia del descenso de gradiente")
    p = plot(xlabel="Iteración / época", ylabel="Pérdida", title=titulo,
             yscale = escala_log ? :log10 : :identity, legend=:topright)
    for (i, h) in enumerate(historiales)
        nombre = isnothing(nombres) ? "Modelo $i" : nombres[i]
        plot!(p, 1:length(h.perdidas), h.perdidas, label=nombre, linewidth=2)
    end
    return p
end

"""
Mapa de contorno de la superficie de pérdida MSE (para 1 característica, es decir
2 pesos w₀ y w₁) con la trayectoria que sigue el descenso de gradiente encima.
"""
function graficar_superficie_perdida(X::AbstractMatrix, y::AbstractVector, historial; niveles::Int=25)
    size(X, 2) == 1 || error("Requiere 1 característica (2 pesos: w₀ y w₁)")
    x = vec(X)
    w0 = [w[1] for w in historial.pesos]
    w1 = [w[2] for w in historial.pesos]

    pad0 = 0.2 * (maximum(w0) - minimum(w0)) + 0.5
    pad1 = 0.2 * (maximum(w1) - minimum(w1)) + 0.5
    r0 = range(minimum(w0) - pad0, maximum(w0) + pad0, length=80)
    r1 = range(minimum(w1) - pad1, maximum(w1) + pad1, length=80)
    Z = [mean((y .- (a .+ b .* x)) .^ 2) for b in r1, a in r0]     # filas: w₁, columnas: w₀

    p = contour(r0, r1, Z, fill=true, levels=niveles, xlabel="w₀ (bias)", ylabel="w₁",
                title="Superficie de pérdida MSE")
    plot!(p, w0, w1, label="Trayectoria", linewidth=2, color=:white, marker=:circle, markersize=2)
    scatter!(p, [w0[1]], [w1[1]], label="Inicio", markersize=7, color=:lime, markershape=:star5)
    scatter!(p, [w0[end]], [w1[end]], label="Final", markersize=7, color=:red, markershape=:star5)
    return p
end


#============================================================================
4. EVALUACIÓN Y MÉTRICAS
============================================================================#

# ----------------------------------------------------------------------------
# 4.1 MATRIZ DE CONFUSIÓN (BINARIA Y MULTICLASE)
# ----------------------------------------------------------------------------

"""
Matriz de confusión: filas = clase real, columnas = clase predicha.
`clases` es la lista ordenada de etiquetas.
"""
struct MatrizConfusion
    matriz::Matrix{Int}
    clases::Vector
end

"""
    matriz_confusion(y_true, y_pred; clases = etiquetas ordenadas)

Construye la matriz de confusión para cualquier número de clases y de cualquier
tipo (números o texto).
"""
function matriz_confusion(y_true::AbstractVector, y_pred::AbstractVector;
                          clases=sort(unique(vcat(y_true, y_pred))))
    length(y_true) == length(y_pred) || error("y_true e y_pred deben tener la misma longitud")
    idx = Dict(c => i for (i, c) in enumerate(clases))
    k = length(clases)
    M = zeros(Int, k, k)
    for (t, p) in zip(y_true, y_pred)
        M[idx[t], idx[p]] += 1
    end
    return MatrizConfusion(M, collect(clases))
end

function Base.show(io::IO, cm::MatrizConfusion)
    etiquetas = string.(cm.clases)
    w = maximum(length.(vcat(etiquetas, string.(vec(cm.matriz))))) + 2
    println(io, "Matriz de confusión (filas = real, columnas = predicho)")
    println(io, lpad("", w), join(lpad.(etiquetas, w)))
    for (i, e) in enumerate(etiquetas)
        println(io, lpad(e, w), join(lpad.(string.(cm.matriz[i, :]), w)))
    end
    print(io, "Accuracy global: ", round(accuracy(cm), digits=4))
end

"Accuracy global: proporción de aciertos (diagonal / total)."
accuracy(cm::MatrizConfusion) = sum(diag(cm.matriz)) / sum(cm.matriz)

"""
Métricas de cada clase tratada como "positiva" frente al resto (uno contra todos).
Devuelve un DataFrame con TP, FP, FN, TN, Precision, Recall, Especificidad y F1.
Sirve igual para 2 clases (mira la fila de la clase positiva) que para multiclase.
"""
function metricas_por_clase(cm::MatrizConfusion)
    M = cm.matriz
    total = sum(M)
    dividir(a, b) = b == 0 ? 0.0 : a / b

    filas = map(1:size(M, 1)) do i
        tp = M[i, i]
        fn = sum(M[i, :]) - tp
        fp = sum(M[:, i]) - tp
        tn = total - tp - fn - fp
        prec = dividir(tp, tp + fp)
        rec = dividir(tp, tp + fn)
        (Clase = cm.clases[i], TP = tp, FP = fp, FN = fn, TN = tn,
         Precision = prec, Recall = rec, Especificidad = dividir(tn, tn + fp),
         F1 = dividir(2 * prec * rec, prec + rec))
    end
    return DataFrame(filas)
end

"Mapa de calor de la matriz de confusión con los recuentos en cada celda."
function graficar_matriz_confusion(cm::MatrizConfusion; titulo="Matriz de confusión")
    M = cm.matriz
    k = size(M, 1)
    etiquetas = string.(cm.clases)
    p = heatmap(1:k, 1:k, M, yflip=true, c=:Blues, title=titulo,
                xlabel="Predicho", ylabel="Real",
                xticks=(1:k, etiquetas), yticks=(1:k, etiquetas))
    for i in 1:k, j in 1:k
        color = M[i, j] > maximum(M) / 2 ? :white : :black
        annotate!(p, j, i, text(string(M[i, j]), 10, color))
    end
    return p
end

# ----------------------------------------------------------------------------
# 4.2 CURVA ROC Y AUC
# ----------------------------------------------------------------------------

"""
Curva ROC variando el umbral sobre los `scores` (valores continuos, p. ej.
`predecir(X, W)` o `probabilidad(X, W)`). Devuelve (fpr, tpr).
"""
function curva_roc(y_true::AbstractVector, scores::AbstractVector)
    y = a_pm1(y_true)
    orden = sortperm(scores, rev=true)
    ys = y[orden]
    ss = scores[orden]
    npos = count(==(1), ys)
    nneg = length(ys) - npos
    (npos > 0 && nneg > 0) || error("Se necesitan ejemplos de las dos clases")

    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0
    for i in eachindex(ys)
        ys[i] == 1 ? (tp += 1) : (fp += 1)
        # solo se añade un punto cuando cambia el umbral (maneja scores empatados)
        if i == length(ys) || ss[i+1] != ss[i]
            push!(tpr, tp / npos)
            push!(fpr, fp / nneg)
        end
    end
    return fpr, tpr
end

"Área bajo la curva (regla del trapecio)."
auc(fpr, tpr) = sum((fpr[2:end] .- fpr[1:end-1]) .* (tpr[2:end] .+ tpr[1:end-1]) ./ 2)

"Calcula la curva ROC y el AUC. Devuelve (fpr, tpr, auc)."
function roc_auc(y_true::AbstractVector, scores::AbstractVector)
    fpr, tpr = curva_roc(y_true, scores)
    return (fpr = fpr, tpr = tpr, auc = auc(fpr, tpr))
end

"Dibuja una o varias curvas ROC (los resultados de `roc_auc`)."
function graficar_roc(rocs...; nombres=nothing, titulo="Curva ROC")
    p = plot([0, 1], [0, 1], label="Azar", linestyle=:dash, color=:gray,
             xlabel="FPR (1 - especificidad)", ylabel="TPR (recall)", title=titulo,
             xlims=(0, 1), ylims=(0, 1), legend=:bottomright)
    for (i, r) in enumerate(rocs)
        nombre = isnothing(nombres) ? "Modelo $i" : nombres[i]
        plot!(p, r.fpr, r.tpr, label="$nombre (AUC = $(round(r.auc, digits=3)))", linewidth=2)
    end
    return p
end

# ----------------------------------------------------------------------------
# 4.3 VALIDACIÓN CRUZADA K-FOLD
# ----------------------------------------------------------------------------

"Reparte los índices 1:n en k grupos (folds) aleatorios de tamaño similar."
function crear_folds(n::Int, k::Int)
    idx = shuffle(1:n)
    return [idx[i:k:n] for i in 1:k]
end

"""
    validacion_cruzada_kfold(X, y, entrenar, evaluar; k=5)

  * `entrenar(Xtrain, ytrain)` devuelve el modelo entrenado.
  * `evaluar(modelo, Xtest, ytest)` devuelve un número (error, accuracy...).

Devuelve (media, std, resultados) de la métrica en los k folds. Ejemplo:

    validacion_cruzada_kfold(X, y,
        (Xt, yt) -> regresion_lineal(Xt, yt),
        (W, Xv, yv) -> mse(yv, predecir(Xv, W)); k=5)
"""
function validacion_cruzada_kfold(X, y, entrenar::Function, evaluar::Function; k::Int=5)
    n = length(y)
    folds = crear_folds(n, k)
    resultados = Float64[]
    for fold in folds
        tr = setdiff(1:n, fold)
        modelo = entrenar(_filas(X, tr), y[tr])
        push!(resultados, evaluar(modelo, _filas(X, fold), y[fold]))
    end
    return (media = mean(resultados), std = std(resultados), resultados = resultados)
end


#============================================================================
5. SOBREAJUSTE E INFRAAJUSTE
============================================================================#
# Aquí x es un VECTOR (una sola característica) y se amplía con potencias de x.
# Para generar datos y partirlos:
#     x, y = generar_datos(f, 80; ruido=0.3)
#     xtr, ytr, xval, yval = train_test_split(x, y, 0.7)

# ----------------------------------------------------------------------------
# 5.1 POLINOMIOS Y COMPLEJIDAD DEL MODELO
# ----------------------------------------------------------------------------

"Matriz de características polinómicas Φ = [1 x x² ... xᵈ]."
function crear_polinomio(x::AbstractVector, d::Int)
    Φ = ones(length(x), d + 1)
    for j in 1:d
        Φ[:, j+1] = x .^ j
    end
    return Φ
end

"Ajusta un polinomio de grado `d` por mínimos cuadrados (con Ridge si λ > 0). Devuelve W."
ajustar_polinomio(x::AbstractVector, y::AbstractVector, d::Int; λ::Real=0.0) =
    _minimos_cuadrados(crear_polinomio(x, d), y, λ)

"Evalúa el polinomio de coeficientes W en los puntos x."
predecir_polinomio(x::AbstractVector, W::AbstractVector) = crear_polinomio(x, length(W) - 1) * W

# ----------------------------------------------------------------------------
# 5.2 CURVAS DE APRENDIZAJE, COMPLEJIDAD Y REGULARIZACIÓN
# ----------------------------------------------------------------------------
# Todas devuelven errores MSE de entrenamiento y de validación.

function _errores_polinomio(x_tr, y_tr, x_val, y_val, d; λ=0.0)
    W = ajustar_polinomio(x_tr, y_tr, d; λ=λ)
    return mse(y_tr, predecir_polinomio(x_tr, W)), mse(y_val, predecir_polinomio(x_val, W))
end

"""
Error en función del tamaño del conjunto de entrenamiento (grado `d` fijo).
Se entrena con subconjuntos aleatorios de x_train de los tamaños indicados.
"""
function curva_aprendizaje(x_train, y_train, x_val, y_val, d::Int; tamaños=nothing, λ::Real=0.0)
    n = length(x_train)
    if isnothing(tamaños)
        tamaños = unique(clamp.(ceil.(Int, n .* (0.1:0.1:1.0)), 2, n))
    end
    orden = randperm(n)
    et = Float64[]
    ev = Float64[]
    for m in tamaños
        idx = orden[1:m]
        e_tr, e_val = _errores_polinomio(x_train[idx], y_train[idx], x_val, y_val, d; λ=λ)
        push!(et, e_tr)
        push!(ev, e_val)
    end
    return (tamaños = collect(tamaños), errores_train = et, errores_val = ev)
end

"Error en función del grado del polinomio (complejidad del modelo)."
function curva_complejidad(x_train, y_train, x_val, y_val; grados=1:12)
    et = Float64[]
    ev = Float64[]
    for d in grados
        e_tr, e_val = _errores_polinomio(x_train, y_train, x_val, y_val, d)
        push!(et, e_tr)
        push!(ev, e_val)
    end
    return (grados = collect(grados), errores_train = et, errores_val = ev)
end

"Error en función de la regularización λ (grado `d` fijo)."
function curva_regularizacion(x_train, y_train, x_val, y_val, d::Int; lambdas=10.0 .^ (-6:0.5:2))
    et = Float64[]
    ev = Float64[]
    for λ in lambdas
        e_tr, e_val = _errores_polinomio(x_train, y_train, x_val, y_val, d; λ=λ)
        push!(et, e_tr)
        push!(ev, e_val)
    end
    return (lambdas = collect(lambdas), errores_train = et, errores_val = ev)
end

"""
Dibuja error de entrenamiento y de validación frente a `x`, marcando el mínimo
del error de validación. Las funciones `graficar_curva_*` son atajos para
los resultados de las funciones anteriores.
"""
function graficar_errores(x, err_train, err_val; xlabel="", titulo="Errores",
                          xscale::Symbol=:identity)
    p = plot(x, err_train, label="Entrenamiento", linewidth=2, marker=:circle, markersize=3,
             xlabel=xlabel, ylabel="MSE", title=titulo, xscale=xscale, legend=:topright)
    plot!(p, x, err_val, label="Validación", linewidth=2, marker=:square, markersize=3)
    vline!(p, [x[argmin(err_val)]], label="Mínimo validación", color=:gray, linestyle=:dot)
    return p
end

graficar_curva_aprendizaje(r) = graficar_errores(r.tamaños, r.errores_train, r.errores_val;
    xlabel="Tamaño del conjunto de entrenamiento", titulo="Curva de aprendizaje")
graficar_curva_complejidad(r) = graficar_errores(r.grados, r.errores_train, r.errores_val;
    xlabel="Grado del polinomio", titulo="Complejidad del modelo")
graficar_curva_regularizacion(r) = graficar_errores(r.lambdas, r.errores_train, r.errores_val;
    xlabel="λ", titulo="Regularización (Ridge)", xscale=:log10)

# ----------------------------------------------------------------------------
# 5.3 DIAGNÓSTICO Y SELECCIÓN DE MODELO
# ----------------------------------------------------------------------------

"""
Diagnóstico orientativo a partir de los errores de train y validación:
  * ambos por encima de `umbral_alto`             → :infraajuste
  * validación - train > `umbral_gap` (y alto)    → :sobreajuste
  * en otro caso                                  → :buen_ajuste
Los umbrales dependen de la escala de los datos: ajústalos a cada problema.
"""
function diagnosticar_ajuste(err_train::Real, err_val::Real; umbral_alto::Real=0.1, umbral_gap::Real=0.05)
    if err_train > umbral_alto && err_val > umbral_alto
        return :infraajuste, "Error alto en entrenamiento y validación"
    elseif err_val - err_train > umbral_gap && err_val > umbral_alto
        return :sobreajuste, "Gran diferencia entre entrenamiento y validación"
    else
        return :buen_ajuste, "Errores razonables y similares"
    end
end

"Grado de polinomio con menor error de validación."
function seleccionar_mejor_grado(x_train, y_train, x_val, y_val; grados=1:12)
    r = curva_complejidad(x_train, y_train, x_val, y_val; grados=grados)
    i = argmin(r.errores_val)
    return (mejor_grado = r.grados[i], mejor_error = r.errores_val[i], resultado = r)
end

"Valor de λ con menor error de validación (para un grado `d` dado)."
function seleccionar_mejor_lambda(x_train, y_train, x_val, y_val, d::Int; lambdas=10.0 .^ (-6:0.5:2))
    r = curva_regularizacion(x_train, y_train, x_val, y_val, d; lambdas=lambdas)
    i = argmin(r.errores_val)
    return (mejor_lambda = r.lambdas[i], mejor_error = r.errores_val[i], resultado = r)
end

# ----------------------------------------------------------------------------
# 5.4 SESGO Y VARIANZA
# ----------------------------------------------------------------------------

"""
    analisis_sesgo_varianza(f, d; n_train=30, n_experimentos=100, ruido=0.3, λ=0.0)

Repite `n_experimentos` veces: genera un conjunto de entrenamiento nuevo a partir
de la función verdadera `f`, ajusta un polinomio de grado `d` y lo evalúa en una
rejilla de puntos. Devuelve el sesgo², la varianza, el ruido² (error irreducible)
y las predicciones de todos los experimentos (para `graficar_sesgo_varianza`).
"""
function analisis_sesgo_varianza(f::Function, d::Int; n_train::Int=30, n_experimentos::Int=100,
                                 ruido::Real=0.3, λ::Real=0.0, intervalo=(-1.0, 1.0))
    x_test = collect(range(intervalo[1], intervalo[2], length=100))
    y_real = f.(x_test)
    P = zeros(n_experimentos, length(x_test))

    for e in 1:n_experimentos
        x, y = generar_datos(f, n_train; ruido=ruido, intervalo=intervalo)
        W = ajustar_polinomio(x, y, d; λ=λ)
        P[e, :] = predecir_polinomio(x_test, W)
    end

    pred_media = vec(mean(P, dims=1))
    return (sesgo2 = mean((pred_media .- y_real) .^ 2),
            varianza = mean(var(P, dims=1)),
            ruido2 = float(ruido)^2,
            predicciones = P, x_test = x_test, y_real = y_real, pred_media = pred_media)
end

"Dibuja las predicciones de todos los experimentos, su media y la función real."
function graficar_sesgo_varianza(r; titulo="Sesgo y varianza")
    p = plot(xlabel="x", ylabel="y", title=titulo, legend=:topright)
    for e in 1:size(r.predicciones, 1)
        plot!(p, r.x_test, r.predicciones[e, :], color=:lightblue, alpha=0.3, label="")
    end
    plot!(p, r.x_test, r.pred_media, label="Predicción media", color=:red, linewidth=2)
    plot!(p, r.x_test, r.y_real, label="Función real", color=:black, linewidth=2, linestyle=:dash)
    return p
end


#============================================================================
6. EJEMPLOS
Cada función es una práctica corta que puedes ejecutar tal cual o copiar y modificar.
============================================================================#

"Manipulación de un DataFrame: faltantes, categóricas, normalización y partición."
function ejemplo_datos()
   # Dataframe de ejemplo con 10 filas y 3 características (2 numéricas y 1 categórica)
    df = DataFrame(
        edad    = [25, 30, 35, 40, 28, 33, 45, 29, 52, 38],
        salario = [30000, 45000, 50000, 60000, 35000, 48000, 65000, 38000, 72000, 55000],
        ciudad  = ["Madrid", "Barcelona", "Madrid", "Sevilla", "Barcelona",
                   "Madrid", "Sevilla", "Barcelona", "Madrid", "Sevilla"],
        compro  = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    allowmissing!(df, :salario)          # permitimos valores faltantes en la columna salario
    df[2, :salario] = missing            # introducimos un valor faltante

    resumen(df)
    display(graficar_histogramas(df))
    display(graficar_correlaciones(df))

    Random.seed!(1)
    Xtr, ytr, Xte, yte = preparar_datos(df, :compro; faltantes=:mean,
                                        categoricas=:one_hot, normalizacion=:z_score,
                                        train_ratio=0.7)
    println("\nXtrain: ", size(Xtr), "   Xtest: ", size(Xte))
    println("Medias de Xtrain (≈ 0): ", round.(vec(mean(Xtr, dims=1)), digits=3))
end

"Regresión lineal: solución analítica, descenso de gradiente y superficie de pérdida."
function ejemplo_regresion()
    Random.seed!(1)
    x, y = generar_datos(t -> 2t + 1, 100; ruido=0.5)
    xtr, ytr, xte, yte = train_test_split(x, y, 0.8)

    W = regresion_lineal(xtr, ytr)
    y_pred = predecir(xte, W)
    println("W (analítica) = ", round.(W, digits=3))
    println("Test → MSE = ", round(mse(yte, y_pred), digits=4),
            " | MAE = ", round(mae(yte, y_pred), digits=4),
            " | R² = ", round(r2_score(yte, y_pred), digits=4))
    display(graficar_ajuste(xtr, ytr, t -> predecir(t, W); nombres=["Mínimos cuadrados"]))
    display(graficar_predicciones(yte, y_pred))

    # Descenso de gradiente: batch frente a estocástico
    Xtr = reshape(xtr, :, 1)
    W_b, h_b = bgd_mse(Xtr, ytr; lr=0.1, max_iter=300)
    W_s, h_s = sgd_mse(Xtr, ytr; lr=0.01, epochs=50)
    println("W (batch GD) = ", round.(W_b, digits=3), " | W (SGD) = ", round.(W_s, digits=3))
    display(graficar_convergencia(h_b, h_s; nombres=["Batch GD", "SGD"]))
    display(graficar_superficie_perdida(Xtr, ytr, h_b))
end

"Clasificación binaria: perceptrón, regresión logística, matriz de confusión y ROC."
function ejemplo_clasificacion()
    Random.seed!(2)
    X, y = generar_clasificacion_2d(300; separacion=2.5)
    Xtr, ytr, Xte, yte = train_test_split(X, y, 0.75)
    display(graficar_datos(Xtr, ytr; titulo="Datos de entrenamiento"))

    W_p, errores = perceptron(Xtr, ytr; max_iter=50)
    W_l, hist = regresion_logistica_gd(Xtr, ytr; lr=0.1, max_iter=500)
    println("Perceptrón: errores en la última época = ", errores[end])

    pred = clasificar(Xte, W_l)
    matriz = matriz_confusion(yte, pred)
    println(matriz)
    println(metricas_por_clase(matriz))
    display(graficar_matriz_confusion(matriz))
    display(graficar_frontera(Xte, yte, Xm -> clasificar(Xm, W_l); titulo="Regresión logística (test)"))
    display(graficar_convergencia(hist; nombres=["Logística"], escala_log=false))

    r_log = roc_auc(yte, predecir(Xte, W_l))
    r_per = roc_auc(yte, predecir(Xte, W_p))
    println("AUC logística = ", round(r_log.auc, digits=3), " | AUC perceptrón = ", round(r_per.auc, digits=3))
    display(graficar_roc(r_log, r_per; nombres=["Logística", "Perceptrón"]))

    # Validación cruzada de la accuracy de la regresión logística
    cv = validacion_cruzada_kfold(X, y,
            (Xt, yt) -> regresion_logistica_gd(Xt, yt; max_iter=300)[1],
            (W, Xv, yv) -> accuracy(yv, clasificar(Xv, W)); k=5)
    println("Accuracy CV (5 folds) = ", round(cv.media, digits=3), " ± ", round(cv.std, digits=3))
end

"Sobreajuste: complejidad, curvas de aprendizaje, regularización y sesgo-varianza."
function ejemplo_sobreajuste()
    f(t) = sin(3t)
    x, y = generar_datos(f, 60; ruido=0.3, seed=3)
    xtr, ytr, xval, yval = train_test_split(x, y, 0.7)

    # Modelos de distinta complejidad
    grados = [1, 4, 12]
    Ws = [ajustar_polinomio(xtr, ytr, d) for d in grados]
    modelos = [t -> predecir_polinomio(t, W) for W in Ws]
    display(graficar_ajuste(xtr, ytr, modelos...; nombres=["grado $d" for d in grados],
                            titulo="Infraajuste vs sobreajuste"))

    # Error frente a complejidad
    cc = curva_complejidad(xtr, ytr, xval, yval; grados=1:12)
    display(graficar_curva_complejidad(cc))
    mejor = seleccionar_mejor_grado(xtr, ytr, xval, yval; grados=1:12)
    println("Mejor grado según validación: ", mejor.mejor_grado)
    println("Diagnóstico grado 12: ",
            diagnosticar_ajuste(cc.errores_train[12], cc.errores_val[12]))

    # Curva de aprendizaje y regularización con el grado alto
    display(graficar_curva_aprendizaje(curva_aprendizaje(xtr, ytr, xval, yval, 12)))
    reg = seleccionar_mejor_lambda(xtr, ytr, xval, yval, 12)
    println("Mejor λ para grado 12: ", reg.mejor_lambda)
    display(graficar_curva_regularizacion(reg.resultado))

    # Sesgo y varianza según la complejidad
    for d in (1, 3, 10)
        r = analisis_sesgo_varianza(f, d)
        println("Grado $d → sesgo² = ", round(r.sesgo2, digits=4),
                " | varianza = ", round(r.varianza, digits=4))
        display(graficar_sesgo_varianza(r; titulo="Grado $d"))
    end
end

println("ML.jl cargado.")