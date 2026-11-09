#============================================================================

FUNDAMENTOS DE MACHINE LEARNING EN JULIA PURO

Basado en el curso IAIC 2025-26 - Fundamentos de ML
https://www.cs.us.es/cursos/iaic-2025/Temas/Fundamentos_de_ML/

Parte 1: Datos y Características

1.1 Estructura Dataset para organizar datos de ML
1.2 Estadísticas básicas y Normalización
1.3 Visualización de datos bidimensionales

Parte 2: Modelos y Espacios de Hipótesis

2.1 Hipótesis lineales
2.2 Regresión lineal (solución analítica) y Ridge (con regularización L2)
2.3 Clasificador Lineal y Perceptrón
2.4 Métricas R², MSE, MAE
2.5 Visualizaciones de modelos y predicciones

Parte 3: Funciones de Pérdida y Optimización

3.1 Regresión: pérdida cuadrática, absoluta
3.2 Clasificación: pérdida 0/1, bisagra, logística, entropía cruzada
3.3 Riesgo empírico
3.4 Descenso de gradiente (batch y estocástico)
3.5 Regresión logística con optimización
3.6 Visualización y Análisis de funciones de pérdida y convergencia

Parte 4: Evaluación y Métricas

4.1 Matriz de confusión (binaria y multiclase)
4.2 Métricas: Accuracy, Precision, Recall, F1, Specificity, FPR
4.3 Curva ROC y cálculo de AUC
4.4 Validación cruzada K-fold
4.5 División train/test

Parte 5: Sobreajuste e Infraajuste

5.1 Generación de datos con ruido
5.2 Polinomios y complejidad del modelo
5.3 Curvas de aprendizaje
5.4 Detección automática de sobreajuste/infraajuste
5.5 Regularización (Ridge)
5.6 Análisis de sesgo-varianza
============================================================================#

using Statistics
using LinearAlgebra
using Random
using Plots

#============================================================================
1. DATOS Y CARACTERÍSTICAS

1.1 Estructura Dataset para organizar datos de ML
1.2 Estadísticas básicas (media y varianza) y Normalización Z-score y Min-Max
1.3 Funciones kernel (polinómica y RBF) para transformación de características
1.4 Visualización de datos bidimensionales
============================================================================#


# ============================================================================
# 1.1 ESTRUCTURA DE DATOS BÁSICA
# ============================================================================

"""
Estructura para representar un conjunto de datos de ML
- X: matriz de características reales (m × n) donde m es el tamaño de muestra y n el número de características
- y: vector de etiquetas (m elementos)
"""
struct Dataset{T<:Real, S}
    X::Matrix{T}        # Características (m × n)
    y::Vector{S}        # Etiquetas (m elementos)
    feature_names::Vector{String}
    
    function Dataset(X::Matrix{T}, y::Vector{S}, feature_names=nothing) where {T<:Real, S}
        m, n = size(X)
        length(y) == m || error("Las dimensiones de X e Y no coinciden: $(size(X)) vs $(length(y))")
        length(feature_names) == n || error("El número de nombres de características no coincide con el número de X")

        if isnothing(feature_names)
            feature_names = ["x$i" for i in 1:n]
        end
        
        new{T,S}(X, y, feature_names)
    end
end

# Propiedades útiles
tamaño_muestra(d::Dataset) = size(d.X, 1)
num_caracteristicas(d::Dataset) = size(d.X, 2)
Base.length(d::Dataset) = tamaño_muestra(d)

"""
Muestra información sobre el dataset
"""
function info(d::Dataset)
    println("Dataset de ML")
    println("─"^50)
    println("Número de muestras (m): ", length(d))
    println("Número de características (n): ", num_caracteristicas(d))
    println("Ratio m/n: ", round(tamaño_muestra(d)/num_caracteristicas(d), digits=2))
    println("Tipo de etiquetas: ", typeof(d.y[1]))
    println("Características: ", join(d.feature_names, ", "))
end

# ============================================================================
# 1.2 ESTADÍSTICAS BÁSICAS Y NORMALIZACIÓN
# ============================================================================

"""
Calcula estadísticas básicas de las características (media y varianza muestrales)
Relacionado con: Principio de Máxima Verosimilitud
"""
function estadisticas_basicas(X::Matrix)
    μ = mean(X, dims=1)  # Media por columna (característica)
    σ² = var(X, dims=1, corrected=true)  # Varianza muestral
    
    return (media=vec(μ), varianza=vec(σ²))
end

"""
Normalización Z-score de características
Transforma cada característica para tener media 0 y varianza 1
"""
function normalizar_zscore(X::Matrix)
    stats = estadisticas_basicas(X)
    X_norm = (X .- stats.media') ./ sqrt.(stats.varianza')
    return X_norm, stats
end

"""
Normalización Min-Max (escala a [0,1])
"""
function normalizar_minmax(X::Matrix)
    mins = minimum(X, dims=1)
    maxs = maximum(X, dims=1)
    X_norm = (X .- mins) ./ (maxs .- mins)
    return X_norm, (mins=mins, maxs=maxs)
end

# ============================================================================
# 1.3 VISUALIZACIÓN DE DATOS BIDIMENSIONALES
# ============================================================================

"""
Visualiza un dataset de regresión bidimensional (2 características)
Muestra scatter plot de los puntos coloreados por el valor de y
"""
function mostrar_dataset_regresion(d::Dataset; titulo="Dataset de Regresión")
    if num_caracteristicas(d) != 2
        error("Esta función solo funciona con datasets bidimensionales (n=2)")
    end
    
    scatter(d.X[:,1], d.X[:,2], 
            marker_z=d.y,
            color=:viridis,
            xlabel=d.feature_names[1],
            ylabel=d.feature_names[2],
            title=titulo,
            label="Datos",
            colorbar_title="y",
            markersize=6,
            alpha=0.7,
            legend=:topright)
end

"""
Visualiza un dataset de clasificación bidimensional (2 características)
Muestra scatter plot con diferentes colores/formas para cada clase
"""
function mostrar_dataset_clasificacion(d::Dataset; titulo="Dataset de Clasificación")
    if num_caracteristicas(d) != 2
        error("Esta función solo funciona con datasets bidimensionales (n=2)")
    end
    
    # Obtener clases únicas
    clases = unique(d.y)
    
    # Crear el plot
    p = plot(xlabel=d.feature_names[1],
             ylabel=d.feature_names[2],
             title=titulo,
             legend=:topright)
    
    # Colores y marcadores para diferentes clases
    colores = [:blue, :red, :green, :orange, :purple, :yellow]
    marcadores = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5]
    
    # Dibujar cada clase
    for (i, clase) in enumerate(clases)
        indices = d.y .== clase
        scatter!(p, d.X[indices, 1], d.X[indices, 2],
                label="Clase $clase",
                color=colores[mod1(i, length(colores))],
                markershape=marcadores[mod1(i, length(marcadores))],
                markersize=6,
                alpha=0.7)
    end
    
    return p
end

"""
Visualiza las características del dataset de forma automática
Detecta si es clasificación o regresión basándose en el tipo de y
"""
function mostrar_dataset(d::Dataset; titulo="Dataset")
    if num_caracteristicas(d) != 2
        error("Esta función solo funciona con datasets bidimensionales (n=2)")
    end
    
    # Detectar si es clasificación (etiquetas discretas) o regresión
    if eltype(d.y) <: Integer && length(unique(d.y)) < 20
        mostrar_dataset_clasificacion(d, titulo=titulo)
    else
        mostrar_dataset_regresion(d, titulo=titulo)
    end
end

"""
Visualiza la distribución de cada característica con histogramas
"""
function mostrar_histogramas(d::Dataset; titulo="Distribución de Características")
    n = num_caracteristicas(d)
    
    if n == 2
        p1 = histogram(d.X[:,1], 
                      xlabel=d.feature_names[1],
                      ylabel="Frecuencia",
                      title="Distribución $(d.feature_names[1])",
                      legend=false,
                      bins=20,
                      color=:skyblue,
                      alpha=0.7)
        
        p2 = histogram(d.X[:,2], 
                      xlabel=d.feature_names[2],
                      ylabel="Frecuencia",
                      title="Distribución $(d.feature_names[2])",
                      legend=false,
                      bins=20,
                      color=:lightcoral,
                      alpha=0.7)
        
        plot(p1, p2, layout=(1,2), size=(800,300), plot_title=titulo)
    else
        error("Esta función está optimizada para datasets bidimensionales")
    end
end

"""
Visualización completa del dataset: scatter plot + histogramas
"""
function mostrar_dataset_completo(d::Dataset; titulo="Análisis del Dataset")
    if num_caracteristicas(d) != 2
        error("Esta función solo funciona con datasets bidimensionales (n=2)")
    end
    
    # Plot principal
    p1 = mostrar_dataset(d, titulo="Datos")
    
    # Histogramas
    p2 = histogram(d.X[:,1], 
                  xlabel=d.feature_names[1],
                  ylabel="Frecuencia",
                  title="Distribución $(d.feature_names[1])",
                  legend=false,
                  bins=20,
                  color=:skyblue,
                  alpha=0.7)
    
    p3 = histogram(d.X[:,2], 
                  xlabel=d.feature_names[2],
                  ylabel="Frecuencia",
                  title="Distribución $(d.feature_names[2])",
                  legend=false,
                  bins=20,
                  color=:lightcoral,
                  alpha=0.7)
    
    # Distribución de y
    if eltype(d.y) <: Integer && length(unique(d.y)) < 20
        # Para clasificación: gráfico de barras
        clases = unique(d.y)
        conteos = [count(==(c), d.y) for c in clases]
        p4 = bar(string.(clases), conteos,
                xlabel="Clase",
                ylabel="Frecuencia",
                title="Distribución de Clases",
                legend=false,
                color=:lightgreen,
                alpha=0.7)
    else
        # Para regresión: histograma de y
        p4 = histogram(d.y,
                      xlabel="y",
                      ylabel="Frecuencia",
                      title="Distribución de y",
                      legend=false,
                      bins=20,
                      color=:lightgreen,
                      alpha=0.7)
    end
    
    plot(p1, p2, p3, p4, 
         layout=@layout([a{0.5w} grid(3,1)]),
         size=(1000,600),
         plot_title=titulo)
end

#============================================================================
2. MODELOS Y ESPACIOS DE HIPÓTESIS

2.1 Hipótesis lineales
2.2 Regresión lineal (solución analítica) y Ridge (con regularización L2)
2.3 Regresión polinómica
2.4 Clasificador Lineal y Perceptrón
2.5 Métricas R², MSE, MAE
2.6 Visualizaciones de modelos y predicciones
============================================================================#


# ============================================================================
# 2.1 HIPÓTESIS LINEALES
# ============================================================================

"""
Hipótesis lineal: h_W(x) = W^T x
donde W ∈ ℝⁿ es el vector de pesos

Para problemas de regresión: h: ℝⁿ → ℝ
Para problemas de clasificación: aplicamos sign(h(x))
"""
struct HipotesisLineal{T<:Real}
    W::Vector{T}
    bias::Bool
    
    function HipotesisLineal(W::Vector{T}, bias::Bool=true) where T<:Real
        new{T}(W, bias)
    end
end

"""
Predice usando la hipótesis lineal
"""
function (h::HipotesisLineal)(X::Matrix)
    if h.bias
        # Añade columna de unos para el término de sesgo
        X_aug = hcat(ones(size(X,1)), X)
    else
        X_aug = X
    end
    return X_aug * h.W
end

(h::HipotesisLineal)(x::Vector) = h(x')'[1]

"""
Dimensión del espacio de hipótesis lineal
Para H = {h: ℝⁿ → ℝ : h(x) = Wx}, d_eff(H) = n
"""
dimension_efectiva(h::HipotesisLineal) = length(h.W)

# ============================================================================
# 2.2 REGRESIÓN LINEAL (Método de Mínimos Cuadrados)
# ============================================================================

"""
Ajusta una hipótesis lineal que minimiza el error cuadrático

min_W ∑ᵢ (yⁱ - W^T xⁱ)²

Solución analítica: W = (X^T X)⁻¹ X^T y
"""
function regresion_lineal(X::Matrix, y::Vector; bias::Bool=true)
    if bias
        X_aug = hcat(ones(size(X,1)), X)
    else
        X_aug = X
    end
    
    # Solución de mínimos cuadrados
    W = (X_aug' * X_aug) \ (X_aug' * y)
    
    return HipotesisLineal(W, bias)
end

"""
Regresión lineal con regularización Ridge (L2)

min_W ∑ᵢ (yⁱ - W^T xⁱ)² + λ||W||²

Solución: W = (X^T X + λI)⁻¹ X^T y
"""
function regresion_ridge(X::Matrix, y::Vector, λ::Float64=1.0; bias::Bool=true)
    if bias
        X_aug = hcat(ones(size(X,1)), X)
    else
        X_aug = X
    end
    
    n = size(X_aug, 2)
    W = (X_aug' * X_aug + λ * I(n)) \ (X_aug' * y)
    
    return HipotesisLineal(W, bias)
end

# ============================================================================
# 2.3 CLASIFICACIÓN LINEAL
# ============================================================================

"""
Clasificador lineal binario
Aplica sign(W^T x) para clasificar en {-1, +1}
"""
struct ClasificadorLineal{T<:Real}
    h::HipotesisLineal{T}
end

"""
Clasifica datos (devuelve -1 o +1)
"""
function clasificar(c::ClasificadorLineal, X::Matrix)
    scores = c.h(X)
    return sign.(scores)
end

clasificar(c::ClasificadorLineal, x::Vector) = clasificar(c, reshape(x, 1, :))[1]

"""
Entrena clasificador lineal usando método de Perceptrón
(Versión simple iterativa)
"""
function perceptron(X::Matrix, y::Vector, max_iter::Int=100, lr::Float64=0.1)
    m, n = size(X)
    X_aug = hcat(ones(m), X)
    W = zeros(n + 1)
    
    for iter in 1:max_iter
        errores = 0
        for i in 1:m
            pred = sign(dot(W, X_aug[i, :]))
            if pred != y[i]
                W .+= lr * y[i] * X_aug[i, :]
                errores += 1
            end
        end
        if errores == 0
            break
        end
    end
    
    return ClasificadorLineal(HipotesisLineal(W, true))
end

# ============================================================================
# 2.4 MÉTRICAS DE EVALUACIÓN
# ============================================================================

"""
Calcula coeficiente de determinación R²
R² = 1 - SS_res/SS_tot
donde SS_res = ∑(yⁱ - ŷⁱ)² y SS_tot = ∑(yⁱ - ȳ)²
"""
function r2_score(y_true::Vector, y_pred::Vector)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res/ss_tot
end

"""
Error cuadrático medio (MSE)
"""
mse(y_true::Vector, y_pred::Vector) = mean((y_true .- y_pred).^2)

"""
Error absoluto medio (MAE)
"""
mae(y_true::Vector, y_pred::Vector) = mean(abs.(y_true .- y_pred))

"""
Precisión de clasificación
"""
precision_clasificacion(y_true::Vector, y_pred::Vector) = mean(y_true .== y_pred)

# ============================================================================
# 2.5 VISUALIZACIONES
# ============================================================================

"""
Visualiza regresión lineal en 2D (una característica)
Muestra datos y línea de regresión
"""
function visualizar_regresion_1d(X::Matrix, y::Vector, h::HipotesisLineal; titulo="Regresión Lineal")
    if size(X, 2) != 1
        error("Esta función solo funciona con 1 característica")
    end
    
    # Ordenar para visualización
    idx = sortperm(X[:, 1])
    X_sorted = X[idx, :]
    y_sorted = y[idx]
    y_pred_sorted = h(X_sorted)
    
    scatter(X[:, 1], y,
            label="Datos",
            xlabel="x",
            ylabel="y",
            title=titulo,
            markersize=5,
            alpha=0.6,
            legend=:topright)
    
    plot!(X_sorted[:, 1], y_pred_sorted,
          label="Predicción",
          linewidth=2,
          color=:red)
end

"""
Visualiza comparación entre predicciones y valores reales
"""
function visualizar_predicciones(y_true::Vector, y_pred::Vector; titulo="Predicciones vs Valores Reales")
    # Línea ideal y = x
    y_min, y_max = minimum([y_true; y_pred]), maximum([y_true; y_pred])
    
    scatter(y_true, y_pred,
            xlabel="Valores Reales",
            ylabel="Predicciones",
            title=titulo,
            label="Predicciones",
            markersize=5,
            alpha=0.6,
            legend=:bottomright)
    
    plot!([y_min, y_max], [y_min, y_max],
          label="Predicción perfecta",
          linewidth=2,
          linestyle=:dash,
          color=:red)
end

"""
Visualiza clasificador lineal en 2D
Muestra puntos de datos, frontera de decisión y regiones
"""
function visualizar_clasificador(X::Matrix, y::Vector, c::ClasificadorLineal; titulo="Clasificador Lineal")
    if size(X, 2) != 2
        error("Esta función solo funciona con 2 características")
    end
    
    # Crear grid para frontera de decisión
    x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
    
    x1_grid = range(x1_min, x1_max, length=100)
    x2_grid = range(x2_min, x2_max, length=100)
    
    # Evaluar clasificador en el grid
    Z = zeros(length(x2_grid), length(x1_grid))
    for (i, x2) in enumerate(x2_grid)
        for (j, x1) in enumerate(x1_grid)
            X_point = reshape([x1, x2], 1, 2)
            Z[i, j] = clasificar(c, X_point)[1]
        end
    end
    
    # Plotear contorno de decisión
    p = contour(x1_grid, x2_grid, Z,
                levels=[-0.5, 0.5],
                linewidth=2,
                color=:black,
                linestyle=:dash,
                xlabel="x₁",
                ylabel="x₂",
                title=titulo,
                legend=false)
    
    # Plotear regiones
    contourf!(p, x1_grid, x2_grid, Z,
              levels=[-1, 0, 1],
              alpha=0.2,
              color=[:blue, :red])
    
    # Plotear puntos
    for clase in unique(y)
        idx = y .== clase
        scatter!(p, X[idx, 1], X[idx, 2],
                label="Clase $clase",
                markersize=6,
                alpha=0.7,
                color=clase > 0 ? :red : :blue,
                markerstrokewidth=1,
                markerstrokecolor=:black)
    end
    
    return p
end

"""
Visualiza comparación entre diferentes modelos de regresión
"""
function comparar_modelos(X::Matrix, y::Vector, modelos::Vector; nombres::Vector{String}=String[], titulo="Comparación de Modelos")
    if size(X, 2) != 1
        error("Esta función solo funciona con 1 característica")
    end
    
    if isempty(nombres)
        nombres = ["Modelo $i" for i in 1:length(modelos)]
    end
    
    # Crear puntos para curva suave
    x_min, x_max = minimum(X), maximum(X)
    x_range = range(x_min - 0.5, x_max + 0.5, length=200)
    X_range = reshape(collect(x_range), :, 1)
    
    # Plotear datos
    p = scatter(X[:, 1], y,
                label="Datos",
                xlabel="x",
                ylabel="y",
                title=titulo,
                markersize=5,
                alpha=0.6,
                legend=:topright)
    
    # Plotear cada modelo
    colores = [:red, :blue, :green, :orange, :purple]
    for (i, modelo) in enumerate(modelos)
        y_pred_range = modelo(X_range)
        plot!(p, x_range, y_pred_range,
              label=nombres[i],
              linewidth=2,
              color=colores[mod1(i, length(colores))])
    end
    
    return p
end

"""
Visualiza métricas de rendimiento para múltiples modelos
"""
function visualizar_metricas(nombres::Vector{String}, r2::Vector{Float64}, mse_vals::Vector{Float64})
    p1 = bar(nombres, r2,
             title="Coeficiente R²",
             ylabel="R²",
             legend=false,
             color=:skyblue,
             ylims=(0, 1))
    
    p2 = bar(nombres, mse_vals,
             title="Error Cuadrático Medio",
             ylabel="MSE",
             legend=false,
             color=:lightcoral)
    
    plot(p1, p2, layout=(1, 2), size=(800, 300))
end

#============================================================================
3. FUNCIONES DE PÉRDIDA Y OPTIMIZACIÓN

3.1 Regresión: pérdida cuadrática, absoluta
3.2 Clasificación: pérdida 0/1, bisagra, logística, entropía cruzada
3.3 Riesgo empírico
3.4 Descenso de gradiente (batch y estocástico)
3.5 Regresión logística con optimización
3.6 Visualización y Análisis de funciones de pérdida y convergencia
============================================================================#


# ============================================================================
# 3.1 FUNCIONES DE PÉRDIDA PARA REGRESIÓN
# ============================================================================

"""
Pérdida de Error Cuadrático
L(y, ŷ) = (y - ŷ)²
"""
perdida_cuadratica(y::Real, y_pred::Real) = (y - y_pred)^2
perdida_cuadratica(y::Vector, y_pred::Vector) = (y .- y_pred).^2

"""
Pérdida de Error Cuadrático Medio (MSE)
L̂(h|D) = (1/m) ∑ᵢ (yⁱ - h(xⁱ))²
"""
mse_perdida(y::Vector, y_pred::Vector) = mean((y .- y_pred).^2)

"""
Pérdida de Error Absoluto
L(y, ŷ) = |y - ŷ|
"""
perdida_absoluta(y::Real, y_pred::Real) = abs(y - y_pred)
perdida_absoluta(y::Vector, y_pred::Vector) = abs.(y .- y_pred)

"""
Pérdida de Error Absoluto Medio (MAE)
"""
mae_perdida(y::Vector, y_pred::Vector) = mean(abs.(y .- y_pred))

# ============================================================================
# 3.2 FUNCIONES DE PÉRDIDA PARA CLASIFICACIÓN BINARIA
# ============================================================================

"""
Pérdida 0/1
L((x,y), h) = 1 si y ≠ ŷ, 0 en caso contrario
"""
function perdida_01(y::Vector{<:Integer}, y_pred::Vector{<:Real})
    y_pred_class = sign.(y_pred)
    return Float64.(y .!= y_pred_class)
end

"""
Pérdida Bisagra (Hinge Loss)
L((x,y), h) = max{0, 1 - y·h(x)}

Usada en SVM. Es convexa pero no diferenciable en y·h(x) = 1
"""
perdida_bisagra(y::Real, score::Real) = max(0, 1 - y * score)
perdida_bisagra(y::Vector, scores::Vector) = max.(0, 1 .- y .* scores)

"""
Pérdida Logística
L((x,y), h) = log(1 + exp(-y·h(x)))

Convexa y diferenciable. Usada en regresión logística.
"""
perdida_logistica(y::Real, score::Real) = log(1 + exp(-y * score))
perdida_logistica(y::Vector, scores::Vector) = log.(1 .+ exp.(-y .* scores))

"""
Pérdida de Entropía Cruzada Binaria (Cross-Entropy)
L(y, p) = -y·log(p) - (1-y)·log(1-p)
donde y ∈ {0,1} y p es la probabilidad predicha
"""
function perdida_entropia_cruzada(y::Real, prob::Real)
    # Clip para evitar log(0)
    p = clamp(prob, 1e-10, 1 - 1e-10)
    return -y * log(p) - (1 - y) * log(1 - p)
end

function perdida_entropia_cruzada(y::Vector, probs::Vector)
    [perdida_entropia_cruzada(y[i], probs[i]) for i in 1:length(y)]
end

# ============================================================================
# 3.3 RIESGO EMPÍRICO
# ============================================================================

"""
Riesgo Empírico
L̂(h|D) = (1/m) ∑ᵢ L((xⁱ, yⁱ), h)

El riesgo empírico es el promedio de la pérdida sobre el conjunto de datos
"""
function riesgo_empirico(perdidas::Vector{<:Real})
    return mean(perdidas)
end

"""
Calcula el riesgo empírico para una hipótesis dada
"""
function calcular_riesgo(h, X::Matrix, y::Vector, perdida_fn)
    y_pred = h(X)
    perdidas = perdida_fn(y, y_pred)
    return riesgo_empirico(perdidas)
end

# ============================================================================
# 3.4 DESCENSO DE GRADIENTE PARA OPTIMIZACIÓN
# ============================================================================

"""
Estructura para almacenar el historial de optimización
"""
struct HistorialOptimizacion
    perdidas::Vector{Float64}
    pesos::Vector{Vector{Float64}}
end

"""
Descenso de Gradiente para regresión lineal con pérdida MSE

∇L(W) = -(2/m) X^T (y - Xw)

Actualización: W ← W - η·∇L(W)
"""
function bgd_mse(X::Matrix, y::Vector; 
                                 bias::Bool=true,
                                 lr::Float64=0.01, 
                                 max_iter::Int=1000,
                                 tol::Float64=1e-6)
    if bias
        X_aug = hcat(ones(size(X,1)), X)
    else
        X_aug = X
    end
    
    m, n = size(X_aug)
    W = zeros(n)
    
    historial = HistorialOptimizacion(Float64[], Vector{Float64}[])
    
    for iter in 1:max_iter
        # Predicción
        y_pred = X_aug * W
        
        # Pérdida actual
        perdida = mse_perdida(y, y_pred)
        push!(historial.perdidas, perdida)
        push!(historial.pesos, copy(W))
        
        # Gradiente
        grad = -(2/m) * X_aug' * (y - y_pred)
        
        # Actualización
        W_new = W - lr * grad
        
        # Criterio de parada
        if norm(W_new - W) < tol
            W = W_new
            break
        end
        
        W = W_new
    end
    
    return W, historial
end

"""
Descenso de Gradiente Estocástico (SGD)
Actualiza pesos usando un solo ejemplo a la vez
"""
function sgd_mse(X::Matrix, y::Vector;
                 bias::Bool=true,
                 lr::Float64=0.01,
                 epochs::Int=100)
    if bias
        X_aug = hcat(ones(size(X,1)), X)
    else
        X_aug = X
    end
    
    m, n = size(X_aug)
    W = zeros(n)
    
    historial = HistorialOptimizacion(Float64[], Vector{Float64}[])
    
    for epoch in 1:epochs
        # Mezclar datos
        indices = shuffle(1:m)
        
        for i in indices
            # Predicción para este ejemplo
            y_pred = dot(X_aug[i,:], W)
            
            # Gradiente para este ejemplo
            grad = -2 * X_aug[i,:] * (y[i] - y_pred)
            
            # Actualización
            W -= lr * grad
        end
        
        # Calcular pérdida de la época
        y_pred_all = X_aug * W
        perdida = mse_perdida(y, y_pred_all)
        push!(historial.perdidas, perdida)
        push!(historial.pesos, copy(W))
    end
    
    return W, historial
end

# ============================================================================
# 3.5 REGRESIÓN LOGÍSTICA CON OPTIMIZACIÓN
# ============================================================================

"""
Regresión Logística con Descenso de Gradiente
Minimiza la pérdida logística para clasificación binaria
"""
function regresion_logistica_gd(X::Matrix, y::Vector{<:Integer};
                                 bias::Bool=true,
                                 lr::Float64=0.1,
                                 max_iter::Int=1000)
    # Convertir etiquetas a {-1, 1} si no lo están
    y_transformed = [yi ∈ [0, -1] ? -1 : 1 for yi in y]
    
    if bias
        X_aug = hcat(ones(size(X,1)), X)
    else
        X_aug = X
    end
    
    m, n = size(X_aug)
    W = zeros(n)
    
    historial = HistorialOptimizacion(Float64[], Vector{Float64}[])
    
    for iter in 1:max_iter
        # Scores
        scores = X_aug * W
        
        # Pérdida logística
        perdidas = perdida_logistica(y_transformed, scores)
        perdida_total = mean(perdidas)
        push!(historial.perdidas, perdida_total)
        push!(historial.pesos, copy(W))
        
        # Gradiente de la pérdida logística
        # ∇L = -(1/m) ∑ᵢ yⁱ·xⁱ·σ(-yⁱ·scores)
        # donde σ(z) = 1/(1 + exp(-z))
        σ = 1 ./ (1 .+ exp.(y_transformed .* scores))
        ∇L = -(1/m) * X_aug' * (y_transformed .* σ)
        
        # Actualización
        W -= lr * ∇L
    end
    
    return W, historial
end

# ============================================================================
# 3.6 VISUALIZACIÓN Y ANÁLISIS
# ============================================================================

"""
Visualiza diferentes funciones de pérdida para regresión
"""
function visualizar_perdidas_regresion()
    errores = range(-3, 3, length=100)
    
    # Calcular pérdidas
    l_cuad = [perdida_cuadratica(0.0, e) for e in errores]
    l_abs = [perdida_absoluta(0.0, e) for e in errores]
    
    plot(errores, l_cuad,
         label="Cuadrática",
         xlabel="Error (y - ŷ)",
         ylabel="Pérdida",
         title="Comparación de Funciones de Pérdida (Regresión)",
         linewidth=2,
         legend=:topleft)
    
    plot!(errores, l_abs,
          label="Absoluta (L1)",
          linewidth=2)
end

"""
Visualiza diferentes funciones de pérdida para clasificación
"""
function visualizar_perdidas_clasificacion()
    # Margen = y·h(x) para y=1
    margenes = range(-2, 3, length=100)
    
    # Calcular pérdidas
    l_01 = [m < 0 ? 1.0 : 0.0 for m in margenes]
    l_bis = [perdida_bisagra(1.0, m) for m in margenes]
    l_log = [perdida_logistica(1.0, m) for m in margenes]
    
    plot(margenes, l_01,
         label="0/1",
         xlabel="Margen (y·h(x))",
         ylabel="Pérdida",
         title="Comparación de Funciones de Pérdida (Clasificación)",
         linewidth=2,
         legend=:topright)
    
    plot!(margenes, l_bis,
          label="Bisagra (Hinge)",
          linewidth=2)
    
    plot!(margenes, l_log,
          label="Logística",
          linewidth=2,
          linestyle=:dash)
    
    # Marcar región de margen y error
    vline!([0], label="Frontera de decisión", color=:red, linestyle=:dot, linewidth=1)
    vline!([1], label="Margen = 1", color=:orange, linestyle=:dot, linewidth=1)
end

"""
Visualiza la convergencia del descenso de gradiente
"""
function visualizar_convergencia(historial::HistorialOptimizacion; titulo="Convergencia del Descenso de Gradiente")
    plot(1:length(historial.perdidas), historial.perdidas,
         xlabel="Iteración",
         ylabel="Pérdida (MSE)",
         title=titulo,
         label="Pérdida",
         linewidth=2,
         marker=:circle,
         markersize=2,
         yscale=:log10,
         legend=:topright)
end

"""
Visualiza la trayectoria de los pesos durante la optimización (para 2 pesos)
"""
function visualizar_trayectoria_pesos(historial::HistorialOptimizacion, W_optimo::Vector{Float64}=Float64[])
    if length(historial.pesos[1]) != 2 && length(historial.pesos[1]) != 3
        println("Visualización solo disponible para 2 o 3 pesos")
        return
    end
    
    # Extraer coordenadas (ignorando bias si existe)
    if length(historial.pesos[1]) == 3
        w1 = [w[2] for w in historial.pesos]
        w2 = [w[3] for w in historial.pesos]
        opt_idx = 2:3
    else
        w1 = [w[1] for w in historial.pesos]
        w2 = [w[2] for w in historial.pesos]
        opt_idx = 1:2
    end
    
    # Plot de trayectoria
    p = plot(w1, w2,
             xlabel="w₁",
             ylabel="w₂",
             title="Trayectoria de los Pesos durante Optimización",
             label="Trayectoria GD",
             linewidth=2,
             marker=:circle,
             markersize=3,
             alpha=0.7,
             legend=:topright)
    
    # Marcar inicio y fin
    scatter!(p, [w1[1]], [w2[1]],
             label="Inicio",
             markersize=8,
             color=:green,
             markershape=:star5)
    
    scatter!(p, [w1[end]], [w2[end]],
             label="Final",
             markersize=8,
             color=:red,
             markershape=:star5)
    
    # Marcar óptimo si se proporciona
    if !isempty(W_optimo)
        scatter!(p, [W_optimo[opt_idx[1]]], [W_optimo[opt_idx[2]]],
                label="Óptimo analítico",
                markersize=10,
                color=:blue,
                markershape=:diamond)
    end
    
    return p
end

"""
Visualiza superficie de pérdida y trayectoria de optimización
"""
function visualizar_superficie_perdida(X::Matrix, y::Vector, historial::HistorialOptimizacion)
    if size(X, 2) != 1
        println("Esta visualización requiere 1 característica")
        return
    end
    
    X_aug = hcat(ones(size(X, 1)), X)
    
    # Extraer trayectoria de pesos
    w0_vals = [w[1] for w in historial.pesos]
    w1_vals = [w[2] for w in historial.pesos]
    
    # Crear grid para superficie
    w0_range = range(minimum(w0_vals) - .1, maximum(w0_vals) + .1, length=50)
    w1_range = range(minimum(w1_vals) - .1, maximum(w1_vals) + .1, length=50)
    
    # Calcular pérdida en cada punto del grid
    Z = zeros(length(w1_range), length(w0_range))
    for (i, w1) in enumerate(w1_range)
        for (j, w0) in enumerate(w0_range)
            W_temp = [w0, w1]
            y_pred = X_aug * W_temp
            Z[i, j] = mse_perdida(y, y_pred)
        end
    end
    
    # Plot de contorno
    contour(w0_range, w1_range, Z,
            xlabel="w₀ (bias)",
            ylabel="w₁",
            title="Superficie de Pérdida MSE",
            fill=true,
            levels=20,
            colorbar=true)
    
    # Superponer trayectoria
    plot!(w0_vals, w1_vals,
          label="Trayectoria GD",
          linewidth=3,
          color=:white,
          marker=:circle,
          markersize=4)
    
    scatter!([w0_vals[1]], [w1_vals[1]],
            label="Inicio",
            markersize=8,
            color=:lime,
            markershape=:star5)
    
    scatter!([w0_vals[end]], [w1_vals[end]],
            label="Final",
            markersize=8,
            color=:red,
            markershape=:star5)
end

"""
Visualiza comparación entre batch GD y SGD
"""
function comparar_bgd_sgd(hist_bgd::HistorialOptimizacion, hist_sgd::HistorialOptimizacion)
    p1 = plot(1:length(hist_bgd.perdidas), hist_bgd.perdidas,
              label="Batch GD",
              xlabel="Iteración/Época",
              ylabel="Pérdida (MSE)",
              title="Batch GD vs SGD",
              linewidth=2,
              yscale=:log10)
    
    plot!(p1, 1:length(hist_sgd.perdidas), hist_sgd.perdidas,
          label="SGD",
          linewidth=2,
          linestyle=:dash)
    
    return p1
end

"""
Visualiza la frontera de decisión de regresión logística
"""
function visualizar_regresion_logistica(X::Matrix, y::Vector, W::Vector; titulo="Regresión Logística")
    if size(X, 2) != 2
        error("Esta función solo funciona con 2 características")
    end
    
    # Convertir etiquetas a {-1, 1} si no lo están
    y_plot = [yi ∈ [0, -1] ? -1 : 1 for yi in y]
    
    # Crear grid
    x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
    
    x1_grid = range(x1_min, x1_max, length=100)
    x2_grid = range(x2_min, x2_max, length=100)
    
    # Calcular probabilidades
    Z = zeros(length(x2_grid), length(x1_grid))
    for (i, x2) in enumerate(x2_grid)
        for (j, x1) in enumerate(x1_grid)
            score = W[1] + W[2]*x1 + W[3]*x2
            # Probabilidad de clase positiva: σ(score)
            Z[i, j] = 1 / (1 + exp(-score))
        end
    end
    
    # Plot de contorno de probabilidad
    p = contourf(x1_grid, x2_grid, Z,
                 levels=20,
                 xlabel="x₁",
                 ylabel="x₂",
                 title=titulo,
                 colorbar_title="P(y=1)",
                 alpha=0.6)
    
    # Frontera de decisión (probabilidad = 0.5)
    contour!(p, x1_grid, x2_grid, Z,
             levels=[0.5],
             linewidth=3,
             color=:black,
             linestyle=:dash,
             label="Frontera (P=0.5)")
    
    # Plotear puntos
    for clase in unique(y_plot)
        idx = y_plot .== clase
        scatter!(p, X[idx, 1], X[idx, 2],
                label="Clase $clase",
                markersize=6,
                alpha=0.8,
                color=clase > 0 ? :red : :blue,
                markerstrokewidth=1,
                markerstrokecolor=:black)
    end
    
    return p
end

"""
Compara diferentes funciones de pérdida para regresión
"""
function comparar_perdidas_regresion(errores::Vector{Float64})
    println("\nComparación de funciones de pérdida (Regresión)")
    println("─"^60)
    println("Error\t\tCuadrática\tAbsoluta")
    println("─"^60)
    
    for e in errores
        y_true = 0.0
        y_pred = e
        l_cuad = perdida_cuadratica(y_true, y_pred)
        l_abs = perdida_absoluta(y_true, y_pred)
        
        println("$(rpad(round(e, digits=2), 8))\t",
                "$(rpad(round(l_cuad, digits=2), 12))\t",
                "$(rpad(round(l_abs, digits=2), 10))\t")
    end
end

"""
Compara diferentes funciones de pérdida para clasificación
"""
function comparar_perdidas_clasificacion(margenes::Vector{Float64})
    println("\nComparación de funciones de pérdida (Clasificación)")
    println("Margen = y·h(x)")
    println("─"^60)
    println("Margen\t\t0/1\t\tBisagra\t\tLogística")
    println("─"^60)
    
    for m in margenes
        y = 1.0
        score = m
        
        l_01 = m < 0 ? 1.0 : 0.0
        l_bis = perdida_bisagra(y, score)
        l_log = perdida_logistica(y, score)
        
        println("$(rpad(round(m, digits=2), 10))\t",
                "$(rpad(l_01, 8))\t",
                "$(rpad(round(l_bis, digits=3), 10))\t",
                "$(round(l_log, digits=3))")
    end
end

#============================================================================
4. EVALUACIÓN Y MÉTRICAS

4.1 Matriz de confusión (binaria y multiclase)
4.2 Métricas: Accuracy, Precision, Recall, F1, Specificity, FPR
4.3 Curva ROC y cálculo de AUC
4.4 Validación cruzada K-fold
4.5 División train/test
============================================================================#

# ============================================================================
# 4.1 MATRIZ DE CONFUSIÓN
# ============================================================================

"""
Estructura para almacenar la matriz de confusión
Para clasificación binaria con etiquetas {-1, 1} o {0, 1}
"""
struct MatrizConfusion
    TP::Int  # True Positives (Verdaderos Positivos)
    TN::Int  # True Negatives (Verdaderos Negativos)
    FP::Int  # False Positives (Falsos Positivos)
    FN::Int  # False Negatives (Falsos Negativos)
    
    function MatrizConfusion(y_true::Vector{<:Integer}, y_pred::Vector{<:Integer})
        # Normalizar etiquetas a {-1, 1}
        y_t = [yi ∈ [0, -1] ? -1 : 1 for yi in y_true]
        y_p = [yi ∈ [0, -1] ? -1 : 1 for yi in y_pred]
        
        TP = sum((y_t .== 1) .& (y_p .== 1))
        TN = sum((y_t .== -1) .& (y_p .== -1))
        FP = sum((y_t .== -1) .& (y_p .== 1))
        FN = sum((y_t .== 1) .& (y_p .== -1))
        
        new(TP, TN, FP, FN)
    end
end

"""
Imprime la matriz de confusión de forma legible
"""
function Base.show(io::IO, cm::MatrizConfusion)
    println(io, "\nMatriz de Confusión:")
    println(io, "                  Predicho")
    println(io, "               Pos      Neg")
    println(io, "    Real Pos   $(lpad(cm.TP, 4))     $(lpad(cm.FN, 4))")
    println(io, "         Neg   $(lpad(cm.FP, 4))     $(lpad(cm.TN, 4))")
end

"""
Matriz de confusión para clasificación multiclase
"""
struct MatrizConfusionMulticlase
    matriz::Matrix{Int}
    clases::Vector{Int}
    
    function MatrizConfusionMulticlase(y_true::Vector{<:Integer}, y_pred::Vector{<:Integer})
        clases = sort(unique(vcat(y_true, y_pred)))
        k = length(clases)
        
        # Mapear etiquetas a índices
        clase_a_idx = Dict(c => i for (i, c) in enumerate(clases))
        
        matriz = zeros(Int, k, k)
        for i in 1:length(y_true)
            idx_true = clase_a_idx[y_true[i]]
            idx_pred = clase_a_idx[y_pred[i]]
            matriz[idx_true, idx_pred] += 1
        end
        
        new(matriz, clases)
    end
end

function Base.show(io::IO, cm::MatrizConfusionMulticlase)
    println(io, "\nMatriz de Confusión (Multiclase):")
    println(io, "Filas = Real, Columnas = Predicho")
    println(io, "\t\t", join([string(c) for c in cm.clases], "\t"))
    for (i, clase) in enumerate(cm.clases)
        println(io, "Clase $clase\t\t", join(cm.matriz[i,:], "\t"))
    end
    println(io, "\nAccuracy: ", round(accuracy_multiclase(cm) * 100, digits=2), "%")
end


# ============================================================================
# 4.2 MÉTRICAS DE CLASIFICACIÓN
# ============================================================================

"""
Accuracy (Exactitud)
Porcentaje de predicciones correctas
Accuracy = (TP + TN) / (TP + TN + FP + FN)
"""
accuracy(cm::MatrizConfusion) = (cm.TP + cm.TN) / (cm.TP + cm.TN + cm.FP + cm.FN)

"""
Precision (Precisión)
De todas las predicciones positivas, cuántas son correctas
Precision = TP / (TP + FP)
"""
precision(cm::MatrizConfusion) = cm.TP / (cm.TP + cm.FP)

"""
Recall / Sensitivity / TPR (True Positive Rate)
De todos los casos realmente positivos, cuántos se detectan
Recall = TP / (TP + FN)
"""
recall(cm::MatrizConfusion) = cm.TP / (cm.TP + cm.FN)
sensitivity(cm::MatrizConfusion) = recall(cm)
tpr(cm::MatrizConfusion) = recall(cm)

"""
Specificity (Especificidad)
De todos los casos realmente negativos, cuántos se detectan
Specificity = TN / (TN + FP)
"""
specificity(cm::MatrizConfusion) = cm.TN / (cm.TN + cm.FP)

"""
FPR (False Positive Rate)
Tasa de falsos positivos
FPR = FP / (TN + FP) = 1 - Specificity
"""
fpr(cm::MatrizConfusion) = cm.FP / (cm.TN + cm.FP)

"""
F1 Score
Media armónica entre Precision y Recall
F1 = 2 * (Precision * Recall) / (Precision + Recall)
"""
function f1_score(cm::MatrizConfusion)
    prec = precision(cm)
    rec = recall(cm)
    return 2 * (prec * rec) / (prec + rec)
end

"""
Calcula todas las métricas principales
"""
function metricas_clasificacion(cm::MatrizConfusion)
    return (
        accuracy = accuracy(cm),
        precision = precision(cm),
        recall = recall(cm),
        specificity = specificity(cm),
        f1 = f1_score(cm),
        fpr = fpr(cm)
    )
end

"""
Imprime un resumen de todas las métricas
"""
function resumen_metricas(cm::MatrizConfusion)
    println(cm)
    println("\nMétricas de Rendimiento:")
    println("─"^50)
    m = metricas_clasificacion(cm)
    println("Accuracy:     ", round(m.accuracy * 100, digits=2), "%")
    println("Precision:    ", round(m.precision * 100, digits=2), "%")
    println("Recall/TPR:   ", round(m.recall * 100, digits=2), "%")
    println("Specificity:  ", round(m.specificity * 100, digits=2), "%")
    println("F1 Score:     ", round(m.f1, digits=4))
    println("FPR:          ", round(m.fpr * 100, digits=2), "%")
end

"""
Accuracy para clasificación multiclase
"""
function accuracy_multiclase(cm::MatrizConfusionMulticlase)
    total = sum(cm.matriz)
    correctos = sum(cm.matriz[i,i] for i in 1:size(cm.matriz, 1))
    return correctos / total
end

# ============================================================================
# 4.3 CURVA ROC Y AUC
# ============================================================================

"""
Calcula puntos de la curva ROC variando el umbral
"""
function curva_roc(y_true::Vector{<:Integer}, scores::Vector{<:Real})
    # Normalizar etiquetas a {-1, 1}
    y_t = [yi ∈ [0, -1] ? -1 : 1 for yi in y_true]
    
    # Ordenar por scores (de mayor a menor)
    indices = sortperm(scores, rev=true)
    y_sorted = y_t[indices]
    scores_sorted = scores[indices]
    
    # Contar positivos y negativos totales
    n_pos = sum(y_sorted .== 1)
    n_neg = sum(y_sorted .== -1)
    
    # Vectores para la curva ROC
    tpr_vals = Float64[0.0]
    fpr_vals = Float64[0.0]
    
    tp = 0
    fp = 0
    
    # Recorrer cada posible umbral
    for i in 1:length(y_sorted)
        if y_sorted[i] == 1
            tp += 1
        else
            fp += 1
        end
        
        # Agregar punto a la curva
        push!(tpr_vals, tp / n_pos)
        push!(fpr_vals, fp / n_neg)
    end
    
    return fpr_vals, tpr_vals
end

"""
Calcula el AUC (Area Under Curve) usando la regla del trapecio
"""
function calcular_auc(fpr_vals::Vector{Float64}, tpr_vals::Vector{Float64})
    auc = 0.0
    for i in 2:length(fpr_vals)
        # Área del trapecio
        ancho = fpr_vals[i] - fpr_vals[i-1]
        altura = (tpr_vals[i] + tpr_vals[i-1]) / 2
        auc += ancho * altura
    end
    return auc
end

"""
Calcula la curva ROC y el AUC
"""
function roc_auc(y_true::Vector{<:Integer}, scores::Vector{<:Real})
    fpr_vals, tpr_vals = curva_roc(y_true, scores)
    auc_val = calcular_auc(fpr_vals, tpr_vals)
    return (fpr=fpr_vals, tpr=tpr_vals, auc=auc_val)
end

# ============================================================================
# 4.4 VALIDACIÓN CRUZADA
# ============================================================================

"""
Divide datos en K folds para validación cruzada
"""
function crear_folds(n::Int, k::Int; shuffle_data::Bool=true)
    indices = collect(1:n)
    
    if shuffle_data
        indices = Random.shuffle(indices)
    end
    
    fold_size = div(n, k)
    folds = Vector{Vector{Int}}()
    
    for i in 1:k
        inicio = (i-1) * fold_size + 1
        fin = i == k ? n : i * fold_size
        push!(folds, indices[inicio:fin])
    end
    
    return folds
end

"""
Validación cruzada K-fold
"""
function validacion_cruzada_kfold(X::Matrix, y::Vector, modelo_fn, evaluar_fn; k::Int=5)
    n = size(X, 1)
    folds = crear_folds(n, k)
    
    resultados = Float64[]
    
    for i in 1:k
        # Separar train/test
        test_idx = folds[i]
        train_idx = setdiff(1:n, test_idx)
        
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        # Entrenar modelo
        modelo = modelo_fn(X_train, y_train)
        
        # Evaluar
        resultado = evaluar_fn(modelo, X_test, y_test)
        push!(resultados, resultado)
    end
    
    return (
        media = mean(resultados),
        std = std(resultados),
        resultados = resultados
    )
end

# ============================================================================
# 4.5 DIVISIÓN TRAIN/TEST
# ============================================================================

"""
División simple train/test
"""
function train_test_split(X::Matrix, y::Vector, test_size::Float64=0.2; shuffle_data::Bool=true)
    n = size(X, 1)
    n_test = Int(round(n * test_size))
    n_train = n - n_test
    
    indices = collect(1:n)
    if shuffle_data
        Random.shuffle!(indices)
    end
    
    train_idx = indices[1:n_train]
    test_idx = indices[n_train+1:end]
    
    return (
        X_train = X[train_idx, :],
        y_train = y[train_idx],
        X_test = X[test_idx, :],
        y_test = y[test_idx]
    )
end

#============================================================================
5. SOBREAJUSTE E INFRAAJUSTE

5.1 Generación de datos con ruido
5.2 Polinomios y complejidad del modelo
5.3 Curvas de aprendizaje
5.4 Detección automática de sobreajuste/infraajuste
5.5 Regularización (Ridge)
5.6 Análisis de sesgo-varianza
============================================================================#

# ============================================================================
# 5.1. GENERACIÓN DE DATOS CON RUIDO
# ============================================================================

"""
Genera datos para experimentos de sobreajuste/infraajuste
"""
function generar_datos_polinomicos(n::Int=100; grado_real::Int=3, ruido::Float64=0.5)
    Random.seed!(42)
    
    X = sort(2 * rand(n) .- 1)  # X en [-1, 1]
    
    # Función verdadera: polinomio de grado específico
    if grado_real == 2
        y_true = X.^2
    elseif grado_real == 3
        y_true = 0.5 * X.^3 - 0.3 * X
    else
        y_true = sin(3 * X)
    end
    
    y = y_true .+ ruido * randn(n)
    
    return X, y, y_true
end

# ============================================================================
# 5.2 POLINOMIOS Y COMPLEJIDAD DEL MODELO
# ============================================================================

"""
Crea características polinómicas hasta grado d
"""
function crear_polinomio(X::Vector, d::Int)
    n = length(X)
    Phi = ones(n, d+1)
    
    for i in 1:d
        Phi[:, i+1] = X.^i
    end
    
    return Phi
end

"""
Ajusta un polinomio de grado d usando mínimos cuadrados
"""
function ajustar_polinomio(X::Vector, y::Vector, d::Int; λ::Float64=0.0)
    Phi = crear_polinomio(X, d)
    
    if λ > 0
        # Con regularización L2 (Ridge)
        W = (Phi' * Phi + λ * I(d+1)) \ (Phi' * y)
    else
        # Sin regularización
        W = (Phi' * Phi) \ (Phi' * y)
    end
    
    return W
end

"""
Predice usando un modelo polinómico
"""
function predecir_polinomio(X::Vector, W::Vector)
    d = length(W) - 1
    Phi = crear_polinomio(X, d)
    return Phi * W
end

# ============================================================================
# 5.3. CURVAS DE APRENDIZAJE
# ============================================================================

"""
Calcula error de entrenamiento y validación para diferentes tamaños de muestra
"""
function curva_aprendizaje(X_train::Vector, y_train::Vector, 
                          X_val::Vector, y_val::Vector,
                          d::Int;
                          tamaños::Union{Nothing, Vector{Int}}=nothing)
    n_train = length(X_train)
    
    if isnothing(tamaños)
        tamaños = unique(Int[ceil(n_train * p) for p in 0.1:0.1:1.0])
    end
    
    errores_train = Float64[]
    errores_val = Float64[]
    
    for m in tamaños
        if m > n_train
            break
        end
        
        # Entrenar con primeros m ejemplos
        X_m = X_train[1:m]
        y_m = y_train[1:m]
        
        W = ajustar_polinomio(X_m, y_m, d)
        
        # Error en train
        y_pred_train = predecir_polinomio(X_m, W)
        err_train = mean((y_m .- y_pred_train).^2)
        push!(errores_train, err_train)
        
        # Error en validación
        y_pred_val = predecir_polinomio(X_val, W)
        err_val = mean((y_val .- y_pred_val).^2)
        push!(errores_val, err_val)
    end
    
    return (
        tamaños = tamaños[1:length(errores_train)],
        errores_train = errores_train,
        errores_val = errores_val
    )
end

"""
Calcula error para diferentes grados del polinomio (complejidad del modelo)
"""
function curva_complejidad(X_train::Vector, y_train::Vector,
                          X_val::Vector, y_val::Vector;
                          grados::Vector{Int}=collect(1:15))
    errores_train = Float64[]
    errores_val = Float64[]
    
    for d in grados
        W = ajustar_polinomio(X_train, y_train, d)
        
        # Error en train
        y_pred_train = predecir_polinomio(X_train, W)
        err_train = mean((y_train .- y_pred_train).^2)
        push!(errores_train, err_train)
        
        # Error en validación
        y_pred_val = predecir_polinomio(X_val, W)
        err_val = mean((y_val .- y_pred_val).^2)
        push!(errores_val, err_val)
    end
    
    return (
        grados = grados,
        errores_train = errores_train,
        errores_val = errores_val
    )
end

# ============================================================================
# 5.4 DETECCIÓN DE SOBREAJUSTE/INFRAAJUSTE
# ============================================================================

"""
Diagnostica si el modelo está sobreajustado o infraajustado
"""
function diagnosticar_ajuste(err_train::Float64, err_val::Float64; 
                             umbral_alto::Float64=0.1,
                             umbral_gap::Float64=0.05)
    gap = err_val - err_train
    
    if err_train > umbral_alto && err_val > umbral_alto
        return :infraajuste, "Alto error en train y validación"
    elseif gap > umbral_gap && err_val > umbral_alto
        return :sobreajuste, "Gran diferencia entre train y validación"
    else
        return :buen_ajuste, "Errores razonables y similares"
    end
end

"""
Encuentra el mejor grado de polinomio
"""
function seleccionar_mejor_modelo(X_train::Vector, y_train::Vector,
                                  X_val::Vector, y_val::Vector;
                                  grados::Vector{Int}=collect(1:10))
    resultado = curva_complejidad(X_train, y_train, X_val, y_val, grados=grados)
    
    # Encontrar mínimo error de validación
    idx_min = argmin(resultado.errores_val)
    mejor_grado = resultado.grados[idx_min]
    mejor_err_val = resultado.errores_val[idx_min]
    
    return (
        mejor_grado = mejor_grado,
        mejor_error = mejor_err_val,
        resultado = resultado
    )
end

# ============================================================================
# 5.5 REGULARIZACIÓN
# ============================================================================

"""
Compara modelos con diferentes valores de regularización
"""
function curva_regularizacion(X_train::Vector, y_train::Vector,
                              X_val::Vector, y_val::Vector,
                              d::Int;
                              lambdas::Vector{Float64}=10.0.^(-4:0.5:2))
    errores_train = Float64[]
    errores_val = Float64[]
    
    for λ in lambdas
        W = ajustar_polinomio(X_train, y_train, d, λ=λ)
        
        # Error en train
        y_pred_train = predecir_polinomio(X_train, W)
        err_train = mean((y_train .- y_pred_train).^2)
        push!(errores_train, err_train)
        
        # Error en validación
        y_pred_val = predecir_polinomio(X_val, W)
        err_val = mean((y_val .- y_pred_val).^2)
        push!(errores_val, err_val)
    end
    
    return (
        lambdas = lambdas,
        errores_train = errores_train,
        errores_val = errores_val
    )
end

"""
Encuentra el mejor valor de regularización
"""
function seleccionar_mejor_lambda(X_train::Vector, y_train::Vector,
                                  X_val::Vector, y_val::Vector,
                                  d::Int;
                                  lambdas::Vector{Float64}=10.0.^(-4:0.5:2))
    resultado = curva_regularizacion(X_train, y_train, X_val, y_val, d, lambdas=lambdas)
    
    idx_min = argmin(resultado.errores_val)
    mejor_lambda = resultado.lambdas[idx_min]
    mejor_err_val = resultado.errores_val[idx_min]
    
    return (
        mejor_lambda = mejor_lambda,
        mejor_error = mejor_err_val,
        resultado = resultado
    )
end

# ============================================================================
# 5.6 ANÁLISIS DE SESGO-VARIANZA
# ============================================================================

"""
Realiza múltiples entrenamientos con diferentes muestras para medir varianza
"""
function analisis_sesgo_varianza(X_base::Vector, y_base_fn::Function,
                                 d::Int, n_train::Int, n_experimentos::Int=50;
                                 ruido::Float64=0.5)
    # Puntos de test fijos
    X_test = collect(range(-1, 1, length=100))
    y_test_true = y_base_fn(X_test)
    
    # Almacenar predicciones
    predicciones = zeros(n_experimentos, length(X_test))
    
    for exp in 1:n_experimentos
        # Generar muestra de entrenamiento con ruido
        idx_sample = rand(1:length(X_base), n_train)
        X_train = X_base[idx_sample]
        y_train = y_base_fn(X_train) .+ ruido * randn(n_train)
        
        # Entrenar modelo
        W = ajustar_polinomio(X_train, y_train, d)
        
        # Predecir en test
        predicciones[exp, :] = predecir_polinomio(X_test, W)
    end
    
    # Calcular sesgo y varianza
    pred_media = mean(predicciones, dims=1)[1, :]
    sesgo = mean((pred_media .- y_test_true).^2)
    varianza = mean(var(predicciones, dims=1))
    
    return (
        sesgo = sesgo,
        varianza = varianza,
        predicciones = predicciones,
        X_test = X_test,
        y_test_true = y_test_true
    )
end

println("Módulo de Fundamentos ML cargado correctamente.")

