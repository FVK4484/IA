include("../ML.jl")
include("ML_Basicos.jl")
using Statistics
using LinearAlgebra
using CSV
using DataFrames
using Plots
using Colors
using FileIO
using ImageIO
using Snowball

# Función para normalizar datos a N(0,1)
function normalizar(datos)
    # Extraer características
    features = [x for (x, _) in datos]
    n_dims = length(features[1])
    
    # Calcular media y desv. estándar por dimensión
    medias = [mean([f[i] for f in features]) for i in 1:n_dims]
    stds = [std([f[i] for f in features]) for i in 1:n_dims]
    
    # Función para normalizar un vector
    norm_vec(x) = [(x[i] - medias[i]) / stds[i] for i in 1:n_dims]
    
    # Normalizar todos los datos
    datos_norm = [(norm_vec(x), y) for (x, y) in datos]
    
    return datos_norm, medias, stds
end

# Función auxiliar para normalizar un punto de prueba
function normalizar_punto(x::Vector, medias::Vector, stds::Vector)
    return [(x[i] - medias[i]) / stds[i] for i in 1:length(x)]
end


#=====================================================================
KNN (k-Nearest Neighbors) CLASIFICACIÓN
=====================================================================#

# Ejemplo 1: Clasificación de puntos en el plano
# ----------------------------------------------
datos_puntos = [
    # Grupo A (esquina inferior izquierda)
    ([1.0, 1.0], "A"), ([1.5, 1.2], "A"), ([0.8, 1.5], "A"), ([1.2, 0.9], "A"),
    # Grupo B (esquina superior derecha)
    ([9.0, 9.0], "B"), ([8.8, 9.2], "B"), ([9.2, 8.9], "B"), ([8.9, 9.1], "B"),
    # Grupo C (esquina inferior derecha)
    ([9.0, 1.0], "C"), ([8.9, 1.1], "C"), ([9.1, 0.9], "C"),
]

# Normalizar datos
datos_puntos_norm, medias_puntos, stds_puntos = normalizar(datos_puntos)

puntos_a_clasificar = [
    [1.1, 1.3],
    [9.0, 9.1],
    [8.95, 1.05],
    [5.0, 5.0]  # Punto en el medio (ambiguo)
]

for punto in puntos_a_clasificar
    punto_norm = normalizar_punto(punto, medias_puntos, stds_puntos)
    clase = knn(3, datos_puntos_norm, punto_norm)
    println("Punto $punto → Clase: $clase")
end

# Ejemplo 2: Análisis de deporte/actividad física
# -----------------------------------------------

# Características: [frecuencia cardíaca (bpm), VO2 (ml/kg/min)]
datos_actividad = [
    ([60, 35], "Sedentario"),
    ([65, 37], "Sedentario"),
    ([62, 34], "Sedentario"),
    ([100, 50], "Entrenamiento moderado"),
    ([105, 52], "Entrenamiento moderado"),
    ([102, 49], "Entrenamiento moderado"),
    ([150, 70], "Entrenamiento intenso"),
    ([155, 72], "Entrenamiento intenso"),
    ([152, 68], "Entrenamiento intenso"),
]

# Normalizar datos
datos_actividad_norm, medias_actividad, stds_actividad = normalizar(datos_actividad)

personas = [
    ([63, 36], "¿Persona 1?"),
    ([103, 51], "¿Persona 2?"),
    ([153, 71], "¿Persona 3?"),
    ([85, 42], "¿Persona 4 (intermedio)?")
]

for (vitales, persona) in personas
    vitales_norm = normalizar_punto(vitales, medias_actividad, stds_actividad)
    clase = knn(3, datos_actividad_norm, vitales_norm)
    println("$persona → $clase")
end


# Ejemplo 3: Clasificación simple de flores (dataset iris)
# --------------------------------------------------------

df_iris = CSV.read("Practicas/datasets/iris.csv", DataFrame, header = true)
datos_flores = [ (collect(df_iris[j, 1:4]), df_iris[j, 5]) for j in 1:size(df_iris, 1) ]

# Normalizar datos
datos_flores_norm, medias_flores, stds_flores = normalizar(datos_flores)

# Clasificar una nueva flor
nueva_flor = [6.5, 3.1, 5.0, 1.8]
nueva_flor_norm = normalizar_punto(nueva_flor, medias_flores, stds_flores)
clase = knn(3, datos_flores_norm, nueva_flor_norm)

# Matriz de confusión (k = 3, validación leave-one-out: cada flor se clasifica
# con las otras 149, para no ser su propio vecino)

etiquetas_flores = [y for (_, y) in datos_flores_norm]

# Función para calcular la predicción de kNN con validación leave-one-out
# esta función devuelve un vector de predicciones para cada punto en el conjunto 
# de datos
function knn_loo(k, datos)
    [knn(k, datos[[1:i-1; i+1:end]], datos[i][1]) for i in eachindex(datos)]
end

pred_k3 = knn_loo(3, datos_flores_norm)
cm_k3 = matriz_confusion(etiquetas_flores, pred_k3)
println(cm_k3)
println(metricas_por_clase(cm_k3))
display(graficar_matriz_confusion(cm_k3; titulo = "kNN iris (k = 3)"))

# Comparación de distintos valores de k (impares para evitar empates)
ks = 1:2:41
preds = Dict(k => knn_loo(k, datos_flores_norm) for k in ks)
accs = [accuracy(etiquetas_flores, preds[k]) for k in ks]

println("\n  k   accuracy")
for (k, a) in zip(ks, accs)
    println(lpad(k, 3), "   ", round(a, digits = 4))
end

p_acc = plot(collect(ks), accs, marker = :circle, legend = false,
             xlabel = "k", ylabel = "Accuracy (leave-one-out)",
             title = "kNN en iris: accuracy según k")
display(p_acc)


# Fiabilidad de cada punto: fracción de valores de k para los que se clasifica bien
correcto = [preds[k][i] == etiquetas_flores[i] for i in eachindex(etiquetas_flores), k in ks]
fiabilidad = vec(mean(correcto, dims = 2))

println("\nPuntos clasificados siempre bien: ", count(==(1.0), fiabilidad))
println("Puntos poco fiables (acierto < 100 % de los k):")
for i in findall(<(1.0), fiabilidad)
    println("  flor $i ($(etiquetas_flores[i])): acierto en $(round(100*fiabilidad[i], digits=1)) % de los k")
end

# Proyección a 2D (PCA) de los datos normalizados para visualizar
X = reduce(vcat, [x' for (x, _) in datos_flores_norm])
V = svd(X .- mean(X, dims = 1)).V[:, 1:2]
Z = X * V

especies = unique(etiquetas_flores)
colores = Dict(e => c for (e, c) in zip(especies, [:royalblue, :darkorange, :seagreen]))

# (a) Mapa de fiabilidad: color = fracción de k con acierto
p_fiab = scatter(Z[:, 1], Z[:, 2], zcolor = fiabilidad, c = :RdYlGn, clims = (0, 1),
                 markersize = 6, markerstrokewidth = 0.3, label = "",
                 colorbar_title = "Fiabilidad", xlabel = "PC1", ylabel = "PC2",
                 title = "Fiabilidad de cada punto (todos los k)")
dudosos = findall(<(1.0), fiabilidad)
scatter!(p_fiab, Z[dudosos, 1], Z[dudosos, 2], markershape = :circle, markersize = 10,
         markercolor = :transparent, markerstrokecolor = :black, label = "poco fiables")

# (b) Errores para k = 1, 3, 15 y 41 sobre el mapa de especies
paneles = map([1, 3, 15, 41]) do k
    mal = findall(preds[k] .!= etiquetas_flores)
    p = plot(title = "k = $k  (errores: $(length(mal)))", xlabel = "PC1", ylabel = "PC2")
    for e in especies
        idx = findall(==(e), etiquetas_flores)
        scatter!(p, Z[idx, 1], Z[idx, 2], c = colores[e], label = string(e),
                 markersize = 4, markerstrokewidth = 0)
    end
    scatter!(p, Z[mal, 1], Z[mal, 2], markershape = :xcross, markersize = 9,
             markerstrokewidth = 2, c = :red, label = "error")
    p
end
display(plot(p_fiab, paneles..., layout = (3, 2), size = (1000, 1200)))

# (c) Matrices de confusión para k extremos
display(plot([graficar_matriz_confusion(matriz_confusion(etiquetas_flores, preds[k]);
                                        titulo = "k = $k") for k in (1, 15, 41)]...,
             layout = (1, 3), size = (1500, 400)))

#



# ============================================
# EJEMPLOS DE REGRESIÓN
# ============================================

# Ejemplo 1: Predicción de precio de vivienda
# -------------------------------------------
# Características: las 13 variables del conjunto Boston Housing
# Valor: mediana del precio de vivienda en miles de dólares

df_housing = CSV.read("Practicas/datasets/housing.data", DataFrame;
                      header = false, delim = ' ', ignorerepeated = true)
datos_viviendas = [ (collect(df_housing[j, 1:13]), df_housing[j, 14]) for j in 1:size(df_housing, 1) ]

dtr = datos_viviendas[1:400]
dts = datos_viviendas[401:end]  # Resto para test

# Normalizar datos
datos_viviendas_norm, medias_viviendas, stds_viviendas = normalizar(dtr)

println("\n=== Predicción de precio de vivienda ===")
for (caracteristicas, valor_real) in dts
    vivienda_norm = normalizar_punto(caracteristicas, medias_viviendas, stds_viviendas)
    precio = knn_regresion(3, datos_viviendas_norm, vivienda_norm)
    println("  Valor real: $(round(valor_real, digits=1)) → Precio predicho: $(round(precio, digits=1)) miles de dólares")
end

# Ejemplo 8: Predicción de consumo de energía
# -------------------------------------------
# Características: [temperatura media (°C), horas de luz solar, día laborable (0/1)]
# Valor: consumo de energía en kWh
datos_energia = [
    ([15, 8, 1], 45.0),
    ([16, 8, 1], 46.0),
    ([14, 7, 0], 40.0),
    ([20, 10, 1], 50.0),
    ([22, 10, 1], 52.0),
    ([25, 11, 1], 58.0),
    ([28, 12, 1], 65.0),
    ([30, 12, 1], 70.0),
    ([32, 12, 0], 68.0),
]

# Normalizar datos
datos_energia_norm, medias_energia, stds_energia = normalizar(datos_energia)

condiciones_energia = [
    ([18, 9, 1], "Día laborable templado"),
    ([27, 11, 1], "Día laborable cálido"),
    ([35, 12, 0], "Fin de semana caluroso"),
]

for (condicion, descripcion) in condiciones_energia
    condicion_norm = normalizar_punto(condicion, medias_energia, stds_energia)
    consumo = knn_regresion(3, datos_energia_norm, condicion_norm)
    println("  $descripcion → Consumo predicho: $(round(consumo, digits=1)) kWh")
end

#=====================================================================
K-Medias
=====================================================================#

# Ejemplo 1: Agrupación de puntos en el plano
# --------------------------------------------
datos_kmedias_2d = [
    [1.0, 1.0], [1.2, 0.8], [0.8, 1.3], [1.1, 1.4],
    [8.0, 8.0], [8.3, 7.8], [7.7, 8.2], [8.1, 8.4],
]

C_2d, Cls_2d, pots_2d = k_medias(datos_kmedias_2d, 2)

println("\n=== K-medias en 2D (K = 2) ===")
for k in eachindex(C_2d)
    puntos_cluster = [datos_kmedias_2d[i] for i in Cls_2d[k]]
    println("Cluster $k: centroide $(round.(C_2d[k], digits=2)), puntos $puntos_cluster")
end
println("Potencial final: ", round(pots_2d[end], digits=3))

p_kmedias_2d = plot(xlabel = "x₁", ylabel = "x₂", title = "K-medias en 2D")
for k in eachindex(C_2d)
    puntos = [datos_kmedias_2d[i] for i in Cls_2d[k]]
    scatter!(p_kmedias_2d, [p[1] for p in puntos], [p[2] for p in puntos];
             label = "Cluster $k", markersize = 6)
    scatter!(p_kmedias_2d, [C_2d[k][1]], [C_2d[k][2]];
             label = "Centroide $k", marker = :star5, markersize = 10)
end
display(p_kmedias_2d)

# Ejemplo 2: Agrupación de observaciones en una dimensión
# -------------------------------------------------------
datos_kmedias_1d = [[1.0], [1.2], [1.4], [5.0], [5.2], [5.4], [9.0], [9.2], [9.4]]
C_1d, Cls_1d, pots_1d = k_medias(datos_kmedias_1d, 3)

println("\n=== K-medias en 1D (K = 3) ===")
for k in eachindex(C_1d)
    println("Cluster $k: centroide $(round(C_1d[k][1], digits=2)), índices $(Cls_1d[k])")
end
println("Potencial final: ", round(pots_1d[end], digits=3))


# Ejercicio: comparar K-medias con las especies de Iris
# -----------------------------------------------------
#=
Usa el conjunto Iris para comprobar si K Medias hace una clusterización de los 
datos (en 3 clusters de puntos de dimensión 4) que sea coherente con la 
clasificación de los mismos, es decir, que los clusters encontrados coincidan 
con las especies de las flores.
Se usan las cuatro características normalizadas, sin incluir las etiquetas.
Como la inicialización es aleatoria, se elige entre varios reinicios el de
menor potencial. Los números de cluster no tienen una correspondencia fija
con los nombres de las especies.
=#

# Para este ejercicio tiene sentido trabajar sobre datos normalizados. 
# K-medias usa distancias euclídeas y, sin normalización, las características 
# con mayor variabilidad pueden dominar la agrupación. En Iris, al estandarizar 
# las cuatro medidas, cada una contribuye en una escala comparable; las 
# etiquetas de especie no se usan para agrupar, solo para evaluar el resultado 
# después.

# Normalizar en K-medias no es una regla universal: al normalizar cambias qué 
# significa “cerca”. Si quisieras que las variables de mayor variación pesaran 
# más, podrías usar los datos originales.
D_iris = [x for (x, _) in datos_flores_norm]
especies_iris = unique(etiquetas_flores)
K_iris = length(especies_iris)

# Lanzamos K-medias varias veces y nos quedamos con el resultado de menor 
# potencial
# Observa que k_medias devuelve una tupla (C, Cls, pots), donde pots es un 
# vector con los potenciales de cada iteración. El último valor es el potencial 
# final.
resultados_iris = [k_medias(D_iris, K_iris) for _ in 1:20]
idx_mejor_iris = argmin([resultado[3][end] for resultado in resultados_iris])
C_iris, Cls_iris, pots_iris = resultados_iris[idx_mejor_iris]

# Asignamos cada punto a un cluster según los índices de Cls_iris. 
# Creamos un vector de asignaciones, donde asignaciones_iris[i] = k significa 
# que el punto i pertenece al cluster k.
asignaciones_iris = zeros(Int, length(D_iris))
for k in 1:K_iris
    for i in Cls_iris[k]
        asignaciones_iris[i] = k
    end
end

# La tabla de contingencia cruza las especies reales (filas) con los clusters 
# (columnas). Como los números de cluster son arbitrarios — el cluster 1 no 
# significa necesariamente la especie 1 —, primero hay que encontrar una 
# correspondencia entre ambos. 

# El código siguiente busca la correspondencia uno-a-uno que maximiza los 
# aciertos; después de asignar los nombres según esa correspondencia, la tabla 
# puede interpretarse como una matriz de confusión.

# Filas = especies reales; columnas = clusters encontrados.
contingencia_iris = zeros(Int, K_iris, K_iris)
indice_especie_iris = Dict(especie => i for (i, especie) in enumerate(especies_iris))
for i in eachindex(etiquetas_flores)
    fila = indice_especie_iris[etiquetas_flores[i]]
    contingencia_iris[fila, asignaciones_iris[i]] += 1
end

# Mostramos la tabla de contingencia.
tabla_iris_clusters = DataFrame(Especie = string.(especies_iris))
for k in 1:K_iris
    tabla_iris_clusters[!, Symbol("Cluster $k")] = contingencia_iris[:, k]
end

# Buscar la mejor correspondencia uno-a-uno entre clusters y especies.

# Primero, enumeramos todas las permutaciones posibles de los clusters (K! 
# permutaciones)
permutaciones_iris = (
    (1, 2, 3), (1, 3, 2), (2, 1, 3),
    (2, 3, 1), (3, 1, 2), (3, 2, 1),
)
# Después, para cada permutación, calculamos el número de aciertos (suma de la
# diagonal de la tabla de contingencia reordenada según la permutación)
aciertos_por_permutacion = [
    sum(contingencia_iris[permutacion[k], k] for k in 1:K_iris)
    for permutacion in permutaciones_iris
]
# Elegimos la permutación que da el máximo número de aciertos
mejor_permutacion_iris = permutaciones_iris[argmax(aciertos_por_permutacion)]
aciertos_iris = maximum(aciertos_por_permutacion)

# Mostramos los resultados
println("\n=== K-medias en Iris (4 características, K = $K_iris) ===")
println("Mejor resultado de 20 inicializaciones, elegido por potencial.")
println("Tabla de contingencia (filas = especie, columnas = cluster):")
display(tabla_iris_clusters)
for k in 1:K_iris
    println("Cluster $k corresponde mejor a $(especies_iris[mejor_permutacion_iris[k]])")
end
println("Coincidencia tras la mejor correspondencia: $aciertos_iris / $(length(etiquetas_flores)) ",
        "($(round(100 * aciertos_iris / length(etiquetas_flores), digits=1)) %)")
println("Potencial final: ", round(pots_iris[end], digits=3))
if aciertos_iris == length(etiquetas_flores)
    println("La agrupación coincide exactamente con las especies.")
else
    println("La agrupación es parcial: algunas especies aparecen mezcladas en los clusters.")
end

# Como sabes, Iris usa 4 medidas para cada muestra de flor (sl, sw, pl y pw), 
# dos de ellas para las dimensiones de los sépalos y dos para los pétalos. 
# Encuentra para cada una de las 4 variables una clusterización (con el número 
# de clusters que consideres adecuado) que discretice el dataset, pasando de un 
# conjunto de muestras con 4 valores continuos a uno de muestras con 4 valores 
# discretos (para cada medida, se sustituye el valor continuo por el cluster al 
# que pertenece). Usa Knn para comprobar si estas discretizaciones son 
# coherentes con la clasificación de las flores.

# Discretización univariante de las cuatro medidas de Iris
# ---------------------------------------------------------
# Se usan tres niveles por medida para mantener la misma escala discreta.
# Los códigos 1, 2 y 3 se asignan según el orden de los centroides, de menor a
# mayor medida; así la distancia usada por KNN respeta el orden de los valores.
# Los valores numéricos están en el orden habitual sl, sw, pl, pw y se toman
# por posición, sin depender de los nombres de cabecera del CSV.
nombres_medidas_iris = ["sl", "sw", "pl", "pw"]
K_bins_iris = 3

# Se discretizan las cuatro medidas por separado.
columnas_discretizadas = Vector{Vector{Int}}()
centroides_discretizacion = Vector{Vector{Float64}}()

# Para cada medida, se ejecuta K-medias varias veces y se elige la 
# clusterización de menor potencial. Después, se asigna a cada muestra el 
# código del cluster al que pertenece.
for j in 1:4
    valores = [Float64(df_iris[i, j]) for i in 1:size(df_iris, 1)]
    datos_univariantes = [[valor] for valor in valores]
    # Se ejecuta K-medias varias veces y se elige la de menor potencial
    resultados = [
        k_medias(datos_univariantes, K_bins_iris; dist = eucl)
        for _ in 1:20
    ]
    idx_mejor = argmin([resultado[3][end] for resultado in resultados])
    C_medida, Cls_medida, _ = resultados[idx_mejor]
    orden_centros = sortperm([centro[1] for centro in C_medida])

    # Se asigna a cada muestra el código del cluster al que pertenece
    codigos = zeros(Int, length(valores))
    for (codigo, cluster) in enumerate(orden_centros)
        for i in Cls_medida[cluster]
            codigos[i] = codigo
        end
    end
    # Se guardan los códigos y los centroides ordenados para esta medida
    push!(columnas_discretizadas, codigos)
    push!(centroides_discretizacion, [C_medida[k][1] for k in orden_centros])
    println("$(nombres_medidas_iris[j]): centroides de los bins = ",
            round.(centroides_discretizacion[end], digits=2))
end

# Se crean los datos de las flores discretizados
datos_flores_discretos = [
    ([columnas_discretizadas[j][i] for j in 1:4], etiquetas_flores[i])
    for i in eachindex(etiquetas_flores)
]
# Se crea un DataFrame con las cuatro medidas discretizadas y la especie
tabla_flores_discretas = DataFrame(
    sl = columnas_discretizadas[1],
    sw = columnas_discretizadas[2],
    pl = columnas_discretizadas[3],
    pw = columnas_discretizadas[4],
    especie = string.(etiquetas_flores),
)
println("Primeras diez muestras discretizadas [sl, sw, pl, pw]:")
display(first(tabla_flores_discretas, 10))

# Comparar KNN sobre las medidas continuas normalizadas y sobre los bins.
preds_discretos = Dict(k => knn_loo(k, datos_flores_discretos) for k in ks)
accs_discretos = [accuracy(etiquetas_flores, preds_discretos[k]) for k in ks]
println("Accuracy con k=3: continuo = ",
        round(accuracy(etiquetas_flores, preds[3]), digits=4),
        ", discretizado = ",
        round(accuracy(etiquetas_flores, preds_discretos[3]), digits=4))

p_acc_comparacion = plot(collect(ks), accs, marker = :circle,
                         label = "Continuas normalizadas",
                         xlabel = "k", ylabel = "Accuracy (leave-one-out)",
                         title = "kNN en Iris: continuo vs. discretizado")
plot!(p_acc_comparacion, collect(ks), accs_discretos, marker = :diamond,
      label = "Discretizadas (3 bins por medida)")
display(p_acc_comparacion)

# Repite el punto 2 pero clusterizando bidimensionalmente, es decir, encontrar 
# un clustering para las dimensiones de los sépalos, y otro para los pétalos. 
# De esta forma, en vez de trabajar con 4 dimensiones continuas convertidas en 
# 4 dimensiones discretas, pasamos a 2 dimensiones discretas.

# Clusterizamos por separado las dos medidas de sépalos y las dos de pétalos.
# Se estandarizan las coordenadas dentro de cada par para que una medida no
# domine por tener una variabilidad mayor. Usamos tres clusters en cada par.
nombres_pares_iris = ["Sépalos (sl, sw)", "Pétalos (pl, pw)"]
datos_pares_iris = [
    [Float64.(collect(df_iris[i, 1:2])) for i in 1:size(df_iris, 1)],
    [Float64.(collect(df_iris[i, 3:4])) for i in 1:size(df_iris, 1)],
]
K_clusters_par = 3
clusters_pares_iris = Vector{Vector{Int}}()

for (nombre, datos_par) in zip(nombres_pares_iris, datos_pares_iris)
    datos_par_norm, medias_par, stds_par = normalizar([(x, nothing) for x in datos_par])
    D_par_norm = [x for (x, _) in datos_par_norm]
    resultados_par = [
        k_medias(D_par_norm, K_clusters_par; dist = eucl)
        for _ in 1:20
    ]
    idx_mejor_par = argmin([resultado[3][end] for resultado in resultados_par])
    C_par, Cls_par, _ = resultados_par[idx_mejor_par]

    asignaciones_par = zeros(Int, length(D_par_norm))
    for cluster in 1:K_clusters_par
        for i in Cls_par[cluster]
            asignaciones_par[i] = cluster
        end
        centro_original = C_par[cluster] .* stds_par .+ medias_par
        println("$nombre, cluster $cluster: centroide ",
                round.(centro_original, digits=2))
    end
    push!(clusters_pares_iris, asignaciones_par)
end

# La representación discreta tiene dos categorías: cluster de sépalos y de
# pétalos. Se conserva así en la tabla de salida.
datos_flores_2d_discretos = [
    ([clusters_pares_iris[1][i], clusters_pares_iris[2][i]], etiquetas_flores[i])
    for i in eachindex(etiquetas_flores)
]
tabla_flores_2d_discretas = DataFrame(
    cluster_sepalos = clusters_pares_iris[1],
    cluster_petalo = clusters_pares_iris[2],
    especie = string.(etiquetas_flores),
)
println("\nPrimeras diez muestras [cluster sépalos, cluster pétalos]:")
display(first(tabla_flores_2d_discretas, 10))

# Los identificadores de cluster son categorías sin orden. Para usar el KNN
# euclídeo existente, codificamos cada categoría one-hot: dos clusters distintos
# tienen la misma distancia, sin depender de sus identificadores arbitrarios.
datos_flores_2d_codificados = [
    (vcat(
        [Float64(clusters_pares_iris[1][i] == cluster) for cluster in 1:K_clusters_par],
        [Float64(clusters_pares_iris[2][i] == cluster) for cluster in 1:K_clusters_par],
    ), etiquetas_flores[i])
    for i in eachindex(etiquetas_flores)
]
preds_2d_discretos = Dict(k => knn_loo(k, datos_flores_2d_codificados) for k in ks)
accs_2d_discretos = [
    accuracy(etiquetas_flores, preds_2d_discretos[k]) for k in ks
]

println("Accuracy leave-one-out con k=3:")
println("  4 medidas continuas: ",
        round(accuracy(etiquetas_flores, preds[3]), digits=4))
println("  4 medidas discretizadas por separado: ",
        round(accuracy(etiquetas_flores, preds_discretos[3]), digits=4))
println("  2 clusters bidimensionales: ",
        round(accuracy(etiquetas_flores, preds_2d_discretos[3]), digits=4))

p_acc_comparacion_2d = plot(
    collect(ks), accs;
    marker = :circle, label = "4 continuas normalizadas",
    xlabel = "k", ylabel = "Accuracy (leave-one-out)",
    title = "kNN en Iris: discretizaciones",
)
plot!(p_acc_comparacion_2d, collect(ks), accs_discretos;
      marker = :diamond, label = "4 discretas")
plot!(p_acc_comparacion_2d, collect(ks), accs_2d_discretos;
      marker = :utriangle, label = "2 clusters 2D")
display(p_acc_comparacion_2d)


## 
# ============================================
# Método del Codo
# ============================================
#=
Usando las funciones definidas en las tareas anteriores, vamos a dar un 
procedimiento (no muy bueno) para decidir el número de clusters que podríamos 
considerar adecuado para un conjunto de datos. Para ello:

Crea una función que reciba como entrada D y calcule para cada valor de K 
(1 ≤ K ≤ |D|) el potencial de una clusterización de D por medio de K-medias. 
Como el algoritmo de K-medias puede dar distintos resultados dependiendo de la 
elección inicial de centroides, puede ser interesante que esta función no haga 
una sola ejecución para cada valor de K, sino unas cuantas y se quede con la 
media de los potenciales obtenidos.

Antes de continuar, intenta responder a las siguientes cuestiones de forma 
razonada:
1. ¿Cuánto vale el potencial si K=|D|?
2. ¿Qué puedes decir del comportamiento respecto a K (crece, decrece,...)?

Representa los de valores del potencial respecto a K para ver el comportamiento 
de la evolución del número de clusters en el ejemplo `blob_data.csv`.
=#

# D  : Dataset de puntos
# N  : Los potenciales se van a calcular de 1 a N clusters (N ≤ |D|)
# rep: Nº de repeticiones para cada valor de K

function Codo(D, N, rep; grafica = nothing, etiqueta = "")
	potenciales = zeros(N)
	for k in 1:N
		pk = 0
		for _ in 1:rep
			(C, Cls, pots_k) = k_medias(D, k)
			pk  = pk + pots_k[end]
		end
		potenciales[k] = pk / rep
	end
	if isnothing(grafica)
		return plot(potenciales; label = etiqueta, xlabel = "Número de clusters (K)",
		            ylabel = "Potencial", title = "Método del codo")
	end
	plot!(grafica, potenciales; label = etiqueta)
end

df = CSV.read("Practicas/datasets/blob.csv", DataFrame, header = false)
D  = [ collect(df[j, 2:3]) for j in 1:size(df, 1) ]
datos_blob_norm, _, _ = normalizar([(p, nothing) for p in D])
D_norm = [x for (x, _) in datos_blob_norm]

Codo(D, 10, 20; etiqueta = "Datos originales")
Codo(D_norm, 10, 20;  etiqueta = "Datos normalizados")

# ============================================
# EJEMPLOS DE CLASIFICACIÓN
# ============================================

# Ejercicio: Reducción de colores en una imagen con K-medias
# ----------------------------------------------------------
# Utiliza algún algoritmo de Clustering para rebajar adecuadamente el número de 
# colores que se usa en una imagen. Para ello, ten en cuenta que una imagen es 
# un conjunto de pixels (en forma de matriz), cada uno con un color RGB.

# Cargar una imagen PNG (ImageIO también permite leer otros formatos comunes).
ruta_imagen = "Practicas/ML_Basicos/img_robot/sensores.png"
imagen_rgb = FileIO.load(ruta_imagen)
pixeles = vec(imagen_rgb)

# Para que el coste sea razonable en imágenes grandes, se ajustan los centroides
# con una muestra repartida por toda la imagen y luego se cuantizan todos los píxeles.
K_colores = 8
max_pixeles_muestra = 10_000
n_pixeles_muestra = min(max_pixeles_muestra, length(pixeles))
indices_muestra = unique(round.(Int, range(1, length(pixeles); length = n_pixeles_muestra)))

vector_rgb(pixel) = [Float64(red(pixel)), Float64(green(pixel)), Float64(blue(pixel))]
datos_muestra_rgb = [vector_rgb(pixeles[i]) for i in indices_muestra]
K_colores = min(K_colores, length(datos_muestra_rgb))

C_rgb, Cls_rgb, pots_rgb = k_medias(datos_muestra_rgb, K_colores; iter = 50, dist = eucl)

# Reemplazar cada color por el centroide RGB más cercano.
asignaciones_rgb = Vector{Int}(undef, length(pixeles))
for i in eachindex(pixeles)
    color = vector_rgb(pixeles[i])
    asignaciones_rgb[i] = argmin([eucl2(color, centro) for centro in C_rgb])
end

pixeles_reducidos = [
    RGB(C_rgb[asignaciones_rgb[i]][1],
        C_rgb[asignaciones_rgb[i]][2],
        C_rgb[asignaciones_rgb[i]][3])
    for i in eachindex(pixeles)
]
imagen_reducida = reshape(pixeles_reducidos, size(imagen_rgb))
ruta_imagen_reducida = "Practicas/ML_Basicos/img_robot/sensores_reducida.png"
FileIO.save(ruta_imagen_reducida, imagen_reducida)

println("\n=== Reducción de colores con K-medias ===")
println("Imagen: $ruta_imagen")
println("Dimensiones: $(size(imagen_rgb, 2)) × $(size(imagen_rgb, 1))")
println("Colores distintos antes: $(length(Set(pixeles)))")
println("Colores después: $K_colores")
println("Potencial final: ", round(pots_rgb[end], digits=4))
println("Imagen reducida guardada en: $ruta_imagen_reducida")

display(plot(
    plot(imagen_rgb; axis = nothing, ticks = nothing, title = "Original"),
    plot(imagen_reducida; axis = nothing, ticks = nothing, title = "$K_colores colores"),
    layout = (1, 2), size = (1000, 400),
))

# Ejercicio: Agrupación de textos cortos por similaridad
# ------------------------------------------------------
# Idea un sistema en el que k-medias pueda usarse para agrupar textos cortos por 
# similaridad en su contenido.

const STEMMER_ESPANOL = Stemmer("spanish")
raiz_snowball(token::AbstractString) = stem(STEMMER_ESPANOL, lowercase(token))

"""
    kmedias_textos(ruta, K; iter=100, reinicios=20, dimensiones_latentes=20)

Lee un texto corto por línea, reduce sus palabras a la raíz con Snowball,
las representa con TF-IDF, reduce la representación mediante análisis
semántico latente (LSA) y las agrupa con K-medias. Snowball hace stemming,
no una lematización lingüística completa; LSA detecta coocurrencias, pero no
comprende sinónimos ni el significado general de las frases.
"""
function kmedias_textos(
    ruta::AbstractString,
    K::Int;
    iter::Int=100,
    reinicios::Int=20,
    dimensiones_latentes::Int=20,
)
    textos = [strip(linea) for linea in eachline(ruta) if !isempty(strip(linea))]
    isempty(textos) && throw(ArgumentError("El fichero no contiene textos."))
    1 <= K <= length(textos) ||
        throw(ArgumentError("K debe estar entre 1 y el número de textos ($(length(textos)))."))
    iter > 0 || throw(ArgumentError("iter debe ser mayor que cero."))
    reinicios > 0 || throw(ArgumentError("reinicios debe ser mayor que cero."))
    dimensiones_latentes > 0 ||
        throw(ArgumentError("dimensiones_latentes debe ser mayor que cero."))

    palabras_vacias = Set([
        raiz_snowball(palabra) for palabra in split(
            "a al algo algunas algunos ante antes como con contra cual cuando de del desde " *
            "donde durante e el ella ellas ellos en entre era es esa esas ese eso esos esta " *
            "estaba estaban estas este esto estos fue fueron ha hasta hay la las le les lo " *
            "los mas me mi mis mucho muy no nos o otra otras otro otros para pero por porque " *
            "que quien se sin sobre su sus te tu tus un una unas uno unos y ya",
        )
    ])
    tokens_por_texto = [
        filter!(
            token -> !(token in palabras_vacias),
            [raiz_snowball(token) for token in split(lowercase(texto), r"[^\p{L}\p{N}]+")
             if !isempty(token)],
        )
        for texto in textos
    ]
    any(isempty, tokens_por_texto) &&
        throw(ArgumentError("Cada línea no vacía debe contener al menos una palabra."))

    vocabulario = sort!(collect(Set(Iterators.flatten(tokens_por_texto))))
    indice_vocabulario = Dict(token => i for (i, token) in enumerate(vocabulario))

    frecuencias_documentales = zeros(Int, length(vocabulario))
    for tokens in tokens_por_texto
        for token in unique(tokens)
            frecuencias_documentales[indice_vocabulario[token]] += 1
        end
    end
    n_textos = length(textos)
    idf = [log((n_textos + 1) / (frecuencia + 1)) + 1
           for frecuencia in frecuencias_documentales]

    matriz_tfidf = zeros(Float64, n_textos, length(vocabulario))
    for (i, tokens) in enumerate(tokens_por_texto)
        for token in tokens
            matriz_tfidf[i, indice_vocabulario[token]] += 1
        end
    end
    matriz_tfidf .*= reshape(idf, 1, length(idf))
    for i in 1:n_textos
        norma = norm(view(matriz_tfidf, i, :))
        norma > 0 && (matriz_tfidf[i, :] ./= norma)
    end

    rango = min(dimensiones_latentes, size(matriz_tfidf, 1), size(matriz_tfidf, 2))
    descomposicion = svd(matriz_tfidf; full = false)
    matriz_latente = descomposicion.U[:, 1:rango] *
                     Diagonal(descomposicion.S[1:rango])
    vectores_latentes = [collect(matriz_latente[i, :]) for i in 1:n_textos]
    for vector in vectores_latentes
        norma = norm(vector)
        norma > 0 && (vector ./= norma)
    end

    # LSA reduce la dispersión de TF-IDF agrupando términos que coaparecen.
    resultados = [
        k_medias(vectores_latentes, K; iter = iter, dist = eucl)
        for _ in 1:reinicios
    ]
    C, Cls, pots = resultados[argmin([resultado[3][end] for resultado in resultados])]

    grupos = zeros(Int, n_textos)
    for cluster in 1:K, i in Cls[cluster]
        grupos[i] = cluster
    end

    return (
        textos = textos,
        grupos = grupos,
        centroides = C,
        potenciales = pots,
        vocabulario = vocabulario,
        dimensiones_latentes = rango,
    )
end

# Fichero de ejemplo: una frase por línea. Cambia la ruta y K para otros textos.
ruta_textos = "Practicas/ML_Basicos/textos_ejemplo2.txt"
K_textos = 10
resultado_textos = kmedias_textos(ruta_textos, K_textos)

println("\n=== Agrupación de textos con K-medias ===")
println("Fichero: $ruta_textos")
println("Vocabulario: $(length(resultado_textos.vocabulario)) términos")
println("Dimensiones tras LSA: $(resultado_textos.dimensiones_latentes)")
for cluster in 1:K_textos
    indices = findall(==(cluster), resultado_textos.grupos)
    println("\nCluster $cluster ($(length(indices)) textos):")
    for i in indices
        println("  - ", resultado_textos.textos[i])
    end
end