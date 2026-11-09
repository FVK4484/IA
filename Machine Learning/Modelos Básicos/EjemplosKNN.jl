include("KNN.jl")
using Statistics

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

# ============================================
# EJEMPLOS DE CLASIFICACIÓN
# ============================================

# Ejemplo 1: Clasificación simple de flores (características: longitud sépalo, ancho sépalo)
# ------------------------------------------------------------------------------------------
datos_flores = [
    ([5.1, 3.5], "Setosa"),
    ([4.9, 3.0], "Setosa"),
    ([6.2, 2.9], "Setosa"),
    ([7.0, 3.2], "Versicolor"),
    ([6.4, 3.2], "Versicolor"),
    ([6.9, 3.1], "Versicolor"),
    ([7.6, 3.0], "Virginica"),
    ([7.3, 2.9], "Virginica"),
    ([7.2, 3.6], "Virginica"),
]

# Normalizar datos
datos_flores_norm, medias_flores, stds_flores = normalizar(datos_flores)

# Clasificar una nueva flor
nueva_flor = [6.5, 3.1]
nueva_flor_norm = normalizar_punto(nueva_flor, medias_flores, stds_flores)
clase = knn(3, datos_flores_norm, nueva_flor_norm)

# Ejemplo 2: Clasificación de puntos en el plano
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

# Ejemplo 3: Análisis de deporte/actividad física
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

# ============================================
# EJEMPLOS DE REGRESIÓN
# ============================================

# Ejemplo 5: Predicción de precio de vivienda
# -------------------------------------------
# Características: [tamaño en m², antigüedad en años]
# Valor: precio en miles de euros
datos_viviendas = [
    ([80, 5], 150.0),
    ([85, 6], 155.0),
    ([90, 4], 165.0),
    ([120, 10], 200.0),
    ([125, 12], 195.0),
    ([130, 8], 210.0),
    ([180, 15], 300.0),
    ([185, 18], 295.0),
    ([190, 20], 310.0),
]

# Normalizar datos
datos_viviendas_norm, medias_viviendas, stds_viviendas = normalizar(datos_viviendas)

viviendas_a_predecir = [
    [95, 7],
    [115, 9],
    [175, 16],
]

println("\n=== Predicción de precio de vivienda ===")
for vivienda in viviendas_a_predecir
    vivienda_norm = normalizar_punto(vivienda, medias_viviendas, stds_viviendas)
    precio = knn_regresion(3, datos_viviendas_norm, vivienda_norm)
    println("  Vivienda $vivienda → Precio predicho: $(round(precio, digits=1))k€")
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

