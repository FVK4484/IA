# Distancia Euclidiana entre dos puntos
dist_euclidiana(x1::Vector, x2::Vector) = sum((x1 .- x2) .^ 2)

"""
    knn(k::Int, datos::Vector{Tuple}, x::Vector)

Clasifica un punto x usando el algoritmo KNN (k-Nearest Neighbors).

Argumentos:
    - k::Int: número de vecinos más cercanos a considerar
    - datos::Vector{Tuple}: vector de pares (x, y) donde:
        * x es un vector numérico de características
        * y es la clase/clasificación del punto
    - x::Vector: punto a clasificar

Retorna:
    La clase más frecuente entre los k vecinos más cercanos

Importante: 
    Suele funciona mejor si las características están normalizadas o estandarizadas

Ejemplo:
    datos = [([1.0, 2.0], "A"), ([1.5, 2.5], "A"), ([5.0, 6.0], "B")]
    knn(2, datos, [1.2, 2.1])  # Retorna "A"
"""
function knn(k::Int, datos::Vector, x::Vector)
    distancias = [ (dist_euclidiana(x, x_i), y_i) for (x_i, y_i) in datos ]
    
    # Ordenar por distancia (menor primero)
    sort!(distancias, by = d -> d[1])
    
    # Obtener los k vecinos más cercanos
    k_vecinos = distancias[1:k]
    
    # Contar las clases de los k vecinos
    clases_count = Dict()
    for (_, clase) in k_vecinos
        if haskey(clases_count, clase)
            clases_count[clase] += 1
        else
            clases_count[clase] = 1
        end
    end
    
    # Retornar la clase más frecuente
    # Si hay empate, retorna una de ellas
    clase_predicha = argmax(clases_count)
    
    return clase_predicha
end

# Función para regresión con KNN ponderado por distancia
"""
    knn_regresion(k::Int, datos::Vector{Tuple}, x::Vector)

Predice un valor numérico usando KNN con ponderación inversa por distancia.

Argumentos:
    - k::Int: número de vecinos más cercanos a considerar
    - datos::Vector{Tuple}: vector de pares (x, y) donde:
        * x es un vector numérico de características
        * y es un valor numérico (continuo)
    - x::Vector: punto para el que predecir

Retorna:
    Valor predicho como media ponderada inversamente por la distancia

El peso de cada vecino se calcula como: w_i = 1 / (1 + distancia_i)
Esto proporciona un balance entre vecinos cercanos y lejanos

Ejemplo:
    datos = [([1.0, 2.0], 5.0), ([2.0, 3.0], 7.0), ([5.0, 6.0], 15.0)]
    knn_regresion(2, datos, [1.5, 2.5])  # Predice valor cercano a 6.0
"""
function knn_regresion(k::Int, datos::Vector, x::Vector)   
    # Calcular distancias euclidianas a todos los puntos
    distancias = [ (dist_euclidiana(x, x_i), y_i) for (x_i, y_i) in datos ]    
    
    # Ordenar por distancia (menor primero)
    sort!(distancias, by = d -> d[1])
    
    # Obtener los k vecinos más cercanos
    k_vecinos = distancias[1:k]
    
    # Calcular pesos inversamente proporcionales a la distancia
    pesos = []
    valores = []
    
    for (dist, valor) in k_vecinos
        peso = 1.0 / (1.0 + dist)
        push!(pesos, peso)
        push!(valores, valor)
    end
    
    # Normalizar los pesos para que sumen 1
    peso_total = sum(pesos)
    pesos = pesos ./ peso_total
    
    # Calcular la predicción como media ponderada
    prediccion = sum(pesos .* valores)
    
    return prediccion
end
