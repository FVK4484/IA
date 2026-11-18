#=====================================================================
Utilidades
=====================================================================#

# Centroide de un conjunto de puntos/vectores
function centroide(D)
	sum(D)/length(D)
end

# Distancia euclídea2 (euclídea al cuadrado)
function eucl2(x, y)
	# ∑ (xᵢ - yᵢ)²
	sum((x .- y).^2)
	# sum( (xi - yi)^2 for (xi, yi) in zip(x, y) )
	
end

# distancia euclídea
function eucl(x, y)
	sqrt(eucl2(x, y))
end

# distancia manhattan
function manh(x, y)
	# ∑ |xᵢ - yᵢ|
	sum(abs.(x .- y))
	# sum( abs(xi - yi) for (xi, yi) in zip(x, y) )
end

#=====================================================================
KNN (k-Nearest Neighbors)
=====================================================================#

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
    distancias = [ (eucl2(x, x_i), y_i) for (x_i, y_i) in datos ]
    
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
    distancias = [ (eucl2(x, x_i), y_i) for (x_i, y_i) in datos ]    
    
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

#=====================================================================
K-Medias
=====================================================================#

#= 
Define la función `k_medias` que recibe:

* `D`           : Vector de puntos de muestra.
* `K`           : Número de clusters a buscar.
* `iter = 100`     : Número de pasos a ejecutar del algoritmo.
* `dist = eucl2`: distancia que se usará en `D`.

Devuelve una tupla `(C, Cls, pots)`, donde:
1.  C = [c₁, ..., cₖ] es un vector de los K centroides para los clusters.
2.  Cls es la colección de K clusters asociados a los centroides de C.
3.  pots es un vector con los potenciales de cada iteración.

Asignar a cada 1 ≤ i ≤ K el vector de índices de D que forman el cluster de cᵢ. 
Es decir, si ci está formado por {D[j₁], ..., D[jₙ]}, entonces en el diccionario 
se asocia la clave i con el conjunto de índices [j₁, ..., jₙ]
=#

# D: Dataset de Puntos
# C: Centroides
# K: |C|
# iter: Número de iteraciones (por defecto, 100)
# dist: distancia en D (por defecto, eucl2)

function k_medias(D, K ; iter = 100, dist = eucl2)
	# Inicialmente, los centroides se sitúan en K puntos al azar de D
	C   = rand(D, K)
	N   = length(D)
	Cls = Dict()
   pots = []
	# El algoritmo se ejecuta un número prefijado de pasos
	for _ in 1:iter
		# Cls es un diccionario de K clusters, inicialmente, vacíos
		Cls = Dict( [k => [] for k in 1:K] )
		# Decidimos el cluster en el que meter cada elemento de D (el índice)
		for i in 1:N
			push!( Cls[argmin([dist(D[i], c) for c in C])], i)
		end
		# Calculamos los nuevos centroides por el contenido de los clusters
      # y los potenciales asociados
      pot = 0
		for k in 1:K
			if !isempty( Cls[k] )
				C[k] = centroide( D[i] for i in Cls[k] )
            pot = pot + sum( [ dist(D[j], C[k])^2 for j in Cls[k] ] )
			end
		end
		push!(pots, pot)
	end
	# Devolvemos Clusters y centroides
	return (C, Cls, pots)
end

println("Algoritmos Básicos cargados correctamente")