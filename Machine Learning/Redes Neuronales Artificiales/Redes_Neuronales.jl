using Random, CSV, DataFrames, Plots

# ============================================================
# 0. CARGA DE LA LIBRERÍA
# ============================================================
include("NN.jl")
using .NN

# Float32 es el tipo estándar en Deep Learning por eficiencia y compatibilidad con GPUs
const F = Float32


# ============================================================
# UTILIDADES ADAPTADAS
# ============================================================

#=
Estas funciones auxiliares proporcionan operaciones comunes en Machine Learning:
- Escalado/normalización de datos
- Codificación one-hot de labels
- Conversión de formatos (DataFrame ↔ Matrix)
- Métricas de evaluación (error, matriz de confusión)

Algunas funciones son similares a las que ya están disponibles en ML.jl, aunque
adaptadas a las necesidades específicas de este tema, pero se ha preferido 
mantener en esta forma particular por claridad pedagógica.
=#

"""
    escala(a1, b1, a2, b2)

Crea una función de escalado lineal del intervalo [a1, b1] al intervalo [a2, b2].

# Motivación
En redes neuronales, es crucial normalizar los datos:
- Acelera la convergencia del entrenamiento
- Evita problemas numéricos (overflow/underflow)
- Hace que todas las features tengan la misma escala de importancia

# Uso común
- Normalizar a [0,1] para funciones sigmoid: escala(min, max, 0, 1)
- Normalizar a [-1,1] para funciones tanh: escala(min, max, -1, 1)
- Desnormalizar predicciones: escala(0, 1, min_original, max_original)

# Retorna
Una función que mapea valores de [a1,b1] a [a2,b2] linealmente.

# Ejemplo
```julia
f = escala(0, 100, 0, 1)  # Normalizar porcentajes a [0,1]
f(50)  # → 0.5
f(100) # → 1.0
```
"""
function escala(a1, b1, a2, b2)
    function _escala(x)
        # Paso 1: Normalizar x a [0,1] relativo a [a1,b1]
        x1 = (x - a1) / (b1 - a1)
        # Paso 2: Escalar de [0,1] a [a2,b2]
        x1 * (b2 - a2) + a2
    end
    return _escala
end

"""
    onehot(i::Integer, n::Integer) -> Vector{F}

Convierte un índice entero a su representación one-hot.

# Motivación
One-hot encoding es fundamental en clasificación multiclase:
- Convierte categorías discretas en formato numérico
- Compatible con funciones de activación como softmax
- Permite calcular cross-entropy loss de forma vectorizada
- Evita ordenamiento artificial de categorías (0,1,2... no implican orden)

# Argumentos
- `i`: Índice de la clase (1-based en Julia)
- `n`: Número total de clases

# Ejemplo
```julia
onehot(2, 4)  # → [0, 1, 0, 0]  (clase 2 de 4)
```

# Ver también
- `onehot(labels, classes)`: versión para vectores de etiquetas
"""
function onehot(i::Integer, n::Integer)
    res = zeros(F, n)  # Vector de ceros
    res[i] = 1         # Activar posición i
    return res
end

"""
    onehot(labels::AbstractVector, classes::AbstractVector) -> Matrix{F}

Convierte un vector de etiquetas categóricas a matriz one-hot.

# Motivación
Para entrenar clasificadores multiclase, necesitamos:
- Convertir labels simbólicos ("perro", "gato") a formato numérico
- Crear una matriz donde cada columna es una muestra
- Cada fila representa una clase (1 si pertenece, 0 si no)

# Argumentos
- `labels`: Vector de etiquetas (puede ser String, Symbol, Int, etc.)
- `classes`: Vector con todas las clases únicas (define el orden)

# Retorna
Matriz (n_classes, n_samples) donde:
- Cada columna es el one-hot de una muestra
- Fila i = 1 si la muestra pertenece a classes[i]

# Ejemplo
```julia
labels = ["gato", "perro", "gato", "pez"]
classes = ["gato", "perro", "pez"]
Y = onehot(labels, classes)
# Y = [1 0 1 0     # gato
#      0 1 0 0     # perro  
#      0 0 0 1]    # pez
```

# Nota
El orden en `classes` determina qué fila corresponde a cada clase.
Es crucial usar el mismo orden en entrenamiento y predicción.
"""
function onehot(labels::AbstractVector, classes::AbstractVector)
    n = length(classes)  # Número de clases
    m = length(labels)   # Número de muestras
    Y = zeros(F, n, m)   # Matriz one-hot inicializada con ceros
    
    # Para cada muestra, encontrar su clase y activar la fila correspondiente
    for (j, lab) in enumerate(labels)
        i = findfirst(==(lab), classes)  # Índice de la clase
        Y[i, j] = 1  # Activar posición (clase, muestra)
    end
    return Y
end

"""
    df_to_matrix(df::DataFrame) -> Matrix{F}

Convierte un DataFrame de features a Matrix con formato (dimensiones, muestras).

# Motivación
DataFrames son excelentes para manipulación de datos, pero las redes neuronales
necesitan matrices numéricas con formato específico:
- Cada COLUMNA = una muestra/ejemplo
- Cada FILA = una característica/feature
- Tipo Float32 (F) para eficiencia y compatibilidad con GPUs

Este formato permite operaciones matriciales eficientes: Y = W·X + b

# Operación
1. Convierte DataFrame a Matrix (filas=muestras, cols=features)
2. Transpone (') para obtener (features, muestras)
3. Convierte a Float32

# Ejemplo
```julia
df = DataFrame(altura=[1.7, 1.8], peso=[70, 80])
# DataFrame: 2 filas (muestras), 2 columnas (features)

X = df_to_matrix(df)
# Matrix: (2, 2) = 2 features × 2 muestras
# X = [1.7  1.8     # altura
#      70   80]     # peso
```

# Convención en Deep Learning
- Formato columnas = batch de datos facilita álgebra lineal
- Y = σ(W·X + b) procesa todas las muestras a la vez (vectorización)
- Compatible con frameworks como PyTorch, TensorFlow (aunque usan (batch, features))
"""
function df_to_matrix(df::DataFrame)::Matrix{F}
    return Matrix{F}(df)'  # Convertir y transponer
end

"""
    calcula_error(X::Matrix{F}, Y::Matrix{F}, net::NN.Network) -> Float32

Calcula la tasa de error de clasificación (proporción de predicciones incorrectas).

# Motivación
La tasa de error (o error rate) es una métrica simple e interpretable:
- Error = 0.0 → clasificación perfecta (100% accuracy)
- Error = 0.1 → 90% de las muestras bien clasificadas
- Error = 1.0 → todas las predicciones incorrectas

# Diferencia con Loss
- Loss (cross-entropy): mide qué tan "segura" está la red (diferenciable)
- Error: cuenta predicciones correctas/incorrectas (no diferenciable)
- Loss se usa para entrenar, Error para evaluar el rendimiento final

# Método
1. Predecir todas las muestras: preds = net(X)
2. Para cada muestra: comparar argmax(predicción) vs argmax(real)
   - argmax selecciona la clase con mayor probabilidad
   - Para one-hot [0.1, 0.8, 0.1], argmax=2 (segunda clase)
3. Contar coincidencias
4. Error = 1 - (correctas / total)

# Argumentos
- `X`: Matrix (n_features, n_samples) de datos de entrada
- `Y`: Matrix (n_classes, n_samples) con labels one-hot
- `net`: Red neuronal entrenada

# Ejemplo
```julia
# Y_real = [1, 0, 0] = clase 1
# Y_pred = [0.7, 0.2, 0.1]
# argmax(Y_real) = 1, argmax(Y_pred) = 1 → correcto ✓
```

# Ver también
- `matriz_confusion`: para análisis detallado de errores por clase
"""
function calcula_error(X::Matrix{F}, Y::Matrix{F}, net::NN.Network)::F
    preds = net(X)  # Predicciones de la red: (n_classes, n_samples)
    correct = 0
    
    # Comparar clase predicha vs clase real para cada muestra
    for j in 1:size(Y, 2)  # Iterar sobre muestras (columnas)
        # argmax devuelve el índice de la clase con mayor valor
        if argmax(preds[:, j]) == argmax(Y[:, j])
            correct += 1
        end
    end
    
    # Tasa de error = 1 - accuracy
    return F(1) - F(correct) / F(size(Y, 2))
end

"""
    matriz_confusion(X::Matrix{F}, Y::Matrix{F}, net::NN.Network) -> Matrix{Int}

Calcula y visualiza la matriz de confusión para un clasificador multiclase.

# Motivación
La matriz de confusión revela información valiosa que el error global oculta:
- ¿Qué clases se confunden entre sí?
- ¿Hay clases particularmente difíciles de clasificar?
- ¿El error está balanceado o concentrado en ciertas clases?

Ejemplo: En reconocimiento de dígitos, el 7 y el 1 suelen confundirse.

# Estructura
Matriz M donde M[i,j] = número de veces que:
- Clase real = i
- Clase predicha = j

```
       Predicción
        0   1   2
    0 [45   2   1]   # 45 ceros correctos, 2 confundidos con 1, 1 con 2
R   1 [ 3  48   0]   # 3 unos confundidos con 0, 48 correctos
e   2 [ 0   1  49]   # 1 dos confundido con 1, 49 correctos
al
```

# Interpretación
- **Diagonal**: Predicciones correctas (idealmente altos)
- **Fuera diagonal**: Confusiones (idealmente ceros)
- **Fila i**: Muestra cómo se dispersan las predicciones para la clase real i
- **Columna j**: Muestra qué clases reales se predicen como j

# Métricas derivadas
De la matriz de confusión se pueden calcular:
- **Precision[j]** = M[j,j] / sum(M[:,j]) → de las predichas como j, cuántas son j
- **Recall[i]** = M[i,i] / sum(M[i,:]) → de las reales i, cuántas se detectan
- **F1-score** = 2·(Precision·Recall)/(Precision+Recall)

# Visualización
Se genera un heatmap donde:
- Colores oscuros = valores altos
- Diagonal oscura = buen clasificador
- Celdas oscuras fuera diagonal = confusiones frecuentes

# Retorna
Matriz de confusión (n_classes × n_classes) de enteros
"""
function matriz_confusion(X::Matrix{F}, Y::Matrix{F}, net::NN.Network)
    n_classes = size(Y, 1)
    M = zeros(Int, n_classes, n_classes)  # Matriz de confusión
    preds = net(X)  # Predicciones
    
    # Llenar la matriz contando coincidencias (real, predicha)
    for j in 1:size(Y, 2)  # Para cada muestra
        y_real = argmax(Y[:, j])       # Clase real
        y_pred = argmax(preds[:, j])   # Clase predicha
        M[y_real, y_pred] += 1         # Incrementar contador
    end
    
    # Visualizar como heatmap
    heatmap(1:n_classes, 1:n_classes, M, 
            color  = :greys, 
            xlabel = "Predicción", 
            ylabel = "Real", 
            title  = "Matriz de Confusión")
    
    return M
end

# ============================================================
# 1. APROXIMACIÓN DE FUNCIONES
# ============================================================

#=
Genera y entrena redes neuronales adecuadas para aproximar las siguientes 
funciones. En las que puedas, haz una representación gráfica comparada de la 
función y de la aproximación obtenida:
=#


# ---------- 1.1 f1(x) = x^2 en [-1,1] ----------
#=
ENUNCIADO: Aproximar la función f1(x) = x² en el intervalo [-1,1]

SOLUCIÓN:
1. Generar datos de entrenamiento: 51 puntos equiespaciados en [-1,1]
2. Crear una red neuronal con arquitectura [1, 3, 3, 1]:
   - 1 neurona de entrada (x)
   - 2 capas ocultas con 3 neuronas cada una
   - 1 neurona de salida (f(x))
   - Activación sigmoid en todas las capas
3. La función x² en [-1,1] tiene rango [0,1], compatible con sigmoid
4. Entrenar con SGD, validación cruzada del 20%
5. Visualizar la aproximación vs función real
=#
function ejercicio_f1()
    println("\n=== Ejercicio 1.1: f1(x) = x^2 ===")
    
    # Definir la función objetivo y generar datos de entrenamiento (X, Y)

    f1(x) = x^2
    xs = range(-1f0, 1f0, length=51)  # 51 puntos en [-1,1]
    X = reshape(collect(xs), 1, :)    # Forma (1, 51): 1 característica, 51 muestras
    Y = reshape(f1.(xs), 1, :)        # Valores reales de f1(x)

    # Crear red neuronal: 1 entrada -> 3 -> 3 -> 1 salida
    # No necesita normalización: x² en [-1,1] da [0,1], compatible con sigmoid
    
    net = NN.Network([1, 3, 3, 1], [NN.sigmoid, NN.sigmoid, NN.sigmoid])

    println("Arquitectura de la red:")
    NN.summary(net)

    # Gradient check: verificar que el cálculo del gradiente es correcto
    # Compara el gradiente analítico (backpropagation) con el numérico
    # Es una herramienta docente fundamental para validar la implementación

    println("\nGradient check:")
    println(NN.gradient_check(net, X[:, 1:5], Y[:, 1:5]))

    # Entrenamiento de la red con los siguientes hiperparámetros:
    # - epochs = 2000: número de pasadas completas por el dataset
    # - batch_size = 10: tamaño del minibatch para SGD
    # - optimizer = SGD(1.5): descenso de gradiente estocástico con η = 1.5
    # - validation_split = 0.2: separar 20% de datos para validación (80% train)

    history = NN.fit!(net, X, Y;
        epochs=2000, batch_size=10, optimizer=NN.SGD(1.5f0),
        validation_split=0.2, verbose=false)

    # Visualización de resultados:
    # Comparar la función real con la aproximación de la red neuronal

    y_pred = vec(net(X))
    p = plot(xs, vec(Y), label= "f1(x)", linewidth=2, title="Aproximación f_1(x)=x²")
    plot!(xs, y_pred, label="NN Aprox.", linewidth=2)
    scatter!(xs, vec(Y), label="datos", markersize=2)
    display(p)

    # Curvas de aprendizaje: permiten detectar sobreajuste (overfitting)
    # Si val_loss aumenta mientras train_loss disminuye, hay overfitting

    p2 = plot(history["loss"], label="train", title="Curva de aprendizaje f1")
    plot!(history["val_loss"], label="val")
    display(p2)

    return net, history
end

ejercicio_f1()

# ---------- 1.2 f2(x) = x^3 + x^2 - log(x^2+1) ----------
#=
ENUNCIADO: Aproximar la función f2(x) = x³ + x² - log(x²+1) en [-1,1]

SOLUCIÓN:
1. Esta función tiene un rango más amplio que [0,1], por lo que necesita normalización
2. Calcular min y max de la función en el intervalo
3. Normalizar salidas a [0,1] para que sean compatibles con sigmoid
4. Entrenar la red con una arquitectura [1, 4, 4, 1]
5. Desnormalizar las predicciones para visualizarlas en escala original
=#
function ejercicio_f2()
    println("\n=== Ejercicio 1.2: f2(x) = x^3 + x^2 - log(x^2+1) ===")
    
    # Definir función y generar datos
    f2(x) = x^3 + x^2 - log(x^2 + 1)
    xs = range(-1f0, 1f0, length=50)
    X = reshape(collect(xs), 1, :)
    Y_raw = f2.(xs)  # Valores originales sin normalizar

    # Normalización necesaria: f2 no está en [0,1]
    # 1. Calcular el rango de la función
    miny, maxy = minimum(Y_raw), maximum(Y_raw)
    # 2. Crear función de escalado lineal de [miny, maxy] a [0, 1]
    esc = escala(miny, maxy, 0, 1)
    # 3. Aplicar normalización a los datos de salida
    Y = reshape(esc.(Y_raw), 1, :)

    # Red con 2 capas ocultas de 4 neuronas cada una
    net = NN.Network([1, 4, 4, 1], [NN.sigmoid, NN.sigmoid, NN.sigmoid])
    # Tasa de aprendizaje más alta (3.3) y más épocas (5000) por mayor complejidad
    history = NN.fit!(net, X, Y; epochs=5000, batch_size=10, optimizer=NN.SGD(3.3f0), validation_split=0.2, verbose=false)

    # Desnormalización: convertir predicciones de [0,1] a escala original
    y_pred_norm = vec(net(X))  # Predicciones normalizadas
    y_pred = escala(0, 1, miny, maxy).(y_pred_norm)  # Predicciones en escala original

    p = plot(xs, Y_raw, label="f2(x)", linewidth=2, title="Aproximación f2")
    plot!(xs, y_pred, label="NN Aprox.", linewidth=2)
    display(p)

    p2 = plot(history["loss"], label="train", title="Curva f2")
    plot!(history["val_loss"], label="val")
    display(p2)

    return net, history
end

ejercicio_f2()

# ---------- 1.3 f3(x,y) = (x^3 + xy + y^2)/3 ----------
#=
ENUNCIADO: Aproximar la función f3(x,y) = (x³ + xy + y²)/3 en [-1,1]×[-1,1]

SOLUCIÓN:
1. Función de 2 variables -> 2 neuronas de entrada
2. Generar un grid 20×20 = 400 puntos de entrenamiento
3. Crear matriz X de tamaño (2, 400): cada columna es un punto (x,y)
4. Normalizar salidas a [0,1]
5. Usar ReLU en capas ocultas (converge mejor que sigmoid para funciones complejas)
6. Visualizar con gráfico de superficie 3D
=#
function ejercicio_f3()
    println("\n=== Ejercicio 1.3: f3(x,y) = (x^3 + xy + y^2)/3 ===")
    f3(x, y) = (x^3 + x*y + y^2) / 3
    x1s = range(-1f0, 1f0, length=20)
    x2s = range(-1f0, 1f0, length=20)

    # Construir dataset con grid 2D:
    # - X tiene 2 filas (coordenadas x, y) y 400 columnas (puntos)
    # - Y tiene 1 fila (valor de f3) y 400 columnas
    n = length(x1s) * length(x2s)  # 20 × 20 = 400
    X = Matrix{F}(undef, 2, n)
    Y = Matrix{F}(undef, 1, n)
    idx = 1
    # Iterar sobre todos los pares (x1, x2) en el grid
    for x1 in x1s, x2 in x2s
        X[:, idx] = [x1, x2]  # Punto de entrada
        Y[1, idx] = f3(x1, x2)  # Valor objetivo
        idx += 1
    end

    # Normalizar salidas a [0,1]
    miny, maxy = minimum(Y), maximum(Y)
    Y = (Y .- miny) ./ (maxy - miny)

    # Arquitectura: 2 entradas -> 4 -> 4 -> 1 salida
    # ReLU en capas ocultas: f(x) = max(0,x), mejor para funciones no lineales complejas
    # Sigmoid en salida: asegura que la salida esté en [0,1]
    net = NN.Network([2, 4, 4, 1], [NN.relu, NN.relu, NN.sigmoid])
    # Batch más grande (32) porque tenemos más datos (400 puntos)
    history = NN.fit!(net, X, Y; epochs=2000, batch_size=32, optimizer=NN.SGD(0.1f0), validation_split=0.2, verbose=false)

    # Generar superficie aproximada
    y_pred = net(X)  # Predicciones normalizadas
    y_pred_denorm = y_pred .* (maxy - miny) .+ miny  # Desnormalizar

    # Reorganizar datos en forma de matriz para surface plot
    Z_approx = reshape(y_pred_denorm, length(x2s), length(x1s))
    Z_real = [f3(x1, x2) for x2 in x2s, x1 in x1s]

    p = surface(x1s, x2s, Z_approx, title="Aproximación NN f3", alpha=0.9)
    surface!(x1s, x2s, Z_real, alpha=0.5)
    display(p)

    p2 = plot(history["loss"], label="train", title="Curva f3")
    plot!(history["val_loss"], label="val")
    display(p2)

    return net, history
end

ejercicio_f3()

# ---------- 1.4 f4(x,y,z) = (x^2+y^2, e^y-e^x) ----------
#=
ENUNCIADO: Aproximar f4(x,y,z) = (x²+y², e^y-e^x) en [-1,1]³

SOLUCIÓN:
1. Función con 3 entradas y 2 salidas
2. Generar grid 10×10×10 = 1000 puntos (grid más pequeño por memoria)
3. Red: 3 neuronas entrada -> 8 -> 8 -> 2 neuronas salida
4. Normalizar CADA salida de forma independiente
5. Notar que la variable z no afecta la salida (solo x,y) - la red lo aprenderá
=#
function ejercicio_f4()
    println("\n=== Ejercicio 1.4: f4(x,y,z) = (x^2+y^2, e^y-e^x) ===")
    # Función vectorial: devuelve un vector de 2 componentes
    f4(x, y, z) = [x^2 + y^2, exp(y) - exp(x)]

    # Grid 3D: 10×10×10 = 1000 puntos
    # Grid más pequeño que en f3 porque el problema es más complejo (3D -> 2D)
    xs = range(-1f0, 1f0, length=10)
    n = length(xs)^3  # 1000 puntos
    X = Matrix{F}(undef, 3, n)  # 3 coordenadas de entrada
    Y = Matrix{F}(undef, 2, n)  # 2 valores de salida
    idx = 1
    # Triple bucle para generar todos los puntos del cubo 3D
    for x in xs, y in xs, z in xs
        X[:, idx] = [x, y, z]
        Y[:, idx] = f4(x, y, z)
        idx += 1
    end

    # Normalizar cada salida de forma INDEPENDIENTE
    # Esto es importante cuando las salidas tienen diferentes escalas
    for i in 1:2
        miny, maxy = minimum(Y[i, :]), maximum(Y[i, :])
        Y[i, :] = (Y[i, :] .- miny) ./ (maxy - miny)
    end

    # Red más grande: 3 -> 8 -> 8 -> 2
    # ReLU ayuda con la capacidad de representación
    net = NN.Network([3, 8, 8, 2], [NN.relu, NN.relu, NN.sigmoid])
    # Batch grande (64) y más épocas por mayor complejidad
    history = NN.fit!(net, X, Y; epochs=3000, batch_size=64, optimizer=NN.SGD(0.1f0), validation_split=0.2, verbose=false)

    loss_final = NN.compute_loss(net, X, Y)
    println("Pérdida final f4: $loss_final")

    p = plot(history["loss"], label="train", title="Curva f4")
    plot!(history["val_loss"], label="val")
    display(p)

    return net, history
end

ejercicio_f4()

# ---------- 1.5 f5(x) = 1 si x∈[0,1], 0 e.o.c ----------
#=
ENUNCIADO: Aproximar la función escalón (función indicadora del intervalo [0,1])

SOLUCIÓN:
1. Función discontinua: un desafío para redes neuronales
2. Red con 2 capas ocultas de 10 neuronas cada una
3. Más neuronas necesarias para aproximar discontinuidades
4. Muchas épocas (5000) y tasa de aprendizaje moderada (0.5)
5. El resultado será una aproximación suave de la función escalón
   (sigmoid no puede representar exactamente una discontinuidad)
=#
function ejercicio_f5()
    println("\n=== Ejercicio 1.5: Función escalón f5 ===")
    # Función indicadora del intervalo [0,1]
    f5(x) = (0 <= x <= 1) ? 1.0f0 : 0.0f0
    # Rango más amplio [-1, 2] para capturar el comportamiento fuera del intervalo
    xs = range(-1f0, 2f0, length=100)
    X = reshape(collect(xs), 1, :)
    Y = reshape(f5.(xs), 1, :)

    # Red con más capacidad: 10 neuronas por capa
    # Las discontinuidades necesitan más parámetros para ser aproximadas
    net = NN.Network([1, 10, 10, 1], [NN.sigmoid, NN.sigmoid, NN.sigmoid])
    # Muchas épocas para refinar la aproximación de la discontinuidad
    history = NN.fit!(net, X, Y; epochs=5000, batch_size=20, optimizer=NN.SGD(0.5f0), validation_split=0.2, verbose=false)

    y_pred = vec(net(X))
    # La aproximación será una función sigmoide suave, no un escalón exacto
    p = plot(xs, vec(Y), label="f5(x)", linewidth=2, title="Aproximación escalón")
    plot!(xs, y_pred, label="NN Aprox.", linewidth=2)
    display(p)

    return net, history
end

ejercicio_f5()

# ============================================================
# 2. CLASIFICACIÓN: IRIS
# ============================================================

#=
ENUNCIADO EJERCICIO 2: Clasificación multiclase con el dataset Iris

El dataset Iris contiene 150 muestras de flores con 4 características:
- Longitud del sépalo (sepal length)
- Anchura del sépalo (sepal width)
- Longitud del pétalo (petal length)
- Anchura del pétalo (petal width)

Y 3 clases (especies):
- Setosa
- Versicolor
- Virginica

SOLUCIÓN:
1. Cargar datos con CSV y DataFrames
2. Separar features (4) de labels (especies)
3. Codificar labels con one-hot encoding: 
   - Setosa     -> [1, 0, 0]
   - Versicolor -> [0, 1, 0]
   - Virginica  -> [0, 0, 1]
4. Normalizar features dividiéndolas por 8 (escala típica de Iris)
5. Dividir en train (80%) y test (20%)
6. Entrenar red: 4 entradas -> 4 ocultas -> 3 salidas
7. Evaluar con matriz de confusión y accuracy
=#


function ejercicio_iris()
    println("\n=== Ejercicio 2: Dataset Iris ===")

    # 1. Cargar datos desde CSV
    df = CSV.read("./Practicas/datasets/iris.csv", DataFrame)

    # 2. Preparar datos en formato matricial
    # Separar features (todas las columnas excepto 'class')
    X_raw = df_to_matrix(select(df, Not(:class)))  # (4, 150): 4 features, 150 muestras
    
    # Obtener las clases únicas y codificar con one-hot
    classes = unique(df.class)  # ["Setosa", "Versicolor", "Virginica"]
    Y = onehot(df.class, classes)  # (3, 150): 3 clases one-hot encoded

    # 3. Normalización de features: dividir por 8 es típico para Iris
    # (los valores de Iris suelen estar en el rango 0-8)
    X = X_raw ./ 8.0f0

    # 4. División train/test (80%/20%) con semilla aleatoria fija
    Random.seed!(42)  # Para reproducibilidad
    n = size(X, 2)  # Número total de muestras (150)
    perm = randperm(n)  # Permutación aleatoria de índices
    n_train = floor(Int, 0.8 * n)  # 120 muestras para entrenamiento
    idx_train = perm[1:n_train]
    idx_test = perm[n_train+1:end]  # 30 muestras para test

    X_train, Y_train = X[:, idx_train], Y[:, idx_train]
    X_test, Y_test = X[:, idx_test], Y[:, idx_test]

    # 5. Crear arquitectura de red: 4 -> 4 -> 3
    # - 4 entradas: las 4 características de la flor
    # - 4 neuronas ocultas: capacidad suficiente para Iris (problema simple)
    # - 3 salidas: una por cada clase (one-hot)
    net = NN.Network([4, 4, 3], [NN.sigmoid, NN.sigmoid])
    println("\nArquitectura Iris:")
    NN.summary(net)

    # 6. Gradient check: validar la implementación del backpropagation
    println("\nGradient check (5 samples):")
    println(NN.gradient_check(net, X_train[:, 1:5], Y_train[:, 1:5]))

    # 7. Entrenamiento con monitoreo de validación
    # validation_split=0.2: 20% del train se usa para validación
    # Esto da: 96 train, 24 validación, 30 test
    history = NN.fit!(net, X_train, Y_train;
        epochs=1000, batch_size=10, optimizer=NN.SGD(0.4f0),
        validation_split=0.2, verbose=true)

    # 8. Visualizar curvas de aprendizaje
    # Loss (pérdida): debe bajar en train y validación
    p1 = plot(history["loss"], label="train loss", title="Iris - Curvas de aprendizaje")
    plot!(history["val_loss"], label="val loss")
    display(p1)

    # Accuracy (precisión): debe subir en train y validación
    p2 = plot(history["acc"], label="train acc")
    plot!(history["val_acc"], label="val acc")
    display(p2)

    # 9. Evaluación final en conjuntos de train y test
    # Error de clasificación: proporción de predicciones incorrectas
    println("\nError train: ", calcula_error(X_train, Y_train, net))
    println("Error test:  ", calcula_error(X_test, Y_test, net))

    # 10. Matriz de confusión: muestra qué clases se confunden entre sí
    # Filas = clases reales, Columnas = clases predichas
    # Diagonal = predicciones correctas
    println("\nMatriz de confusión (test):")
    M = matriz_confusion(X_test, Y_test, net)
    display(M)

    # 11. Ejemplo de predicción individual
    # Predecir la especie para una flor con medidas específicas
    println("\nPredicción muestra [5.1, 3.5, 1.4, 0.2]/8:")
    # La red devuelve probabilidades para cada clase
    println(net([5.1f0, 3.5f0, 1.4f0, 0.2f0] ./ 8.0f0))

    return net, history
end

ejercicio_iris()

# ============================================================
# 3. CLASIFICACIÓN: MNIST
# ============================================================

#=
ENUNCIADO EJERCICIO 3: Reconocimiento de dígitos escritos a mano (MNIST)

Dataset MNIST:
- 10,000 ejemplos en mnist_test.csv (o 60,000 en mnist_train.csv)
- Cada ejemplo: 785 componentes
  * Columna 1: etiqueta del dígito (0-9)
  * Columnas 2-785: imagen 28×28 píxeles aplanada (valores 0-255)

Objetivo: Clasificar dígitos (10 clases) a partir de imágenes

SOLUCIÓN:
1. Cargar datos y separar labels de píxeles
2. Normalizar píxeles a [0,1] dividiendo por 255
3. Codificar labels con one-hot (10 clases)
4. Dividir en train/test (80/20)
5. Red: 784 entradas -> 30 ocultas -> 10 salidas
6. Entrenar y evaluar con matriz de confusión
=#


"""Visualiza un dígito de MNIST a partir de su vector de 784 píxeles."""
function show_digito(ps::Vector{F})
    # Invertir colores: 255 (blanco) - valor normalizado
    pixels = [255 - 255 * j for j in ps]
    # Reorganizar vector (784,) en matriz 28×28 y transponer
    img = reshape(pixels, 28, 28)'
    # Normalizar a [0,1] y mostrar en escala de grises
    plot(Gray.(img ./ 255), size=(200, 200), axis=false, ticks=false)
end

function ejercicio_mnist()
    println("\n=== Ejercicio 3: Dataset MNIST ===")

    # 1. Cargar datos (sin headers)
    df = CSV.read("./Practicas/datasets/mnist_test.csv", DataFrame, header=false)

    # 2. Separar labels (columna 1) de píxeles (columnas 2-785)
    y_labels = df[:, 1]  # Labels: 0, 1, 2, ..., 9
    X_raw = df_to_matrix(df[:, 2:end])  # (784, 10000): 784 píxeles, 10K imágenes

    # 3. Normalizar píxeles de [0,255] a [0,1]
    # Importante para compatibilidad con sigmoid y convergencia más rápida
    X = X_raw ./ 255.0f0

    # 4. One-hot encoding de las 10 clases
    # Label 0 -> [1,0,0,0,0,0,0,0,0,0]
    # Label 1 -> [0,1,0,0,0,0,0,0,0,0]
    # ...
    # Label 9 -> [0,0,0,0,0,0,0,0,0,1]
    Y = zeros(F, 10, size(X, 2))
    for j in 1:size(X, 2)
        Y[y_labels[j] + 1, j] = 1  # +1 porque labels son 0-9, índices 1-10
    end

    # 5. División train/test (80%/20%)
    # Nota: Si tienes mnist_train.csv, úsalo para train y este para test
    Random.seed!(42)
    n = size(X, 2)  # 10,000 muestras
    perm = randperm(n)
    n_train = floor(Int, 0.8 * n)  # 8,000 para train
    idx_train = perm[1:n_train]
    idx_test = perm[n_train+1:end]  # 2,000 para test

    X_train, Y_train = X[:, idx_train], Y[:, idx_train]
    X_test, Y_test = X[:, idx_test], Y[:, idx_test]

    # 6. Arquitectura de red: 784 -> 30 -> 10
    # - 784 entradas: una por cada píxel
    # - 30 neuronas ocultas: compresión significativa (784 -> 30)
    # - 10 salidas: una por cada dígito (0-9)
    Random.seed!(123)
    net = NN.Network([784, 30, 10], [NN.sigmoid, NN.sigmoid])
    NN.summary(net)

    # 7. Entrenamiento
    # Menos épocas (100) porque MNIST es grande y toma tiempo
    history = NN.fit!(net, X_train, Y_train;
        epochs=100, batch_size=10, optimizer=NN.SGD(0.4f0),
        validation_split=0.1, verbose=true)

    # 8. Visualizar convergencia
    p = plot(history["loss"], label="train", title="MNIST - Curvas")
    plot!(history["val_loss"], label="val")
    display(p)

    # 9. Evaluación final
    println("\nError train: ", calcula_error(X_train, Y_train, net))
    println("Error test:  ", calcula_error(X_test, Y_test, net))

    # 10. Matriz de confusión: ver qué dígitos se confunden
    # Por ejemplo, 7 y 1 suelen confundirse, igual que 4 y 9
    println("\nMatriz de confusión (test):")
    M = matriz_confusion(X_test, Y_test, net)

    # 11. Inspección visual de una predicción
    # Seleccionar una imagen aleatoria del test set
    i = rand(1:size(X_test, 2))
    x = X_test[:, i]
    y_real = argmax(Y_test[:, i]) - 1  # -1 para convertir índice a label
    y_pred = argmax(net(x)) - 1
    println("\nMuestra test $i: real=$y_real, pred=$y_pred")
    # Visualizar la imagen
    p1 = show_digito(x)
    display(p1)

    return net, history
end

ejercicio_mnist()

# ============================================================
# 4. APRENDIZAJE DE REPRESENTACIÓN: AUTOENCODER CON IRIS
# ============================================================

#=
ENUNCIADO EJERCICIO 4: Autoencoder para Aprendizaje de Representación (Iris)

Concepto de Autoencoder:
- Red neuronal que aprende a reconstruir su entrada: f(x) ≈ x
- Arquitectura simétrica con cuello de botella (bottleneck) en el centro
- Ejemplo: [4, 10, 2, 10, 4]
  * Encoder: 4 -> 10 -> 2 (compresión)
  * Decoder: 2 -> 10 -> 4 (reconstrucción)

Objetivo:
- La capa del medio (2 dimensiones) captura la información esencial
- Si la red puede reconstruir x desde 2 dimensiones, esas 2 dim contienen
  toda la información relevante comprimida

Aplicación:
- Reducción de dimensionalidad (4D -> 2D)
- Visualización de datos en 2D
- Detección de anomalías (muestras que no se reconstruyen bien)
- Pre-entrenamiento para otras tareas

SOLUCIÓN:
1. Usar solo las features (X), no las labels
2. Entrenar red a reconstruir: X -> encoder -> bottleneck -> decoder -> X̂
3. Minimizar error de reconstrucción: ||X - X̂||²
4. Extraer representación 2D del bottleneck y visualizar
5. Colorear por clase para ver si el autoencoder separa especies
=#

function ejercicio_autoencoder_iris()
    println("\n=== Ejercicio 4: Autoencoder Iris ===")

    # 1. Cargar datos (solo features, sin labels)
    df = CSV.read("./Practicas/datasets/iris.csv", DataFrame)
    X_raw = df_to_matrix(select(df, Not(:class)))  # (4, 150)
    X = X_raw ./ 8.0f0  # Normalizar

    # Guardar clases para colorear visualizaciones
    classes = unique(df.class)
    Y_labels = df.class

    # 2. Crear Autoencoder simétrico: 4 -> 10 -> 2 -> 10 -> 4
    # - Compresión: 4 dimensiones -> 2 dimensiones (bottleneck)
    # - Reconstrucción: 2 dimensiones -> 4 dimensiones
    # - Objetivo: X̂ ≈ X, donde X̂ = decoder(encoder(X))
    ae = NN.Autoencoder([4, 10, 2, 10, 4], NN.sigmoid)
    println("\nAutoencoder Iris:")
    NN.summary(ae.net)

    # 3. Entrenar a reconstruir la entrada
    # La pérdida es el error de reconstrucción: MSE(X, X̂)
    history = NN.fit_autoencoder!(ae, X;
        epochs=3000, batch_size=10, optimizer=NN.SGD(1.0f0),
        validation_split=0.2, verbose=true)

    # 4. Visualizar curva de aprendizaje (error de reconstrucción)
    p = plot(history["loss"], label="train", title="AE Iris - Reconstrucción")
    plot!(history["val_loss"], label="val")
    display(p)

    # 5. Verificar calidad de reconstrucción en muestras individuales
    println("\nReconstrucciones (primeras 5 muestras):")
    for j in 1:5
        x = X[:, j]  # Entrada original
        x̂ = ae(x)  # Reconstrucción: ae(x) hace encoding y luego decoding
        err = sum((x .- x̂).^2)  # Error cuadrático
        println("  x=$(round.(x, digits=2)) -> x̂=$(round.(x̂, digits=2)), error=$err")
    end

    # 6. Extraer codificación 2D (representación del bottleneck)
    # ae(X, Val(:encode)) ejecuta solo el encoder: 4D -> 2D
    H = ae(X, Val(:encode))  # (2, 150): 2 dimensiones, 150 muestras
    
    # 7. Visualizar espacio latente 2D coloreado por especie
    # Si las especies se separan bien, el autoencoder ha aprendido
    # características discriminativas (aunque fue entrenado sin labels!)
    colors = [:red, :green, :blue]
    cidx = [findfirst(==(lab), classes) for lab in Y_labels]
    markcolors = [colors[i] for i in cidx]

    p = scatter(H[1, :], H[2, :], mc=markcolors, legend=false,
                title="Codificación 2D Autoencoder Iris",
                xlabel="z1", ylabel="z2")
    display(p)

    return ae, history
end

ejercicio_autoencoder_iris()  

# ============================================================
# 5. APRENDIZAJE DE REPRESENTACIÓN: AUTOENCODER CON MNIST
# ============================================================

#=
ENUNCIADO EJERCICIO 5: Autoencoder para MNIST

Objetivo: Comprimir imágenes de dígitos de 784 dimensiones a 2 dimensiones

Desafío: 
- Compresión extrema: 784 -> 2 (factor de 392)
- ¿Pueden 2 números capturar la esencia de un dígito?

Arquitectura: 784 -> 30 -> 2 -> 30 -> 784
- Encoder: comprime gradualmente 784 -> 30 -> 2
- Decoder: reconstruye 2 -> 30 -> 784

Aplicaciones:
1. Visualización: mapear dígitos a plano 2D
2. Interpolación: generar dígitos "intermedios"
3. Compresión: almacenar solo 2 números en vez de 784
=#

function ejercicio_autoencoder_mnist()
    println("\n=== Ejercicio 5: Autoencoder MNIST ===")

    # 1. Cargar y preparar datos (solo píxeles, sin labels)
    df = CSV.read("./Practicas/datasets/mnist_test.csv", DataFrame, header=false)
    X_raw = df_to_matrix(df[:, 2:end])  # (784, 10000)
    X = X_raw ./ 255.0f0  # Normalizar a [0,1]

    # 2. Autoencoder con compresión extrema: 784 -> 30 -> 2 -> 30 -> 784
    # Solo 2 números para representar cada dígito!
    ae = NN.Autoencoder([784, 30, 2, 30, 784], NN.sigmoid)
    NN.summary(ae.net)

    # 3. Entrenar (menos épocas porque MNIST es grande)
    history = NN.fit_autoencoder!(ae, X;
        epochs=100, batch_size=10, optimizer=NN.SGD(0.4f0),
        validation_split=0.1, verbose=true)

    # 4. Visualizar reconstrucción de una imagen aleatoria
    i = rand(1:size(X, 2))
    x = X[:, i]  # Imagen original (784 píxeles)
    x̂ = ae(x)  # Reconstrucción desde 2 dimensiones
    p1 = show_digito(x)
    p2 = show_digito(x̂)
    p = plot(p1, p2, layout=(1, 2), title=["Original" "Reconstrucción"])
    display(p)

    # 5. Interpolación en espacio latente:
    # Crear dígitos "intermedios" entre dos imágenes
    println("\nInterpolación entre dos dígitos:")
    a1, a2 = 1, 10  # Índices de dos imágenes
    # Codificar ambas imágenes a 2D
    h1 = ae(X[:, a1:a1], Val(:encode))[:, 1]  # Vector 2D del dígito 1
    h2 = ae(X[:, a2:a2], Val(:encode))[:, 1]  # Vector 2D del dígito 2

    # Generar animación: interpolar en el espacio latente 2D
    # h(t) = (1-t)*h1 + t*h2, con t ∈ [0,1]
    # Cada h intermedio se decodifica a una imagen 784D
    anim = Animation()
    for t in 0.0f0:0.1f0:1.0f0  # 11 frames de h1 a h2
        h = (1-t) * h1 + t * h2  # Interpolación lineal en espacio latente
        # Decodificar: pasar h por la segunda mitad del autoencoder
        x_interp = NN.forward_from(ae.net, reshape(h, :, 1), ae.bottleneck + 1)
        show_digito(vec(x_interp))
        frame(anim)
    end
    # Guardar como GIF: muestra la transformación gradual entre dígitos
    gif(anim, "mnist_interpolacion.gif", fps=10)

    return ae, history
end

ejercicio_autoencoder_mnist()

# ============================================================
# 6. DENOISING AUTOENCODER
# ============================================================

#=
ENUNCIADO EJERCICIO 6: Autoencoder para eliminación de ruido (Denoising)

Problema: Imágenes corruptas por ruido
Solución: Entrenar autoencoder que:
  1. Recibe imagen ruidosa: X_noisy = X + ruido
  2. Reconstruye imagen limpia: X̂ ≈ X

Diferencia con autoencoder normal:
- Normal: X -> encoder -> decoder -> X̂ ≈ X
- Denoising: X_noisy -> encoder -> decoder -> X̂ ≈ X (imagen limpia!)

Aplicaciones:
- Restauración de imágenes
- Pre-procesamiento robusto
- Regularización (evita overfitting)

Técnica:
- Añadir ruido gaussiano: X_noisy = X + ε, donde ε ~ N(0, σ²)
- Entrenar a reconstruir X desde X_noisy
- Red aprende a ignorar ruido y capturar estructura esencial
=#

function ejercicio_denoising_mnist()
    println("\n=== Ejercicio 6: Denoising Autoencoder MNIST ===")

    # 1. Cargar datos
    df = CSV.read("./Practicas/datasets/mnist_test.csv", DataFrame, header=false)
    X_raw = df_to_matrix(df[:, 2:end])
    X = X_raw ./ 255.0f0

    # 2. Crear autoencoder (misma arquitectura que ejercicio 5)
    ae = NN.Autoencoder([784, 30, 2, 30, 784], NN.sigmoid)

    # 3. Entrenamiento con ruido
    # noise_level=0.2: desviación estándar del ruido gaussiano
    # En cada época: X_noisy = X + N(0, 0.2²)
    # Objetivo: minimizar ||X̂ - X||, NO ||X̂ - X_noisy||
    history = NN.fit_denoising!(ae, X;
        noise_level=0.2f0, epochs=100, batch_size=10,
        optimizer=NN.SGD(0.4f0), validation_split=0.1, verbose=true)

    # 4. Prueba de denoising en una imagen
    i = rand(1:size(X, 2))
    x_clean = X[:, i]  # Imagen original limpia
    # Añadir ruido artificial
    x_noisy = NN.add_noise(reshape(x_clean, :, 1), 0.2f0)[:, 1]
    # Reconstruir (eliminar ruido)
    x_recovered = ae(x_noisy)

    # 5. Comparación visual: Original vs Ruidosa vs Denoised
    p1 = show_digito(x_clean)
    p2 = show_digito(x_noisy)
    p3 = show_digito(x_recovered)
    p = plot(p1, p2, p3, layout=(1, 3),
             title=["Original" "Ruidosa (+0.2)" "Denoised"])
    display(p)

    # 6. Métricas cuantitativas
    println("Error noisy vs clean:  ", sum((x_noisy .- x_clean).^2))
    println("Error recovered vs clean:", sum((x_recovered .- x_clean).^2))
    # La imagen recuperada debe tener menor error que la ruidosa

    return ae, history
end

ejercicio_denoising_mnist()

# ============================================================
# 7. COMPARACIÓN DE OPTIMIZADORES
# ============================================================

#=
ENUNCIADO EJERCICIO 7: Comparar SGD vs Momentum

Optimizadores:
1. SGD (Stochastic Gradient Descent):
   w ← w - η ∇L(w)
   - Simple pero puede oscilar
   - Sensible a tasa de aprendizaje

2. Momentum:
   v ← βv + ∇L(w)
   w ← w - ηv
   - Acumula inercia en la dirección del gradiente
   - Reduce oscilaciones
   - Acelera convergencia en direcciones consistentes
   - β ∈ [0,1] (típico: 0.9)

Experimento: Entrenar dos redes idénticas con diferentes optimizadores
y comparar velocidad de convergencia y estabilidad.
=#

function ejercicio_optimizadores()
    println("\n=== Ejercicio 7: Comparación SGD vs Momentum ===")

    # Dataset simple para comparación rápida: f1(x)=x²
    xs = range(-1f0, 1f0, length=51)
    X = reshape(collect(xs), 1, :)
    Y = reshape(xs.^2, 1, :)

    # Red 1: Entrenar con SGD puro
    net_sgd = NN.Network([1, 3, 3, 1], [NN.sigmoid, NN.sigmoid, NN.sigmoid])
    hist_sgd = NN.fit!(net_sgd, X, Y; epochs=500, batch_size=10,
                       optimizer=NN.SGD(0.5f0), validation_split=0.2, verbose=false)

    # Red 2: Entrenar con Momentum
    # Momentum(net, η, β): η=tasa aprendizaje, β=factor momentum (0.9)
    net_mom = NN.Network([1, 3, 3, 1], [NN.sigmoid, NN.sigmoid, NN.sigmoid])
    opt_mom = NN.Momentum(net_mom, 0.5f0, 0.9f0)
    hist_mom = NN.fit!(net_mom, X, Y; epochs=500, batch_size=10,
                       optimizer=opt_mom, validation_split=0.2, verbose=false)

    # Red 3: Adam
      net_adam = NN.Network([1, 3, 3, 1], [NN.sigmoid, NN.sigmoid, NN.sigmoid])
      # net_adam = NN.Network([784, 30, 10], [NN.relu, NN.sigmoid])
      opt_adam = NN.Adam(net_adam, eta=0.001f0)  # beta1, beta2, epsilon por defecto
      hist_adam = NN.fit!(net_adam, X, Y; epochs=500, optimizer=opt_adam, validation_split=0.1, verbose=false)

    # Comparación visual de convergencia
    # Momentum suele converger más rápido y con menos oscilaciones
    p = plot(hist_sgd["loss"], label="SGD", title="SGD vs Momentum",
             linewidth=2)
    plot!(hist_mom["loss"], label="Momentum", linewidth=2)
    plot!(hist_adam["loss"], label="Adam", linewidth=2)
    plot!(hist_sgd["val_loss"], label="SGD val", linestyle=:dash)
    plot!(hist_mom["val_loss"], label="Momentum val", linestyle=:dash)
    plot!(hist_adam["val_loss"], label="Adam val", linestyle=:dash)
    display(p)

    println("Loss final SGD:     ", hist_sgd["loss"][end])
    println("Loss final Momentum:", hist_mom["loss"][end])
    println("Loss final Adam:    ", hist_adam["loss"][end])

    return hist_sgd, hist_mom, hist_adam
end

ejercicio_optimizadores()

# ============================================================
# 8. WORD2VEC
# ============================================================

#=
ENUNCIADO EJERCICIO 8: Word2Vec - Embeddings de palabras

Concepto:
- Representar palabras como vectores densos de números reales
- Palabras con significados similares tienen vectores similares
- Captura relaciones semánticas y sintácticas

Modelo Skip-gram:
- Input: palabra central (e.g., "gato")
- Output: palabras de contexto (e.g., "el", "come", "pescado")
- Red aprende embeddings que predicen contexto

Negative Sampling:
- Técnica de optimización eficiente
- En vez de calcular softmax sobre todo el vocabulario,
  solo comparar con k palabras negativas aleatorias

Propiedades interesantes:
- Similaridad: cos("gato", "perro") > cos("gato", "mesa")
- Analogías: rey - hombre + mujer ≈ reina

Aplicaciones:
- Pre-entrenamiento para NLP
- Búsqueda semántica
- Clustering de documentos
- Sistemas de recomendación
=#

function ejercicio_word2vec()
    println("\n=== Ejercicio 8: Word2Vec ===")

    # 1. Definir vocabulario (pequeño para demo)
    # Cada palabra tiene un ID único
    vocab = Dict("el"=>1, "gato"=>2, "come"=>3, "pescado"=>4,
                 "perro"=>5, "carne"=>6, "corre"=>7, "rápido"=>8)

    # 2. Corpus de entrenamiento (secuencia de palabras)
    # En aplicaciones reales: millones de palabras
    corpus_text = ["el", "gato", "come", "pescado", "el", "perro", "come", "carne",
                   "el", "perro", "corre", "rápido", "el", "gato", "corre", "rápido"]
    # Convertir palabras a IDs
    corpus = [vocab[w] for w in corpus_text]

    # 3. Crear modelo Word2Vec
    # Parámetros:
    # - vocab_size=8: número de palabras únicas
    # - embed_dim=10: dimensión de los vectores de embedding
    # - window=2: contexto = 2 palabras a cada lado
    # - negative_samples=5: usar 5 palabras negativas por muestra
    w2v = NN.Word2Vec(8, 10, window=2, negative_samples=5)
    
    # 4. Entrenar con Skip-gram
    # Para cada palabra en el corpus, predecir palabras cercanas
    NN.fit_skipgram!(w2v, corpus; epochs=2000, eta=0.01f0)

    # 5. Consultar palabras más similares usando similaridad coseno
    # cos(θ) = (u·v) / (||u|| ||v||)
    println("\nPalabras más similares a 'gato' (idx 2):")
    sims = NN.most_similar(w2v, 2; n=3)  # Top 3 más similares
    for (idx, sim) in sims
        word = findfirst(==(idx), vocab)
        println("  $word -> sim=$sim")
    end
    # Esperamos: "perro" sea similar a "gato" (ambos animales)

    # 6. Analogías: resolver "A es a B como C es a ?"
    # Mediante aritmética vectorial: vec(?) ≈ vec(B) - vec(A) + vec(C)
    println("\nAnalogía: gato:pescado :: perro:?")
    # Gato come pescado, ¿qué come el perro?
    res = NN.analogy(w2v, vocab["gato"], vocab["pescado"], vocab["perro"]; n=1)
    word = findfirst(==(res[1][1]), vocab)
    println("  Resultado: $word (sim=$(res[1][2]))")
    # Esperamos: "carne" (porque "perro come carne" en el corpus)

    return w2v
end

ejercicio_word2vec()

# ============================================================
# 9. GAN
# ============================================================

#=
ENUNCIADO EJERCICIO 9: GAN (Generative Adversarial Network)

Concepto:
- Dos redes neuronales compitiendo:
  1. Generador (G): crea datos falsos desde ruido aleatorio
  2. Discriminador (D): distingue datos reales de falsos

Dinámica de entrenamiento:
- D intenta maximizar: log(D(x_real)) + log(1 - D(G(z)))
  (clasificar correctamente reales y falsos)
- G intenta minimizar: log(1 - D(G(z)))
  (engañar al discriminador)

Proceso:
1. Entrenar D con datos reales (label=1) y falsos (label=0)
2. Entrenar G para que D clasifique sus generaciones como reales
3. Repetir hasta equilibrio: D no puede distinguir, G genera datos realistas

Aplicaciones:
- Generación de imágenes (DeepFake, arte)
- Aumento de datos (data augmentation)
- Super-resolución
- Traducción imagen-a-imagen

Ejemplo: Generar puntos en un círculo 2D
- Datos reales: puntos en círculo
- Ruido: vectores aleatorios 2D
- G aprende a transformar ruido en puntos del círculo
=#

function ejercicio_gan()
    println("\n=== Ejercicio 9: GAN (Generative Adversarial Network) ===")

    # 1. Generar datos reales: puntos en un círculo 2D con ruido
    n = 2000
    angles = range(0f0, 2π, length=n)  # Ángulos equiespaciados
    # Crear puntos (cos(θ), sin(θ)) + ruido gaussiano pequeño
    X_real = hcat([[cos(a) + 0.02f0*randn(), sin(a) + 0.02f0*randn()] for a in angles]...)
    X_real = NN.normalize01(F.(X_real))  # Normalizar a [0,1]

    # 2. Crear GAN
    # Parámetros:
    # - latent_dim=2: dimensión del ruido de entrada
    # - data_dim=2: dimensión de los datos (puntos 2D)
    # - hidden=[16]: capa oculta con 16 neuronas
    # - act=sigmoid: función de activación
    gan = NN.GAN(2, 2, [16], act=NN.sigmoid)

    # Mostrar arquitecturas
    println("Arquitectura GAN:")
    println("Generator:")
    NN.summary(gan.generator)  # Ruido (2) -> 16 -> Datos (2)
    println("Discriminator:")
    NN.summary(gan.discriminator)  # Datos (2) -> 16 -> Probabilidad (1)

    # 3. Entrenar GAN
    # Proceso alternado:
    # - Cada k=1 pasos del discriminador, 1 paso del generador
    # - epochs=500: número de épocas de entrenamiento
    # - batch_size=64: tamaño del minibatch
    # - eta=0.1: tasa de aprendizaje
    NN.fit_gan!(gan, X_real; epochs=500, batch_size=64, eta=0.1f0, k=1, verbose=true)

    # 4. Generar muestras sintéticas
    # gan(n) genera n puntos desde ruido aleatorio
    X_gen = gan(500)  # 500 puntos generados

    # 5. Visualización: comparar distribución real vs generada
    # Si GAN funcionó bien, los puntos generados formarán un círculo
    p = scatter(X_real[1, :], X_real[2, :], label="Reales", alpha=0.5,
                title="GAN: Reales vs Generados")
    scatter!(X_gen[1, :], X_gen[2, :], label="Generados", alpha=0.5)
    display(p)

    return gan
end

ejercicio_gan()

# ============================================================
# 10. EJECUCIÓN DE TODOS LOS EJERCICIOS
# ============================================================

#=
Función auxiliar para ejecutar todos los ejercicios de forma secuencial.
Útil para:
- Validar que todos los ejercicios funcionan correctamente
- Generar todos los resultados de una vez
- Demostración completa de las capacidades de la librería

NOTA: La ejecución completa puede tomar varios minutos dependiendo
de la velocidad del ordenador, especialmente los ejercicios de MNIST
que entrenan con más datos.
=#

function ejecutar_todos()
    println("="^60)
    println("PRÁCTICA NN REFACTORIZADA - EJECUCIÓN COMPLETA")
    println("="^60)

    # 1. Aproximación de funciones (1-5)
    # Demuestran capacidad de las redes para aproximar funciones arbitrarias
    ejercicio_f1()
    ejercicio_f2()
    ejercicio_f3()
    ejercicio_f4()
    ejercicio_f5()

    # 2. Clasificación (6-7)
    # Clasificación multiclase con diferentes datasets
    ejercicio_iris()
    ejercicio_mnist()

    # 3. Aprendizaje de Representación (8-9)
    # Autoencoders para compresión y visualización
    ejercicio_autoencoder_iris()
    ejercicio_autoencoder_mnist()

    # 4. Funcionalidades Avanzadas (10-13)
    # Técnicas modernas de deep learning
    ejercicio_denoising_mnist()
    ejercicio_optimizadores()
    ejercicio_word2vec()
    ejercicio_gan()

    println("\n" * "="^60)
    println("PRÁCTICA COMPLETADA")
    println("="^60)
end

# Descomenta la siguiente línea para ejecutar todos los ejercicios:
# ejecutar_todos()

# O ejecuta ejercicios individuales para exploración:
# ejercicio_f1()
# ejercicio_iris()
# ejercicio_autoencoder_iris()
# ejercicio_gan()
