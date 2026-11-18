module NN

import LinearAlgebra
import Printf
import Random

# ============================================================
# 0. TIPO BASE Y ACTIVACIONES
# ============================================================

# Float32 es el tipo estándar en Deep Learning por eficiencia y compatibilidad con GPUs
const F = Float32

"""
Estructura para almacenar una función de activación y su derivada.
Ambas funciones deben poder operar tanto sobre escalares como sobre matrices.
"""
struct Activation
    f::Function   # Función de activación: σ(x)
    d::Function   # Derivada: σ'(x)
end

# --- Funciones de Activación ---

# SIGMOID: σ(x) = 1/(1+e^(-x))
# Rango: (0, 1). Usada en salidas de probabilidad binaria
sigma(x) = one(x) / (one(x) + exp(-x))
# Derivada: σ'(x) = σ(x)(1-σ(x))
sigmad(x) = sigma(x) * (one(x) - sigma(x))
const sigmoid = Activation(sigma, sigmad)

# RELU: ReLU(x) = max(0, x)
# Rango: [0, ∞). La más usada en capas ocultas por simplicidad y eficiencia
reluf(x) = max(x, zero(x))
# Derivada: ReLU'(x) = 1 si x>0, 0 en otro caso
relud(x) = x > zero(x) ? one(x) : zero(x)
const relu = Activation(reluf, relud)

# TANH: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
# Rango: (-1, 1). Centrada en cero, mejor que sigmoid en capas ocultas
tanhf_(x) = tanh(x)
# Derivada: tanh'(x) = 1 - tanh²(x)
tanhd_(x) = one(x) - tanhf_(x)^2
const tanh_act = Activation(tanhf_, tanhd_)

# ============================================================
# 1. FUNCIONES DE COSTE (operan sobre matrices donde columnas = muestras)
# ============================================================

"""
Estructura para funciones de coste.
* f: calcula el coste promedio sobre un batch
* delta: calcula el gradiente inicial para backpropagation
"""
struct Cost
    f::Function      # (A, Y) -> escalar (coste promedio del batch)
    delta::Function  # (Z, A, Y) -> matriz con ∂L/∂Z para iniciar backprop
end

"""
Coste Cuadrático (Mean Squared Error):
L = (1/2m) Σ ||a_i - y_i||²

Usado habitualmente en problemas de regresión.
El factor 1/2 simplifica la derivada.
"""
function quadratic_cost(act::Activation)::Cost
    Cost(
        # Coste: promedio sobre el batch
        (A, Y) -> F(0.5) * sum((A .- Y).^2) / size(Y, 2),
        # Delta: ∂L/∂z = (a - y) ⊙ σ'(z)
        (Z, A, Y) -> (A .- Y) .* act.d.(Z)
    )
end

"""
Entropía Cruzada Binaria (Binary Cross-Entropy):
L = -(1/m) Σ [y*log(a) + (1-y)*log(1-a)]

Usada en clasificación binaria con sigmoid en la salida.
El epsilon (1e-8) evita log(0).
"""
const cross_entropy = Cost(
    # Coste: suma negativa de log-probabilidades
    (A, Y) -> -sum(Y .* log.(A .+ F(1e-8)) .+ (1 .- Y) .* log.(1 .- A .+ F(1e-8))) / size(Y, 2),
    # Delta simplificado: cuando usamos sigmoid + BCE, la derivada se simplifica a (a - y)
    (Z, A, Y) -> A .- Y
)

# ============================================================
# 2. CAPA DENSE (el bloque fundamental de las redes neuronales)
# ============================================================

"""
`Dense(in_dim, out_dim, activation; init)`

Capa totalmente conectada (fully-connected o dense).

**Estructura**: Cada neurona de salida está conectada a todas las entradas.

**Parámetros aprendibles**:
* W: Matriz de pesos (out_dim, in_dim)
* b: Vector de sesgos (out_dim,)

**Caches** (para forward/backward):
* Z: Pre-activación, Z = W*A + b
* A: Post-activación, A = σ(Z)

**Gradientes**:
* dW: ∂L/∂W
* db: ∂L/∂b

**Formato matricial**:
- Cada columna = una muestra
- X: (in_dim, batch_size)
- Z, A: (out_dim, batch_size)

**Inicialización de pesos**:
- "xavier": Para sigmoide/tanh. Escala = sqrt(2/(in+out))
- "he": Para ReLU. Escala = sqrt(2/in)
- otro: Escala pequeña (0.01)
"""
mutable struct Dense
    # Parámetros aprendibles
    W::Matrix{F}   # Pesos (out_dim, in_dim)
    b::Vector{F}   # Sesgos (out_dim,)
    
    # Función de activación
    activation::Activation
    
    # Caches para forward/backward (se redimensionan dinámicamente)
    Z::Matrix{F}   # Pre-activación (out_dim, batch_size)
    A::Matrix{F}   # Post-activación (out_dim, batch_size)
    
    # Gradientes
    dW::Matrix{F}  # ∂L/∂W (out_dim, in_dim)
    db::Vector{F}  # ∂L/∂b (out_dim,)

    function Dense(in_dim::Int, out_dim::Int, act::Activation; init::String="xavier")
        # Seleccionar escala de inicialización según el método
        if init == "xavier"
            # Xavier/Glorot: bueno para sigmoid/tanh
            scale = sqrt(F(2) / (in_dim + out_dim))
        elseif init == "he"
            # He: bueno para ReLU (considera solo entrada)
            scale = sqrt(F(2) / in_dim)
        else
            # Inicialización pequeña genérica
            scale = F(0.01)
        end
        
        # Inicializar pesos con distribución normal escalada
        W = randn(F, out_dim, in_dim) * scale
        
        # Inicializar sesgos en cero (práctica común)
        b = zeros(F, out_dim)
        
        # Inicializar caches vacías (se redimensionan en forward)
        new(W, b, act, 
            Matrix{F}(undef, 0, 0), Matrix{F}(undef, 0, 0),
            Matrix{F}(undef, 0, 0), Vector{F}(undef, 0))
    end
end

# ============================================================
# 3. NETWORK (vector de capas)
# ============================================================

mutable struct Network
    layers::Vector{Dense}
    cost::Cost
end

"""
`Network(dims, acts; cost, init)`

Constructor. `dims` = tamaños de capas, `acts` = activaciones por capa.
Ejemplo: `Network([2, 16, 1], [relu, sigmoid])`
"""
function Network(dims::Vector{Int}, acts::Vector{Activation};
                 cost::Cost=quadratic_cost(acts[end]), init::String="xavier")
    layers = [Dense(dims[i], dims[i+1], acts[i], init=init) for i in 1:length(dims)-1]
    return Network(layers, cost)
end

"""
`forward!(net::Network, X::Matrix{F})::Matrix{F}`

Propagación hacia adelante (Forward Pass).

**Entrada**: X con shape (in_dim, batch_size)
- Cada columna es una muestra
- in_dim es la dimensión de entrada

**Proceso**: Para cada capa l:
1. Z[l] = W[l] * A[l-1] + b[l]  (combinación lineal)
2. A[l] = σ(Z[l])              (aplicar activación)

**Salida**: Activaciones de la última capa (predicciones)

**Importante**: Guarda Z y A en cada capa para usarlas en backward!
"""
function forward!(net::Network, X::Matrix{F})::Matrix{F}
    # Iniciar con las entradas
    A = X
    
    # Propagar por todas las capas
    for layer in net.layers
        # Combinación lineal: Z = W*A + b
        # W*A: (out_dim, in_dim) * (in_dim, batch) = (out_dim, batch)
        # b se broadcastea automáticamente sobre columnas
        layer.Z = layer.W * A .+ layer.b
        
        # Aplicar función de activación elemento a elemento
        layer.A = layer.activation.f.(layer.Z)
        
        # La salida de esta capa es la entrada de la siguiente
        A = layer.A
    end
    
    # Devolver las activaciones finales (predicciones de la red)
    return A
end

"""
`backward!(net::Network, X::Matrix{F}, Y::Matrix{F})`

Propagación hacia atrás (Backpropagation).

Calcula los gradientes ∂L/∂W y ∂L/∂b para todas las capas
y los almacena en layer.dW y layer.db.

**Algoritmo**:
1. Última capa: δ[L] = cost.delta(Z[L], A[L], Y)
2. Capas anteriores: δ[l] = (W[l+1]^T * δ[l+1]) ⊙ σ'(Z[l])
3. Gradientes: dW[l] = (1/m) * δ[l] * A[l-1]^T
               db[l] = (1/m) * sum(δ[l] por columnas)

donde m = batch_size, ⊙ = producto elemento a elemento
"""
function backward!(net::Network, X::Matrix{F}, Y::Matrix{F})
    # m = número de muestras en el batch
    m = size(X, 2)
    L = length(net.layers)

    # === PASO 1: Última capa ===
    # Calcular el "error" inicial usando la función de coste
    # Para MSE: δ = (A - Y) ⊙ σ'(Z)
    # Para BCE+sigmoid: δ = A - Y (simplificado)
    dZ = net.cost.delta(net.layers[L].Z, net.layers[L].A, Y)
    
    # Activaciones de la capa anterior (o entrada si es la primera capa)
    A_prev = L == 1 ? X : net.layers[L-1].A
    
    # Calcular gradientes de la última capa:
    # dW = (1/m) * δ * A_prev^T
    # Shapes: (out, batch) * (batch, in) = (out, in)
    net.layers[L].dW = dZ * A_prev' / m
    
    # db = (1/m) * sum(δ por filas)
    # vec convierte matriz (out, 1) a vector (out,)
    net.layers[L].db = vec(sum(dZ, dims=2)) / m

    # === PASO 2: Propagación hacia atrás por capas ocultas ===
    for l in L-1:-1:1
        # Propagar el error a la capa anterior:
        # dA = W[l+1]^T * δ[l+1]
        dA = net.layers[l+1].W' * dZ
        
        # Calcular nuevo δ multiplicando por la derivada de la activación:
        # δ[l] = dA ⊙ σ'(Z[l])
        dZ = dA .* net.layers[l].activation.d.(net.layers[l].Z)
        
        # Activaciones de la capa anterior
        A_prev = l == 1 ? X : net.layers[l-1].A
        
        # Calcular gradientes de esta capa
        net.layers[l].dW = dZ * A_prev' / m
        net.layers[l].db = vec(sum(dZ, dims=2)) / m
    end
end

"""
`update!(net::Network, eta::F)`

Actualización de parámetros usando Descenso del Gradiente.

Aplica la regla de actualización:
* W = W - η * dW
* b = b - η * db

donde η (eta) es la tasa de aprendizaje (learning rate).

Un η muy grande puede causar divergencia.
Un η muy pequeño hace el aprendizaje muy lento.
Valores típicos: 0.001 - 0.1
"""
function update!(net::Network, eta::F)
    for layer in net.layers
        # Actualizar pesos: W = W - η * ∂L/∂W
        layer.W .-= eta * layer.dW
        
        # Actualizar sesgos: b = b - η * ∂L/∂b
        layer.b .-= eta * layer.db
    end
end

"""
`compute_loss(net::Network, X::Matrix{F}, Y::Matrix{F})::F`

Calcula la pérdida (loss) sobre un dataset completo.

Proceso:
1. Forward pass para obtener predicciones
2. Calcular coste usando la función de coste de la red

Devuelve un escalar que representa el error promedio.
"""
function compute_loss(net::Network, X::Matrix{F}, Y::Matrix{F})::F
    A = forward!(net, X)
    return net.cost.f(A, Y)
end

"""
`accuracy(net::Network, X::Matrix{F}, Y::Matrix{F})::F`

Calcula la precisión para clasificación binaria.

Proceso:
1. Predicciones: a >= 0.5 -> clase 1, a < 0.5 -> clase 0
2. Comparar con targets
3. Contar aciertos / total

Devuelve un valor entre 0.0 (0%) y 1.0 (100%).
"""
function accuracy(net::Network, X::Matrix{F}, Y::Matrix{F})::F
    A = forward!(net, X)
    # Umbral 0.5 para clasificación binaria
    preds = A .>= F(0.5)
    targets = Y .>= F(0.5)
    return sum(preds .== targets) / length(Y)
end

# ============================================================
# 4. OPTIMIZADORES (desacoplados de la red)
# ============================================================

# Tipo base para todos los optimizadores
abstract type Optimizer end

"""
`SGD(eta)`

Descenso del Gradiente Estocástico (Stochastic Gradient Descent).

El optimizador más básico: actualiza parámetros directamente
con el gradiente multiplicado por la tasa de aprendizaje.

* eta: Learning rate (tasa de aprendizaje)
"""
struct SGD <: Optimizer
    eta::F
end

"""
`Momentum(net, eta, beta)`

SGD con Momento (Momentum).

Acelera el aprendizaje acumulando un "momento" de actualizaciones previas:
v_t = β*v_{t-1} + ∇L
θ_t = θ_{t-1} - η*v_t

Ventajas:
- Suaviza oscilaciones en direcciones con alta curvatura
- Acelera el progreso en direcciones consistentes
- Ayuda a escapar de mínimos locales

* eta: Learning rate
* beta: Factor de momento (típicamente 0.9)
* vW, vb: Velocidades acumuladas para cada parámetro
"""
struct Momentum <: Optimizer
    eta::F
    beta::F
    vW::Vector{Matrix{F}}  # Velocidades para W de cada capa
    vb::Vector{Vector{F}}  # Velocidades para b de cada capa

    function Momentum(net::Network, eta::F, beta::F=F(0.9))
        # Inicializar velocidades en cero
        vW = [zeros(F, size(l.W)) for l in net.layers]
        vb = [zeros(F, size(l.b)) for l in net.layers]
        new(eta, beta, vW, vb)
    end
end

"""
`Adam(net; eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`

**Adaptive Moment Estimation** - El optimizador más popular en Deep Learning.

**Motivación**:
Adam combina las mejores ideas de dos optimizadores:
- **Momentum**: Acumula velocidad en direcciones consistentes (primer momento)
- **RMSProp**: Adapta el learning rate por parámetro (segundo momento)

Resultado: Converge rápido, es robusto a la elección de η, y funciona bien
en la mayoría de problemas sin necesidad de ajustar hiperparámetros.

**Algoritmo**:
Para cada parámetro θ y su gradiente g:

1. **Primer momento** (media exponencial de gradientes):
   m_t = β₁·m_{t-1} + (1-β₁)·g_t

2. **Segundo momento** (media exponencial de gradientes al cuadrado):
   v_t = β₂·v_{t-1} + (1-β₂)·g_t²

3. **Corrección de sesgo** (importante en las primeras iteraciones):
   m̂_t = m_t / (1 - β₁ᵗ)
   v̂_t = v_t / (1 - β₂ᵗ)

4. **Actualización adaptativa**:
   θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)

**¿Por qué funciona tan bien?**
- **m̂**: Da dirección suavizada (reduce ruido del gradiente)
- **√v̂**: Escala el learning rate por parámetro (pequeño si g² es grande)
- **Corrección de sesgo**: Evita sesgo hacia cero en los primeros pasos

**Hiperparámetros**:
* `eta`: Learning rate (típico: 0.001, rango: 0.0001 - 0.01)
* `beta1`: Decay del primer momento, controla la "memoria" de direcciones
            pasadas (típico: 0.9, rango: 0.8 - 0.95)
* `beta2`: Decay del segundo momento, controla adaptación del learning rate
            (típico: 0.999, rango: 0.99 - 0.9999)
* `epsilon`: Término de estabilidad numérica para evitar división por cero
             (típico: 1e-8)
* `t`: Contador de iteraciones (para corrección de sesgo)

**Estados internos** (se inicializan en cero):
* `mW, mb`: Primer momento para pesos y sesgos
* `vW, vb`: Segundo momento para pesos y sesgos

**Ventajas sobre otros optimizadores**:
- vs SGD: Converge mucho más rápido
- vs Momentum: Adapta η automáticamente por parámetro
- vs RMSProp: Añade momento para suavizar la trayectoria
- Funciona bien "out of the box" sin tuning exhaustivo

**Cuándo usarlo**:
- Default para la mayoría de problemas de Deep Learning
- Especialmente bueno con redes profundas
- Cuando no tienes tiempo/recursos para ajustar hiperparámetros
"""
mutable struct Adam <: Optimizer
    eta::F       # learning rate (típico: 0.001)
    beta1::F     # decay del primer momento (típico: 0.9)
    beta2::F     # decay del segundo momento (típico: 0.999)
    epsilon::F   # estabilidad numérica (típico: 1e-8)
    t::Int       # contador de pasos (inicializado en 0)
    
    # Momentos acumulados (uno por capa)
    mW::Vector{Matrix{F}}  # primer momento (media) de W
    vW::Vector{Matrix{F}}  # segundo momento (varianza) de W
    mb::Vector{Vector{F}}  # primer momento de b
    vb::Vector{Vector{F}}  # segundo momento de b
end

"""
Constructor para Adam.

Inicializa todos los momentos (m, v) en cero para cada parámetro de la red.
El contador t se inicializa en 0 y se incrementa en cada llamada a step!.

**Nota sobre inicialización**:
Empezar en cero es correcto gracias a la corrección de sesgo.
Sin ella, los primeros pasos estarían sesgados hacia cero.

**Ejemplo**:
```julia
net = Network([784, 128, 10], [relu, sigmoid])
opt = Adam(net, eta=F(0.001))

# En el bucle de entrenamiento:
for epoch in 1:epochs
    for batch in batches
        forward!(net, X_batch)
        backward!(net, X_batch, Y_batch)
        step!(opt, net)  # Actualización Adam
    end
end
```
"""
function Adam(net::Network; eta::F=F(0.001), beta1::F=F(0.9), beta2::F=F(0.999), epsilon::F=F(1e-8))
    # Inicializar primer momento (m) en cero para todos los parámetros
    mW = [zeros(F, size(l.W)) for l in net.layers]
    mb = [zeros(F, size(l.b)) for l in net.layers]
    
    # Inicializar segundo momento (v) en cero para todos los parámetros
    vW = [zeros(F, size(l.W)) for l in net.layers]
    vb = [zeros(F, size(l.b)) for l in net.layers]
    
    # t=0 se incrementará a 1 en el primer step!
    return Adam(eta, beta1, beta2, epsilon, 0, mW, vW, mb, vb)
end

"""Aplica un paso de optimización SGD estándar."""
function step!(opt::SGD, net::Network)
    update!(net, opt.eta)
end

"""Aplica un paso de optimización con Momentum."""
function step!(opt::Momentum, net::Network)
    for i in eachindex(net.layers)
        # Actualizar velocidades: v = β*v + ∇L
        opt.vW[i] = opt.beta .* opt.vW[i] .+ net.layers[i].dW
        opt.vb[i] = opt.beta .* opt.vb[i] .+ net.layers[i].db
        
        # Actualizar parámetros usando velocidades
        net.layers[i].W .-= opt.eta * opt.vW[i]
        net.layers[i].b .-= opt.eta * opt.vb[i]
    end
end

"""
`step!(opt::Adam, net::Network)`

Aplica un paso de optimización Adam.

**Proceso** (para cada capa):

1. **Extraer gradientes**: Los calculó backward! y están en dW, db

2. **Actualizar momentos con decaimiento exponencial**:
   - Primer momento (m): Media móvil de gradientes
     m = β₁·m + (1-β₁)·g
   - Segundo momento (v): Media móvil de gradientes al cuadrado
     v = β₂·v + (1-β₂)·g²

3. **Corrección de sesgo**:
   Problema: m y v están inicializados en 0, así que en las primeras
   iteraciones están sesgados hacia 0.
   
   Solución: Dividir por (1 - βᵗ) donde t es el número de iteraciones.
   Esto compensa el sesgo. Conforme t→∞, (1-βᵗ)→1 y la corrección desaparece.
   
   m̂ = m / (1 - β₁ᵗ)
   v̂ = v / (1 - β₂ᵗ)

4. **Actualización adaptativa**:
   θ = θ - η · m̂ / (√v̂ + ε)
   
   - m̂: Dirección (gradiente suavizado)
   - √v̂: Escala (learning rate adaptado por parámetro)
   - ε: Evita división por cero

**Interpretación**:
- Si un parámetro tiene gradientes grandes y consistentes:
  * m̂ será grande (señal fuerte)
  * √v̂ será grande (normaliza el paso)
  * Resultado: Paso moderado y estable
  
- Si un parámetro tiene gradientes pequeños:
  * m̂ será pequeño
  * √v̂ será pequeño (no normaliza mucho)
  * Resultado: Paso efectivo relativo al gradiente
"""
function step!(opt::Adam, net::Network)
    # Incrementar contador de iteraciones (necesario para bias correction)
    opt.t += 1
    
    # Procesar cada capa independientemente
    for i in eachindex(net.layers)
        # === PASO 1: Extraer gradientes ===
        gW = net.layers[i].dW  # Gradiente de pesos (out, in)
        gb = net.layers[i].db  # Gradiente de sesgos (out,)
        
        # === PASO 2: Actualizar primer momento (media de gradientes) ===
        # m_t = β₁·m_{t-1} + (1-β₁)·g_t
        # β₁ típicamente 0.9 -> retiene 90% del momento anterior
        opt.mW[i] = opt.beta1 .* opt.mW[i] .+ (1 - opt.beta1) .* gW
        opt.mb[i] = opt.beta1 .* opt.mb[i] .+ (1 - opt.beta1) .* gb
        
        # === PASO 3: Actualizar segundo momento (media de gradientes²) ===
        # v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        # β₂ típicamente 0.999 -> retiene 99.9% del momento anterior
        # Esto estima la varianza (o magnitud típica) del gradiente
        opt.vW[i] = opt.beta2 .* opt.vW[i] .+ (1 - opt.beta2) .* (gW.^2)
        opt.vb[i] = opt.beta2 .* opt.vb[i] .+ (1 - opt.beta2) .* (gb.^2)
        
        # === PASO 4: Corrección de sesgo ===
        # Sin esto, m y v estarían sesgados hacia 0 en las primeras iteraciones
        # porque se inicializaron en 0.
        # 
        # Ejemplo: Con β₁=0.9, después del primer paso:
        #   m₁ = 0.9·0 + 0.1·g₁ = 0.1·g₁  <- Muy pequeño!
        # Con corrección:
        #   m̂₁ = 0.1·g₁ / (1-0.9¹) = 0.1·g₁ / 0.1 = g₁  <- Correcto!
        mW_hat = opt.mW[i] ./ (1 - opt.beta1^opt.t)
        vW_hat = opt.vW[i] ./ (1 - opt.beta2^opt.t)
        mb_hat = opt.mb[i] ./ (1 - opt.beta1^opt.t)
        vb_hat = opt.vb[i] ./ (1 - opt.beta2^opt.t)
        
        # === PASO 5: Actualización adaptativa de parámetros ===
        # θ = θ - η · m̂ / (√v̂ + ε)
        #
        # Interpretación de la fórmula:
        # - m̂: Dirección suavizada del gradiente
        # - √v̂: Escala típica del gradiente (adaptación per-parameter)
        # - División: Learning rate efectivo = η / √v̂
        #   * Si gradientes grandes -> √v̂ grande -> paso más conservador
        #   * Si gradientes pequeños -> √v̂ pequeño -> paso más agresivo
        # - ε: Evita división por cero (típicamente 1e-8)
        net.layers[i].W .-= opt.eta .* mW_hat ./ (sqrt.(vW_hat) .+ opt.epsilon)
        net.layers[i].b .-= opt.eta .* mb_hat ./ (sqrt.(vb_hat) .+ opt.epsilon)
    end
end

# ============================================================
# 5. ENTRENAMIENTO CON HISTORIA Y VALIDACIÓN
# ============================================================

"""
`fit!(net, X, Y; epochs, batch_size, optimizer, validation_split, verbose)`

Entrena la red neuronal.

**Parámetros**:
* net: Red a entrenar
* X: Datos de entrada (dim, n_samples)
* Y: Etiquetas/targets (out_dim, n_samples)
* epochs: Número de épocas (pasadas completas por los datos)
* batch_size: Tamaño del mini-batch
* optimizer: Optimizador a usar (SGD, Momentum, etc.)
* validation_split: Fracción de datos para validación (0.0 = sin validación)
* verbose: Mostrar progreso

**Proceso por época**:
1. Mezclar datos aleatoriamente
2. Dividir en mini-batches
3. Para cada batch: forward -> backward -> update
4. Calcular métricas (loss, accuracy)
5. Si hay validación, evaluar en datos de validación

**Devuelve**: Diccionario con historial de métricas
- "loss": pérdida en entrenamiento
- "acc": precisión en entrenamiento
- "val_loss": pérdida en validación (si validation_split > 0)
- "val_acc": precisión en validación (si validation_split > 0)
"""
function fit!(net::Network, X::Matrix{F}, Y::Matrix{F};
              epochs::Int=100,
              batch_size::Int=32,
              optimizer::Optimizer=SGD(F(0.01)),
              validation_split::Float64=0.0,
              verbose::Bool=true)::Dict{String, Vector{F}}

    n = size(X, 2)
    # Inicializar historial de métricas
    history = Dict("loss"=>F[], "val_loss"=>F[], "acc"=>F[], "val_acc"=>F[])

    X_train, Y_train = X, Y
    X_val, Y_val = nothing, nothing

    # Dividir datos en entrenamiento y validación si se solicita
    if validation_split > 0
        n_train = floor(Int, n * (1 - validation_split))
        perm = Random.randperm(n)  # Permutación aleatoria
        X_train = X[:, perm[1:n_train]]
        Y_train = Y[:, perm[1:n_train]]
        X_val = X[:, perm[n_train+1:end]]
        Y_val = Y[:, perm[n_train+1:end]]
    end

    # Bucle de entrenamiento
    for epoch in 1:epochs
        # Mezclar datos al inicio de cada época (importante para SGD)
        perm = Random.randperm(size(X_train, 2))
        
        # Procesar mini-batches
        for k in 1:batch_size:size(X_train, 2)
            # Seleccionar índices del batch actual
            idx = perm[k:min(k+batch_size-1, size(X_train, 2))]
            X_batch = X_train[:, idx]
            Y_batch = Y_train[:, idx]

            # Paso de entrenamiento: forward + backward + update
            forward!(net, X_batch)
            backward!(net, X_batch, Y_batch)
            step!(optimizer, net)
        end

        # Calcular métricas en datos de entrenamiento
        loss = compute_loss(net, X_train, Y_train)
        acc = accuracy(net, X_train, Y_train)
        push!(history["loss"], loss)
        push!(history["acc"], acc)

        # Evaluar en validación si existe
        if validation_split > 0
            vloss = compute_loss(net, X_val, Y_val)
            vacc = accuracy(net, X_val, Y_val)
            push!(history["val_loss"], vloss)
            push!(history["val_acc"], vacc)
            verbose && @info @Printf.sprintf("Epoch %d/%d - loss: %.4f - acc: %.2f%% - val_loss: %.4f - val_acc: %.2f%%",
                                             epoch, epochs, loss, acc*100, vloss, vacc*100)
        else
            verbose && @info @Printf.sprintf("Epoch %d/%d - loss: %.4f - acc: %.2f%%", epoch, epochs, loss, acc*100)
        end
    end

    return history
end

# ============================================================
# 6. UTILIDADES DOCENTES
# ============================================================

"""
`summary(net::Network)`

Imprime un resumen de la arquitectura de la red.

Muestra:
- Número de capa
- Tipo (Dense)
- Dimensiones de entrada -> salida
- Función de activación
- Número de parámetros (pesos + sesgos)
- Total de parámetros entrenables

Útil para verificar que la red se construyó correctamente.
"""
function summary(net::Network)
    total = 0
    println("═" ^ 60)
    println("Network Summary")
    println("═" ^ 60)
    for (i, l) in enumerate(net.layers)
        in_d, out_d = size(l.W, 2), size(l.W, 1)
        # Contar parámetros: pesos + sesgos
        p = length(l.W) + length(l.b)
        total += p
        # Identificar función de activación
        act_name = l.activation == sigmoid ? "σ" : l.activation == relu ? "ReLU" : "tanh"
        @Printf.printf("Layer %-2d  Dense(%3d -> %3d)  %-6s  Params: %d\n", i, in_d, out_d, act_name, p)
    end
    println("─" ^ 60)
    @Printf.printf("Total de parámetros entrenables: %d\n", total)
    println("═" ^ 60)
end

"""
`gradient_check(net, X, Y; epsilon)`

Verifica la implementación de backpropagation usando diferencias finitas.

**¿Por qué es importante?**
Los errores en backprop son difíciles de detectar: la red puede
entrenar pero converger mal. Esta prueba verifica que los gradientes
son matemáticamente correctos.

**Método**:
Para cada parámetro θ:
1. Gradiente analítico: dL/dθ (de backprop)
2. Gradiente numérico: [L(θ+ε) - L(θ-ε)] / (2ε)
3. Comparar ambos

Si la diferencia máxima < 1e-5, backprop es correcto.

**ADVERTENCIA**: Es computacionalmente costoso (O(n²)).
Usar solo con redes pequeñas y pocos datos para debugging.
"""
function gradient_check(net::Network, X::Matrix{F}, Y::Matrix{F}; epsilon::F=F(1e-4))::Bool
    # Guardar estado original de la red
    W0 = [copy(l.W) for l in net.layers]
    b0 = [copy(l.b) for l in net.layers]

    # Calcular gradientes analíticos (backprop)
    forward!(net, X)
    backward!(net, X, Y)
    gradW = [copy(l.dW) for l in net.layers]
    gradb = [copy(l.db) for l in net.layers]

    max_diff = F(0)

    # Verificar gradientes de W en cada capa
    for l in eachindex(net.layers)
        # Verificar cada elemento de W
        for i in 1:size(net.layers[l].W, 1), j in 1:size(net.layers[l].W, 2)
            # L(θ + ε)
            net.layers[l].W[i,j] += epsilon
            lp = compute_loss(net, X, Y)
            
            # L(θ - ε)
            net.layers[l].W[i,j] -= 2*epsilon
            lm = compute_loss(net, X, Y)
            
            # Restaurar valor original
            net.layers[l].W[i,j] += epsilon
            
            # Gradiente numérico: [L(+ε) - L(-ε)] / (2ε)
            approx = (lp - lm) / (2*epsilon)
            
            # Comparar con gradiente analítico
            max_diff = max(max_diff, abs(approx - gradW[l][i,j]))
        end
        
        # Verificar cada elemento de b
        for i in 1:length(net.layers[l].b)
            net.layers[l].b[i] += epsilon
            lp = compute_loss(net, X, Y)
            net.layers[l].b[i] -= 2*epsilon
            lm = compute_loss(net, X, Y)
            net.layers[l].b[i] += epsilon
            approx = (lp - lm) / (2*epsilon)
            max_diff = max(max_diff, abs(approx - gradb[l][i]))
        end
    end

    # Restaurar estado original
    for l in eachindex(net.layers)
        net.layers[l].W .= W0[l]
        net.layers[l].b .= b0[l]
    end

    @info @Printf.sprintf("Gradient check - max diff: %.8f (threshold: 1e-5)", max_diff)
    return max_diff < F(1e-5)
end

"""Guarda pesos en un archivo JLD2/serialize (usamos serialize nativo)."""
function save_weights(net::Network, filename::String)
    weights = [(W=copy(l.W), b=copy(l.b)) for l in net.layers]
    open(filename, "w") do io
        serialize(io, weights)
    end
end

function load_weights!(net::Network, filename::String)
    weights = open(filename, "r") do io
        deserialize(io)
    end
    for (i, w) in enumerate(weights)
        net.layers[i].W .= w.W
        net.layers[i].b .= w.b
    end
end

# ============================================================
# 7. AUTOENCODER (extensión directa de Network)
# ============================================================

"""
`Autoencoder`

Red neuronal para aprendizaje no supervisado de representaciones.

Un autoencoder tiene una estructura simétrica:
- Encoder: comprime la entrada a una representación latente (bottleneck)
- Decoder: reconstruye la entrada original desde el bottleneck

Ejemplo: [784, 256, 64, 256, 784]
         entrada -> encoder -> bottleneck (64) -> decoder -> salida

El objetivo es minimizar el error de reconstrucción.
Esto fuerza al bottleneck a capturar las características más importantes.

* net: Red completa (encoder + decoder)
* bottleneck: Índice de la última capa del encoder
"""
struct Autoencoder
    net::Network
    bottleneck::Int  # índice de la última capa del encoder
end

"""
Constructor conveniente. Calcula automáticamente la capa bottleneck
como la mitad de las capas (redondeando hacia abajo).
"""
function Autoencoder(dims::Vector{Int}, act::Activation=sigmoid; init::String="xavier")
    acts = fill(act, length(dims)-1)
    net = Network(dims, acts, cost=quadratic_cost(act), init=init)
    # Bottleneck = mitad de las capas
    bottleneck = div(length(dims)-1, 2)
    return Autoencoder(net, bottleneck)
end

"""
Propaga la entrada solo hasta la capa `stop`.
Útil para extraer la representación del bottleneck (encoding).
"""
function forward_until(net::Network, X::Matrix{F}, stop::Int)::Matrix{F}
    A = X
    for i in 1:stop
        # Propagar solo hasta la capa especificada
        A = net.layers[i].activation.f.(net.layers[i].W * A .+ net.layers[i].b)
    end
    return A
end

"""
Propaga desde la capa `start` hasta el final.
Útil para reconstruir desde el bottleneck (decoding).
"""
function forward_from(net::Network, X::Matrix{F}, start::Int)::Matrix{F}
    A = X
    for i in start:length(net.layers)
        # Propagar desde la capa de inicio hasta el final
        A = net.layers[i].activation.f.(net.layers[i].W * A .+ net.layers[i].b)
    end
    return A
end

"""Codifica: entrada -> representación latente (bottleneck)."""
encode(ae::Autoencoder, X::Matrix{F}) = forward_until(ae.net, X, ae.bottleneck)

"""Decodifica: representación latente -> reconstrucción."""
decode(ae::Autoencoder, H::Matrix{F}) = forward_from(ae.net, H, ae.bottleneck + 1)

"""Reconstrucción completa: entrada -> encoding -> decoding -> salida."""
reconstruct(ae::Autoencoder, X::Matrix{F}) = forward!(ae.net, X)

"""
Entrena el autoencoder. El target es la propia entrada (X = Y).
La red aprende a reconstruir su entrada pasando por el bottleneck.
"""
function fit_autoencoder!(ae::Autoencoder, X::Matrix{F}; kwargs...)
    # Entrenar con X como entrada y X como target
    return fit!(ae.net, X, X; kwargs...)
end

"""
Añade ruido gaussiano a los datos.

Usado en Denoising Autoencoders para hacer el modelo más robusto.
El ruido fuerza a la red a aprender características esenciales,
ignorando variaciones aleatorias.

* level: Desviación estándar del ruido (típicamente 0.1-0.3)
"""
function add_noise(X::Matrix{F}, level::F)::Matrix{F}
    return X + level * randn(F, size(X))
end

"""
Entrena un Denoising Autoencoder.

Proceso:
1. Añadir ruido a los datos de entrada
2. Entrenar la red para reconstruir los datos LIMPIOS
   (entrada = datos con ruido, target = datos sin ruido)

Esto hace que la red aprenda representaciones robustas
que ignoran el ruido y capturan la estructura subyacente.
"""
function fit_denoising!(ae::Autoencoder, X::Matrix{F}; noise_level::F=F(0.1), kwargs...)
    # Generar versión ruidosa de los datos
    X_noisy = add_noise(X, noise_level)
    # Entrenar: entrada ruidosa -> target limpio
    return fit!(ae.net, X_noisy, X; kwargs...)
end

# ============================================================
# 8. WORD2VEC (mantiene lógica sample-by-sample, es inherentemente secuencial)
# ============================================================

mutable struct Word2Vec
    vocab_size::Int
    embedding_dim::Int
    embeddings::Matrix{F}   # (dim, vocab)
    context::Matrix{F}      # (dim, vocab)
    window::Int
    negative_samples::Int
end

function Word2Vec(vocab_size::Int, embedding_dim::Int;
                  window::Int=5, negative_samples::Int=5)
    emb = randn(F, embedding_dim, vocab_size) * F(0.01)
    ctx = randn(F, embedding_dim, vocab_size) * F(0.01)
    return Word2Vec(vocab_size, embedding_dim, emb, ctx, window, negative_samples)
end

function fit_skipgram!(w2v::Word2Vec, corpus::Vector{Int};
                       epochs::Int=100, eta::F=F(0.01))
    n = length(corpus)
    for epoch in 1:epochs
        for i in 1:n
            center = corpus[i]
            ws = max(1, i - w2v.window)
            we = min(n, i + w2v.window)
            for j in ws:we
                j == i && continue
                context = corpus[j]

                # Positivo
                s_pos = LinearAlgebra.dot(w2v.context[:, context], w2v.embeddings[:, center])
                sig_pos = sigma(s_pos)
                g_emb = (sig_pos - 1) * w2v.context[:, context]
                g_ctx = (sig_pos - 1) * w2v.embeddings[:, center]

                # Negativas
                for _ in 1:w2v.negative_samples
                    neg = rand(1:w2v.vocab_size)
                    neg == context && continue
                    s_neg = LinearAlgebra.dot(w2v.context[:, neg], w2v.embeddings[:, center])
                    sig_neg = sigma(s_neg)
                    g_emb += sig_neg * w2v.context[:, neg]
                    w2v.context[:, neg] .-= eta * sig_neg * w2v.embeddings[:, center]
                end

                w2v.embeddings[:, center] .-= eta * g_emb
                w2v.context[:, context] .-= eta * g_ctx
            end
        end
        @info @Printf.sprintf("Word2Vec epoch %d/%d", epoch, epochs)
    end
    return w2v
end

function cosine_similarity(u::Vector{F}, v::Vector{F})::F
    return LinearAlgebra.dot(u, v) / (LinearAlgebra.norm(u) * LinearAlgebra.norm(v) + F(1e-8))
end

function most_similar(w2v::Word2Vec, word_idx::Int; n::Int=5)
    target = w2v.embeddings[:, word_idx]
    sims = [(i, cosine_similarity(target, w2v.embeddings[:, i]))
            for i in 1:w2v.vocab_size if i != word_idx]
    sort!(sims, by=x->x[2], rev=true)
    return sims[1:n]
end

function analogy(w2v::Word2Vec, a::Int, b::Int, c::Int; n::Int=1)
    vec = w2v.embeddings[:, b] .- w2v.embeddings[:, a] .+ w2v.embeddings[:, c]
    sims = [(i, cosine_similarity(vec, w2v.embeddings[:, i]))
            for i in 1:w2v.vocab_size if i ∉ (a,b,c)]
    sort!(sims, by=x->x[2], rev=true)
    return sims[1:n]
end

# ============================================================
# 9. GAN (Generative Adversarial Network)
# ============================================================

struct GAN
    generator::Network
    discriminator::Network
    latent_dim::Int
end

function GAN(input_dim::Int, latent_dim::Int, hidden::Vector{Int};
             act::Activation=sigmoid, init::String="xavier")
    gen_dims = vcat([latent_dim], hidden, [input_dim])
    gen_acts = fill(act, length(gen_dims)-1)
    gen = Network(gen_dims, gen_acts, cost=quadratic_cost(act), init=init)

    disc_dims = vcat([input_dim], reverse(hidden), [1])
    disc_acts = fill(act, length(disc_dims)-1)
    disc = Network(disc_dims, disc_acts, cost=cross_entropy, init=init)

    return GAN(gen, disc, latent_dim)
end

function generate(gan::GAN, n::Int)::Matrix{F}
    Z = randn(F, gan.latent_dim, n)
    return forward!(gan.generator, Z)
end

function fit_gan!(gan::GAN, X_real::Matrix{F};
                  epochs::Int=100, batch_size::Int=32, eta::F=F(0.01),
                  k::Int=1, verbose::Bool=true)

    m = size(X_real, 2)

    for epoch in 1:epochs
        # --- Entrenar Discriminator ---
        for _ in 1:k
            idx = rand(1:m, batch_size)
            X_batch = X_real[:, idx]
            Y_real = ones(F, 1, batch_size)
            Y_fake = zeros(F, 1, batch_size)

            # Real
            forward!(gan.discriminator, X_batch)
            backward!(gan.discriminator, X_batch, Y_real)
            update!(gan.discriminator, eta)

            # Fake
            Z = randn(F, gan.latent_dim, batch_size)
            X_gen = forward!(gan.generator, Z)
            forward!(gan.discriminator, X_gen)
            backward!(gan.discriminator, X_gen, Y_fake)
            update!(gan.discriminator, eta)
        end

        # --- Entrenar Generator (truco non-saturating) ---
        Z = randn(F, gan.latent_dim, batch_size)
        X_gen = forward!(gan.generator, Z)
        forward!(gan.discriminator, X_gen)

        # Queremos que D clasifique como real (1)
        Y_target = ones(F, 1, batch_size)
        backward!(gan.discriminator, X_gen, Y_target)

        # El gradiente en la última capa de D respecto a la salida
        L = length(gan.discriminator.layers)
        dZ = gan.discriminator.cost.delta(gan.discriminator.layers[L].Z,
                                           gan.discriminator.layers[L].A,
                                           Y_target)
        # Propagar hacia atrás hasta la entrada de D
        for l in L:-1:2
            dA = gan.discriminator.layers[l].W' * dZ
            dZ = dA .* gan.discriminator.layers[l-1].activation.d.(gan.discriminator.layers[l-1].Z)
        end
        # Obtener gradiente respecto a la entrada X_gen
        dX = gan.discriminator.layers[1].W' * dZ
        
        # Usamos como target para G: X_gen - alpha * dX
        target_G = X_gen .- F(0.1) .* dX

        forward!(gan.generator, Z)
        backward!(gan.generator, Z, target_G)
        update!(gan.generator, eta)

        if verbose && epoch % 10 == 0
            @info @Printf.sprintf("GAN epoch %d/%d", epoch, epochs)
        end
    end
    return gan
end

# ============================================================
# 10. UTILIDADES DE PREPARACIÓN DE DATOS
# ============================================================

"""
`normalize01(X::Matrix{F})::Matrix{F}`

Normaliza los datos al rango [0, 1] usando min-max scaling:
X_norm = (X - min) / (max - min)

Beneficios:
- Todas las features tienen la misma escala
- Facilita el entrenamiento (gradientes más estables)
- Compatible con activaciones como sigmoid

IMPORTANTE: Guardar min/max para normalizar datos de test.
"""
function normalize01(X::Matrix{F})::Matrix{F}
    min_val = minimum(X)
    max_val = maximum(X)
    range = max_val - min_val
    # Evitar división por cero si todos los valores son iguales
    range = range == 0 ? one(F) : range
    return (X .- min_val) ./ range
end

"""
`to_matrix(data::Vector{Vector{Float64}})::Matrix{F}`

Convierte Vector de Vectores a Matriz.
Cada vector se convierte en una columna.

Ejemplo:
[[1,2], [3,4], [5,6]] -> [1 3 5]
                         [2 4 6]

Útil para compatibilidad con código que usa Vector{Vector}.
"""
function to_matrix(data::Vector{Vector{Float64}})::Matrix{F}
    d = length(data[1])  # dimensión de cada muestra
    n = length(data)     # número de muestras
    X = Matrix{F}(undef, d, n)
    for i in 1:n
        X[:, i] = data[i]
    end
    return X
end

"""
`to_vectors(X::Matrix{F})::Vector{Vector{F}}`

Convierte Matriz a Vector de Vectores.
Cada columna se convierte en un vector.

Inversa de to_matrix().
"""
function to_vectors(X::Matrix{F})::Vector{Vector{F}}
    return [X[:, i] for i in 1:size(X, 2)]
end

# ============================================================
# 11. ATAJOS DE LLAMADA FUNCIONAL (sintaxis R(x))
# ============================================================

"""
`Network(x)`

Atajo para evaluar la red. Acepta:
* `Vector{F}` -> devuelve `Vector{F}` (un solo sample)
* `Matrix{F}` -> devuelve `Matrix{F}` (batch completo)

Ejemplo:
```julia
net = Network([2, 16, 1], [relu, sigmoid])
y = net([0.5f0, -0.2f0])        # un sample
Y = net(X_batch)                 # batch matricial
```
"""
(R::Network)(x::Vector{F}) = vec(forward!(R, reshape(x, :, 1)))
(R::Network)(x::Matrix{F}) = forward!(R, x)

"""
`Autoencoder(x)` -> reconstrucción
`Autoencoder(x, Val(:encode))` -> codificación al bottleneck
"""
(ae::Autoencoder)(x::Vector{F}) = vec(reconstruct(ae, reshape(x, :, 1)))
(ae::Autoencoder)(x::Matrix{F}) = reconstruct(ae, x)
(ae::Autoencoder)(x::Vector{F}, ::Val{:encode}) = vec(encode(ae, reshape(x, :, 1)))
(ae::Autoencoder)(x::Matrix{F}, ::Val{:encode}) = encode(ae, x)

"""
`GAN(n)` -> genera n muestras
"""
(gan::GAN)(n::Int) = generate(gan, n)

end # module
