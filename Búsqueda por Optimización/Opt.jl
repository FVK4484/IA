# ============================================
# Templado Simulado (Simulated Annealing)
# ============================================

"""
Implementación básica del Templado Simulado. Depende de:
- `Energia::Function`: Función de energía/fitness a minimizar (toma estados).
- `s0`: Estado inicial.
- `Rep:Int`: Número de intentos por ciclo (para cada valor de T).
- `vecino::Function`: Función que devuelve un vecino del estado actual.
- `T_min::Float64`: Valor mínimo de T.
- `tasa_T::Float64`: Tasa de enfriamiento (% de disminución de T).
- `ac_eq::Bool`: Indica si se acepta el cambio cuando ΔE=0.
"""
function SA(Energia::Function, s0, Rep::Int, vecino::Function, T_min::Float64,
            tasa_T::Float64, ac_eq::Bool)
    # Datos de partida
    T = 1.0
    sbest = s0 
    Ebest = E0 = Energia(s0)

    # El algoritmo itera hasta alcanzar la T_min
    while T > T_min

        # Hacemos Rep pruebas para un T fijo
        for _ in 1:Rep            
            s1 = vecino(s0)           # Tomamos un vecino (generador)
            E1 = Energia(s1)          # Calculamos su energía
            ΔE = E1 - E0              # y el incremento de energía

            # Si la energía disminuye, o es igual pero se acepta, o con prob∼T
            if ΔE < 0 || (ac_eq && ΔE ==0) || rand() < T                      
                    s0 = s1                 # saltamos a ese estado
                    E0 = E1
                    # Si el nuevo estado es el mejor
                    if E0 < Ebest
                        sbest = s0              # lo almacenamos
                        Ebest = E0
                    end
            end
        end

        # Actualizamos T
        T = T * (1 - tasa_T / 100)  
    end

    # Devolvemos el mejor estado encontrado, y su energía
    return sbest, Ebest
end

println("Templado Simulado cargado correctamente.")

# ============================================
# DESCENSO DE GRADIENTE BÁSICO
# ============================================

using LinearAlgebra

"""
    gradient_descent(f, grad_f, θ₀; η=0.01, max_iter=1000, tol=1e-6)

Descenso de gradiente básico (batch).

# Argumentos
- `f`: función objetivo f(θ) → escalar
- `grad_f`: gradiente ∇f(θ) → vector
- `θ₀`: punto inicial
- `η`: tasa de aprendizaje
- `max_iter`: máximo de iteraciones
- `tol`: tolerancia para convergencia (norma del gradiente)

# Retorna
- `θ_opt`: parámetros optimizados
- `history`: historial de valores [f(θ), ‖∇f‖]
"""
function gradient_descent(f, grad_f, θ₀; η=0.01, max_iter=1000, tol=1e-6)
    θ = copy(θ₀)
    history = Vector{Tuple{Float64, Float64}}()
    
    for t in 1:max_iter
        g = grad_f(θ)
        f_val = f(θ)
        grad_norm = norm(g)
        
        push!(history, (f_val, grad_norm))
        
        # Criterio de parada
        if grad_norm < tol
            println("Convergencia alcanzada en iteración $t")
            return θ, history
        end
        
        # Actualización
        θ = θ .- η .* g
    end
    
    println("Máximo de iteraciones alcanzado")
    return θ, history
end




# ============================================
# DESCENSO ESTOCÁSTICO POR MINI-BATCHES
# ============================================

using Random

"""
    sgd(f_batch, grad_batch, θ₀, data; η=0.01, batch_size=32, epochs=100, tol=1e-6)

SGD con mini-batches.

# Argumentos
- `f_batch`: pérdida para un batch f(θ, batch_indices) → escalar
- `grad_batch`: gradiente para un batch ∇f(θ, batch_indices) → vector
- `θ₀`: punto inicial
- `data`: datos de entrenamiento (tupla o struct)
- `batch_size`: tamaño del mini-batch
- `epochs`: épocas de entrenamiento
"""
function sgd(f_batch, grad_batch, θ₀, data; 
             η=0.01, batch_size=32, epochs=100, tol=1e-6, verbose=true)
    
    θ = copy(θ₀)
    n = size(data[1], 1)  # número de muestras (número de filas)
    n_batches = ceil(Int, n / batch_size)
    history = Vector{Tuple{Int, Float64, Float64}}()  # (iter, loss, grad_norm)
    iter = 0
    
    for epoch in 1:epochs
        # Barajar datos cada época
        indices = shuffle(1:n)
        
        epoch_loss = 0.0
        
        for b in 1:n_batches
            batch_idx = indices[(b-1)*batch_size+1 : min(b*batch_size, n)]
            
            g = grad_batch(θ, batch_idx)
            θ = θ .- η .* g
            
            epoch_loss += f_batch(θ, batch_idx)
            iter += 1
            
            # Guardar historial cada ciertas iteraciones
            if iter % 100 == 0
                # Evaluar en todo el dataset para monitoreo
                full_loss = f_batch(θ, 1:n)
                push!(history, (iter, full_loss, norm(g)))
            end
        end
        
        if verbose && epoch % 10 == 0
            println("Época $epoch: loss promedio = $(epoch_loss/n_batches)")
        end
    end
    
    return θ, history
end




# ============================================
# MOMENTUM (GD con Momentum)
# ============================================

"""
    gd_momentum(f, grad_f, θ₀; η=0.01, β=0.9, max_iter=1000, tol=1e-6)

GD con momentum. β=0 recupera GD estándar.
"""
function gd_momentum(f, grad_f, θ₀; η=0.01, β=0.9, max_iter=1000, tol=1e-6)
    θ = copy(θ₀)
    v = zero(θ)  # velocidad inicial
    history = Vector{Tuple{Float64, Float64}}()
    
    for t in 1:max_iter
        g = grad_f(θ)
        f_val = f(θ)
        grad_norm = norm(g)
        
        push!(history, (f_val, grad_norm))
        
        if grad_norm < tol
            println("Convergencia en iteración $t")
            return θ, history
        end
        
        # Actualizar velocidad y parámetros
        v = β .* v .+ g          # acumular momentum
        θ = θ .- η .* v          # moverse con velocidad
    end
    
    return θ, history
end


# ============================================
# NESTEROV ACCELERATED GRADIENT (NAG)
# ============================================

"""
    nesterov(f, grad_f, θ₀; η=0.01, β=0.9, max_iter=1000, tol=1e-6)

Momentum de Nesterov: evalúa el gradiente en la posición "adelantada".
"""
function nesterov(f, grad_f, θ₀; η=0.01, β=0.9, max_iter=1000, tol=1e-6)
    θ = copy(θ₀)
    v = zero(θ)
    history = Vector{Tuple{Float64, Float64}}()
    
    for t in 1:max_iter
        # Posición "adelantada" donde evaluamos el gradiente
        θ_lookahead = θ .- β .* v
        
        g = grad_f(θ_lookahead)
        f_val = f(θ)
        grad_norm = norm(g)
        
        push!(history, (f_val, grad_norm))
        
        if grad_norm < tol
            println("Convergencia en iteración $t")
            return θ, history
        end
        
        v = β .* v .+ g
        θ = θ .- η .* v
    end
    
    return θ, history
end

# ============================================
# ADAGRAD
# ============================================

"""
    adagrad(f, grad_f, θ₀; η=0.01, ε=1e-8, max_iter=1000, tol=1e-6)

AdaGrad: adapta η por coordenada. Bueno para datos sparse, malo para
optimización profunda (acumulación infinita de gradientes).
"""
function adagrad(f, grad_f, θ₀; η=0.01, ε=1e-8, max_iter=1000, tol=1e-6)
    θ = copy(θ₀)
    r = zero(θ)  # acumulador de gradientes cuadrados
    history = Vector{Tuple{Float64, Float64}}()
    
    for t in 1:max_iter
        g = grad_f(θ)
        f_val = f(θ)
        grad_norm = norm(g)
        
        push!(history, (f_val, grad_norm))
        
        if grad_norm < tol
            println("Convergencia en iteración $t")
            return θ, history
        end
        
        r = r .+ g.^2                    # acumular cuadrados
        θ = θ .- (η ./ (sqrt.(r) .+ ε)) .* g   # adaptar por coordenada
    end
    
    return θ, history
end

# ============================================
# RMSPROP
# ============================================

"""
    rmsprop(f, grad_f, θ₀; η=0.001, β=0.9, ε=1e-8, max_iter=1000, tol=1e-6)

RMSprop: media móvil exponencial de gradientes cuadrados.
"""
function rmsprop(f, grad_f, θ₀; η=0.001, β=0.9, ε=1e-8, max_iter=1000, tol=1e-6)
    θ = copy(θ₀)
    r = zero(θ)  # media móvil de g²
    history = Vector{Tuple{Float64, Float64}}()
    
    for t in 1:max_iter
        g = grad_f(θ)
        f_val = f(θ)
        grad_norm = norm(g)
        
        push!(history, (f_val, grad_norm))
        
        if grad_norm < tol
            println("Convergencia en iteración $t")
            return θ, history
        end
        
        r = β .* r .+ (1 - β) .* g.^2    # EMA de cuadrados
        θ = θ .- (η ./ (sqrt.(r) .+ ε)) .* g
    end
    
    return θ, history
end

# ============================================
# ADAM
# ============================================

"""
    adam(f, grad_f, θ₀; η=0.001, β₁=0.9, β₂=0.999, ε=1e-8, 
         max_iter=1000, tol=1e-6)

Adam: Adaptive Moment Estimation. Combina momentum (primer momento)
y RMSprop (segundo momento) con corrección de sesgo.

# Hiperparámetros estándar (no tocar sin buena razón):
- β₁ = 0.9  (decaimiento del momentum)
- β₂ = 0.999 (decaimiento del segundo momento)
- η = 0.001 (tasa de aprendizaje base)
"""
function adam(f, grad_f, θ₀; η=0.001, β₁=0.9, β₂=0.999, ε=1e-8, 
              max_iter=1000, tol=1e-6)
    θ = copy(θ₀)
    m = zero(θ)  # primer momento (media móvil de gradientes)
    v = zero(θ)  # segundo momento (media móvil de cuadrados)
    history = Vector{Tuple{Float64, Float64}}()
    
    for t in 1:max_iter
        g = grad_f(θ)
        f_val = f(θ)
        grad_norm = norm(g)
        
        push!(history, (f_val, grad_norm))
        
        if grad_norm < tol
            println("Convergencia en iteración $t")
            return θ, history
        end
        
        # Actualizar momentos
        m = β₁ .* m .+ (1 - β₁) .* g
        v = β₂ .* v .+ (1 - β₂) .* g.^2
        
        # Corrección de sesgo (crucial en las primeras iteraciones)
        m_hat = m ./ (1 - β₁^t)
        v_hat = v ./ (1 - β₂^t)
        
        # Actualización
        θ = θ .- (η ./ (sqrt.(v_hat) .+ ε)) .* m_hat
    end
    
    return θ, history
end

println("Descenso por Gradiente (y variantes) cargados correctamente.")
