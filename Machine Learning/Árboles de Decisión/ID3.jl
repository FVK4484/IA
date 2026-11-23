using DataFrames
using StatsBase
using CSV
using Plots
using Random
using Statistics
##########################
## Funciones Auxiliares ##
##########################

# Logaritmo en base 2 seguro (retorna 0 si x ≤ 0)
log2a(x) = (x > 0) ? log2(x) : 0

# Calcula la entropía de Shannon de un vector de etiquetas.
# Usa StatsBase.proportionmap para obtener las proporciones de cada clase
function entropia(y)
    if length(y) == 0
        return 0.0
    end
    p = values(StatsBase.proportionmap(y))
    -sum(p .* log2a.(p))
end

# Encuentra el punto de corte óptimo para un atributo continuo que maximiza la ganancia de información
function encontrar_mejor_corte(data, atributo, obj)
    if nrow(data) <= 1
        return nothing, -Inf
    end
    
    # Ordena valores numéricos
    valores_ordenados = sort(unique(data[:, atributo]))
    if length(valores_ordenados) <= 1
        return nothing, -Inf
    end
    
    # Calcular los posibles puntos de corte (puntos medios entre valores consecutivos)
    puntos_corte = [round((valores_ordenados[i] + valores_ordenados[i+1])/2, digits=2) for i in 1:length(valores_ordenados)-1]

    # Encontrar el punto de corte con la máxima ganancia de información
    mejor_ganancia = -Inf
    mejor_corte   = nothing
    
    for c in puntos_corte
        # Dividir los datos en dos grupos
        mascara   = data[:, atributo] .<= c
        data_izq  = data[mascara, :]
        data_dcha = data[.!mascara, :]
        
        # Calcular la entropía ponderada
        prob_izq  = nrow(data_izq) / nrow(data)
        prob_dcha = nrow(data_dcha) / nrow(data)
        
        entropia_ponderada = prob_izq * entropia(data_izq[:, obj]) +
                             prob_dcha * entropia(data_dcha[:, obj])
        
        # Calcular la ganancia de información
        ganancia = entropia(data[:, obj]) - entropia_ponderada
        
        if ganancia > mejor_ganancia
            mejor_ganancia = ganancia
            mejor_corte = c
        end
    end
    
    return mejor_corte, mejor_ganancia
end

# Calcula la ganancia de información de un atributo (categórico o continuo) respecto al objetivo
function GI(data, atributo, obj)
    # Comprobar si atributo es continuo (numérico)
    es_continuo = eltype(data[:, atributo]) <: Number
    
    if es_continuo
        mejor_corte, mejor_ganancia = encontrar_mejor_corte(data, atributo, obj)
        return mejor_ganancia
    else
        entropia_total     = entropia(data[:, obj])
        entropia_ponderada = 0.0
        valores_atributo   = unique(data[:, atributo])
        
        for v in valores_atributo
            subdata = data[data[:, atributo] .== v, :]
            prob    = nrow(subdata) / nrow(data)
            entropia_ponderada += prob * entropia(subdata[:, obj])
        end
        
        return entropia_total - entropia_ponderada
    end
end

# Selecciona el atributo con mayor ganancia de información y su punto de corte (si es continuo)
function encontrar_mejor_atributo(data, atributos, obj)
    mejor_ganancia = -Inf
    mejor_atr      = first(atributos)
    mejor_corte    = nothing
    
    for atr in atributos
        es_continuo = eltype(data[:, atr]) <: Number
        
        if es_continuo
            punto_corte, ganancia = encontrar_mejor_corte(data, atr, obj)
            if ganancia > mejor_ganancia
                mejor_ganancia = ganancia
                mejor_atr = atr
                mejor_corte = punto_corte
            end
        else
            ganancia = GI(data, atr, obj)
            if ganancia > mejor_ganancia
                mejor_ganancia = ganancia
                mejor_atr = atr
                mejor_corte = nothing
            end
        end
    end
    
    return mejor_atr, mejor_corte
end

# Adaptador: usa la partición de ML y recompone los DataFrame que espera ID3.
function train_test_split_id3(
    data::DataFrame,
    objetivo::Symbol;
    train_ratio::Real=0.8,
    seed::Int=42
)
    Random.seed!(seed)

    X_train, y_train, X_test, y_test =
        train_test_split(data, objetivo, train_ratio)

    train_data = copy(X_train)
    train_data[!, objetivo] = y_train
    train_data = select(train_data, propertynames(data))

    test_data = copy(X_test)
    test_data[!, objetivo] = y_test
    test_data = select(test_data, propertynames(data))

    return train_data, test_data
end

###############################################
## Estructura para la construcción del árbol ##
###############################################

# Estructura para representar un nodo del árbol. Puede ser de tipo 
#   Atributo    (nodo de decisión)
#   Hoja        (nodo de respuesta)
#
# Si es de tipo atributo y continuo tiene un punto_corte e hijos
# Si es de tipo atributo y categórico tiene solo hijos
# Si es de tipo hoja, tiene un valor de clase
# Los hijos son un diccionario que mapea valores (del atributo) a nodos hijos
#
# Ejemplo Hoja:
#  
# Treenode(
#     nothing,                  # es hoja, así que no tiene atributo
#     nothing,                  # tampoco tiene punto de corte
#     Dict{Any, Treenode}(),    # ni hijos...
#     String3("No"))            # pero sí tiene valor de hoja (respuesta)
#   
# Ejemplo Atributo Categórico (que acaba en hojas):
#
# Treenode(
#     :Estilo,                  # es de decisión, así que tiene atributo
#     nothing,                  # es categórico, así que no tiene punto_corte
#     Dict{Any, Treenode}       # es de tipo atributo, así que tiene hijos
#         (
#             String7("Moderno") => Treenode(
#                 nothing, 
#                 nothing,
#                 Dict{Any, Treenode}(), 
#                 String3("No")), 
#             String7("Clásico") => Treenode(
#                 nothing,
#                 nothing, 
#                 Dict{Any, Treenode}(), 
#                 String3("Yes"))), 
#     nothing)                  # es de tipo atributo, así que no tiene hoja
#
# Ejemplo Atributo Continuo (que acaba en hojas):
#
# Treenode(
#     :Temperatura,             # es de decisión, así que tiene atributo
#     32.0,                     # es continuo, así que tiene punto_corte
#     Dict{Any, Treenode}       # es de tipo atributo, así que tiene hijos
#         (
#             String7("> 32.0") => Treenode(
#                 nothing, 
#                 nothing,
#                 Dict{Any, Treenode}(), 
#                 String3("No")), 
#             String7("≤ 32.0") => Treenode(
#                 nothing,
#                 nothing, 
#                 Dict{Any, Treenode}(), 
#                 String3("Yes"))), 
#     nothing)                  # es de tipo atributo, así que no tiene hoja

mutable struct Treenode
    atributo::Union{Symbol, Nothing}
    punto_corte::Union{Float64, Nothing} 
    hijos::Dict{Any, Treenode}
    hoja::Union{Any, Nothing}
    
    Treenode() = new(nothing, nothing, Dict(), nothing)
end

########################
## Implementación ID3 ##
########################

# Construye un árbol de decisión ID3 usando el DataFrame, atributos y objetivo dados
function id3_train(
    data::DataFrame, 
    atributos::Vector{Symbol}, 
    objetivo::Symbol, 
    min_muestras::Int = 2)

    nodo = Treenode()
    
    # Base cases
    if length(unique(data[:, objetivo])) == 1
        nodo.hoja = first(data[:, objetivo])
        return nodo
    end
    
    if isempty(atributos) || nrow(data) < min_muestras
        nodo.hoja = mode(data[:, objetivo])
        return nodo
    end
    
    # Encontrar mejor atributo y punto de corte
    mejor_at, punto_corte = encontrar_mejor_atributo(data, atributos, objetivo)
    nodo.atributo = mejor_at
    nodo.punto_corte = punto_corte
    
    # Tratar de forma diferente los atributos continuos y los categóricos
    if !isnothing(punto_corte)  # Atributos continuos
        # División binaria basada en el punto de corte
        mascara = data[:, mejor_at] .<= punto_corte
        subdatas = Dict(
            "≤ $(punto_corte)" => data[mascara, :],
            "> $(punto_corte)" => data[.!mascara, :]
        )
    else  # Atributos categóricos
        # Cortar por valores únicos
        subdatas = Dict(v => data[data[:, mejor_at] .== v, :] 
                      for v in unique(data[:, mejor_at]))
        # Los atributos categóricos usados ya no se consideran en subárboles
        atributos = filter(f -> f != mejor_at, atributos)
    end

    # Crear subárboles
    for (v, subdata) in subdatas
        if nrow(subdata) == 0
            hijo = Treenode()
            hijo.hoja = mode(data[:, objetivo])
        else
            hijo = id3_train(subdata, atributos, objetivo, min_muestras)
        end
        nodo.hijos[v] = hijo
    end
    
    # Crear subárboles: ¡¡¡¡ Incompleta !!!!
    ### Esta versión es incompleta porque no permite preguntar varias veces 
    ### por el mismo atributo numérico con diferentes cortes.
    ### Es una implementación natural para atributos categóricos, pero no
    ### para atributos numéricos:
    #
    # resto_atributos = filter(f -> f != mejor_at, atributos)
    #
    # for (v, subdata) in subdatas
    #     if nrow(subdata) == 0
    #         hijo = Treenode()
    #         hijo.hoja = mode(data[:, objetivo])
    #     else
    #         hijo = id3_train(subdata, resto_atributos, objetivo, min_muestras)
    #     end
    #     nodo.hijos[v] = hijo
    # end
    
    return nodo
end

# Predice la clase de una muestra individual recorriendo el árbol de decisión
function ID3_predict(nodo::Treenode, muestra::DataFrameRow)
    if !isnothing(nodo.hoja)
        return nodo.hoja
    end
    
    # Manejar atributos continuos
    if !isnothing(nodo.punto_corte)
        v = muestra[nodo.atributo] <= nodo.punto_corte ? "≤ $(nodo.punto_corte)" : "> $(nodo.punto_corte)"
    else
        v = muestra[nodo.atributo]
    end
    
    if !haskey(nodo.hijos, v)
        hojas = [hijo.hoja for hijo in values(nodo.hijos) if !isnothing(hijo.hoja)]
        return isempty(hojas) ? nothing : mode(hojas)
    end
    
    return ID3_predict(nodo.hijos[v], muestra)
end

# Predice las clases de múltiples muestras en un DataFrame
function ID3_predict(nodo::Treenode, data::DataFrame)
    return [ID3_predict(nodo, data[i, :]) for i in 1:nrow(data)]
end

########################
## Representación ID3 ##
########################

# Genera una representación textual del árbol con estadísticas e indentación jerárquica
function show_arbol(nodo::Treenode, depth::Int=0, prefix::String="")
    function tree_to_string(nodo::Treenode, prof::Int)
        if isnothing(nodo.atributo)
            return "$(prefix * "   "^prof)     ✅ $(nodo.hoja)"
        end
        
        # Mostrar punto de corte para atributos continuos
        split_info = isnothing(nodo.punto_corte) ? "" : " (corte en $(round(nodo.punto_corte, digits=2)))"
        nodo_str = "$(prefix * "  "^prof)   ❓: $(nodo.atributo)$(split_info)?\n"
        
        for (valor, hijo) in nodo.hijos
            hijo_str = tree_to_string(hijo, prof + 1)
            nodo_str *= "$(prefix * "  "^(prof+1)) └─ $(nodo.atributo) $(valor):\n"
            nodo_str *= "$(hijo_str)\n"
        end
        
        return nodo_str
    end
    
    function tree_info(nodo::Treenode)
        function contar_nodos(nodo::Treenode)
            if isnothing(nodo.atributo)
                return (nodos = 1, hojas = 1, prof = 0)
            end
            
            total_nodos, total_hojas, max_prof = 1, 0, 0
            
            for hijo in values(nodo.hijos)
                hijo_info = contar_nodos(hijo)
                total_nodos += hijo_info.nodos
                total_hojas += hijo_info.hojas
                max_prof = max(max_prof, hijo_info.prof)
            end
            
            return (nodos = total_nodos, hojas = total_hojas, prof = max_prof + 1)
        end
        
        info = contar_nodos(nodo)
        
        return """
        Información del árbol: 
            $(info.nodos) nodos ($(info.hojas) hojas), Profundidad $(info.prof)
        """
    end
    
    cadena_arbol = """
    Representación del Árbol de Decisión
    
    $(tree_info(nodo))
    
    Estructura del árbol:
    $(tree_to_string(nodo, 0))
    """
    
    return cadena_arbol
end

####################
## Evaluación ID3 ##
####################

# Calcula la matriz de confusión y métricas (accuracy, precision, recall, F1) por clase
function evalua_modelo(true_labels, pred_labels)
    clases = unique(vcat(true_labels, pred_labels))
    cm = matriz_confusion(true_labels, pred_labels; clases=clases)
    metricas_df = metricas_por_clase(cm)
    metricas = Dict(
        fila.Clase => (
            precision = fila.Precision,
            recall = fila.Recall,
            f1_score = fila.F1
        )
        for fila in eachrow(metricas_df)
    )

    return (
        MC = cm.matriz,
        class_metricas = metricas,
        accuracy = accuracy(cm),
        labels = cm.clases,
        confusion = cm
    )
end

# Muestra en pantalla las métricas de evaluación y la matriz de confusión
function show_rendimiento(evaluacion)
    println("\nMetricas de Evaluación:")
    println("Accuracy: ", round(evaluacion.accuracy, digits=4))

    println("\nMétricas por Clase:")
    for (label, metrica) in evaluacion.class_metricas
        println("Clase $label:")
        println("  Precisión:  ", round(metrica.precision, digits=4))
        println("  Recall:     ", round(metrica.recall, digits=4))
        println("  F1 Score:   ", round(metrica.f1_score, digits=4))
    end

    println("\nMatriz de Confusión:")
    display(evaluacion.MC)
end

# Visualiza la matriz de confusión como un heatmap con anotaciones de valores
function plot_MC(evaluacion)
    return graficar_matriz_confusion(
        evaluacion.confusion;
        titulo="Matriz de confusión"
    )
end

# ===========================================================================
# Representación de la estructura del árbol de decisión ID3
# ===========================================================================

using Plots

function _id3_label(nodo)
    if isnothing(nodo.atributo)
        return "Clase: $(nodo.hoja)"
    end

    isnothing(nodo.punto_corte) &&
        return "Atributo: $(nodo.atributo)"

    return "Atributo: $(nodo.atributo)\nCorte: $(round(nodo.punto_corte, digits=2))"
end

_id3_children(nodo) = sort!(collect(nodo.hijos); by=par -> string(first(par)))

function _id3_collect!(nodo, nodos, aristas)
    push!(nodos, nodo)
    for (etiqueta, hijo) in _id3_children(nodo)
        push!(aristas, (nodo, hijo, string(etiqueta)))
        _id3_collect!(hijo, nodos, aristas)
    end
    return nothing
end

function _id3_layout!(nodo, profundidad, hoja_actual, n_hojas, separacion_x,
                      separacion_y, posiciones)
    hijos = _id3_children(nodo)

    if isempty(hijos)
        hoja_actual[] += 1
        x = (hoja_actual[] - (n_hojas + 1) / 2) * separacion_x
    else
        xs = [_id3_layout!(last(par), profundidad + 1, hoja_actual, n_hojas,
                           separacion_x, separacion_y, posiciones) for par in hijos]
        x = (first(xs) + last(xs)) / 2
    end

    y = -profundidad * separacion_y
    posiciones[nodo] = (x, y)
    return x
end

"""
    plot_arbol_id3(raiz)

Devuelve una visualización `Plots.Plot` de un árbol ID3 con nodos `Treenode`.
Las hojas muestran la clase; las aristas muestran los valores o condiciones
que conducen a cada hijo.
"""
function plot_arbol_id3(raiz)
    all(hasproperty(raiz, campo) for campo in (:atributo, :punto_corte, :hijos, :hoja)) ||
        throw(ArgumentError("La raíz debe tener los campos de un nodo ID3"))

    nodos = Any[]
    aristas = Tuple{Any, Any, String}[]
    _id3_collect!(raiz, nodos, aristas)

    etiquetas = IdDict{Any, String}()
    max_caracteres = 1
    max_lineas = 1
    for nodo in nodos
        etiqueta = _id3_label(nodo)
        etiquetas[nodo] = etiqueta
        lineas = split(etiqueta, '\n')
        max_caracteres = max(max_caracteres, maximum(length, lineas))
        max_lineas = max(max_lineas, length(lineas))
    end

    n_hojas = count(nodo -> isempty(nodo.hijos), nodos)
    max_profundidad = maximum(
        function profundidad(nodo)
            hijos = _id3_children(nodo)
            isempty(hijos) ? 0 : 1 + maximum(profundidad(last(par)) for par in hijos)
        end,
        (raiz,)
    )

    ancho_nodo = max(1.8, 0.12 * max_caracteres + 0.6)
    alto_nodo = max(0.9, 0.35 * max_lineas + 0.5)
    separacion_x = ancho_nodo + 0.9
    separacion_y = alto_nodo + 1.2
    posiciones = IdDict{Any, Tuple{Float64, Float64}}()
    _id3_layout!(raiz, 0, Ref(0), n_hojas, separacion_x, separacion_y, posiciones)

    ancho_figura = clamp(240 * n_hojas, 900, 2400)
    alto_figura = clamp(170 * (max_profundidad + 1), 400, 1800)
    mitad_x = max(separacion_x / 2, (n_hojas - 1) * separacion_x / 2) +
              ancho_nodo / 2 + 0.5
    limite_y = max_profundidad * separacion_y + alto_nodo / 2 + 0.5

    p = Plots.plot(
        ;
        legend=false,
        axis=false,
        grid=false,
        framestyle=:none,
        xlims=(-mitad_x, mitad_x),
        ylims=(-limite_y, alto_nodo / 2 + 0.5),
        size=(ancho_figura, alto_figura),
        background_color=:white
    )

    for (padre, hijo, etiqueta) in aristas
        xp, yp = posiciones[padre]
        xh, yh = posiciones[hijo]
        inicio_y = yp - alto_nodo / 2
        fin_y = yh + alto_nodo / 2

        Plots.plot!(p, [xp, xh], [inicio_y, fin_y];
                    linecolor=:gray55, linewidth=1.5, label="")
        Plots.annotate!(p, (xp + xh) / 2, (inicio_y + fin_y) / 2,
                        Plots.text(etiqueta, 8, :center, :gray25))
    end

    for nodo in nodos
        x, y = posiciones[nodo]
        es_hoja = isempty(nodo.hijos)
        color_fondo = es_hoja ? :honeydew : :aliceblue
        color_borde = es_hoja ? :darkgreen : :steelblue
        forma = Plots.Shape(
            [x - ancho_nodo / 2, x + ancho_nodo / 2,
             x + ancho_nodo / 2, x - ancho_nodo / 2],
            [y - alto_nodo / 2, y - alto_nodo / 2,
             y + alto_nodo / 2, y + alto_nodo / 2]
        )

        Plots.plot!(p, forma; fillcolor=color_fondo, linecolor=color_borde,
                    linewidth=1.5, label="")
        Plots.annotate!(p, x, y,
                        Plots.text(etiquetas[nodo], 9, :center, color_borde))
    end

    return p
end

"""
    guardar_arbol_id3(raiz, archivo="arbol_id3.png")

Guarda la visualización del árbol en el formato indicado por la extensión.
"""
function guardar_arbol_id3(raiz, archivo::AbstractString="arbol_id3.png")
    Plots.savefig(plot_arbol_id3(raiz), archivo)
    return archivo
end

"""
    mostrar_arbol_id3(raiz)

Muestra y devuelve la visualización del árbol.
"""
function mostrar_arbol_id3(raiz)
    p = plot_arbol_id3(raiz)
    display(p)
    return p
end

"""
    plot_arbol_dot(raiz)

Alias compatible con el código anterior. Devuelve un gráfico de Plots, no un
grafo DOT; no requiere Graphviz ni GraphvizDotLang.
"""
plot_arbol_dot(raiz) = plot_arbol_id3(raiz)

# =========================================================
println("ID3.jl cargado.")
