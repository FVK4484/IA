include("CSP.jl")
using .CSP_Module

# -----------------------------------------
# Ejemplo 4: Problema de coloreado de mapas
# -----------------------------------------

#=
El siguiente problema clásico que veremos: Cómo colorear un mapa de forma que 2 países con frontera común queden coloreados con colores distintos.

Aunque se suele plantear como "encontrar el menor número de colores", aquí nos vamos a centrar en: dado un número fijo de colores, encontrar, si existe, una coloración válida usando como mucho ese número de colores.

Este problema, que se da en 2D, se generaliza al problema de coloreado de grafos: Dado un grafo G=(V,E), donde V es el conjunto de vértices y E el de aristas, y dado un conjunto de colores, C, se trata de encontrar una función de coloración que asigne a cada vértice un color de forma que no haya dos vértices adyacentes del mismo color.

Vamos a resolver el problema para un caso simple y particular: las regiones de Australia, pero la solución es general y fácilmente extrapolable al caso de colorear grafos cualesquiera. Y lo vamos a resolver con Backtracking estándar.
=#

function coloreado_australia()
    println("=== PROBLEMA DEL COLOREADO DE AUSTRALIA ===")
    println("    ===================================")
    println("Colorear las regiones de Australia con 3 colores sin que regiones adyacentes tengan el mismo color\n")
    
    # Variables: cada región
    variables = [:WA, :NT, :Q, :NSW, :V, :SA, :T]  # Western Australia, Northern Territory, etc.
    
    # Dominios: 3 colores disponibles
    colores = ["Rojo", "Verde", "Azul"]
    domains = Dict(region => colores for region in variables)
    
    # Restricciones: regiones adyacentes deben tener colores diferentes
    adyacencias = [
        (:WA, :NT), (:WA, :SA),
        (:NT, :WA), (:NT, :SA), (:NT, :Q),
        (:SA, :WA), (:SA, :NT), (:SA, :Q), (:SA, :NSW), (:SA, :V),
        (:Q, :NT), (:Q, :SA), (:Q, :NSW),
        (:NSW, :Q), (:NSW, :SA), (:NSW, :V),
        (:V, :SA), (:V, :NSW)
        # Tasmania (:T) es una isla, no tiene adyacencias
    ]
    
    constraints = Constraint[]
    for (r1, r2) in adyacencias
        push!(constraints, Constraint([r1, r2],
        function(assignment)
            if haskey(assignment, r1) && haskey(assignment, r2)
                return assignment[r1] != assignment[r2]
            end
            return true
        end, "$r1 ≠ $r2"
        ))
    end
    
    csp = CSP(variables, domains, constraints)
    
    show_CSP(csp)
    println("Resolviendo con Backtracking...")
    solucion = solve_BT(csp)
    
    if !isnothing(solucion)
        println("¡Solución encontrada!")
        for region in [:WA, :NT, :SA, :Q, :NSW, :V, :T]
            println("$region: $(solucion[region])")
        end
    else
        println("No se encontró solución")
    end
    println()
    
    return csp, solucion
end

coloreado_australia()