module CSP_Module

export Constraint, CSP, is_consistent, is_full_assignment_consistent,
       is_partially_consistent, AC3!, solve_BT_with_AC3,
       solve_brute_force, solve_BT, solve_BT_with_GAC,
       GAC!, compare_AC3_vs_GAC
export AC3_optimized!, GAC_optimized!, solve_BT_with_AC3_optimized, 
       solve_BT_with_GAC_optimized, compare_optimizations, BitsetDomain
export several_methods, show_CSP

# --- Definición de la estructura básica de un CSP ---

"""
    Constraint

Representa una restricción en un CSP.
- `vars`: Vector de símbolos que representan las variables involucradas.
- `check_func`: Una función que toma un diccionario (asignación parcial o completa)
                y devuelve true si la restricción se cumple para los valores
                asignados de las variables en `vars`. Debe manejar asignaciones parciales.
- `func_string`: Una representación en cadena de la función para facilitar la visualización.
"""
struct Constraint
    vars::Vector{Symbol}
    check_func::Function
    func_string::String
end

"""
    CSP

Representa un Problema de Satisfacción de Restricciones.
- `variables`: Vector de símbolos que representan las variables.
- `domains`: Diccionario que mapea cada variable a un vector de sus posibles valores.
- `constraints`: Vector de objetos Constraint.
"""
struct CSP
    variables::Vector{Symbol}
    domains::Dict{Symbol, Vector{Any}}
    constraints::Vector{Constraint}
end

# --- Funciones auxiliares para verificar consistencia ---

"""
    is_consistent(assignment::Dict{Symbol, Any}, constraint::Constraint)

Verifica si una restricción específica es consistente con una asignación dada.
Si la asignación no contiene todas las variables de la restricción, devuelve true
(asume que las restricciones parciales son consistentes hasta que se demuestre lo contrario).
"""
function is_consistent(assignment::Dict{Symbol, Any}, constraint::Constraint)
    # Solo verificar la restricción si todas sus variables están en la asignación
    if all(v in keys(assignment) for v in constraint.vars)
        return constraint.check_func(assignment)
    end
    return true # Es consistente si hay variables involucradas sin asignación todavía
end

"""
    is_full_assignment_consistent(assignment::Dict{Symbol, Any}, csp::CSP)

Verifica si una asignación COMPLETA satisface TODAS las restricciones del CSP.
Se utiliza principalmente en el método de Fuerza Bruta.
"""
function is_full_assignment_consistent(assignment::Dict{Symbol, Any}, csp::CSP)
    # Asegurarse de que sea una asignación completa
    if length(assignment) != length(csp.variables)
        return false # No es una asignación completa
    end
    # Verificar cada restricción
    for C in csp.constraints
        if !all(v in keys(assignment) for v in C.vars)
            println("Error: Asignación completa no contiene todas las variables de una restricción.")
            return false
        end
        if !C.check_func(assignment)
            return false # Falla la restricción
        end
    end
    return true # Todas las restricciones satisfechas
end

"""
    is_partially_consistent(assignment::Dict{Symbol, Any}, csp::CSP)

Verifica si una asignación PARCIAL es consistente con TODAS las restricciones
que involucran ÚNICAMENTE variables ya asignadas.
Se utiliza en algoritmos de búsqueda como Backtracking.
"""
function is_partially_consistent(assignment::Dict{Symbol, Any}, csp::CSP)
    for C in csp.constraints
        # Verificar solo las restricciones cuyas variables están todas en la asignación parcial
        if all(v in keys(assignment) for v in C.vars)
            if !C.check_func(assignment)
                return false # Falla la restricción con la asignación parcial actual
            end
        end
    end
    return true # Consistente con las restricciones relevantes hasta ahora
end

# --- Métodos de Resolución Básicos ---

# Implementación del método de Fuerza Bruta
function solve_brute_force(csp::CSP)
    # Generar todas las posibles asignaciones completas (producto cartesiano de dominios)
    Ds = [csp.domains[v] for v in csp.variables]
    
    # Usar Iterators.product para generar todas las combinaciones
    for values_tuple in Iterators.product(Ds...)
        assignment = Dict{Symbol, Any}(zip(csp.variables, values_tuple))
        # Comprobar si la asignación es globalmente consistente
        if is_full_assignment_consistent(assignment, csp)
            return assignment
        end
    end

    # Si no se encuentra ninguna solución después de verificar todas
    return nothing
end


# Implementación del método de Backtracking
function BT(assignment::Dict{Symbol, Any}, unassigned_vars::Vector{Symbol}, csp::CSP)
    # Si la asignación es completa (todas las variables asignadas)
    if isempty(unassigned_vars)
        return assignment # Devolver la asignación (es una solución)
    end

    # Seleccionar una variable sin asignar (la primera para simplicidad)
    x = unassigned_vars[1]
    remaining_vars = unassigned_vars[2:end]

    # Para cada valor en el dominio de x
    for v in csp.domains[x]
        # Crear nueva asignación con x = v
        new_assignment = copy(assignment)
        new_assignment[x] = v

        # Verificar si la nueva asignación parcial es consistente
        if is_partially_consistent(new_assignment, csp)
            # Si es consistente, hacer la llamada recursiva
            result = BT(new_assignment, remaining_vars, csp)

            # Si la llamada recursiva encontró una solución
            if result !== nothing
                return result # Devolver la solución
            end
        end
        # Si no es consistente o la llamada recursiva falló, probar el siguiente valor
    end

    # Si no se encuentra ningún valor para x que conduzca a una solución
    return nothing # Devolver nothing (fallo)
end

# Función wrapper para iniciar el backtracking
function solve_BT(csp::CSP)
    initial_assignment = Dict{Symbol, Any}()
    initial_unassigned_vars = copy(csp.variables)
    return BT(initial_assignment, initial_unassigned_vars, csp)
end

# --- Algoritmo AC-3 (Arc Consistency) ---

"""
    Arc

Representa un arco en AC-3.
Un arco (Xi, Xj) representa la restricción que conecta las variables Xi y Xj.
"""
struct Arc
    xi::Symbol  # Variable fuente
    xj::Symbol  # Variable destino
    constraint::Constraint  # La restricción que conecta xi y xj
end

"""
    revise!(domains::Dict{Symbol, Vector{Any}}, arc::Arc)

Función REVISE del algoritmo AC-3. Dado el arco (Xi, Xj) y su restricción, elimina 
de Dom(Xi) los valores que no tienen soporte en Dom(Xj). Modifica el dominio de Xi.
Devuekve true si el dominio de Xi fue modificado, false en caso contrario.
"""
function revise!(Dom::Dict{Symbol, Vector{Any}}, arc::Arc)
    revised = false
    xi, xj = arc.xi, arc.xj
    constraint = arc.constraint
    
    # Verificar que ambas variables existan en los dominios
    if !haskey(Dom, xi) || !haskey(Dom, xj)
        error("Variables $xi o $xj no encontradas en dominios. Variables disponibles: $(keys(Dom))")
    end
    
    # Crear una copia del dominio de xi para iterar (evitar modificación durante iteración)
    xi_values_to_check = copy(Dom[xi])
    
    for xi_value in xi_values_to_check
        # Verificar si existe al menos un valor en Dom(Xj) que satisfaga la restricción
        has_support = false
        
        for xj_value in Dom[xj]
            # Crear asignación temporal para verificar la restricción
            temp_assignment = Dict{Symbol, Any}(xi => xi_value, xj => xj_value)
            
            # Para restricciones n-arias, necesitamos verificar solo las variables del arco
            # pero la función de restricción puede necesitar otras variables
            try
                # Si la restricción se satisface, xi_value tiene soporte
                if constraint.check_func(temp_assignment)
                    has_support = true
                    break
                end
            catch e
                # Si la restricción requiere más variables de las que tenemos en el arco,
                # asumimos que tiene soporte (será verificado cuando todas las variables estén asignadas)
                if isa(e, KeyError)
                    has_support = true
                    break
                else
                    rethrow(e)
                end
            end
        end
        
        # Si xi_value no tiene soporte en Dom(Xj), eliminarlo de Dom(Xi)
        if !has_support
            filter!(x -> x != xi_value, Dom[xi])
            revised = true
        end
    end
    
    return revised
end

"""
    arcs_from(csp::CSP)

Genera todos los arcos a partir de las restricciones del CSP.
Solo crea arcos para restricciones binarias. Las restricciones n-arias (n > 2)
son manejadas de manera especial.
"""
function arcs_from(csp::CSP)
    arcs = Arc[]
    
    for C in csp.constraints
        # Solo procesamos restricciones binarias para AC-3
        if length(C.vars) == 2
            xi = C.vars[1]
            xj = C.vars[2]
            
            # Verificar que las variables existen en el CSP
            if xi in csp.variables && xj in csp.variables
                # Crear arcos bidireccionales
                push!(arcs, Arc(xi, xj, C))
                push!(arcs, Arc(xj, xi, C))
            end
        end
        # Para restricciones n-arias (n > 2), podríamos crear arcos entre todos los pares,
        # pero esto es más complejo y puede no ser eficiente
        # Por ahora, AC-3 solo maneja restricciones binarias eficientemente
    end
    
    return arcs
end

"""
    neighbor_arcs(xi::Symbol, xj::Symbol, all_arcs::Vector{Arc})

Obtiene todos los arcos (Xk, Xi) donde Xk ≠ Xi y Xk ≠ Xj.
Esto se usa en AC-3 para agregar arcos a la cola cuando el dominio de Xi cambia.
"""
function neighbor_arcs(xi::Symbol, xj::Symbol, all_arcs::Vector{Arc})
    return [arc for arc in all_arcs if arc.xi == xi && arc.xj != xj]
end

"""
    AC3!(csp::CSP)

Implementa el algoritmo AC-3 (Arc Consistency 3) modificado para manejar restricciones n-arias.
Modifica in-place los dominios del CSP para hacerlos arc-consistent.
Retorna true si el CSP sigue siendo consistente, false si algún dominio queda vacío.

NOTA: Esta implementación de AC-3 solo procesa restricciones binarias de manera eficiente.
Las restricciones n-arias requieren algoritmos más sofisticados como GAC (Generalized Arc Consistency).
"""
function AC3!(csp::CSP)
    # Crear una copia profunda de los dominios para no modificar el CSP original
    D = Dict{Symbol, Vector{Any}}()
    for (x, Dx) in csp.domains
        D[x] = copy(Dx)
    end
    
    # Obtener todos los arcos del CSP (solo restricciones binarias)
    all_arcs = arcs_from(csp)
    
    if isempty(all_arcs)
        println("Advertencia: No se encontraron restricciones binarias para AC-3. El algoritmo no reducirá dominios.")
        return true, D
    end
    
    # Inicializar la cola con todos los arcos
    queue = copy(all_arcs)
    
    while !isempty(queue)
        # Tomar un arco de la cola
        arc = popfirst!(queue)
        
        # Aplicar REVISE al arco
        if revise!(D, arc)
            # Si el dominio de Xi queda vacío, no hay solución
            if isempty(D[arc.xi])
                return false, D
            end
            
            # Agregar a la cola todos los arcos (Xk, Xi) donde Xk ≠ Xi, Xk ≠ Xj
            n_arcs = neighbor_arcs(arc.xi, arc.xj, all_arcs)
            for n_arc in n_arcs
                if !(n_arc in queue)  # Evitar duplicados
                    push!(queue, n_arc)
                end
            end
        end
    end
    
    return true, D
end

"""
    solve_BT_with_AC3(csp::CSP)

Resuelve un CSP aplicando primero AC-3 para reducir los dominios,
y luego usando backtracking en el CSP simplificado.
"""
function solve_BT_with_AC3(csp::CSP)
    # Aplicar AC-3 primero
    println("Aplicando AC-3 para reducir dominios...")
    
    # Mostrar dominios originales
    println("Dominios originales:")
    for (x, Dx) in csp.domains
        println("  $x: $(length(Dx)) valores")
    end
    
    consistent, reduced_domains = AC3!(csp)
    
    if !consistent
        println("AC-3 detectó inconsistencia - no hay solución posible.")
        return nothing
    end
    
    # Mostrar dominios reducidos
    println("Dominios después de AC-3:")
    total_original = prod(length(D) for D in values(csp.domains))
    total_reduced = prod(length(D) for D in values(reduced_domains))
    
    for (x, Dx) in reduced_domains
        original_size = length(csp.domains[x])
        new_size = length(Dx)
        reduction = original_size - new_size
        println("  $x: $new_size valores (reducido en $reduction)")
    end
    
    ratio = total_reduced / total_original
    println("Reducción total: $total_original → $total_reduced valores ($(total_original - total_reduced) eliminados, ratio=$(round(ratio, digits=4)))")
    
    # Crear un nuevo CSP con los dominios reducidos
    reduced_csp = CSP(csp.variables, reduced_domains, csp.constraints)
    
    # Aplicar backtracking al CSP reducido
    println("Aplicando backtracking al CSP reducido...")
    return solve_BT(reduced_csp)
end


# --- Extensión para GAC (Generalized Arc Consistency) ---

"""
    GAC_arc

Representa un arco generalizado en GAC.
Un arco (Xi, C) representa una variable Xi y una restricción C que la involucra.
"""
struct GAC_arc
    xi::Symbol              # Variable a revisar
    constraint::Constraint  # Restricción que involucra xi
    other_vars::Vector{Symbol}  # Otras variables en la restricción (sin xi)
end

"""
    GAC_arcs_from(csp::CSP)

Genera todos los arcos GAC a partir de las restricciones del CSP.
Para cada restricción C que involucra variables {X1, X2, ..., Xn},
crea n arcos: (X1, C), (X2, C), ..., (Xn, C).
"""
function GAC_arcs_from(csp::CSP)
    arcs = GAC_arc[]
    
    for C in csp.constraints
        # Para cada variable en la restricción, crear un arco GAC
        for xi in C.vars
            # Las "otras variables" son todas las de la restricción excepto xi
            other_vars = filter(v -> v != xi, C.vars)
            
            # Verificar que xi está en las variables del CSP
            if xi in csp.variables
                push!(arcs, GAC_arc(xi, C, other_vars))
            end
        end
    end
    
    return arcs
end

"""
    revise_GAC!(domains::Dict{Symbol, Vector{Any}}, arc::GAC_arc)

Función REVISE generalizada para GAC.
Elimina de Dom(Xi) los valores que no tienen soporte en la restricción.
Un valor v de Xi tiene soporte si existe al menos una combinación de valores
para las otras variables que satisface la restricción.
"""
function revise_GAC!(D::Dict{Symbol, Vector{Any}}, arc::GAC_arc)
    revised = false
    xi = arc.xi
    C = arc.constraint
    other_vars = arc.other_vars
    
    # Verificar que todas las variables existan en los dominios
    if !haskey(D, xi)
        error("Variable $xi no encontrada en dominios")
    end
    
    for other_var in other_vars
        if !haskey(D, other_var)
            error("Variable $other_var no encontrada en dominios")
        end
    end
    
    # Crear una copia del dominio de xi para iterar
    xi_values_to_check = copy(D[xi])
    
    for xi_value in xi_values_to_check
        has_support = false
        
        # Si no hay otras variables, solo verificar la restricción unaria
        if isempty(other_vars)
            temp_assignment = Dict{Symbol, Any}(xi => xi_value)
            if C.check_func(temp_assignment)
                has_support = true
            end
        else
            # Generar todas las combinaciones posibles de las otras variables
            other_domains = [D[x] for x in other_vars]
            
            # Usar Iterators.product para generar todas las combinaciones
            for values_tuple in Iterators.product(other_domains...)
                # Crear asignación temporal
                temp_assignment = Dict{Symbol, Any}(xi => xi_value)
                for (var, val) in zip(other_vars, values_tuple)
                    temp_assignment[var] = val
                end
                
                # Verificar si la restricción se satisface
                try
                    if C.check_func(temp_assignment)
                        has_support = true
                        break
                    end
                catch e
                    # Si hay un error (ej: variable faltante), continuar
                    if !isa(e, KeyError)
                        rethrow(e)
                    end
                end
            end
        end
        
        # Si xi_value no tiene soporte, eliminarlo
        if !has_support
            filter!(x -> x != xi_value, D[xi])
            revised = true
        end
    end
    
    return revised
end

"""
    get_neighbor_GAC_arcs(xi::Symbol, affected_constraints::Vector{Constraint}, all_arcs::Vector{GAC_arc})

Obtiene todos los arcos GAC (Xk, C) donde:
- Xk ≠ Xi
- C es una restricción que involucra tanto Xi como Xk
Esto se usa para agregar arcos a la cola cuando el dominio de Xi cambia.
"""
function get_neighbor_GAC_arcs(xi::Symbol, all_arcs::Vector{GAC_arc})
    neighbor_arcs = GAC_arc[]
    
    for arc in all_arcs
        # Si el arco no es sobre xi, pero la restricción involucra xi
        if arc.xi != xi && xi in arc.constraint.vars
            push!(neighbor_arcs, arc)
        end
    end
    
    return neighbor_arcs
end

"""
    GAC!(csp::CSP)

Implementa el algoritmo GAC (Generalized Arc Consistency).
Maneja restricciones de cualquier aridad (unarias, binarias, n-arias).
"""
function GAC!(csp::CSP)
    # Crear una copia profunda de los dominios
    D = Dict{Symbol, Vector{Any}}()
    for (x, Dx) in csp.domains
        D[x] = copy(Dx)
    end
    
    # Obtener todos los arcos GAC
    all_arcs = GAC_arcs_from(csp)
    
    if isempty(all_arcs)
        println("Advertencia: No se encontraron restricciones para GAC.")
        return true, D
    end
    
    # Inicializar la cola con todos los arcos
    queue = copy(all_arcs)
    
    iterations = 0
    while !isempty(queue)
        iterations += 1
        
        # Tomar un arco de la cola
        arc = popfirst!(queue)
        
        # Aplicar REVISE al arco
        if revise_GAC!(D, arc)
            # Si el dominio de Xi queda vacío, no hay solución
            if isempty(D[arc.xi])
                println("GAC detectó inconsistencia después de $iterations iteraciones")
                return false, D
            end
            
            # Agregar a la cola todos los arcos vecinos
            n_arcs = get_neighbor_GAC_arcs(arc.xi, all_arcs)
            for n_arc in n_arcs
                if !(n_arc in queue)  # Evitar duplicados
                    push!(queue, n_arc)
                end
            end
        end
    end
    
    println("GAC completado en $iterations iteraciones")
    return true, D
end

"""
    solve_BT_with_GAC(csp::CSP)

Resuelve un CSP aplicando primero GAC para reducir los dominios,
y luego usando backtracking en el CSP simplificado.
"""
function solve_BT_with_GAC(csp::CSP)
    # Aplicar GAC primero
    println("Aplicando GAC para reducir dominios...")
    
    # Mostrar dominios originales
    println("Dominios originales:")
    for (x, Dx) in csp.domains
        println("  $x: $(length(Dx)) valores")
    end
    
    consistent, reduced_domains = GAC!(csp)
    
    if !consistent
        println("GAC detectó inconsistencia - no hay solución posible.")
        return nothing
    end
    
    # Mostrar dominios reducidos
    println("Dominios después de GAC:")
    total_original = prod(length(Dx) for Dx in values(csp.domains))
    total_reduced = prod(length(Dx) for Dx in values(reduced_domains))
    
    for (x, Dx) in reduced_domains
        original_size = length(csp.domains[x])
        new_size = length(Dx)
        reduction = original_size - new_size
        println("  $x: $new_size valores (reducido en $reduction)")
    end

    ratio = total_reduced / total_original    
    println("Reducción total: $total_original → $total_reduced valores ($(total_original - total_reduced) eliminados, , ratio=$(round(ratio, digits=4)))")
    
    # Crear un nuevo CSP con los dominios reducidos
    reduced_csp = CSP(csp.variables, reduced_domains, csp.constraints)
    
    # Aplicar backtracking al CSP reducido
    println("Aplicando backtracking al CSP reducido...")
    return solve_BT(reduced_csp)
end

# --- Función de comparación entre AC-3 y GAC ---

"""
    compare_AC3_vs_GAC(csp::CSP)

Compara la efectividad de AC-3 vs GAC en un CSP dado.
"""
function compare_AC3_vs_GAC(csp::CSP)
    println("=== COMPARACIÓN AC-3 vs GAC ===")
    println("===============================")
    
    # Probar AC-3
    println("\n--- Ejecutando AC-3 ---")
    @time begin
        consistent_AC3, domains_AC3 = AC3!(csp)
        total_AC3 = consistent_AC3 ? prod(length(Dx) for Dx in values(domains_AC3)) : 0
    end
    
    if consistent_AC3
        println("AC-3: Dominios reducidos a $total_AC3 valores totales")
    else
        println("AC-3: Detectó inconsistencia")
    end
    
    # Probar GAC
    println("\n--- Ejecutando GAC ---")
    @time begin
        consistent_GAC, domains_GAC = GAC!(csp)
        total_GAC = consistent_GAC ? prod(length(Dx) for Dx in values(domains_GAC)) : 0
    end
    
    if consistent_GAC
        println("GAC: Dominios reducidos a $total_GAC valores totales")
    else
        println("GAC: Detectó inconsistencia")
    end
    
    # Comparación
    if consistent_AC3 && consistent_GAC
        original_total = prod(length(Dx) for Dx in values(csp.domains))
        reduction_AC3 = original_total - total_AC3
        reduction_GAC = original_total - total_GAC
        ratio_AC3 = reduction_AC3 / original_total
        ratio_GAC = reduction_GAC / original_total
        println("\n--- Resumen ---")
        println("  Dominios originales: $original_total valores")
        println("  AC-3 eliminó: $reduction_AC3 valores (ratio = $(round(ratio_AC3, digits=4)))")
        println("  GAC eliminó: $reduction_GAC valores (ratio = $(round(ratio_GAC, digits=4)))")
        println("  GAC es $(reduction_GAC - reduction_AC3) valores más efectivo que AC-3")
        
        if reduction_GAC > reduction_AC3
            println("✓ GAC fue más efectivo para reducir dominios")
        elseif reduction_GAC == reduction_AC3
            println("= Ambos algoritmos fueron igualmente efectivos")
        else
            println("✓ AC-3 fue más efectivo (inusual)")
        end
    end
end

# --- EXTENSIONES OPTIMIZADAS PARA EL CSP_MODULE ---

# --- Estructuras de datos optimizadas ---

"""
    BitsetDomain

Representación optimizada de dominios usando bitsets para enteros en rangos pequeños.
Útil cuando los dominios son enteros consecutivos (ej: 1:n).
"""
struct BitsetDomain
    bitset::BitSet
    min_val::Int
    max_val::Int
end

# Constructor para Vector{Int}
function BitsetDomain(values::Vector{Int})
    if isempty(values)
        throw(ArgumentError("No se puede crear BitsetDomain con vector vacío"))
    end
    min_val = minimum(values)
    max_val = maximum(values)
    bitset = BitSet(values)
    return BitsetDomain(bitset, min_val, max_val)
end

# Constructor para Vector{Any} - NUEVO
function BitsetDomain(values::Vector{Any})
    if isempty(values)
        throw(ArgumentError("No se puede crear BitsetDomain con vector vacío"))
    end
    
    # Verificar que todos los valores sean enteros
    int_values = Int[]
    for val in values
        if isa(val, Int)
            push!(int_values, val)
        elseif isa(val, Number) && isinteger(val)
            push!(int_values, Int(val))
        else
            throw(ArgumentError("BitsetDomain solo acepta valores enteros, encontrado: $(typeof(val))"))
        end
    end
    
    min_val = minimum(int_values)
    max_val = maximum(int_values)
    bitset = BitSet(int_values)
    return BitsetDomain(bitset, min_val, max_val)
end

# Constructor para rangos
function BitsetDomain(range::UnitRange{Int})
    values = collect(range)
    return BitsetDomain(values)
end

# Funciones auxiliares mejoradas
function Base.length(bd::BitsetDomain)
    return length(bd.bitset)
end

function Base.isempty(bd::BitsetDomain)
    return isempty(bd.bitset)
end

function Base.in(value::Int, bd::BitsetDomain)
    return value in bd.bitset
end

function Base.in(value::Any, bd::BitsetDomain)
    if isa(value, Int)
        return value in bd.bitset
    elseif isa(value, Number) && isinteger(value)
        return Int(value) in bd.bitset
    else
        return false
    end
end

function remove_value!(bd::BitsetDomain, value::Int)
    delete!(bd.bitset, value)
end

function remove_value!(bd::BitsetDomain, value::Any)
    if isa(value, Int)
        delete!(bd.bitset, value)
    elseif isa(value, Number) && isinteger(value)
        delete!(bd.bitset, Int(value))
    end
end

function to_vector(bd::BitsetDomain)
    return collect(bd.bitset)
end

# Función auxiliar para verificar si un dominio es compatible con bitsets
function is_bitset_compatible(domain::Vector{Any})
    if isempty(domain)
        return false
    end
    
    # Verificar que todos sean enteros
    for val in domain
        if !(isa(val, Int) || (isa(val, Number) && isinteger(val)))
            return false
        end
    end
    
    # Verificar que estén en un rango razonable para bitsets
    int_values = [isa(val, Int) ? val : Int(val) for val in domain]
    min_val = minimum(int_values)
    max_val = maximum(int_values)
    
    # Bitsets son eficientes para rangos no muy grandes
    return (max_val - min_val) <= 10000 && min_val >= -1000 && max_val <= 10000
end

# --- Versiones optimizadas de AC3 y GAC ---

"""
    AC3_optimized!(csp::CSP; use_bitsets::Bool=false, order_by_domain_size::Bool=true)

Versión optimizada de AC-3 con las siguientes mejoras:
- Ordenamiento de arcos por tamaño de dominio (opcional)
- Uso de bitsets para dominios pequeños (opcional)
- Evita duplicados en la cola de manera más eficiente
- Contadores de rendimiento para análisis educativo
"""
function AC3_optimized!(csp::CSP; use_bitsets::Bool=false, order_by_domain_size::Bool=true)
    # Contadores para análisis de rendimiento
    stats = Dict(
        :revisions => 0,
        :domain_reductions => 0,
        :queue_operations => 0,
        :duplicate_arcs_avoided => 0
    )
    
    # Preparar dominios (bitsets o vectores normales)
    D = Dict{Symbol, Any}()
    
    if use_bitsets
        println("Verificando compatibilidad con bitsets...")
        
        # Verificar cuáles dominios son compatibles con bitsets
        compatible_vars = Symbol[]
        for (x, Dx) in csp.domains
            if is_bitset_compatible(Dx)
                push!(compatible_vars, x)
            end
        end
        
        if !isempty(compatible_vars)
            println("Variables compatibles con bitsets: $compatible_vars")
            
            for (x, Dx) in csp.domains
                if x in compatible_vars
                    try
                        D[x] = BitsetDomain(Dx)
                        println("  $x: convertido a BitsetDomain")
                    catch e
                        println("  $x: error al convertir, usando vector normal: $e")
                        D[x] = copy(Dx)
                    end
                else
                    D[x] = copy(Dx)
                end
            end
        else
            println("Ningún dominio es compatible con bitsets - usando vectores normales")
            use_bitsets = false  # Desactivar bitsets
            for (x, Dx) in csp.domains
                D[x] = copy(Dx)
            end
        end
    else
        for (x, Dx) in csp.domains
            D[x] = copy(Dx)
        end
    end
    
    # Obtener todos los arcos
    all_arcs = arcs_from(csp)
    
    if isempty(all_arcs)
        println("Advertencia: No se encontraron restricciones binarias para AC-3.")
        return true, D, stats
    end
    
    # Función para obtener el tamaño de dominio
    domain_size(x) = length(D[x])
    
    # Ordenar arcos iniciales por tamaño de dominio si está habilitado
    if order_by_domain_size
        sort!(all_arcs, by = arc -> (domain_size(arc.xi) + domain_size(arc.xj)))
        println("Arcos ordenados por tamaño de dominio")
    end
    
    # Usar un Set para evitar duplicados más eficientemente
    queue_set = Set(all_arcs)
    queue = collect(all_arcs)
    
    println("Iniciando AC-3 optimizado con $(length(queue)) arcos...")
    
    while !isempty(queue)
        # Reordenar cola periódicamente por tamaño de dominio
        if order_by_domain_size && length(queue) > 10 && stats[:queue_operations] % 50 == 0
            sort!(queue, by = arc -> domain_size(arc.xi))
        end
        
        # Tomar arco (el primero si está ordenado, será el de menor dominio)
        arc = popfirst!(queue)
        delete!(queue_set, arc)
        stats[:queue_operations] += 1
        
        # Aplicar REVISE optimizado
        revised = false
        if use_bitsets && isa(D[arc.xi], BitsetDomain) && isa(D[arc.xj], BitsetDomain)
            revised = revise_bitset!(D, arc, stats)
        else
            revised = revise_optimized!(D, arc, stats)
        end
        
        if revised
            stats[:domain_reductions] += 1
            
            # Verificar si el dominio quedó vacío
            if (isa(D[arc.xi], BitsetDomain) && isempty(D[arc.xi])) ||
               (!isa(D[arc.xi], BitsetDomain) && isempty(D[arc.xi]))
                println("Dominio vacío detectado para variable $(arc.xi)")
                return false, D, stats
            end
            
            # Agregar arcos vecinos sin duplicados
            n_arcs = neighbor_arcs(arc.xi, arc.xj, all_arcs)
            for n_arc in n_arcs
                if !(n_arc in queue_set)
                    push!(queue, n_arc)
                    push!(queue_set, n_arc)
                else
                    stats[:duplicate_arcs_avoided] += 1
                end
            end
        end
    end
    
    println("AC-3 optimizado completado exitosamente")
    return true, D, stats
end

"""
    revise_optimized!(domains::Dict{Symbol, Any}, arc::Arc, stats::Dict)

Versión optimizada de revise con mejor manejo de memoria y contadores.
"""
function revise_optimized!(D::Dict{Symbol, Any}, arc::Arc, stats::Dict)
    stats[:revisions] += 1
    revised = false
    xi, xj = arc.xi, arc.xj
    C = arc.constraint
    
    # Pre-calcular valores que no cambian en el bucle
    D_xj = D[xj]
    
    # Filtrar en lugar de crear copia y luego filtrar
    original_size = length(D[xi])
    D[xi] = filter(D[xi]) do xi_value
        # Buscar soporte para xi_value
        for xj_value in D_xj
            temp_assignment = Dict{Symbol, Any}(xi => xi_value, xj => xj_value)
            try
                if C.check_func(temp_assignment)
                    return true  # Mantener este valor
                end
            catch e
                if isa(e, KeyError)
                    return true  # Mantener si hay variables faltantes
                else
                    rethrow(e)
                end
            end
        end
        return false  # Eliminar este valor
    end
    
    revised = length(D[xi]) < original_size
    return revised
end

"""
    revise_bitset!(domains::Dict{Symbol, Any}, arc::Arc, stats::Dict)

Versión especializada de revise para dominios con bitsets.
"""
function revise_bitset!(D::Dict{Symbol, Any}, arc::Arc, stats::Dict)
    stats[:revisions] += 1
    revised = false
    xi, xj = arc.xi, arc.xj
    C = arc.constraint
    
    D_xi = D[xi]::BitsetDomain
    D_xj = D[xj]::BitsetDomain
    
    # Iterar sobre una copia de los valores para poder modificar durante iteración
    xi_values = to_vector(D_xi)
    
    for xi_value in xi_values
        has_support = false
        
        # Convertir explícitamente a Vector{Any} si es necesario
        xj_values = to_vector(D_xj)
        
        for xj_value in xj_values
            temp_assignment = Dict{Symbol, Any}(xi => xi_value, xj => xj_value)
            try
                if C.check_func(temp_assignment)
                    has_support = true
                    break
                end
            catch e
                if isa(e, KeyError)
                    has_support = true
                    break
                else
                    println("Error en constraint check: $e")
                    println("Assignment: $temp_assignment")
                    rethrow(e)
                end
            end
        end
        
        if !has_support
            remove_value!(D_xi, xi_value)
            revised = true
        end
    end
    
    return revised
end

"""
    GAC_optimized!(csp::CSP; order_by_domain_size::Bool=true, use_incremental::Bool=true)

Versión optimizada de GAC con mejoras similares a AC3_optimized.
"""
function GAC_optimized!(csp::CSP; order_by_domain_size::Bool=true, use_incremental::Bool=true)
    # Contadores para análisis
    stats = Dict(
        :revisions => 0,
        :domain_reductions => 0,
        :queue_operations => 0,
        :duplicate_arcs_avoided => 0
    )
    
    # Crear copia de dominios
    D = Dict{Symbol, Vector{Any}}()
    for (x, Dx) in csp.domains
        D[x] = copy(Dx)
    end
    
    # Obtener arcos GAC
    all_arcs = GAC_arcs_from(csp)
    
    if isempty(all_arcs)
        println("Advertencia: No se encontraron restricciones para GAC.")
        return true, D, stats
    end
    
    # Función auxiliar para tamaño de dominio
    domain_size(x) = length(D[x])
    
    # Ordenamiento inicial
    if order_by_domain_size
        sort!(all_arcs, by = arc -> domain_size(arc.xi))
        println("Arcos GAC ordenados por tamaño de dominio")
    end
    
    # Cola con control de duplicados
    queue_set = Set(all_arcs)
    queue = collect(all_arcs)
    
    while !isempty(queue)
        # Reordenamiento periódico
        if order_by_domain_size && length(queue) > 10 && stats[:queue_operations] % 30 == 0
            sort!(queue, by = arc -> domain_size(arc.xi))
        end
        
        arc = popfirst!(queue)
        delete!(queue_set, arc)
        stats[:queue_operations] += 1
        
        # REVISE optimizado para GAC
        if revise_GAC_optimized!(D, arc, stats)
            stats[:domain_reductions] += 1
            
            if isempty(D[arc.xi])
                return false, D, stats
            end
            
            # Agregar arcos vecinos
            if use_incremental
                # Solo agregar arcos realmente afectados
                n_arcs = get_affected_GAC_arcs(arc.xi, arc.constraint, all_arcs)
            else
                n_arcs = get_neighbor_GAC_arcs(arc.xi, all_arcs)
            end
            
            for n_arc in n_arcs
                if !(n_arc in queue_set)
                    push!(queue, n_arc)
                    push!(queue_set, n_arc)
                else
                    stats[:duplicate_arcs_avoided] += 1
                end
            end
        end
    end
    
    return true, D, stats
end

"""
    revise_GAC_optimized!(domains::Dict{Symbol, Vector{Any}}, arc::GAC_arc, stats::Dict)

Versión optimizada de revise_GAC con mejor gestión de memoria.
"""
function revise_GAC_optimized!(D::Dict{Symbol, Vector{Any}}, arc::GAC_arc, stats::Dict)
    stats[:revisions] += 1
    xi = arc.xi
    C = arc.constraint
    other_vars = arc.other_vars
    
    # Filtrado in-place más eficiente
    original_size = length(D[xi])
    
    D[xi] = filter(D[xi]) do xi_value
        # Caso unario
        if isempty(other_vars)
            temp_assignment = Dict{Symbol, Any}(xi => xi_value)
            try
                return C.check_func(temp_assignment)
            catch
                return false
            end
        end
        
        # Caso n-ario: buscar soporte
        other_domains = [D[var] for var in other_vars]
        
        # Optimización: si algún dominio está vacío, no hay soporte
        if any(isempty, other_domains)
            return false
        end
        
        # Buscar combinación que dé soporte
        for values_tuple in Iterators.product(other_domains...)
            temp_assignment = Dict{Symbol, Any}(xi => xi_value)
            for (var, val) in zip(other_vars, values_tuple)
                temp_assignment[var] = val
            end
            
            try
                if C.check_func(temp_assignment)
                    return true  # Encontrado soporte
                end
            catch e
                if !isa(e, KeyError)
                    rethrow(e)
                end
            end
        end
        
        return false  # No se encontró soporte
    end
    
    return length(D[xi]) < original_size
end

"""
    get_affected_GAC_arcs(xi::Symbol, changed_constraint::Constraint, all_arcs::Vector{GAC_arc})

Versión más inteligente que solo devuelve arcos realmente afectados por el cambio en xi.
"""
function get_affected_GAC_arcs(xi::Symbol, changed_constraint::Constraint, all_arcs::Vector{GAC_arc})
    affected_arcs = GAC_arc[]
    
    for arc in all_arcs
        # El arco está afectado si:
        # 1. No es sobre la misma variable xi
        # 2. Su restricción involucra xi
        # 3. Su restricción es diferente a la que acabamos de procesar (para evitar redundancia inmediata)
        if arc.xi != xi && 
           xi in arc.constraint.vars && 
           arc.constraint !== changed_constraint
            push!(affected_arcs, arc)
        end
    end
    
    return affected_arcs
end

# --- Funciones de resolución con optimizaciones ---

"""
    solve_BT_with_AC3_optimized(csp::CSP; use_bitsets::Bool=false, order_by_domain_size::Bool=true)

Versión optimizada del solver que combina AC3 optimizado con backtracking.
"""
function solve_BT_with_AC3_optimized(csp::CSP; use_bitsets::Bool=false, order_by_domain_size::Bool=true)
    println("=== RESOLVIENDO CON AC-3 OPTIMIZADO ===")
    
    # Mostrar configuración
    println("Configuración:")
    println("  - Bitsets: $use_bitsets")
    println("  - Ordenamiento por tamaño: $order_by_domain_size")
    
    # Aplicar AC-3 optimizado
    println("\nAplicando AC-3 optimizado...")
    @time begin
        consistent, reduced_domains, stats = AC3_optimized!(csp; 
            use_bitsets=use_bitsets, 
            order_by_domain_size=order_by_domain_size)
    end
    
    # Mostrar estadísticas
    println("\nEstadísticas de AC-3 optimizado:")
    println("  - Revisiones: $(stats[:revisions])")
    println("  - Reducciones de dominio: $(stats[:domain_reductions])")
    println("  - Operaciones de cola: $(stats[:queue_operations])")
    println("  - Arcos duplicados evitados: $(stats[:duplicate_arcs_avoided])")
    
    if !consistent
        println("AC-3 optimizado detectó inconsistencia.")
        return nothing
    end
    
    # Convertir dominios bitset de vuelta a vectores para backtracking
    D = Dict{Symbol, Vector{Any}}()
    for (x, Dx) in reduced_domains
        if isa(Dx, BitsetDomain)
            D[x] = to_vector(Dx)
        else
            D[x] = Dx
        end
    end
    
    # Mostrar reducción
    original_total = prod(length(csp.domains[x]) for x in csp.variables)
    reduced_total = prod(length(D[x]) for x in csp.variables)
    println("\nReducción de espacio de búsqueda:")
    println("  Original: $original_total combinaciones")
    println("  Reducido: $reduced_total combinaciones")
    println("  Factor de reducción: $(round(original_total/reduced_total, digits=2))x")
    
    # Crear CSP reducido y resolver con backtracking
    reduced_csp = CSP(csp.variables, D, csp.constraints)
    
    println("\nAplicando backtracking...")
    @time solution = solve_BT(reduced_csp)
    
    return solution
end

"""
    solve_BT_with_GAC_optimized(csp::CSP; order_by_domain_size::Bool=true, use_incremental::Bool=true)

Versión optimizada del solver que combina GAC optimizado con backtracking.
"""
function solve_BT_with_GAC_optimized(csp::CSP; order_by_domain_size::Bool=true, use_incremental::Bool=true)
    println("=== RESOLVIENDO CON GAC OPTIMIZADO ===")
    
    println("Configuración:")
    println("  - Ordenamiento por tamaño: $order_by_domain_size")
    println("  - Propagación incremental: $use_incremental")
    
    println("\nAplicando GAC optimizado...")
    @time begin
        consistent, reduced_domains, stats = GAC_optimized!(csp; 
            order_by_domain_size=order_by_domain_size,
            use_incremental=use_incremental)
    end
    
    println("\nEstadísticas de GAC optimizado:")
    println("  - Revisiones: $(stats[:revisions])")
    println("  - Reducciones de dominio: $(stats[:domain_reductions])")
    println("  - Operaciones de cola: $(stats[:queue_operations])")
    println("  - Arcos duplicados evitados: $(stats[:duplicate_arcs_avoided])")
    
    if !consistent
        println("GAC optimizado detectó inconsistencia.")
        return nothing
    end
    
    # Mostrar reducción
    original_total = prod(length(csp.domains[var]) for var in csp.variables)
    reduced_total = prod(length(reduced_domains[var]) for var in csp.variables)
    println("\nReducción de espacio de búsqueda:")
    println("  Original: $original_total combinaciones")
    println("  Reducido: $reduced_total combinaciones")
    println("  Factor de reducción: $(round(original_total/reduced_total, digits=2))x")
    
    reduced_csp = CSP(csp.variables, reduced_domains, csp.constraints)
    
    println("\nAplicando backtracking...")
    @time solution = solve_BT(reduced_csp)
    
    return solution
end

"""
    compare_optimizations(csp::CSP)

Compara todas las versiones (originales vs optimizadas) en un CSP dado.
"""
function compare_optimizations(csp::CSP)
    println("=== COMPARACIÓN COMPLETA DE OPTIMIZACIONES ===")
    println("==============================================")
    
    results = Dict()
    
    # AC-3 original
    println("\n1. AC-3 Original:")
    @time begin
        consistent_orig, domains_orig = AC3!(csp)
        results["AC3_original"] = consistent_orig ? prod(length(d) for d in values(domains_orig)) : 0
    end
    
    # AC-3 optimizado (sin bitsets)
    println("\n2. AC-3 Optimizado (sin bitsets):")
    @time begin
        consistent_opt1, domains_opt1, stats1 = AC3_optimized!(csp; use_bitsets=false)
        results["AC3_optimized"] = consistent_opt1 ? prod(length(d) for d in values(domains_opt1)) : 0
    end
    println("   Revisiones: $(stats1[:revisions]), Duplicados evitados: $(stats1[:duplicate_arcs_avoided])")
    
    # AC-3 optimizado con bitsets (si es posible)
    can_use_bitsets = all(
        all(isa(val, Int) && 0 <= val <= 1000 for val in domain) 
        for domain in values(csp.domains)
    )
    
    if can_use_bitsets
        println("\n3. AC-3 Optimizado (con bitsets):")
        @time begin
            consistent_opt2, domains_opt2, stats2 = AC3_optimized!(csp; use_bitsets=true)
            final_domains = Dict{Symbol, Vector{Any}}()
            for (var, domain) in domains_opt2
                if isa(domain, BitsetDomain)
                    final_domains[var] = to_vector(domain)
                else
                    final_domains[var] = domain
                end
            end
            results["AC3_bitsets"] = consistent_opt2 ? prod(length(d) for d in values(final_domains)) : 0
        end
        println("   Revisiones: $(stats2[:revisions]), Duplicados evitados: $(stats2[:duplicate_arcs_avoided])")
    end
    
    # GAC original
    println("\n4. GAC Original:")
    @time begin
        consistent_gac_orig, domains_gac_orig = GAC!(csp)
        results["GAC_original"] = consistent_gac_orig ? prod(length(d) for d in values(domains_gac_orig)) : 0
    end
    
    # GAC optimizado
    println("\n5. GAC Optimizado:")
    @time begin
        consistent_gac_opt, domains_gac_opt, stats_gac = GAC_optimized!(csp)
        results["GAC_optimized"] = consistent_gac_opt ? prod(length(d) for d in values(domains_gac_opt)) : 0
    end
    println("   Revisiones: $(stats_gac[:revisions]), Duplicados evitados: $(stats_gac[:duplicate_arcs_avoided])")
    
    # Resumen de resultados
    println("\n--- RESUMEN ---")
    original_space = prod(length(domain) for domain in values(csp.domains))
    println(" Espacio de búsqueda original: $original_space")
    
    for (method, final_space) in results
        if final_space > 0
            reduction = original_space - final_space
            ratio = reduction / original_space
            println(" $method: $final_space (reducción: $(round(ratio*100, digits=2))%)")
        else
            println(" $method: INCONSISTENTE")
        end
    end
end

# Funciones auxiliares: 
#   Permite realizar una comparativa de diversos algoritmos sobre un mismo problema
#   En general, evitamos la resolución por fuerza bruta en problemas complejos, 
#   ya que es extremadamente lenta, salvo para problemas muy pequeños

function several_methods(prob_CSP, verifica)
    methods = [
        #(solve_brute_force,"Fuerza Bruta"), 
        (solve_BT,"Backtracking"), 
        (solve_BT_with_AC3, "AC3"), 
        (solve_BT_with_GAC, "GAC")
        ]
    println("=== RESOLUCIÓN DE CSP POR DIVERSOS MÉTODOS ===")
    println("==============================================")
    println("")
    show_CSP(prob_CSP)

    for (method, name) in methods
        println("\nResolución por $name")
        println("---------------------------")
        @time sol = method(prob_CSP)
        
        # Verificar si se encontró solución antes de validar consistencia
        if sol === nothing
            success = false
            println("\n   ¿Ha encontrado solución? $success (No se encontró solución)\n")
        else
            success = is_full_assignment_consistent(sol, prob_CSP)
            println("\n   ¿Ha encontrado solución? $success\n")
        end
        
        verifica(sol)
    end
    println()
    # Comparación de AC3 vs GAC para el CSP
    compare_AC3_vs_GAC(prob_CSP)
    # Comparación de las versiones originales vs optimizadas
    # println()
    # compare_optimizations(prob_CSP)
end

# --- Funciones de visualización ---

function show_CSP(csp::CSP)
    println("=== CSP ===")
    println(" Variables: ", csp.variables)
    println(" Dominios:")
    for (var, dom) in csp.domains
        println("   $var => $dom")
    end
    println(" Restricciones:")
    for (i, constr) in enumerate(csp.constraints)
        println("   R$i: Variables : $(constr.vars), Función : $(constr.func_string)")
    end
    println()
end

end #module
