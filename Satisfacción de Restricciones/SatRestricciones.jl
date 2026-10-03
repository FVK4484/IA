include("CSP.jl")
using .CSP_Module

# ======================== Ejemplos =========================

# ---------------------
# Ejemplo 1: CSP simple
# ---------------------

#=
Vamos a comenzar por un ejemplo muy simple, pero solo para ver de qué forma se usa la librería para modelar problemas genéricos.

Debe tenerse presente que este problema es tan simple que los diversos métodos y optimizaciones no son representativos. Es necesario que el problema aumente en complejidad para evaluar adecuadamente las diferencias entre los enfoques.

    Variables: X, Y, Z 
    Dominios: X ∈ {1,2,3}, Y ∈ {1,2,3}, Z ∈ {1,2,3}
    Restricciones: X ≠ Y, Y < Z, X + Y = Z

=#

# Definimos una función para crearlo (y reutilizarlo con los distintos métodos)

function create_simple_csp()    
    variables = [:X, :Y, :Z]                # Las variables son símbolos
    dominios = Dict(                        # Los dominios son colecciones de 
        :X => [1, 2, 3],                    #   datos para cada variable
        :Y => [1, 2, 3], 
        :Z => [1, 2, 3]
    )
    
    restricciones = [                       # Las restricciones indican qué 
                                            # variables intervienen y la función
                                            # que deben satisfacer
        Constraint([:X, :Y], a -> a[:X] != a[:Y], "X ≠ Y"),             # X ≠ Y (binaria)
        Constraint([:Y, :Z], a -> a[:Y] < a[:Z], "Y < Z"),              # Y < Z (binaria)
        Constraint([:X, :Y, :Z], a -> a[:X] + a[:Y] == a[:Z], "X + Y = Z")  # X + Y = Z (ternaria)
    ]   # Observa que la función (aquí, anónimas) actúa sobre las asignaciones
    
    return CSP(variables, dominios, restricciones)  # Devuelve el CSP
end

# Función de verificación para la solución del CSP simple

function verifica_simple(sol)
    if sol !== nothing
        println("Solución encontrada: $(sol)")
        x, y, z = sol[:X], sol[:Y], sol[:Z]
        println("Verificación: X=$x, Y=$y, Z=$z")
        println("  X ≠ Y:     $(x != y)")
        println("  Y < Z:     $(y < z)")  
        println("  X + Y = Z: $(x + y == z)")
    else
        println("No se encontró solución para el CSP simple.")
    end
end

# Creación del problema
simple_csp = create_simple_csp()
show_CSP(simple_csp)

# Resolución del problema simple con distintos métodos

@time sol = solve_brute_force(simple_csp)
several_methods(simple_csp, verifica_simple)

# ------------------------------------------------------
# Ejemplo 2: Problema Criptoaritmético 'send+more=money'
# ------------------------------------------------------

#=
Este problema es un clásico de los CSP: el problema de la suma de palabras. Concretamente, se trata de encontrar una asignación de dígitos a letras que satisfaga la ecuación:

       send + more = money

Aunque su representación como CSP más directa es:

      10^3 * s + 10^2 * e + 10^1 * n + 10^0 * d
    + 10^3 * m + 10^2 * o + 10^1 * r + 10^0 * e 
    = 10^4 * m + 10^3 * o + 10^2 * n + 10^2 * e + 10^0 * y

esta opción solo añade una gran restricción, en la que intervienen todas las variables, y obliga a una búsqueda exhaustiva muy poco eficiente. En su lugar, se suele utilizar una representación más adecuada del problema:

              s  e  n  d 
            + m  o  r  e
          ---------------
           m  o  n  e  y

donde se deben tener en cuenta los acarreos posibles por cada columna (añade 3 variables de acarreo, pero también mayor número de restricciones más pequeñas que ayudan a podar la búsqueda con métodos más eficientes)
=#

function create_send_more_money_csp()
    variables = [:s, :e, :n, :d, :m, :o, :r, :y, :c1, :c2, :c3]
    domains = Dict(
        :s => collect(1:9),  # s no puede ser 0 (primer dígito)
        :m => [1],           # m debe ser 1 (resultado de acarreo máximo)
        :c1 => [0, 1],       # acarreo binario
        :c2 => [0, 1],       # acarreo binario
        :c3 => [0, 1],       # acarreo binario
        :d => collect(0:9),
        :e => collect(0:9),
        :n => collect(0:9),
        :o => collect(0:9),
        :r => collect(0:9),
        :y => collect(0:9)
    )

    # Creamos la lista de restricciones poco a poco:

    constraints = Constraint[]

    # Restricción 1: Todas las variables {s, e, n, d, m, o, r, y} son distintas 
    #    (muchas restricciones binarias)
    letter_vars = [:s, :e, :n, :d, :m, :o, :r, :y]
    for i in (1):length(letter_vars)
        for j in (i+1):length(letter_vars)
            v1 = letter_vars[i]
            v2 = letter_vars[j]
            push!(constraints, Constraint([v1, v2], a -> a[v1] != a[v2], "$v1 ≠ $v2"))
        end
    end

    # Restricciones aritméticas (suma columna correcta) - son n-arias
    #
    #   d + e = y + 10 * c1
    push!(constraints, Constraint([:d, :e, :y, :c1], 
                                 a -> a[:d] + a[:e] == a[:y] + 10 * a[:c1], "d + e = y + 10 * c1"))
    
    # c1 + n + r = e + 10 * c2  
    push!(constraints, Constraint([:c1, :n, :r, :e, :c2], 
                                 a -> a[:c1] + a[:n] + a[:r] == a[:e] + 10 * a[:c2], "c1 + n + r = e + 10 * c2"))
    
    # c2 + e + o = n + 10 * c3
    push!(constraints, Constraint([:c2, :e, :o, :n, :c3], 
                                 a -> a[:c2] + a[:e] + a[:o] == a[:n] + 10 * a[:c3], "c2 + e + o = n + 10 * c3"))
    
    # c3 + s + m = o + 10 * m (con m = 1, esto es c3 + s + 1 = o + 10)
    push!(constraints, Constraint([:c3, :s, :m, :o], 
                                 a -> a[:c3] + a[:s] + a[:m] == a[:o] + 10 * a[:m], "c3 + s + m = o + 10 * m"))

    return CSP(variables, domains, constraints)
end

# Función de verificación para la solución del problema 'send+more=money'

function verifica_send_more_money(sol)
    if sol !== nothing
        println("Solución encontrada:")
        letter_vars = [:s, :e, :n, :d, :m, :o, :r, :y]
        println([v => sol[v] for v in letter_vars])
        
        # Verificación
        send_val = sol[:s]*1000 + sol[:e]*100 + sol[:n]*10 + sol[:d]
        more_val = sol[:m]*1000 + sol[:o]*100 + sol[:r]*10 + sol[:e]
        money_val = sol[:m]*10000 + sol[:o]*1000 + sol[:n]*100 + sol[:e]*10 + sol[:y]
        
        println("Verificación: $send_val + $more_val = $money_val")
        println("¿Es correcta? ", send_val + more_val == money_val)
    else
        println("No se encontró solución por el problema.")
    end
end

# Creación del problema 'send+more=money'

send_more_money_csp = create_send_more_money_csp()
show_CSP(send_more_money_csp)

# Resolución del problema 'send+more=money' con distintos métodos

solve_brute_force(send_more_money_csp)
@time solve_BT(send_more_money_csp)

several_methods(send_more_money_csp, verifica_send_more_money)


# -----------------------------------
# Ejemplo 3: Problema de las N Reinas
# -----------------------------------

#=
Este es otro problema típico de optimización y búsqueda de soluciones. El objetivo es colocar N reinas en un tablero de N×N de manera que no se amenacen entre sí (una reina amenaza cualquier pieza colocada en su misma fila, columna o diagonal).

Como hemos visto en teoría, hay muchas formas de modelar este problema, nosotros vamos a supones que debe haber una reina en cada columna, por lo que solo hemos de decidir su altura (fila) en la que se coloca. Por tanto, cada reina se asocia a una variable, y el dominio de cada variable será el conjunto  de filas posibles.

Como es un problema que depende de N, será nuestro primer ejemplo de CSP parametrizado
=#

function create_n_queens_csp(n::Int)
    variables = [Symbol("r$i") for i in 1:n]             # Una variable por Reina
    domains = Dict(v => collect(1:n) for v in variables) # Dominio de cada Reina: filas posibles

    # Construimos las restricciones paso a paso, añadiendo las distintas 
    # restricciones (que se agrupan por tipos)
    constraints = Constraint[]

    # Restricciones binarias entre cada par de reinas (i, j) con i < j
    # Observa cómo agrupamos la restricción de las dos diagonales en una sola
    # ecuación
    for i in 1:n
        for j in (i+1):n
            v_i = variables[i] # Reina en columna i
            v_j = variables[j] # Reina en columna j
            col_diff = j - i   # Diferencia entre columnas: j-i = |j-i|

            # Restricción: No misma fila (ri != rj)
            push!(constraints, Constraint([v_i, v_j], a -> a[v_i] != a[v_j], "$v_i ≠ $v_j"))

            # Restricción: No misma diagonal (|ri - rj| != |i - j|)
            push!(constraints, Constraint([v_i, v_j], 
                                         a -> abs(a[v_i] - a[v_j]) != col_diff, "|$v_i - $v_j| ≠ $col_diff"))
        end
    end

    return CSP(variables, domains, constraints)
end

#=
El problema que vamos a encontrar en las variables parametrizadas es recoger los valores de cada una de las variables, ya que no conocemos su nombre a  priori, pero sí el patrón que siguen.
=#

function verifica_n_queens(sol, N)
    if sol !== nothing
        println("Solución encontrada:")
        # Ordenar por columna para una salida más clara
        sorted_vars = sort(collect(keys(sol)), by=x -> parse(Int, string(x)[2:end]))
        for var in sorted_vars
            col = parse(Int, string(var)[2:end])
            row = sol[var]
            println("Columna $col: Fila $row")
        end
        
        # Visualización del tablero
        println("\nVisualización del tablero:")
        board = fill("·", N, N)
        for (var, row) in sol
            col = parse(Int, string(var)[2:end])
            board[row, col] = "♕"
        end
        
        for r in 1:N
            println(join(board[r, :], " "))
        end
    else
        println("No se encontró solución.")
    end
end

# Creación del problema de las N Reinas (8 Reinas)

N = 10
n_queens_csp = create_n_queens_csp(N)
show_CSP(n_queens_csp)

# Resolución del problema de las N Reinas con distintos métodos
verifica2(s) = verifica_n_queens(s, N)
several_methods(n_queens_csp, verifica2)

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
        push!(constraints, Constraint(
            [r1, r2],
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

# -----------------------
# Ejemplo 5: SAT como CSP
# -----------------------

#=
Este ejercicio demuestra cómo transformar un problema SAT en un CSP para poder aplicar los algoritmos de resolución de CSPs como Backtracking, AC-3 y GAC al problema de la satisfactibilidad proposicional.

Para su resolución vamos a hacer uso conjunto de las dos librerías vistas hasta el momento en clase:
- PL.jl para manejar fórmulas proposicionales.
- CSP.jl para definir y resolver el problema como un CSP.

La resolución se basa en:
1. Cada variable proposicional de F se convierte en una variable CSP con dominio {true, false}.
2. F se transforma a Forma Clausal.
3. Cada cláusula (disyunción de literales) se convierte en una restricción CSP.
4. F ∈ SAT ⟺ el CSP resultante tiene solución.
5. La asignación de variables CSP que satisface todas las restricciones corresponde a un modelo de F.

¿Porqué convertir F a CNF y no trabajar directamente con F?
- La CNF facilita la creación de restricciones independientes por cláusula.
- Permite una representación más estructurada y manejable.
- Facilita la aplicación de algoritmos de consistencia como AC-3 y GAC.

Ejemplo de transformación:
- F = (p ∨ ¬q ∨ r) ∧ (¬p ∨ q) ∧ (¬r) (en CNF)
- Variables CSP: p, q, r ∈ {true, false}
- Restricciones CSP:
  * C1: p ∨ ¬q ∨ r = true
  * C2: ¬p ∨ q = true  
  * C3: ¬r = true
=#

include("../Lógica/PL.jl")
using .PropositionalLogic

"""
    sat_to_csp(formula::FormulaPL)

Convierte una fórmula proposicional a un CSP equivalente.
Devuelve el CSP y un mapeo de variables proposicionales a símbolos CSP.
"""
function sat_to_csp(formula::FormulaPL)
    # Obtener todas las variables proposicionales
    prop_vars = collect(vars_of(formula))
    
    # Convertir a forma clausal
    clauses = to_CF(formula)
    
    # Crear variables CSP (una por cada variable proposicional)
    csp_variables = [Symbol(var.name) for var in prop_vars]
    
    # Crear dominios (todas las variables son booleanas)
    domains = Dict{Symbol, Vector{Bool}}()
    for var_symbol in csp_variables
        domains[var_symbol] = [true, false]
    end
    
    # Crear restricciones CSP (una por cada cláusula)
    constraints = Constraint[]
    
    for clause in clauses
        # Obtener variables involucradas en esta cláusula
        clause_vars = Set{Var_PL}()
        for literal in clause.literals
            push!(clause_vars, literal.variable)
        end
        clause_var_symbols = [Symbol(var.name) for var in clause_vars]
        
        # Crear función de verificación para la cláusula
        # Una cláusula es verdadera si al menos uno de sus literales es verdadero
        check_func = function(assignment::Dict{Symbol, Any})
            # Verificar que todas las variables de la cláusula están asignadas
            for var in clause_vars
                var_symbol = Symbol(var.name)
                if !haskey(assignment, var_symbol)
                    return true # Si no están todas las variables, asumir consistente
                end
            end
            
            # Evaluar cada literal de la cláusula
            for literal in clause.literals
                var_symbol = Symbol(literal.variable.name)
                var_value = assignment[var_symbol]
                
                # Un literal es verdadero si:
                # - Es positivo y la variable es true, o
                # - Es negativo y la variable es false
                literal_value = literal.positive ? var_value : !var_value
                
                # Si algún literal es verdadero, la cláusula es verdadera
                if literal_value
                    return true
                end
            end
            
            # Si ningún literal es verdadero, la cláusula es falsa
            return false
        end
        
        # Crear la restricción
        constraint = Constraint(clause_var_symbols, check_func, "$(clause)")
        push!(constraints, constraint)
    end
    
    # Crear el CSP
    csp = CSP(csp_variables, domains, constraints)
    
    # Crear mapeo de variables proposicionales a símbolos CSP
    var_mapping = Dict{Var_PL, Symbol}()
    for var in prop_vars
        var_mapping[var] = Symbol(var.name)
    end
    
    return csp, var_mapping
end

"""
    create_sat_example_1()

Ejemplo SAT sencillo: (p ∨ ¬q) ∧ (¬p ∨ q ∨ r) ∧ (¬r)
"""
function create_sat_example_1()
    # Definir variables proposicionales
    p, q, r = vars(:p, :q, :r)
    
    # Crear fórmula SAT en CNF
    formula = (p | !q) & (!p | q | r) & !r
    
    println("🔍 EJEMPLO SAT 1:")
    println("Fórmula: $formula")
    println("Variables: p, q, r")
    
    # Convertir a CSP
    csp, var_mapping = sat_to_csp(formula)

    show_CSP(csp)
    
    return csp, formula, var_mapping
end

create_sat_example_1();

"""
    create_sat_example_2()

Ejemplo SAT más complejo: 3-SAT con 5 variables
(p ∨ q ∨ ¬r) ∧ (¬p ∨ s ∨ t) ∧ (r ∨ ¬s ∨ ¬t) ∧ (¬q ∨ r ∨ s) ∧ (p ∨ ¬s ∨ t)
"""
function create_sat_example_2()
    # Definir variables proposicionales
    p, q, r, s, t = vars(:p, :q, :r, :s, :t)
    
    # Crear fórmula 3-SAT
    formula = (p | q | !r) & (!p | s | t) & (r | !s | !t) & (!q | r | s) & (p | !s | t)
    
    println("🔍 EJEMPLO SAT 2 (3-SAT):")
    println("Fórmula: $formula")
    println("Variables: p, q, r, s, t")
    
    # Convertir a CSP
    csp, var_mapping = sat_to_csp(formula)
    
    show_CSP(csp)
    
    return csp, formula, var_mapping
end

"""
    create_sat_example_3()

Ejemplo SAT insatisfactible: (p) ∧ (¬p)
"""
function create_sat_example_3()
    # Definir variable proposicional
    p = Var_PL("p")
    
    # Crear fórmula insatisfactible
    formula = p & !p
    
    println("🔍 EJEMPLO SAT 3 (INSATISFACTIBLE):")
    println("Fórmula: $formula")
    println("Variable: p")
    
    # Convertir a CSP
    csp, var_mapping = sat_to_csp(formula)
    
    show_CSP(csp)
    
    return csp, formula, var_mapping
end

"""
    verificar_solucion_sat(solucion, formula::FormulaPL, var_mapping)

Verifica que la solución CSP satisface la fórmula SAT original.
"""
function verificar_solucion_sat(solucion, formula::FormulaPL, var_mapping)
    if isnothing(solucion)
        println("❌ No se encontró solución.")
        return false
    end
    
    println("\n✅ SOLUCIÓN ENCONTRADA:")
    
    # Crear valoración a partir de la solución CSP
    valuation = Valuation()
    for (prop_var, csp_symbol) in var_mapping
        valuation[prop_var] = solucion[csp_symbol]
        println("  $(prop_var.name) = $(solucion[csp_symbol])")
    end
    
    # Verificar que la valoración satisface la fórmula original
    satisface = evaluate(formula, valuation)
    println("\n🎯 ¿La solución satisface la fórmula SAT? $satisface")
    
    return satisface
end

"""
    resolver_sat_con_metodos_csp(crear_ejemplo_func, nombre_ejemplo)

Aplica todos los métodos de CSP a un ejemplo SAT.
"""
function resolver_sat_con_metodos_csp(crear_ejemplo_func, nombre_ejemplo)
    println("\n" * "=" ^ 60)
    println("🚀 RESOLVIENDO $nombre_ejemplo CON MÉTODOS CSP")
    println("=" ^ 60)
    
    # Crear el ejemplo
    csp, formula, var_mapping = crear_ejemplo_func()
    
    # Crear función de verificación específica para SAT
    function verificar_sat(solucion)
        return verificar_solucion_sat(solucion, formula, var_mapping)
    end
    
    # Aplicar todos los métodos
    several_methods(csp, verificar_sat)
    
    return csp, formula, var_mapping
end

"""
    test_sat_as_csp()

Ejecuta todos los ejemplos SAT como CSP.
"""
function test_sat_as_csp()
    println("\n🎓 EJERCICIO: PROBLEMAS SAT COMO CSP")
    println("=" ^ 50)

    # Ejemplo 1: SAT sencillo satisfactible
    resolver_sat_con_metodos_csp(create_sat_example_1, "EJEMPLO 1")
    
    # Ejemplo 2: 3-SAT más complejo
    resolver_sat_con_metodos_csp(create_sat_example_2, "EJEMPLO 2")
    
    # Ejemplo 3: SAT insatisfactible
    resolver_sat_con_metodos_csp(create_sat_example_3, "EJEMPLO 3")
    
end

test_sat_as_csp()

# ------------------------------------------------
# Ejemplo 6: Problema de Planificación de horarios
# ------------------------------------------------

#=
En nuestra colección de problemas clásicos no puede faltar un ejemplo de planificación de horarios, en este caso, muy breve y de juguete, solo para mostrar cómo se modela y resuelve con la librería.

En general, los problemas de planificación se pueden modelar como CSP donde las variables son los eventos a programar, los dominios son los posibles horarios y las restricciones aseguran que no haya solapamientos. Aunque la complejidad puede ampliarse añadiendo usuarios, restricción de recursos, etc. 

En este caso concreto, supongamos que hemos de planificar una serie de exámenes para una universidad dentro de los slots de tiempo que tenemos disponibles, y asegurando algunas restricciones, como que ningún estudiante tenga dos exámenes al mismo tiempo.

Las asignaturas de las que deben examinarse son: Matemáticas, Física, Química, Historia, Literatura e Inglés.
Los slots de tiempo disponibles son: Lunes 9am, Lunes 2pm, Martes 9am, Martes 2pm.
Las restricciones son las siguientes:
- Juan toma Matemáticas y Física
- María toma Matemáticas y Química
- Pedro toma Física y Química
- Ana toma Historia y Literatura
- Carlos toma Historia e Inglés
- Sofia toma Literatura e Inglés
- Luis toma Matemáticas e Historia
- Elena toma Física y Literatura

Usa CSP para modelar y resolver este problema.
=#

function crear_horarios_examenes()
    println("=== PROBLEMA DE HORARIOS DE EXÁMENES ===")
    println("Asignar horarios a exámenes evitando conflictos con estudiantes inscritos\n")
    
    # Variables: cada examen
    variables = [:Matematicas, :Fisica, :Quimica, :Historia, :Literatura, :Ingles]
    
    # Dominios: 4 slots de tiempo disponibles
    horarios = ["Lunes_9am", "Lunes_2pm", "Martes_9am", "Martes_2pm"]
    domains = Dict(examen => horarios for examen in variables)
    
    # Restricciones: estudiantes no pueden tener dos exámenes al mismo tiempo
    # Definimos qué estudiantes toman qué materias:
    conflictos_estudiantes = [
        (:Matematicas, :Fisica),      # Juan toma ambas
        (:Matematicas, :Quimica),     # María toma ambas
        (:Fisica, :Quimica),          # Pedro toma ambas
        (:Historia, :Literatura),     # Ana toma ambas
        (:Historia, :Ingles),         # Carlos toma ambas
        (:Literatura, :Ingles),       # Sofia toma ambas
        (:Matematicas, :Historia),    # Luis toma ambas
        (:Fisica, :Literatura)        # Elena toma ambas
    ]
    
    constraints = Constraint[]
    for (examen1, examen2) in conflictos_estudiantes
        push!(constraints, Constraint(
            [examen1, examen2],
            function(assignment)
                if haskey(assignment, examen1) && haskey(assignment, examen2)
                    return assignment[examen1] != assignment[examen2]
                end
                return true
            end, "$examen1 ≠ $examen2"
        ))
    end
    
    csp = CSP(variables, domains, constraints)
    
    show_CSP(csp)

    println("Resolviendo con Backtracking + AC3...")
    solucion = solve_BT_with_AC3(csp)
    
    if !isnothing(solucion)
        println("\n ¡Horario de exámenes encontrado!")
        println("╔══════════════════════════════════════╗")
        for slot in horarios
            examenes_en_slot = [string(ex) for ex in variables if solucion[ex] == slot]
            if !isempty(examenes_en_slot)
                println("║ $(rpad(slot, 11)): $(rpad(join(examenes_en_slot, ", "), 23)) ║")
            end
        end
        println("╚══════════════════════════════════════╝")
    else
        println("No se pudo crear un horario válido")
    end
    println()

    return csp, solucion
end

crear_horarios_examenes();

# ------------------------------------------------
# Ejemplo 7: Configuración de red de servidores
# ------------------------------------------------

#=
Este problema es un ejemplo más complejo y realista que ilustra cómo los CSP pueden aplicarse a problemas de configuración de sistemas. En este caso, se trata de configurar una red de servidores con múltiples restricciones n-arias interconectadas.

Cada servidor debe configurarse con ciertos parámetros (como tipo de servidor, cantidad de CPU, memoria RAM, y ancho de banda de red) que deben cumplir una serie de restricciones tanto individuales como globales.

El objetivo es encontrar una configuración válida para todos los servidores de nuestra red que satisfaga todas las restricciones, como:

-Necesitamos 2 servidores web, 2 de aplicaciones, 2 de Bases de Datos, un servidor de Caché, y otro de Balanceo de Carga.

- Cada servidor debe tener una configuración compatible con su tipo. Las configuraciones posibles son:

    * Servidor Web: Apache, Nginx, IIS
    * Servidor de Aplicaciones: Julia, Python, NodeJS, DotNet
    * Servidor de Base de Datos: MySQL, PostgreSQL, MongoDB
    * Servidor de Caché: Redis, Memcached
    * Servidor de Balanceo de Carga: HAProxy, Nginx, F5

- No todos los servidores son compatibles con las aplicaciones que se van a correr, así que nos dan también una lista de compatibilidades:

    * Apache es compatible con Julia, Python, DotNet
    * Nginx es compatible con Julia, Python, NodeJS
    * IIS es compatible con DotNet

- La suma total de recursos (CPU, RAM, ancho de banda) no debe exceder los límites disponibles en la red. Los recursos disponibles son:

    * 4 tipos de CPU: de 2, 4, 8 y 16 cores. Podemos usar como máximo 64 cores en total.
    * RAM: de 4, 8, 16 y 32 GB. Podemos usar como máximo 128 GB en total.
    * Ancho de banda: de 100, 1000 y 10000 Mbps

- Según el uso de la máquina, hay algunas restricciones adicionales que debemos verificar:

    * Apache: entre 2 y 8 CPUs, entre 4 y 16 GB de RAM, y 1000 Mbps de ancho de banda
    * Nginx: entre 2 y 16 CPUs, entre 4 y 8 GB de RAM, y 1000 Mbps de ancho de banda
    * IIS: entre 4 y 16 CPUs, entre 8 y 32 GB de RAM, y 1000 Mbps de ancho de banda
    * Julia: entre 4 y 16 CPUs, entre 8 y 32 GB de RAM, y 100 Mbps de ancho de banda
    * Python: entre 2 y 8 CPUs, entre 4 y 16 GB de RAM, y 100 Mbps de ancho de banda
    * NodeJS: entre 2 y 8 CPUs, entre 4 y 8 GB de RAM, y 100 Mbps de ancho de banda
    * DotNet: entre 4 y 16 CPUs, entre 8 y 32 GB de RAM, y 100 Mbps de ancho de banda
    * MySQL: entre 4 y 16 CPUs, entre 8 y 32 GB de RAM, y 1000 Mbps de ancho de banda
    * PostgreSQL: entre 4 y 16 CPUs, entre 8 y 32 GB de RAM, y 1000 Mbps de ancho de banda
    * MongoDB: entre 8 y 16 CPUs, entre 16 y 32 GB de RAM, y 1000 Mbps de ancho de banda
    * Redis: entre 2 y 8 CPUs, entre 8 y 32 GB de RAM, y 1000 Mbps de ancho de banda
    * Memcached: entre 2 y 4 CPUs, entre 4 y 16 GB de RAM, y 1000 Mbps de ancho de banda
    * HAProxy: entre 2 y 8 CPUs, entre 4 y 8 GB de RAM, y 10000 Mbps de ancho de banda
    * F5: entre 8 y 16 CPUs, entre 16 y 32 GB de RAM, y 10000 Mbps de ancho de banda

- Para asegurar un balanceo de carga efectivo, los servidores del mismo tipo (web, aplicaciones y BD) deben tener configuraciones similares: no pueden diferenciarse en mas 4 cores y 8 GB de RAM.

- El servidor de Cache debe tener suficientes recursos para soportar las apps: un mínimo de 8 GB de RAM, más 4 GB adicionales por cada app que use Julia o DotNet. Además, si la Cache es Redis, necesita al menos 4 GB adicionales.

- El servidor de Balanceo de Carga debe ser capaz de manejar todos los servidores web, por lo que si es un F5, necesita al menos 10000 Mbps de ancho de banda, y si es HAProxy, al menos 1000 Mbps.

⋮

Es fácil añadir más restrcciones si se desea, pero con estas ya tenemos un problema bastante completo y realista,... aunque la solución no lo sea tanto. Por ejemplo, no hemos considerado la redundancia de servidores, ni la necesidad de diversas aplicaciones corriendo, ni el coste económico,... pero todo esto se puede añadir fácilmente.

Usa CSP para modelar y resolver este problema.
=#

function configuracion_red_servidores()
    println("=== CONFIGURACIÓN DE RED DE SERVIDORES ===")
    println("Problema ideal para GAC: múltiples restricciones n-arias interconectadas\n")
    
    # Datos del problema: configurar una red de 8 servidores
    servidores = ["Web1", "Web2", "App1", "App2", "DB1", "DB2", "Cache1", "LB1"]
    
    # Posibles configuraciones para cada servidor
    configs_web = ["Apache", "Nginx", "IIS"]
    configs_app = ["Julia", "Python", "NodeJS", "DotNet"] 
    configs_db = ["MySQL", "PostgreSQL", "MongoDB"]
    configs_cache = ["Redis", "Memcached"]
    configs_lb = ["HAProxy", "Nginx", "F5"]
    
    # Variables: configuración, CPU, RAM y red para cada servidor
    variables = Symbol[]
    for servidor in servidores
        push!(variables, Symbol("$(servidor)_config"))
        push!(variables, Symbol("$(servidor)_cpu"))
        push!(variables, Symbol("$(servidor)_ram"))
        push!(variables, Symbol("$(servidor)_red"))
    end
    
    # Dominios
    domains = Dict{Symbol, Vector{Any}}()
    
    # Configuraciones específicas por tipo de servidor
    config_dominios = Dict(
        "Web1" => configs_web, "Web2" => configs_web,
        "App1" => configs_app, "App2" => configs_app,
        "DB1" => configs_db, "DB2" => configs_db,
        "Cache1" => configs_cache,
        "LB1" => configs_lb
    )
    
    # Recursos disponibles (limitados globalmente)
    cpus_disponibles = [2, 4, 8, 16]  # cores
    ram_disponible = [4, 8, 16, 32]   # GB
    red_disponible = [100, 1000, 10000] # Mbps
    
    for servidor in servidores
        domains[Symbol("$(servidor)_config")] = config_dominios[servidor]
        domains[Symbol("$(servidor)_cpu")] = cpus_disponibles
        domains[Symbol("$(servidor)_ram")] = ram_disponible
        domains[Symbol("$(servidor)_red")] = red_disponible
    end
    
    # Restricciones de recursos por configuración
    requisitos_config = Dict(
        # Web servers
        "Apache" => (cpu_min=2, cpu_max=8, ram_min=4, ram_max=16, red_min=1000),
        "Nginx" => (cpu_min=2, cpu_max=16, ram_min=4, ram_max=8, red_min=1000),
        "IIS" => (cpu_min=4, cpu_max=16, ram_min=8, ram_max=32, red_min=1000),
        
        # App servers
        "Julia" => (cpu_min=4, cpu_max=16, ram_min=8, ram_max=32, red_min=100),
        "Python" => (cpu_min=2, cpu_max=8, ram_min=4, ram_max=16, red_min=100),
        "NodeJS" => (cpu_min=2, cpu_max=8, ram_min=4, ram_max=8, red_min=100),
        "DotNet" => (cpu_min=4, cpu_max=16, ram_min=8, ram_max=32, red_min=100),
        
        # Database servers
        "MySQL" => (cpu_min=4, cpu_max=16, ram_min=8, ram_max=32, red_min=1000),
        "PostgreSQL" => (cpu_min=4, cpu_max=16, ram_min=8, ram_max=32, red_min=1000),
        "MongoDB" => (cpu_min=8, cpu_max=16, ram_min=16, ram_max=32, red_min=1000),
        
        # Cache servers
        "Redis" => (cpu_min=2, cpu_max=8, ram_min=8, ram_max=32, red_min=1000),
        "Memcached" => (cpu_min=2, cpu_max=4, ram_min=4, ram_max=16, red_min=1000),
        
        # Load balancers
        "HAProxy" => (cpu_min=2, cpu_max=8, ram_min=4, ram_max=8, red_min=10000),
        "F5" => (cpu_min=8, cpu_max=16, ram_min=16, ram_max=32, red_min=10000)
    )
    
    constraints = Constraint[]
    
    # RESTRICCIÓN 1: Compatibilidad de configuración con recursos
    for servidor in servidores
        push!(constraints, Constraint(
            [Symbol("$(servidor)_config"), Symbol("$(servidor)_cpu"), 
             Symbol("$(servidor)_ram"), Symbol("$(servidor)_red")],
            function(assignment)
                vars_necesarias = [Symbol("$(servidor)_config"), Symbol("$(servidor)_cpu"), 
                                 Symbol("$(servidor)_ram"), Symbol("$(servidor)_red")]
                if all(v in keys(assignment) for v in vars_necesarias)
                    config = assignment[Symbol("$(servidor)_config")]
                    cpu = assignment[Symbol("$(servidor)_cpu")]
                    ram = assignment[Symbol("$(servidor)_ram")]
                    red = assignment[Symbol("$(servidor)_red")]
                    
                    req = requisitos_config[config]
                    return (req.cpu_min <= cpu <= req.cpu_max && 
                            req.ram_min <= ram <= req.ram_max && 
                            red >= req.red_min)
                end
                return true
            end, "$(servidor) recursos compatibles con config"
        ))
    end
    
    # RESTRICCIÓN 2: Límites globales de recursos (muy restrictiva)
    total_cpu_disponible = 64
    total_ram_disponible = 128
    
    all_cpu_vars = [Symbol("$(servidor)_cpu") for servidor in servidores]
    push!(constraints, Constraint(
        all_cpu_vars,
        function(assignment)
            cpu_usado = sum(get(assignment, var, 0) for var in all_cpu_vars if haskey(assignment, var))
            return cpu_usado <= total_cpu_disponible
        end, "Total CPU ≤ $total_cpu_disponible"
    ))
    
    all_ram_vars = [Symbol("$(servidor)_ram") for servidor in servidores]
    push!(constraints, Constraint(
        all_ram_vars,
        function(assignment)
            ram_usado = sum(get(assignment, var, 0) for var in all_ram_vars if haskey(assignment, var))
            return ram_usado <= total_ram_disponible
        end, "Total RAM ≤ $total_ram_disponible"
    ))
    
    # RESTRICCIÓN 3: Compatibilidad entre tecnologías (n-arias complejas)    
    # Web servers y App servers deben ser compatibles
    compatibilidades_web_app = Dict(
        "Apache" => ["Julia", "Python", "DotNet"],
        "Nginx" => ["Julia", "Python", "NodeJS"],
        "IIS" => ["DotNet"]
    )
    
    for web_server in ["Web1", "Web2"]
        for app_server in ["App1", "App2"]
            push!(constraints, Constraint(
                [Symbol("$(web_server)_config"), Symbol("$(app_server)_config")],
                function(assignment)
                    if haskey(assignment, Symbol("$(web_server)_config")) && 
                       haskey(assignment, Symbol("$(app_server)_config"))
                        web_config = assignment[Symbol("$(web_server)_config")]
                        app_config = assignment[Symbol("$(app_server)_config")]
                        return app_config in compatibilidades_web_app[web_config]
                    end
                    return true
                end, "$(web_server) compatible con $(app_server)"
            ))
        end
    end
    
    # RESTRICCIÓN 4: Balanceado de carga - servidores similares deben tener configs similares
    pares_similares = [("Web1", "Web2"), ("App1", "App2"), ("DB1", "DB2")]
    
    for (serv1, serv2) in pares_similares
        # Si tienen la misma configuración, deben tener recursos similares
        push!(constraints, Constraint(
            [Symbol("$(serv1)_config"), Symbol("$(serv2)_config"),
             Symbol("$(serv1)_cpu"), Symbol("$(serv2)_cpu"),
             Symbol("$(serv1)_ram"), Symbol("$(serv2)_ram")],
            function(assignment)
                vars_necesarias = [Symbol("$(serv1)_config"), Symbol("$(serv2)_config"),
                                 Symbol("$(serv1)_cpu"), Symbol("$(serv2)_cpu"),
                                 Symbol("$(serv1)_ram"), Symbol("$(serv2)_ram")]
                if all(v in keys(assignment) for v in vars_necesarias)
                    if assignment[Symbol("$(serv1)_config")] == assignment[Symbol("$(serv2)_config")]
                        # Recursos deben ser idénticos o muy similares
                        cpu_diff = abs(assignment[Symbol("$(serv1)_cpu")] - assignment[Symbol("$(serv2)_cpu")])
                        ram_diff = abs(assignment[Symbol("$(serv1)_ram")] - assignment[Symbol("$(serv2)_ram")])
                        return cpu_diff <= 4 && ram_diff <= 8
                    end
                end
                return true
            end, "$(serv1) balanceado con $(serv2)"
        ))
    end
    
    # RESTRICCIÓN 5: Dependencias de rendimiento
    # Cache debe tener suficientes recursos para soportar las apps
    push!(constraints, Constraint(
        [Symbol("App1_config"), Symbol("App2_config"), 
         Symbol("Cache1_config"), Symbol("Cache1_ram")],
        function(assignment)
            vars_necesarias = [Symbol("App1_config"), Symbol("App2_config"), 
                             Symbol("Cache1_config"), Symbol("Cache1_ram")]
            if all(v in keys(assignment) for v in vars_necesarias)
                app1_config = assignment[Symbol("App1_config")]
                app2_config = assignment[Symbol("App2_config")]
                cache_config = assignment[Symbol("Cache1_config")]
                cache_ram = assignment[Symbol("Cache1_ram")]
                
                # Calcular RAM necesaria basada en las apps
                ram_necesaria = 8  # base
                if app1_config in ["Julia", "DotNet"] ram_necesaria += 4 end
                if app2_config in ["Julia", "DotNet"] ram_necesaria += 4 end
                if cache_config == "Redis" ram_necesaria += 4 end
                
                return cache_ram >= ram_necesaria
            end
            return true
        end, "Cache soporta apps"
    ))
    
    # RESTRICCIÓN 6: Load balancer debe poder manejar todos los web servers
    push!(constraints, Constraint(
        [Symbol("LB1_config"), Symbol("LB1_red"), 
         Symbol("Web1_config"), Symbol("Web2_config")],
        function(assignment)
            vars_necesarias = [Symbol("LB1_config"), Symbol("LB1_red"), 
                             Symbol("Web1_config"), Symbol("Web2_config")]
            if all(v in keys(assignment) for v in vars_necesarias)
                lb_config = assignment[Symbol("LB1_config")]
                lb_red = assignment[Symbol("LB1_red")]
                
                # F5 es más potente pero necesita más ancho de banda
                if lb_config == "F5"
                    return lb_red >= 10000
                else
                    # HAProxy puede trabajar con menos, pero debe ser consistente
                    return lb_red >= 1000
                end
            end
            return true
        end, "LB1 soporta Web Servers"
    ))
    
    csp = CSP(variables, domains, constraints)
    
    show_CSP(csp)

    println("📊 Estadísticas del problema:")
    println("   Variables: $(length(variables))")
    println("   Restricciones: $(length(constraints))")
    println("   Espacio de búsqueda: ~$(prod(length(domains[v]) for v in variables))")
    println("   Complejidad: ALTA - Multiple restricciones n-arias interdependientes\n")
    
    println("🧪 COMPARANDO ALGORITMOS:")
    println("=" ^ 50)
    
    # Medir AC3 + Backtracking
    println("\n🔹 Probando Backtracking + AC3...")
    tiempo_inicio = time()
    solucion_ac3 = solve_BT_with_AC3(csp)
    tiempo_ac3 = time() - tiempo_inicio
    
    if !isnothing(solucion_ac3)
        println("   ✅ Solución encontrada en $(round(tiempo_ac3, digits=3))s")
    else
        println("   ❌ No se encontró solución")
    end
    
    # Medir GAC + Backtracking  
    println("\n🔸 Probando Backtracking + GAC...")
    tiempo_inicio = time()
    solucion_gac = solve_BT_with_GAC(csp)
    tiempo_gac = time() - tiempo_inicio
    
    if !isnothing(solucion_gac)
        println("   ✅ Solución encontrada en $(round(tiempo_gac, digits=3))s")
    else
        println("   ❌ No se encontró solución")
    end
    
    # Comparar eficiencia
    println("\n📈 ANÁLISIS DE RENDIMIENTO:")
    println("=" ^ 40)
    println("Backtracking + AC3:  $(round(tiempo_ac3, digits=3))s")
    println("Backtracking + GAC:  $(round(tiempo_gac, digits=3))s")
    
    if tiempo_ac3 > 0 && tiempo_gac > 0
        mejora = tiempo_ac3 / tiempo_gac
        println("Mejora con GAC:      $(round(mejora, digits=2))x más rápido")
        
        if mejora > 2
            println("🎉 GAC es SIGNIFICATIVAMENTE más eficiente!")
        elseif mejora > 1.5
            println("✨ GAC muestra ventajas claras")
        else
            println("📊 Rendimiento similar (problema quizás no lo suficientemente restrictivo)")
        end
    end
    
    # Mostrar una de las soluciones
    solucion_final = !isnothing(solucion_gac) ? solucion_gac : solucion_ac3
    if !isnothing(solucion_final)
        mostrar_configuracion_servidores(solucion_final, servidores)
    end
    
    return csp, solucion_final
end

function mostrar_configuracion_servidores(solucion, servidores)
    println("\n🖥️  CONFIGURACIÓN DE SERVIDORES GENERADA:")
    println("=" ^ 60)
    
    # Agrupar por tipo de servidor
    tipos = Dict(
        "Web Servers" => ["Web1", "Web2"],
        "App Servers" => ["App1", "App2"], 
        "Database Servers" => ["DB1", "DB2"],
        "Cache Server" => ["Cache1"],
        "Load Balancer" => ["LB1"]
    )
    
    total_cpu = 0
    total_ram = 0
    
    for (tipo, servers) in tipos
        println("\n🔧 $tipo:")
        for servidor in servers
            config = solucion[Symbol("$(servidor)_config")]
            cpu = solucion[Symbol("$(servidor)_cpu")]
            ram = solucion[Symbol("$(servidor)_ram")]
            red = solucion[Symbol("$(servidor)_red")]
            
            total_cpu += cpu
            total_ram += ram
            
            println("   📍 $servidor: $config | CPU: $(cpu) cores | RAM: $(ram)GB | Red: $(red)Mbps")
        end
    end
    
    println("\n📊 RESUMEN DE RECURSOS:")
    println("   Total CPU utilizada: $total_cpu/64 cores ($(round(total_cpu/64*100, digits=1))%)")
    println("   Total RAM utilizada: $total_ram/128 GB ($(round(total_ram/128*100, digits=1))%)")
end

configuracion_red_servidores();
