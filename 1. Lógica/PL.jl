module PropositionalLogic

export FormulaPL, Var_PL, Neg_PL, And_PL, Or_PL, Imp_PL, Iff_PL, ⋀, ⋁
export subformulas, formation_tree
export @formula, truth_table, TAUT, UNSAT, SAT, models, LC, EQUIV

export to_CNF, to_DNF, to_CF, apply_val
export evaluate, Valuation, vars_of, vars

export DPLL, DPLL_SAT, DPLL_LC

# Tipos abstractos para las fórmulas
abstract type FormulaPL end

# Tipos concretos
struct Var_PL <: FormulaPL
    name::String        # se representan como cadenas de texto
end

struct Neg_PL <: FormulaPL
    operand::FormulaPL
end

struct And_PL <: FormulaPL
    left::FormulaPL
    right::FormulaPL
end

struct Or_PL <: FormulaPL
    left::FormulaPL
    right::FormulaPL
end

struct Imp_PL <: FormulaPL
    left::FormulaPL
    right::FormulaPL
end

struct Iff_PL <: FormulaPL
    left::FormulaPL
    right::FormulaPL
end

# ==================== CONSTANTES LÓGICAS ====================

"""
    Top_PL() <: FormulaPL

Representa la tautología (⊤), una fórmula siempre verdadera.
"""
struct Top_PL <: FormulaPL end

"""
    Bottom_PL() <: FormulaPL

Representa la contradicción (⊥), una fórmula siempre falsa.
"""
struct Bottom_PL <: FormulaPL end

# Constantes globales para uso conveniente
"""
Constante global para la tautología. Equivale a `Top_PL()`.
"""
const ⊤ = Top_PL()

"""
Constante global para la contradicción. Equivale a `Bottom_PL()`.
"""
const ⊥ = Bottom_PL()

# ==================== SOBRECARGA DE OPERADORES ====================

# Sobrecarga de operadores para crear fórmulas de forma natural

Base.:!(f::FormulaPL) = Neg_PL(f)                       #       !f ≡ ¬f
Base.:-(f::FormulaPL) = Neg_PL(f)                       #       -f ≡ ¬f
Base.:&(f1::FormulaPL, f2::FormulaPL) = And_PL(f1, f2)  # f1 & f2  ≡ f1 ∧ f2
Base.:|(f1::FormulaPL, f2::FormulaPL) = Or_PL(f1, f2)   # f1 | f2  ≡ f1 ∨ f2
Base.:>(f1::FormulaPL, f2::FormulaPL) = Imp_PL(f1, f2)  # f1 > f2  ≡ f1 → f2
Base.:~(f1::FormulaPL, f2::FormulaPL) = Iff_PL(f1, f2)  # f1 ~ f2  ≡ f1 ↔ f2


# Función para comparar variables basados en sus nombres
Base.isless(v1::Var_PL, v2::Var_PL) = isless(v1.name, v2.name)

# Conjunción de todas las fórmulas en Γ
#   ⋀ = \bigwedge
function ⋀(Γ::Vector{<:FormulaPL})
    return reduce(&, Γ) 
end

# Conjunción de todas las fórmulas en Γ
#   ⋁ = \bigvee
function ⋁(Γ::Vector{<:FormulaPL})
    return reduce(|, Γ) 
end


# Función auxiliar para crear variables de forma más sencilla
function vars(names...)
    return [Var_PL(string(name)) for name in names]
end

# Macro para crear fórmulas de forma más natural
macro formula(expr)
    return parse_formula(expr)
end

function parse_formula(expr)
    if isa(expr, Symbol)
        return :(Var_PL($(string(expr))))
    elseif isa(expr, Expr)
        if expr.head == :call
            op = expr.args[1]
            if (op == :! || op == :-) && length(expr.args) == 2
                return :(Neg_PL($(parse_formula(expr.args[2]))))
            elseif op == :& && length(expr.args) == 3
                return :(And_PL($(parse_formula(expr.args[2])), $(parse_formula(expr.args[3]))))
            elseif op == :| && length(expr.args) == 3
                return :(Or_PL($(parse_formula(expr.args[2])), $(parse_formula(expr.args[3]))))
            elseif op == :> && length(expr.args) == 3
                return :(Imp_PL($(parse_formula(expr.args[2])), $(parse_formula(expr.args[3]))))
            elseif op == :~ && length(expr.args) == 3
                return :(Iff_PL($(parse_formula(expr.args[2])), $(parse_formula(expr.args[3]))))
            end
        end
    end
    error("Expresión no reconocida: $expr")
end

# Función para mostrar las fórmulas de forma legible
function Base.show(io::IO, f::Var_PL)
    print(io, f.name)
end

function Base.show(io::IO, f::Neg_PL)
    print(io, "¬$(f.operand)")
end

function Base.show(io::IO, f::And_PL)
    print(io, "($(f.left) ∧ $(f.right))")
end

function Base.show(io::IO, f::Or_PL)
    print(io, "($(f.left) ∨ $(f.right))")
end

function Base.show(io::IO, f::Imp_PL)
    print(io, "($(f.left) → $(f.right))")
end

function Base.show(io::IO, f::Iff_PL)
    print(io, "($(f.left) ↔ $(f.right))")
end

# Función para obtener todas las variables de una fórmula
function vars_of(f::Var_PL)
    return Set([f])
end

function vars_of(f::Neg_PL)
    return vars_of(f.operand)
end

function vars_of(f::Union{And_PL, Or_PL, Imp_PL, Iff_PL})
    return union(vars_of(f.left), vars_of(f.right))
end

# ==================== ÁRBOL DE FORMACIÓN ====================

"""
    formation_tree(f::FormulaPL, prefix::String, is_last::Bool) -> String

Genera una representación visual del árbol de formación de una fórmula.
Muestra la estructura jerárquica de la fórmula con formato de árbol ASCII.

# Ejemplos de salida
```julia
formation_tree((p & q) | r)
∨
├── ∧
│   ├── p
│   └── q
└── r
```

# Argumentos
- `f`: Fórmula a visualizar
- `prefix`: Prefijo para la indentación (uso interno)
- `is_last`: Si es el último hijo en el nivel actual (uso interno)

# Funcionamiento
- Operadores aparecen como nodos internos
- Variables aparecen como hojas
- Usa caracteres Unicode para dibujar las conexiones del árbol
"""
function formation_tree(f::FormulaPL, prefix::String = "", is_last::Bool = true)
    if isa(f, Var_PL)
        return f.name
    elseif isa(f, Top_PL)
        return "⊤"
    elseif isa(f, Bottom_PL)
        return "⊥"
    elseif isa(f, Neg_PL)
        # Para negaciones simples, mostrar en una línea
        if isa(f.operand, Union{Var_PL, Top_PL, Bottom_PL})
            return "¬ $(f.operand)"
        else
            # Para negaciones complejas, crear subárbol
            operand_tree = formation_tree(f.operand, prefix * "    ", true)
            return "¬\n$(prefix)└── $operand_tree"
        end
    else
        # Para operadores binarios
        operator = if isa(f, And_PL)
            "∧"
        elseif isa(f, Or_PL)
            "∨"
        elseif isa(f, Imp_PL)
            "→"
        elseif isa(f, Iff_PL)
            "←→"
        end
        
        # Construir prefijos para los hijos (manejo de indentación)
        left_prefix = prefix * "│   "    # Continúa la línea vertical
        right_prefix = prefix * "    "   # Espacio en blanco (último hijo)
        
        # Construir árboles para los operandos recursivamente
        left_tree = formation_tree(f.left, left_prefix, false)
        right_tree = formation_tree(f.right, right_prefix, true)
        
        # Formatear la salida con caracteres de conexión
        left_part = "├── $left_tree"    # Rama izquierda (no es la última)
        right_part = "└── $right_tree"  # Rama derecha (es la última)
        
        return "$operator\n$prefix$left_part\n$prefix$right_part"
    end
end

"""
    formation_tree(f::FormulaPL)

Versión de conveniencia que imprime directamente el árbol de formación.
Llama a la versión completa con parámetros por defecto y muestra el resultado.

# Ejemplos
```julia
p, q, r = vars("p", "q", "r")
formation_tree((p & q) > r)
```
"""
function formation_tree(f::FormulaPL)
    println(formation_tree(f, "", true))
end

"""
    subformulas(f::FormulaPL) -> Set{FormulaPL}

Obtiene todas las subfórmulas de una fórmula dada.

# Definición
Una subfórmula de φ es cualquier fórmula que aparece como componente de φ,
incluyendo la propia φ.

# Ejemplos
```julia
p, q = vars("p", "q")
formula = p & q
subs = subformulas(formula)  # {p, q, p ∧ q}
```

# Funcionamiento
- Variables son subfórmulas de sí mismas
- Operadores unarios: subfórmulas del operando + la fórmula completa
- Operadores binarios: subfórmulas de ambos operandos + la fórmula completa

# Nota
El resultado incluye siempre la fórmula original como subfórmula de sí misma.
"""
function subformulas(f::FormulaPL)
    if isa(f, Union{Var_PL, Top_PL, Bottom_PL})
        return Set([f])
    elseif isa(f, Neg_PL)
        return union(subformulas(f.operand), Set([f]))
    elseif isa(f, Union{And_PL, Or_PL, Imp_PL, Iff_PL})
        return union(subformulas(f.left), subformulas(f.right), Set([f]))
    else
        return Set{FormulaPL}()
    end
end 


# Una valoración es un diccionario que mapea nombres de variables a valores booleanos:

"""
    Valuation = Dict{Var_PL, Bool}

Tipo alias para representar valoraciones (asignaciones de verdad).
Una valoración mapea variables proposicionales a valores booleanos.

# Ejemplos
```julia
p, q = vars("p", "q")
val = Valuation(p => true, q => false)
```

# Interpretación matemática
Una valoración v: Var → {0,1} asigna valores de verdad a variables,
extendiendo la evaluación a fórmulas complejas de manera composicional.
"""
const Valuation = Dict{Var_PL, Bool}

function Base.show(io::IO, v::Valuation)
    if isempty(v)
        print(io, "∅")
    else
        # Ordenar las variables alfabéticamente para una salida consistente
        sorted_pairs = sort(collect(v), by = pair -> pair[1].name)
        assignments = ["$(var.name) = $(value ? "1" : "0")" for (var, value) in sorted_pairs]
        print(io, "{", join(assignments, ", "), "}")
    end
end

# Función para evaluar una fórmula dada una valoración

function evaluate(f::FormulaPL, val::Valuation)
    if isa(f, Var_PL)
        return get(val, f, false)  # false por defecto para variables no asignadas
    elseif isa(f, Top_PL)
        return true  # ⊤ siempre es verdadero
    elseif isa(f, Bottom_PL)
        return false  # ⊥ siempre es falso
    elseif isa(f, Neg_PL)
        return !evaluate(f.operand, val)
    elseif isa(f, And_PL)
        return evaluate(f.left, val) && evaluate(f.right, val)
    elseif isa(f, Or_PL)
        return evaluate(f.left, val) || evaluate(f.right, val)
    elseif isa(f, Imp_PL)
        return !evaluate(f.left, val) || evaluate(f.right, val)  # φ → ψ ≡ ¬φ ∨ ψ
    elseif isa(f, Iff_PL)
        return evaluate(f.left, val) == evaluate(f.right, val)   # φ ↔ ψ ≡ (φ → ψ) ∧ (ψ → φ)
    else
        return false  # Caso por defecto (no debería alcanzarse)
    end
end

"""
    (v::Valuation)(F) -> Bool

Permite usar una valoración como función aplicada a una fórmula.
Equivale a `evaluate(F, v)` pero con sintaxis más natural.

# Ejemplos
```julia
p, q = vars("p", "q")
formula = p > q
val = Valuation(p => true, q => false)

# Ambas formas son equivalentes:
result1 = evaluate(formula, val)
result2 = val(formula)  # Sintaxis funcional más natural
```

# Ventajas
- Sintaxis más legible en algunos contextos
- Permite tratar valoraciones como funciones matemáticas v: FormulaPL → Bool
"""
function (v::Valuation)(F)
    return evaluate(F, v)
end

# Función para mostrar la tabla de verdad de una o más fórmulas
function truth_table(fs)
    # Obtener todas las variables únicas de todas las fórmulas
    all_vars = Set{Var_PL}()
    for f in fs
        union!(all_vars, vars_of(f))
    end
    vars = sort(collect(all_vars))
    n = length(vars)
    
    # Calcular el ancho necesario para cada columna
    var_width = maximum([length(var.name) for var in vars]; init=3)
    formula_widths = [max(length(string(f)), 3) for f in fs]
    
    println("Tabla de Verdad")
    println("=" ^ (var_width * n + sum(formula_widths) + 3 * (length(fs) - 1) + 5))
    
    # Encabezado con variables
    header_parts = [rpad(var.name, var_width) for var in vars]
    push!(header_parts, "│")  # Separador
    for (i, f) in enumerate(fs)
        push!(header_parts, rpad(string(f), formula_widths[i]))
    end
    println(join(header_parts, " "))
    
    # Línea separadora
    sep_parts = ["-"^var_width for _ in vars]
    push!(sep_parts, "┼")
    for width in formula_widths
        push!(sep_parts, "-"^width)
    end
    println(join(sep_parts, "-"))
    
    # Generar todas las filas
    for i in 0:(2^n - 1)
        val = Valuation()
        binary = string(i, base=2, pad=n)
        
        # Asignar valores a las variables
        for (j, var) in enumerate(vars)
            val[var] = binary[j] == '1'
        end
        
        # Crear la fila
        row_parts = []
        
        # Valores de las variables
        for var in vars
            val_str = val[var] ? "1" : "0"
            push!(row_parts, rpad(val_str, var_width))
        end
        
        push!(row_parts, "│")  # Separador
        
        # Valores de las fórmulas
        for (i, f) in enumerate(fs)
            result = evaluate(f, val)
            result_str = result ? "1" : "0"
            push!(row_parts, rpad(result_str, formula_widths[i]))
        end
        
        println(join(row_parts, " "))
    end
end

# Sobrecarga para una sola fórmula
function truth_table(f::FormulaPL)
    truth_table([f])
end


# Función que devuelve los modelos de una fórmula.
# En concreto, devuelve las asignaciones de variables que hacen que la fórmula sea verdadera.

function models(f::FormulaPL)::Vector{Valuation}
    # El orden de las variables se hereda del orden alfabético de sus nombres.
    # Por eso hemos sobrecargado isless para el tipo Var_PL para que compare por 
    # nombre.
    vars = sort(collect(vars_of(f)))
    n = length(vars)
    
    results = []
    
    # Generar todas las combinaciones posibles
    for i in 0:(2^n - 1)
        val = Valuation()   # Asignación de valores para las variables
        binary = string(i, base=2, pad=n)   # Genera una cadena binaria de n bits
        
        for (j, var) in enumerate(vars)     # Asignar valores a las variables leyendo la cadena binaria
            val[var] = binary[j] == '1'
        end
        
        result = evaluate(f, val)    # Evaluar la fórmula con la asignación actual
        if result
            push!(results, val)              # Guardar la asignación si la fórmula es verdadera
        end
    end
    
    return results  # Devolver las valoraciones que hacen verdadera la fórmula
end

# Función para verificar si una fórmula es tautología
function TAUT(f::FormulaPL)
    vars = collect(vars_of(f))
    n = length(vars)
    
    for i in 0:(2^n - 1)
        val = Valuation()
        binary = string(i, base=2, pad=n)
        
        for (j, var) in enumerate(vars)
            val[var] = binary[j] == '1'
        end
        
        if !evaluate(f, val)
            return false
        end
    end
    return true
end

# Función para verificar si una fórmula es contradicción
function UNSAT(f::FormulaPL)
    return TAUT(-f)
end

# Función para verificar si una fórmula es satisfactible
function SAT(f::FormulaPL)
    return !UNSAT(f)
end

# Función para verificar consecuencia lógica por reducción al absurdo
#       Γ ⊨ φ si y solo si Γ ∪ {¬φ} es insatisfactible
function LC(Γ::Vector{<:FormulaPL}, φ::FormulaPL)
    return UNSAT(⋀([Γ...,!φ])) # [Γ..., !φ] es la unión de Γ y ¬φ
end

# Función para verificar equivalencia lógica
function EQUIV(f1::FormulaPL, f2::FormulaPL)
    return TAUT(f1 ~ f2)
end

# Funciones para convertir a formas normales

# Función auxiliar para aplicar leyes de De Morgan
function demorgan(f::Neg_PL)
    operand = f.operand
    if isa(operand, And_PL)
        return (!(operand.left) | !(operand.right))
    elseif isa(operand, Or_PL)
        return (!(operand.left) & !(operand.right))
    else
        return f
    end
end

# Función para eliminar implicaciones y bicondicionales
function remove_imp(f::Var_PL)
    return f
end

function remove_imp(f::Neg_PL)
    return !(remove_imp(f.operand))
end

function remove_imp(f::And_PL)
    return (remove_imp(f.left) & remove_imp(f.right))
end

function remove_imp(f::Or_PL)
    return (remove_imp(f.left) | remove_imp(f.right))
end

function remove_imp(f::Imp_PL)
    # p → q ≡ ¬p ∨ q
    return (!(remove_imp(f.left)) | remove_imp(f.right))
end

function remove_imp(f::Iff_PL)
    # p ↔ q ≡ (p → q) ∧ (q → p) ≡ (¬p ∨ q) ∧ (¬q ∨ p)
    left_imp  = (!(remove_imp(f.left)) | remove_imp(f.right))
    right_imp = (!(remove_imp(f.right)) | remove_imp(f.left))
    return (left_imp & right_imp)
end

# Función para mover negaciones hacia adentro (Forma Normal Negativa)
function move_!_in(f::Var_PL)
    return f
end

function move_!_in(f::Neg_PL)
    f_in = f.operand
    if isa(f_in, Var_PL)
        return f
    elseif isa(f_in, Neg_PL)
        # Doble negación
        return move_!_in(f_in.operand)
    elseif isa(f_in, And_PL)
        # ¬(p ∧ q) ≡ ¬p ∨ ¬q
        return (move_!_in(!(f_in.left)) | move_!_in(!(f_in.right)))
    elseif isa(f_in, Or_PL)
        # ¬(p ∨ q) ≡ ¬p ∧ ¬q
        return (move_!_in(!(f_in.left)) & move_!_in(!(f_in.right)))
    else
        return !(move_!_in(f_in))
    end
end

function move_!_in(f::And_PL)
    return (move_!_in(f.left) & move_!_in(f.right))
end

function move_!_in(f::Or_PL)
    return (move_!_in(f.left) | move_!_in(f.right))
end

function move_!_in(f::Imp_PL)
    return move_!_in(remove_imp(f))
end

function move_!_in(f::Iff_PL)
    return move_!_in(remove_imp(f))
end

# Función para distribuir conjunciones sobre disyunciones (para CNF)
function dist_and_or(f::FormulaPL)
    if isa(f, Var_PL) || isa(f, Neg_PL)
        return f
    elseif isa(f, And_PL)
        left = dist_and_or(f.left)
        right = dist_and_or(f.right)
        return (left & right)
    elseif isa(f, Or_PL)
        left = dist_and_or(f.left)
        right = dist_and_or(f.right)
        
        # Si uno de los operandos es una conjunción, distribuir
        if isa(left, And_PL)
            # (p ∧ q) ∨ r ≡ (p ∨ r) ∧ (q ∨ r)
            return (dist_and_or(left.left | right) & dist_and_or(left.right | right))
        elseif isa(right, And_PL)
            # p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r)
            return (dist_and_or(left | right.left) & dist_and_or(left | right.right))
        else
            return (left | right)
        end
    elseif isa(f, Imp_PL) || isa(f, Iff_PL)
        # Primero eliminar implicaciones y luego distribuir
        return dist_and_or(remove_imp(f))
    end
end

# Función para distribuir disyunciones sobre conjunciones (para DNF)
function dist_or_and(f::FormulaPL)
    if isa(f, Var_PL) || isa(f, Neg_PL)
        return f
    elseif isa(f, Or_PL)
        left = dist_or_and(f.left)
        right = dist_or_and(f.right)
        return (left | right)
    elseif isa(f, And_PL)
        left = dist_or_and(f.left)
        right = dist_or_and(f.right)
        
        # Si uno de los operandos es una disyunción, distribuir
        if isa(left, Or_PL)
            # (p ∨ q) ∧ r ≡ (p ∧ r) ∨ (q ∧ r)
            return (dist_or_and(left.left & right) | dist_or_and(left.right & right))
        elseif isa(right, Or_PL)
            # p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)
            return (dist_or_and(left & right.left) | dist_or_and(left & right.right))
        else
            return (left & right)
        end
    elseif isa(f, Imp_PL) || isa(f, Iff_PL)
        # Primero eliminar implicaciones y luego distribuir
        return dist_or_and(remove_imp(f))
    end
end

# Función para convertir a Forma Normal Conjuntiva (CNF)
function to_CNF(f::FormulaPL)
    # Paso 1 y 2: Eliminar implicaciones y bicondicionales
    step1 = remove_imp(f)
    
    # Paso 3: Mover negaciones hacia adentro
    step2 = move_!_in(step1)
    
    # Paso 4a: Distribuir conjunciones sobre disyunciones
    step3 = dist_and_or(step2)
    
    return step3
end

# Función para convertir a Forma Normal Disyuntiva (DNF)
function to_DNF(f::FormulaPL)
    # Paso 1 y 2: Eliminar implicaciones y bicondicionales
    step1 = remove_imp(f)
    
    # Paso 3: Mover negaciones hacia adentro
    step2 = move_!_in(step1)
    
    # Paso 4b: Distribuir disyunciones sobre conjunciones
    step3 = dist_or_and(step2)
    
    return step3
end

# ==================== ALGORITMO DPLL ====================

# Tipo para representar literales en DPLL
struct Literal
    variable::Var_PL
    positive::Bool
end

function Base.show(io::IO, L::Literal)
    if L.positive
        print(io, L.variable)
    else
        print(io, "¬$(L.variable)")
    end
end

# Tipo para representar cláusulas (disyunción de literales)
struct Clause
    literals::Set{Literal}
end

function Base.show(io::IO, C::Clause)
    if isempty(C.literals)
        print(io, "☐")  # Cláusula vacía
    else
        print(io, "(", join(C.literals, " ∨ "), ")")
    end
end

# Función para convertir una fórmula a Forma Clausal (vía CNF)
function to_CF(f::FormulaPL)
    f_cnf = to_CNF(f)
    CF1 = clauses_of(f_cnf)
    CF2 = clean_CF(CF1)
    return CF2
end

function clauses_of(f::Var_PL)
    lit = Literal(f, true)
    return [Clause(Set([lit]))]
end

function clauses_of(f::Neg_PL)
    if isa(f.operand, Var_PL)
        lit = Literal(f.operand, false)
        return [Clause(Set([lit]))]
    else
        error("La fórmula no está en FNC correcta")
    end
end

function clauses_of(f::And_PL)
    left_Cs = clauses_of(f.left)
    right_Cs = clauses_of(f.right)
    return vcat(left_Cs, right_Cs)
end

function clauses_of(f::Or_PL)
    # Extraer todos los literales de la disyunción
    Ls = Set{Literal}()
    literals_of!(f, Ls)
    return [Clause(Ls)]
end

function literals_of!(f::Var_PL, Ls::Set{Literal})
    push!(Ls, Literal(f, true))
end

function literals_of!(f::Neg_PL, Ls::Set{Literal})
    if isa(f.operand, Var_PL)
        push!(Ls, Literal(f.operand, false))
    else
        error("La fórmula no está en FNC correcta")
    end
end

function literals_of!(f::Or_PL, Ls::Set{Literal})
    literals_of!(f.left, Ls)
    literals_of!(f.right, Ls)
end

function literals_of!(f::And_PL, Ls::Set{Literal})
    error("Encontrada conjunción dentro de disyunción - la fórmula no está en FNC")
end

# Función para obtener todas las variables de un conjunto de cláusulas
function vars_of(Cs::Vector{Clause})
    return [L.variable for C in Cs for L in C.literals]
end

# Función para aplicar una valoración de una variable a un conjunto de cláusulas
function apply_val(Cs::Vector{Clause}, var::Var_PL, value::Bool)
    new_Cs = Clause[]
    
    for C in Cs
        new_Ls = Set{Literal}()
        C_sat = false
        
        for L in C.literals
            if L.variable == var
                # Si el literal coincide con la asignación, la cláusula se satisface
                if L.positive == value
                    C_sat = true
                    break
                end
                # Si no coincide, el literal se hace falso y se elimina
            else
                # Mantener literales de otras variables
                push!(new_Ls, L)
            end
        end
        
        # Si la cláusula no se satisfizo completamente, agregar la versión simplificada
        if !C_sat
            push!(new_Cs, Clause(new_Ls))
        end
    end
    
    return new_Cs
end

# Función para verificar si una cláusula es una tautología
function is_tautology(C::Clause)
    # Una cláusula es tautología si contiene tanto un literal como su negación
    # Es decir, si existe una variable que aparece tanto positiva como negativa
    variables_seen = Dict{Var_PL, Set{Bool}}()
    
    for L in C.literals
        if haskey(variables_seen, L.variable)
            # Si ya hemos visto esta variable, agregar la polaridad actual
            push!(variables_seen[L.variable], L.positive)
        else
            # Primera vez que vemos esta variable
            variables_seen[L.variable] = Set([L.positive])
        end
        
        # Si encontramos que una variable tiene ambas polaridades, es tautología
        if length(variables_seen[L.variable]) > 1
            return true
        end
    end
    
    return false
end

# Función para limpiar la forma clausal eliminando tautologías
function clean_CF(Cs::Vector{Clause})
    return [C for C in Cs if !is_tautology(C)]
end

# Función alternativa más concisa usando filter
function clean_CF_alt(Cs::Vector{Clause})
    return filter(!is_tautology, Cs)
end

# Función para encontrar cláusulas unitarias (cláusulas con un solo literal)
function unit_clauses(Cs::Vector{Clause})
    return [first(C.literals) for C in Cs if length(C.literals) == 1]
end

# Función para encontrar literales puros (aparecen solo en forma positiva o negativa)
function pure_literals(Cs::Vector{Clause})
    pos_vars = Set{Var_PL}()
    neg_vars = Set{Var_PL}()

    # pos_vars = [L.variable for C in Cs for L in C.literals if L.positive]
    # neg_vars = [L.variable for C in Cs for L in C.literals if !L.positive]
    
    for C in Cs
        for L in C.literals
            if L.positive
                push!(pos_vars, L.variable)
            else
                push!(neg_vars, L.variable)
            end
        end
    end
    
    pure_Ls = Literal[]
    vars = union(pos_vars, neg_vars)
    
    # Literales positivos puros: pos_vars - neg_vars
    # Literales negativos puros: neg_vars - pos_vars
    for var in vars
        if var in pos_vars && !(var in neg_vars)
            push!(pure_Ls, Literal(var, true))
        elseif var in neg_vars && !(var in pos_vars)
            push!(pure_Ls, Literal(var, false))
        end
    end
    
    return pure_Ls
end

# Función principal del algoritmo DPLL
#    es recursiva, así que debe manejar los casos base y la propagación de 
#       literales unitarios y puros.

function DPLL(Cs::Vector{Clause}, val::Valuation = Valuation())
    # Caso base: si no hay cláusulas, la fórmula es satisfactible, con la valoración actual
    if isempty(Cs)
        return true, val
    end
    
    # Si hay una cláusula vacía, la fórmula es insatisfactible. No hay valoración que la satisfaga.
    for C in Cs
        if isempty(C.literals)
            return false, Valuation()
        end
    end
    
    # Propagación de literales unitarios
    for L in unit_clauses(Cs)
        new_val = copy(val)
        new_val[L.variable] = L.positive
        new_Cs = apply_val(Cs, L.variable, L.positive)
        return DPLL(new_Cs, new_val)
    end
    
    # Eliminación de literales puros
    pure_Ls = pure_literals(Cs)
    for L in pure_Ls
        new_val = copy(val)
        new_val[L.variable] = L.positive
        new_Cs = apply_val(Cs, L.variable, L.positive)
        return DPLL(new_Cs, new_val)
    end

    # Si no hay literales unitarios ni puros, procedemos a la división
    
    # Backtracking: elegir una variable y probar ambos valores
    vars = vars_of(Cs)
    unassigned_vars = setdiff(vars, keys(val))
    
    if isempty(unassigned_vars)
        return true, val
    end
    
    # Elegir una variable no asignada (puede ser aleatoria o la primera)
    #  Aquí se puede implementar una heurística más avanzada si se desea
    chosen_var = first(unassigned_vars)
    
    # Probar con valor true
    new_val_true = copy(val)
    new_val_true[chosen_var] = true
    new_Cs_true = apply_val(Cs, chosen_var, true)
    satisfiable_true, result_val = DPLL(new_Cs_true, new_val_true)
    
    if satisfiable_true
        return true, result_val
    end
    
    # Probar con valor false
    new_val_false = copy(val)
    new_val_false[chosen_var] = false
    new_Cs_false = apply_val(Cs, chosen_var, false)
    return DPLL(new_Cs_false, new_val_false)
end

# Función auxiliar para verificar satisfactibilidad usando DPLL
function DPLL_SAT(f::FormulaPL)
    try
        Cs = to_CF(f)
        satisfiable, val = DPLL(Cs)
        return satisfiable
    catch e
        println("Error al convertir a FNC o aplicar DPLL: $e")
        return false
    end
end

# Función para resolver y obtener la asignación satisfactoria
function DPLL(f::FormulaPL)
    Cs = to_CF(f)
    satisfiable, val = DPLL(Cs)
    return satisfiable, val        
end

function DPLL(fs::Vector{<:FormulaPL})
    Cs = vcat([to_CF(f) for f in fs]...)
    satisfiable, val = DPLL(Cs)
    return satisfiable, val
end

#= Explicación de la línea clave: vcat([to_CF(f) for f in fs]...)

1. [to_CF(f) for f in fs]
    * Es una lista por comprehensión.
    * Itera sobre cada elemento f en la colección fs.
    * Para cada f, aplica la función to_CF(f) que devuelve un vector de cláusulas.
    * El resultado es un vector de vectores: [[clausulas1], [clausulas2], [clausulas3], ...]

2. Los tres puntos ... (operador splat)
    * Desempaqueta el vector de vectores.
    * Convierte [[vec1], [vec2], [vec3]] en argumentos separados: [vec1], [vec2], [vec3]

3. vcat(...)
    * Concatena verticalmente todos los vectores desempaquetados.
    * Resultado final: un solo vector plano con todas las cláusulas.
=#

# ----------- Consecuencia lógica con DPLL -----------

# Función para verificar consecuencia lógica usando DPLL
# Γ ⊨ φ si y solo si Γ ∪ {¬φ} es insatisfactible
function DPLL_LC(Γ::Vector{<:FormulaPL}, φ::FormulaPL)
    satisfiable, val = DPLL([Γ..., -φ])
    return !satisfiable
end

# Función auxiliar para casos simples con sintaxis más conveniente
function DPLL_LC(F::FormulaPL, φ::FormulaPL)
    return DPLL_LC([F], φ)
end

end # module

