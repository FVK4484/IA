# ==============
# Librería PL.jl
# ==============

include("PL.jl")
using .PropositionalLogic


#=
El fichero `Pl.jl` contiene el módulo `PropositionalLogic` con las definiciones de las estructuras y funciones que usaremos para aplicar los algoritmos que se estudian en el tema de Lógica. En el módulo hay muchas funciones auxiliares y de uso interno, y solo se exportan aquellas que puedan servir directamente para la construcción de soluciones tal y como se presenta en la teoría.

Se aconseja consultar el contenido del fichero `Pl.jl` para explorar cómo se diseñan las estructuras de datos y se manipulan los objetos de la Lógica Proposicional, así como las implementaciones de los algoritmos presentados en el temario del curso. Debe tenerse presente que la versión de la librería es una simplificación abreviada de una librería mayor que cubre un curso completo de Lógica Computacional (y que se encuentra en desarrollo).

En este documento vamos a hacer el siguiente recorrido para demostrar el uso de este módulo:
 1. Definición de fórmulas y su manipulación.
 2. Aplicación de los algoritmos implementados en ejemplos pequeños que pueden realizarse a mano.
 3. Resolución de problemas de mayor envergadura usando una representación adecuada en la Lógica Proposicional.
=#

# Uso Básico
# ==========

# Funcionalidades básicas que ofrece la librería: declarar variables proposicionales, definir fórmulas, definir valoraciones, calcular tablas de verdad, etc.

## Declaración de Variables y Definición de Fórmulas
## -------------------------------------------------

#=
Si queremos usar una **variable proposicional** debemos declararla previamente, las normas que sigue esta declaración son las habituales para los símbolos de Julia (por ejemplo, que no se pise con ninguna palabra clave del lenguaje, que comience por un símbolo admitido, etc.). Más adelante veremos cómo crear conjuntos parametrizados de variables para resolver problemas más complejos, en este momento las iremos declarando explícitamente.

Además de su declaración directa por medio del tipo Var_PL, disponemos de la función `vars`, que permite declarar varias variables a la vez, devolviendo un vector con las variables declaradas que pueden ser asignadas a variables de Julia. Por ejemplo:
=#

s = Var_PL("s")             # Declaración directa
p, q, r = vars(:p, :q, :r)  # Declaración múltiple conveniente

#=
Una fórmula puede usar las variables proposicionales declaradas, y en su definición se puede usar una notación infija similar a la representación matemática habitual. Para que resulte cómoda su introducción por teclado, para las conectivas se han elegido símbolos habituales en lenguajes de programación y que están relacionados con los operadores booleanos:

 `&` (conjunción, ∧), 
 `|` (disyunción, ∨), 
 `>` (implicación, →), 
 `~` (doble implicación, ⟷), 
 `!` o `-` (negación, ¬). 
=#

# Un ejemplo de fórmula (escrita de forma prefija e infija) es:

And_PL(p, q)    # Forma prefija
p & q           # Forma infija

# Observa que la fórmula se muestra en el terminal con una notación cercana a la matemática, usando los símbolos habituales (∧, ∨, ¬, →, ⟷) en lugar de los operadores que se usan para su introducción por teclado.

# Por supuesto, podemos asignar las fórmulas construidas a nuevas variables de Julia para poder combinarlas creando fórmulas más complejas:

f1 = p & q
f2 = p | q
f3 = p > q
f4 = p ~ q
f5 = !(p & q)
f6 = -(p & q)
f7 = f1 & -f2 > f3 | (f4 & f5)

# Y también disponemos de algunas funciones auxiliares que permiten mostrar el árbol de formación de una fórmula, o el conjunto de subfórmulas que la componen:

formation_tree(f7)
subformulas(f7)

## Tablas de verdad y evaluaciones
## -------------------------------

# Una valoración se estructura como un diccionario que asigna valores booleanos a variables proposicionales

# const Valuation = Dict{Var_PL, Bool}

# Así, una valoración concreta se podría construir con (recuerda que en un diccionario no importa el orden):

val1 = Valuation(p => true, q => false, r => true)  # También: val1 = Valuation(p => 1, q => 0, r => 1)

# La función `evaluate`, dada una fórmula y una valoración, evalúa la fórmula sobre esos valores, y gracias a las capacidades de Julia, podemos convertir las valoraciones en funciones que evalúan las fórmulas, tal y como hacemos en la teoría. Por ejemplo:

evaluate(f1, val1)  # Usando evaluate
val1(f1)            # Usando la valoración como función
val1(f1) == evaluate(f1, val1)
val1(f2)
val1(f3)

# Lo que permite disponer de forma sencilla (e ineficiente) para calcular las tablas de verdad de fórmulas proposicionales o de conjuntos de fórmulas proposicionales. Por ejemplo:

truth_table(f7)
truth_table([p > q, q > r, p > r])

# Modelos, Tautologías, Satisfactibilidad, Equivalencia y Consecuencia
# ====================================================================

# De forma similar, tenemos una función, `models`, que genera todos los modelos (valoraciones) que satisfacen una fórmula dada como un vector de valoraciones. Por ejemplo:

models(f2)

# De forma trivial, se pueden definir funciones para verificar si una fórmula es **Tautología** (TAUT), **Satisfactble** (SAT) o **Contradicción** (**Insatisfactible**, UNSAT):

f_taut = p | -p
f_cont = p & -p
f_sat  = p & q

for f in [f_taut, f_cont, f_sat]
    println("Fórmula: $f")
    truth_table(f)
    println("  ¿Es tautología? $(TAUT(f))")
    println("  ¿Es contradicción? $(UNSAT(f))")
    println("  ¿Es satisfactible? $(SAT(f))")
    println("  Modelos: ", models(f))
    println()
end

# ... ver si dos fórmulas son equivalentes:

g1 = !(p & q)
g2 = -p | -q
println("¿Son equivalentes '$(g1)' y '$(g2)'? $(EQUIV(g1, g2))")

# O si una fórmula es consecuencia lógica de un conjunto de fórmulas:

Γ = [p > q, p]
φ = q
println("¿Se sigue '$φ' de las premisas $Γ? $(LC(Γ, φ))")

# Para facilitar el trabajo con conjuntos de fórmulas, tenemos definidos dos operadores, `⋀` (\bigwedge) y `⋁` (\bigvee), que permiten definir la conjunción y disyunción de un conjunto de fórmulas. Por ejemplo:

Γ = [p > q, p, q]
⋀(Γ)
⋁(Γ)

# Y, por tanto, podemos usar estos operadores para comprobar la consecuencia lógica a partir de, por ejemplo, las tablas de verdad o de la función TAUT:
Γ = [p > q, p]
φ = q
truth_table(⋀(Γ) > φ)
TAUT(⋀(Γ) > φ)

Γ = [p > q, q > r]
φ = q
LC(Γ, φ)

# Formas Normales y Formas Clausales
# ==================================

#=
Disponemos de funciones para obtener formas normales equivalentes a una fórmula dada: 
   **Forma Normal Conjuntiva** (CNF) 
   **Forma Normal Disyuntiva** (DNF)
   **Forma Clausal** (CF) (eliminando las cláusulas que son tautologías)
=#

f7 = (p > q) & (q > r)
println("Fórmula original: $f7")
println("FNC: $(to_CNF(f7))")
println("FND: $(to_DNF(f7))")

f8 = (p ~ q) ~ (q | p)
println("Fórmula original: $f8")
println("Forma clausal:")
Cl = to_CF(f8)

# Algoritmo DPLL
# ==============

#=
Con todo lo anterior ya es fácil dar una implementación para el algoritmo DPLL, que trabaja sobre formas clausales.

La función `DPLL` aplicada a una forma clausal (un vector de cláusulas) devuelve un par:

(true, val):        si es satisfactible, y donde val es una valoración que hace cierta la fórmula, 
(false, nothing):   si no es satisfactible

Es importante destacar el siguiente hecho: **La implementación que tenemos de DPLL no devuelve todas las ramas posibles, sino que para en la primera rama que salga satisfactible (y devuelve el modelo asociado)**.
=#

f8 = (p ~ q) ~ (q | p)
Cl = to_CF(f8)
DPLL(Cl)

# Gracias al **despacho múltiple** de Julia, podemos usar el mismo nombre de función si queremos usar DPLL para una función que no está expresada en forma clausal (internamente, primero convierte la fórmula a forma clausal y entonces aplica el algoritmo DPLL estándar), o un conjunto de fórmulas (internamente, primero convierte cada fórmula a forma clausal y luego une todas las cláusulas en un solo conjunto). Por ejemplo:

DPLL(f8)        # Es lo mismo que: DPLL(to_CF(f8))
DPLL([f7, f8])  # Es lo mismo que: DPLL(to_CF(f8) ⋃ to_CF(f7))

# Obviamente, al igual que hicimos anteriormente, podemos aprovechar el algoritmo de satisfactibilidad que devuelve DPLL para verificar la consecuencia lógica, tautología o insatisfactibilidad. Por ejemplo, si queremos ver si una fórmula es consecuencia lógica de un conjunto de premisas, podemos usar DPLL con el dato de entrada correcto (y la interpretación correcta de su respuesta) o podemos usar una función de conveniencia, `DPLL_LC`, que hace todo el trabajo de forma automática (la función `DPLL_LC` recibe un conjunto de fórmulas Γ y una fórmula φ, y devuelve true si φ es consecuencia lógica de Γ, y false en caso contrario):

Γ = [
    p > q,          # p → q
    q > r,          # q →  r
    p               # p
]

φ = r      # Por tanto r

# Las siguientes comprobaciones son equivalentes:
DPLL(-(⋀(Γ) > φ))       # ¿Es Tautología ⋀(Γ) -> φ? = ¿Es insatisfacible ¬(⋀(Γ) -> φ)?
DPLL(⋀([Γ..., -φ]))     # ¿Es insatisfacible ⋀ (Γ ⋃ {¬φ})?
DPLL([Γ..., -φ])        # ¿Es insatisfactible Γ ⋃ {¬φ}?
DPLL_LC(Γ, φ)           # ¿Es consecuencia lógica φ de Γ? Es una forma abreviada de lo anterior, ya que, 
                        # internamente, construye la fórmula ⋀(Γ) ∧ ¬φ y verifica su insatisfactibilidad.

# Observa que tenemos varios métodos para ver si una fórmula es SAT, TAUT, comprobar la consecuencia lógica, etc. (los que usan DPLL, los que usan la definición directa, las tablas de verdad,...). Todos son correctos, pero el uso de DPLL es, en general, más eficiente.

# Problemas de Argumentación
# ==========================

# Con todas las herramientas anteriores, ya podemos abordar problemas de cierta complejidad, como los que se presentan a continuación.


## Problema 1
## ----------

#=
Formaliza en el lenguaje de la Lógica Proposicional, y comprueba la validez, de la siguiente argumentación:

Si continúa la investigación, surgirán nuevas evidencias. Si surgen nuevas evidencias, entonces varios dirigentes se verán implicados. Si varios dirigentes están implicados, los periódicos dejarán de hablar del caso. Si la continuación de la investigación implica que los periódicos dejen de hablar del caso, entonces, el surgimiento de nuevas evidencias implica que la investigación continúa. La investigación no continúa. Por tanto, no surgirán nuevas evidencias.

Siguiendo una elección de variables similar a la del ejemplo anterior, podríamos llegar a la conclusión tomando como variables proposicionales:

    p : Continúa la investigación

    q : Surgen nuevas evidencias

    r : Varios dirigentes se verán implicados

    s : Los periódicos dejarán de hablar del caso.

De donde podemos sacar las premisas siguientes:

{ p → q , q → r , r → s , (p → s) → (q → p) , ¬p }

y la conclusión: ¬q

El problema de argumentación se ha traducido, por tanto, en:

¿ { p → q , q → r , r → s , (p → s) → (q → p) , ¬p } ⊧ ¬q ?

Vamos a hacer una implementación en el que el nombre de las variables y sus símbolos sean más representativos:

=#

# 1º Definir variables preposicionales.
continua_investigacion, surgen_nuevas_evidencias, varios_dirigentes_implicados, periodicos_dejan_de_hablar = vars(:continua_investigacion, :surgen_nuevas_evidencias, :varios_dirigentes_implicados, :periodicos_dejan_de_hablar)

# 2º Generar premisas.
premisas = [
    continua_investigacion > surgen_nuevas_evidencias,
    surgen_nuevas_evidencias > varios_dirigentes_implicados,
    varios_dirigentes_implicados > periodicos_dejan_de_hablar,
    (continua_investigacion > periodicos_dejan_de_hablar) > (surgen_nuevas_evidencias > continua_investigacion),
    !continua_investigacion
]

conclusion = -surgen_nuevas_evidencias

println("=== Análisis de validez del argumento ===")
is_valid = DPLL_LC(premisas, conclusion)
println("¿Es el argumento válido? $is_valid")

# En la relación de problemas propuesta en la página de la asignatura puedes encontrar otros problemas de argumentación similares que, por su tamaño, puedes realizar a mano y verificar con la librería.

# Fórmulas Parametrizadas
# =======================

#=
En los siguientes problemas trabajaremos con fórmulas parametrizadas, donde el número de variables que intervienen puede depender de la instancia del problema y, por tanto, no pueden ser declaradas a mano de forma sencilla (también puede ocurrir que el número de variables sea muy grande y prefiramos automatizar el proceso). Para ello, usaremos bucles y estructuras de datos de Julia (normalmente, arrays y diccionarios en sus diversas variedades) para crear las variables y las fórmulas que las relacionan.

Se debe tener presente que los problemas pocas veces se describen completamente y algunas de sus restricciones se basan en un conocimiento a priori no explícito que ha de ser representado también formalmente. A esta información imprescindible pero no explícita se le suele llamar "marco" o "contexto" del problema.

Vamos a intentar seguir patrones similares a los que se usan en la teoría para definir las fórmulas, de forma que el paso de la definición matemática a la implementación sea lo más directo posible.

Los pasos que seguiremos para cada problema serán:
   1. Definir las variables proposicionales que intervienen en el problema, normalmente como una colección (vector, matriz, diccionario, etc.) de variables.
   2. Definir las fórmulas, o conjunto de fórmulas, que representan las restricciones y contexto (marco) del problema, normalmente como una conjunción de varias fórmulas parciales.
   3. Usar DPLL para resolver el problema, normalmente verificando la satisfactibilidad de la conjunción de todas las restricciones.
   4. Interpretar la solución obtenida (si la hay) para el problema concreto.
   5. Visualizar la solución (opcional).
   6. Probar con varias instancias del problema.
=#

## Función "exactamente una"
## -------------------------

#=
Comenzamos definiendo una función que significa "exactamente uno" (∃_{=1} = ∃!) y que será de utilidad para los problemas siguientes. Esta función recibe un conjunto de variables proposicionales y devuelve la fórmula proposicional que se verifica cuando exactamente una, y solo una, de las variables de entrada es verdadera:

 ∃_{=1} (v_1,...,v_n) = ( ∃_{>=1} (v_1,...,v_n) )   ∧   ( ∃_{<=1} (v_1,...,v_n) )
 ∃_{=1} (v_1,...,v_n) = ( ⋁_{i=1..n} v_i )          ∧   ( ⋀_{i=1..n-1} ⋀_{j=i+1..n} ¬(v_i ∧ v_j) )

Observa que en el proceso de creación de la fórmula se usan los operadores `⋀` y `⋁` para crear las conjunciones y disyunciones de varias fórmulas, y que internamente hacemos uso de dos "operadores" intermedios: "al menos una" (∃_{>=1}) y "a lo sumo una" (∃_{<=1}).

Por ejemplo: ∃_{=1} (v1,v2,v3) =  (v1 ∨ v2 ∨ v3)   ∧   ( ¬(v1 ∧ v2) ∧ ¬(v1 ∧ v3) ∧ ¬(v2 ∧ v3) )
=# 

function Ex1(vps::Vector{Var_PL})
    if length(vps) == 0
        error("No se puede crear restricción exactamente_una con lista vacía")
    elseif length(vps) == 1
        return vps[1]
    end

    # Al menos una
    al_menos_una = ⋁(vps)

    # A lo sumo una: para cada par de variable vi,vj ∈ vps, se debe verificar: ¬(vi ∧ vj)
    a_lo_sumo_una = ⋀([-(vps[i] & vps[j]) for i in 1:length(vps) for j in (i+1):length(vps)])
    return al_menos_una & a_lo_sumo_una
end

## Problema de las N Reinas
## ------------------------

#=
Vamos a suponer que tenemos una matriz de variables proposicionales r(i,j) con la interpretación:

                r(i,j) = 1 ⟺ hay una reina en la casilla (i,j)

Función que crea la fórmula general que contiene todas las restricciones que definen una solución del problema:

   * Una única reina por fila:         
                ⋀_{i=1..n} ∃_{=1}(r_{i,j})_{j=1..n}

   * Una única reina por columna:      
                ⋀_{j=1..n} ∃_{=1}(r_{i,j})_{i=1..n}

   * A lo sumo, una reina por diagonal principal: 
                ⋀_{i=1..n} ∃_{≤ 1} (D^1_{i}), 
        donde D^1_i es el conjunto de casillas de la diagonal principal i-ésima.

   * A lo sumo, una reina por diagonal secundaria: 
                ⋀_{i=1..n} ∃_{≤ 1} (D^2_{i}), 
        donde D^2_i es el conjunto de casillas de la diagonal secundaria i-ésima.
=#

function formula_n_reinas(n::Int, r)
    # Restricción 1: Exactamente una reina por fila
    Fila(i) = Ex1([r[i, j] for j in 1:n])
    Filas = [Fila(i) for i in 1:n]

    # Restricción 2: Exactamente una reina por columna
    Columna(j) = Ex1([r[i, j] for i in 1:n])
    Columnas = [Columna(j) for j in 1:n]

    # Restricción 3: A lo sumo una reina por diagonal principal (↘)
    Diag = Dict{Int, FormulaPL}()
    idx = 1
    for k in -(n-2):(n-2)  # Diagonales paralelas a la principal
        D(k) = [ r[row, col] for row in 1:n, col in 1:n if row == col - k ]
        dk = length(D(k))
        if dk > 1
            Diag[idx] = ⋀([!(D(k)[i] & D(k)[j]) for i in 1:dk for j in (i+1):dk])
            idx = idx + 1
        end
    end
    Diags1 = [F for (key, F) in Diag]

    # Restricción 4: A lo sumo una reina por diagonal secundaria (↙)
    Diag = Dict{Int, FormulaPL}()
    idx = 1
    for k in 2:(2*n)   # Diagonales paralelas a la secundaria
        D(k) = [ r[row, col] for row in 1:n, col in 1:n if col == k - row]
        dk = length(D(k))
        if dk > 1
            Diag[idx] = ⋀([!(D(k)[i] & D(k)[j]) for i in 1:dk for j in (i+1):dk])
            idx = idx + 1
        end
    end
    Diags2 = [F for (key, F) in Diag]

    return vcat(Filas, Columnas, Diags1, Diags2)
end

# Vamos a probarla para el caso N=4

r = Matrix{Var_PL}(undef, 4, 4)
for i in 1:4, j in 1:4
    r[i,j] = Var_PL("r($i,$j)")
end
formula_4_reinas = formula_n_reinas(4, r)
DPLL(formula_4_reinas)

# Además de resolverlo, vamos a visualizar las soluciones, por lo que vamos a definir una función que es capaz de leer la solución obtenida (que será una valoración) y representa en un tablero las reinas colocadas:

function visual_sol_n_reinas(solution_dict, n::Int, r)
    println("\nVisualización del tablero $n×$n:")
    tablero = fill("▢", n, n)

    # Procesar la solución y proyectarla en las posiciones del tablero
    for i in 1:n, j in 1:n
        v = r[i,j]
        if solution_dict[v] == 1
            tablero[i, j] = "♕"  # Colocar reina en la posición (i, j)
        end
    end

    # Mostrar el tablero
    for i in 1:n
        println(join(tablero[i, :], " "))
    end

    # Contar reinas colocadas
    queen_count = count(==("♕"), tablero)
    println("\nReinas colocadas: $queen_count/$n")

    return tablero
end

# El procedimiento completo para crear una solución se puede dar en forma de función (que depende de N):

function test_n_reinas(n::Int)
    println("Probando N-reinas para N=$n")
    r = Matrix{Var_PL}(undef, n, n)
    for i in 1:n, j in 1:n
        r[i,j] = Var_PL("r($i,$j)")
    end
    # Generar la fórmula para N reinas y almacenarlas en r
    formula = formula_n_reinas(n, r)
    sat, sol = DPLL(formula)

    if sat == false
        println("No hay solución para $n reinas.")
    else
        println("Solución encontrada:")
        visual_sol_n_reinas(sol, n, r)
    end
end

test_n_reinas(10);

## Problema Sudoku
## ---------------

#=
El problema del Sudoku se puede resolver de forma similar, pero teniendo en cuenta ahora que las variables proposicionales serán de la forma $S(i,j,k)$ con la interpretación:

            S(i,j,k) = 1 ⟺ en la casilla (i,j) está el número k

Las restricciones que tenemos que considerar son:

   * Hay exactamente un número en cada casilla:
            ⋀_{i=1..9} ⋀_{j=1..9} ∃_{=1} ({S(i,j,k)}_{k=1..9})

   * Hay exactamente una ocurrencia de cada número por cada fila:
            ⋀_{i=1..9} ⋀_{k=1..9} ∃_{=1} ({S(i,j,k)}_{j=1..9})

   * Hay exactamente una ocurrencia de cada número por cada columna:
            ⋀_{j=1..9} ⋀_{k=1..9} ∃_{=1} ({S(i,j,k)}_{i=1..9})

   * Hay exactamente una ocurrencia de cada número por cada bloque 3×3:
            ⋀_{B ∈ Bloque} ⋀_{(i,j) ∈ B} ∃_{=1} ({S(i,j,k)}_{k=1..9})

   * Además, habrá que añadir las celdas que vienen ya prefijadas/rellenas.
=#

function formula_sudoku(S, Rellenas::Vector{Tuple{Int, Int, Int}} = [])
    # Exactamente un número por celda
    celda(i,j) = Ex1([S[i, j, k] for k in 1:9])
    Celdas = [celda(i, j) for i in 1:9 for j in 1:9]

    # Exactamente una ocurrencia de cada número por fila
    Fila(i,k) = Ex1([S[i, j, k] for j in 1:9])
    Filas = [Fila(i, k) for i in 1:9 for k in 1:9]
    
    # Exactamente una ocurrencia de cada número por columna
    Columna(j,k) = Ex1([S[i, j, k] for i in 1:9])
    Columnas = [Columna(j, k) for j in 1:9 for k in 1:9]
    
    # Exactamente una ocurrencia de cada número por bloque de Sudoku
    Bloque(i, j, k) = Ex1([S[i + bi, j + bj, k] for bi in 0:2 for bj in 0:2])
    Bloques = [Bloque(i, j, k) for i in [1 4 7] for j in [1 4 7] for k in 1:9]

    Rellenas = [S[i, j, k] for (i, j, k) in Rellenas]

    return vcat(Celdas, Filas, Columnas, Bloques, Rellenas)
end

function visual_sol_sudoku(solution_dict, S)
    tablero = fill(0, 9, 9)

    # Procesar la solución y proyectarla en las posiciones del tablero
    for i in 1:9, j in 1:9, k in 1:9
        v = S[i,j,k]
        if solution_dict[v] == 1
            tablero[i, j] = k  # Colocar reina en la posición (i, j)
        end
    end

    # Mostrar el tablero
    for i in 1:9
        println(join(tablero[i, :], " "))
    end

    return tablero
end

function test_sudoku()
    # Variables S(i,j,k) - número k en posición (i,j)
    S = Array{Var_PL}(undef, 9, 9, 9)
    for i in 1:9, j in 1:9, k in 1:9
        S[i,j,k] = Var_PL("S($i,$j,$k)")
    end
    println("Resolviendo Sudoku...")
    formula = formula_sudoku(S, [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4),
                                 (1, 5, 5), (1, 6, 6), (1, 7, 7), (1, 8, 8),
                                 (1, 9, 9)]);
    sat, sol = DPLL(formula)

    if sat == false
        println("No hay solución para el Sudoku.")
    else
        println("Solución encontrada:")
        visual_sol_sudoku(sol, S)
    end
end

test_sudoku();

## Cloreado de Mapas
## -----------------

#=
Un problema similar es el de coloreado de mapas, donde se supone que tenemos un conjunto de paises (P, una colección de cadenas, por ejemplo), indicando qué pares de países comparten frontera (F) y K colores (que podemos considerarlos como números consecutivos). Entonces las restricciones de un coloreado válido serían:

   * Todos los países deben estar coloreados con exactamente un único color:
                ⋀_{p ∈ P} ∃_{=1} ({C(p,c)}_{c=1..K})

   * Dos países que compartan frontera deben tener coloreados distintos:
                ⋀_{(p_1,p_2) ∈ F} ⋀_{c=1..K} ¬( C(p_1,c) ∧ C(p_2,c) )
=#

function formula_coloreado_mapas(C, paises::Vector{String}, fronteras::Vector{Tuple{String, String}}, num_colores::Int = 4)

    # Restricción 1: Cada país debe tener exactamente un color
    pais_color(pais) = Ex1([C[(pais, color)] for color in 1:num_colores])
    Colores_Paises = [pais_color(pais) for pais in paises]

    # Restricción 2: Países que comparten frontera no pueden tener el mismo color
    Fronteras = [ !(C[(p1, c)] & C[(p2, c)]) for (p1, p2) in fronteras for c in 1:num_colores ]

    return vcat(Colores_Paises, Fronteras)
end

function visual_sol_coloreado_mapas(C, solution_dict, paises::Vector{String}, num_colores::Int)
    println("\nVisualización del Coloreado de Mapas:")

    # Diccionario para mapear números de colores a nombres
    nombres_colores = Dict(1 => "Rojo", 2 => "Azul", 3 => "Verde", 4 => "Amarillo",
                          5 => "Naranja", 6 => "Violeta", 7 => "Rosa", 8 => "Marrón")
    coloreado = Dict{String, Int}()

    # Procesar la solución y proyectarla en las posiciones del tablero
    for p in paises, c in 1:num_colores
        v = C[p,c]
        if solution_dict[v] == 1
            coloreado[p] = c
        end
    end

    # Mostrar el coloreado
    println("┌─────────────────────────────┬──────────────┐")
    println("│            País             │    Color     │")
    println("├─────────────────────────────┼──────────────┤")

    for pais in sort(paises)
        if haskey(coloreado, pais)
            color_num = coloreado[pais]
            color_nombre = get(nombres_colores, color_num, "Color $color_num")
            println("│ $(rpad(pais, 27)) │ $(rpad(color_nombre, 12)) │")
        else
            println("│ $(rpad(pais, 27)) │ $(rpad("Sin asignar", 12)) │")
        end
    end
    println("└─────────────────────────────┴──────────────┘")

    # Verificar que se usaron los colores mínimos
    colores_usados = Set(values(coloreado))
    println("\nColores utilizados: $(length(colores_usados))")
    println("Colores: $(join([nombres_colores[c] for c in sort(collect(colores_usados))], ", "))")

    return coloreado
end

function test_coloreado_mapas(paises::Vector{String}, fronteras::Vector{Tuple{String, String}}, num_colores::Int = 4)
    # Generar variables C(pais, color) - país tiene color
    C = Dict{Tuple{String, Int}, Var_PL}()
    for pais in paises, color in 1:num_colores
        C[(pais, color)] = Var_PL("C($pais,$color)")
    end
    println("Probando coloreado de mapas con $(length(paises)) países y $num_colores colores")
    println("Países: $(join(paises, ", "))")
    println("Fronteras: $(length(fronteras)) conexiones")

    formula = formula_coloreado_mapas(C, paises, fronteras, num_colores)
    sat, sol = DPLL(formula)

    if sat == false
        println("No hay solución para el coloreado con $num_colores colores.")
        return false
    else
        println("¡Solución encontrada!")
        visual_sol_coloreado_mapas(C, sol, paises, num_colores)
        return true
    end
end

# Ejemplo 1: Mapa simple de 4 países
begin
    println("=== EJEMPLO 1: Mapa Simple ===")
    paises_simple = ["España", "Francia", "Italia", "Alemania"]
    fronteras_simple = [("España", "Francia"), ("Francia", "Italia"),
                    ("Francia", "Alemania"), ("Italia", "Alemania")]
    test_coloreado_mapas(paises_simple, fronteras_simple, 3)
end

# Ejemplo 2: Mapa más complejo (grafo completo K5 - requiere 5 colores)
begin
    println("\n=== EJEMPLO 2: Grafo Completo K5 ===")
    paises_k5 = ["A", "B", "C", "D", "E"]
    fronteras_k5 = [(paises_k5[i], paises_k5[j]) for i in 1:5 for j in (i+1):5]
    test_coloreado_mapas(paises_k5, fronteras_k5, 4)
    test_coloreado_mapas(paises_k5, fronteras_k5, 5)
end

# Ejemplo 3: Algunos países europeos reales
begin
    println("\n=== EJEMPLO 3: Europa Occidental ===")
    paises_europa = ["España", "Francia", "Alemania", "Italia", "Suiza", "Austria", "Bélgica"]
    fronteras_europa = [
        ("España", "Francia"),
        ("Francia", "Alemania"), ("Francia", "Italia"), ("Francia", "Suiza"), ("Francia", "Bélgica"),
        ("Alemania", "Austria"), ("Alemania", "Suiza"), ("Alemania", "Bélgica"),
        ("Italia", "Suiza"), ("Italia", "Austria"),
        ("Suiza", "Austria")
    ]
    test_coloreado_mapas(paises_europa, fronteras_europa, 4)
end

## Acertijo de Einstein
## --------------------

#=
El famoso **acertijo de Einstein** (también conocido como **Zebra Puzzle**) nos dice que hay 5 casas en fila, cada una de ellas con diferentes características, pero no nos dicen qué característica corresponde a cada casa:
 - Nacionalidades: Británico, Sueco, Danés, Noruego, Alemán
 - Colores: Rojo, Verde, Amarillo, Azul, Blanco
 - Mascotas: Perro, Pájaro, Gato, Caballo, Pez (Pez Cebra, de ahí el nombre del puzzle)
 - Bebidas: Té, Café, Leche, Cerveza, Agua
 - Cigarrillos: Pall Mall, Dunhill, Blend, Blue Master, Prince

Pero sí nos dan información acerca de posibles relaciones y restricciones existente:
 1. El británico vive en la casa roja
 2. El sueco tiene un perro
 3. El danés bebe té
 4. La casa verde está inmediatamente a la izquierda de la casa blanca
 5. El dueño de la casa verde bebe café
 6. La persona que fuma Pall Mall tiene un pájaro
 7. El dueño de la casa amarilla fuma Dunhill
 8. El hombre en la casa del medio bebe leche
 9. El noruego vive en la primera casa
 10. El hombre que fuma Blend vive al lado del que tiene un gato
 11. El hombre que tiene un caballo vive al lado del que fuma Dunhill
 12. El hombre que fuma Blue Master bebe cerveza
 13. El alemán fuma Prince
 14. El noruego vive al lado de la casa azul
 15. El hombre que fuma Blend vive al lado del que bebe agua

 El objetivo es indicar claramente qué características tiene cada casa.
=#

function formula_acertijo_einstein()
    # Definir los conjuntos de características
    nacionalidades = ["Britanico", "Sueco", "Danes", "Noruego", "Aleman"]
    colores = ["Rojo", "Verde", "Amarillo", "Azul", "Blanco"]
    mascotas = ["Perro", "Pajaro", "Gato", "Caballo", "Pez"]
    bebidas = ["Te", "Cafe", "Leche", "Cerveza", "Agua"]
    cigarrillos = ["PallMall", "Dunhill", "Blend", "BlueMaster", "Prince"]

    # Variables: X(casa, caracteristica) - la casa tiene esa característica
    N = Dict{Tuple{Int, String}, Var_PL}()  # Nacionalidades
    C = Dict{Tuple{Int, String}, Var_PL}()  # Colores
    M = Dict{Tuple{Int, String}, Var_PL}()  # Mascotas
    B = Dict{Tuple{Int, String}, Var_PL}()  # Bebidas
    S = Dict{Tuple{Int, String}, Var_PL}()  # Cigarrillos (Smoking)

    # Generar todas las variables
    for casa in 1:5
        for nac in nacionalidades
            N[(casa, nac)] = Var_PL("N($casa,$nac)")
        end
        for col in colores
            C[(casa, col)] = Var_PL("C($casa,$col)")
        end
        for mas in mascotas
            M[(casa, mas)] = Var_PL("M($casa,$mas)")
        end
        for beb in bebidas
            B[(casa, beb)] = Var_PL("B($casa,$beb)")
        end
        for cig in cigarrillos
            S[(casa, cig)] = Var_PL("S($casa,$cig)")
        end
    end

    # RESTRICCIONES BÁSICAS:
    # Cada casa tiene exactamente una nacionalidad, color, mascota, bebida y cigarrillo
    Basicas = vcat([Ex1([N[(casa, nac)] for nac in nacionalidades]) for casa in 1:5],
                   [Ex1([C[(casa, col)] for col in colores]) for casa in 1:5],
                   [Ex1([M[(casa, mas)] for mas in mascotas]) for casa in 1:5],
                   [Ex1([B[(casa, beb)] for beb in bebidas]) for casa in 1:5],
                   [Ex1([S[(casa, cig)] for cig in cigarrillos]) for casa in 1:5])
      
    # Cada característica aparece en exactamente una casa
    Caracteristicas = vcat([Ex1([N[(casa, nac)] for casa in 1:5]) for nac in nacionalidades],
                           [Ex1([C[(casa, col)] for casa in 1:5]) for col in colores],
                           [Ex1([M[(casa, mas)] for casa in 1:5]) for mas in mascotas],
                           [Ex1([B[(casa, beb)] for casa in 1:5]) for beb in bebidas],
                           [Ex1([S[(casa, cig)] for casa in 1:5]) for cig in cigarrillos])

    # RESTRICCIONES DEL ACERTIJO
    rest = Vector{Vector{FormulaPL}}(undef, 15)  # Vector para las restricciones del acertijo

    # 1. El británico vive en la casa roja
    rest[1] = [ (N[(casa, "Britanico")] ~ C[(casa, "Rojo")]) for casa in 1:5 ]

    # 2. El sueco tiene un perro
    rest[2] = [ (N[(casa, "Sueco")] ~ M[(casa, "Perro")]) for casa in 1:5 ]

    # 3. El danés bebe té
    rest[3] = [ (N[(casa, "Danes")] ~ B[(casa, "Te")]) for casa in 1:5 ]

    # 4. La casa verde está inmediatamente a la izquierda de la casa blanca
    rest[4] = vcat([ (C[(casa, "Verde")] > C[(casa+1, "Blanco")]) for casa in 1:4 ],
                   [ !C[(1, "Blanco")] ],    # La casa blanca no puede estar en la posición 1
                   [ !C[(5, "Verde")]  ])    # La casa verde no puede estar en la posición 5

    # 5. El dueño de la casa verde bebe café
    rest[5] = [ (C[(casa, "Verde")] ~ B[(casa, "Cafe")]) for casa in 1:5 ]

    # 6. La persona que fuma Pall Mall tiene un pájaro
    rest[6] = [ (S[(casa, "PallMall")] ~ M[(casa, "Pajaro")]) for casa in 1:5 ]

    # 7. El dueño de la casa amarilla fuma Dunhill
    rest[7] = [ (C[(casa, "Amarillo")] ~ S[(casa, "Dunhill")]) for casa in 1:5 ]

    # 8. El hombre en la casa del medio bebe leche
    rest[8] = [ B[(3, "Leche")] ]  # La casa del medio (casa 3) bebe leche

    # 9. El noruego vive en la primera casa
    rest[9] = [ N[(1, "Noruego")] ]  # El noruego vive en la primera casa
    
    # 10. El hombre que fuma Blend vive al lado del que tiene un gato
    r10 = FormulaPL[]
    for casa in 1:5
        vecinos_gato = FormulaPL[]
        if casa > 1
            push!(vecinos_gato, M[(casa-1, "Gato")])
        end
        if casa < 5
            push!(vecinos_gato, M[(casa+1, "Gato")])
        end
        if length(vecinos_gato) > 0
            push!(r10, (S[(casa, "Blend")] > ⋁(vecinos_gato)))
        end
    end
    rest[10] = r10  # Asegurar que se cumple para todas las casas

    # 11. El hombre que tiene un caballo vive al lado del que fuma Dunhill
    r11 = FormulaPL[]
    for casa in 1:5
        vecinos_dunhill = FormulaPL[]
        if casa > 1
            push!(vecinos_dunhill, S[(casa-1, "Dunhill")])
        end
        if casa < 5
            push!(vecinos_dunhill, S[(casa+1, "Dunhill")])
        end
        if length(vecinos_dunhill) > 0
            push!(r11, (M[(casa, "Caballo")] > ⋁(vecinos_dunhill)))
        end
    end
    rest[11] = r11  # Asegurar que se cumple para todas las casas

    # 12. El hombre que fuma Blue Master bebe cerveza
    rest[12] = [ (S[(casa, "BlueMaster")] ~ B[(casa, "Cerveza")]) for casa in 1:5 ]

    # 13. El alemán fuma Prince
    rest[13] = [ (N[(casa, "Aleman")] ~ S[(casa, "Prince")]) for casa in 1:5 ]

    # 14. El noruego vive al lado de la casa azul
    # Como el noruego está en casa 1, la casa azul debe estar en casa 2
    rest[14] = [ C[(2, "Azul")] ]

    # 15. El hombre que fuma Blend vive al lado del que bebe agua
    r15 = FormulaPL[]
    for casa in 1:5
        vecinos_agua = FormulaPL[]
        if casa > 1
            push!(vecinos_agua, B[(casa-1, "Agua")])
        end
        if casa < 5
            push!(vecinos_agua, B[(casa+1, "Agua")])
        end
        if length(vecinos_agua) > 0
            push!(r15, (S[(casa, "Blend")] > ⋁(vecinos_agua)))
        end
    end
    rest[15] = r15  # Asegurar que se cumple para todas las casas

    Acertijo = vcat([rest[i] for i in 1:15]...)

    return vcat(Basicas, Caracteristicas, Acertijo)
end

function visual_sol_acertijo_einstein(solution_dict)
    println("\n" * "="^71)
    println("                SOLUCIÓN DEL ACERTIJO DE EINSTEIN")
    println("="^71)

    # Crear estructura para almacenar la solución
    casas = [Dict{String, String}() for _ in 1:5]

    # Procesar la solución
    for (var, value) in solution_dict
        if value == 1
            # Parsear variables del tipo X(casa,caracteristica)
            if occursin(r"^[NCMBS]\(\d+,\w+\)$", var.name)
                tipo = var.name[1]
                contenido = var.name[3:end-1]
                partes = split(contenido, ",")
                casa = parse(Int, partes[1])
                caracteristica = string(partes[2])

                categoria = Dict('N' => "Nacionalidad", 'C' => "Color",
                               'M' => "Mascota", 'B' => "Bebida", 'S' => "Cigarrillo")[tipo]
                casas[casa][categoria] = caracteristica
            end
        end
    end

    # Mostrar la tabla
    println("┌──────┬──────────────┬───────────┬──────────┬──────────┬─────────────┐")
    println("│ Casa │ Nacionalidad │   Color   │ Mascota  │  Bebida  │ Cigarrillo  │")
    println("├──────┼──────────────┼───────────┼──────────┼──────────┼─────────────┤")
    for casa in 1:5
        nac = get(casas[casa], "Nacionalidad", "?")
        col = get(casas[casa], "Color", "?")
        mas = get(casas[casa], "Mascota", "?")
        beb = get(casas[casa], "Bebida", "?")
        cig = get(casas[casa], "Cigarrillo", "?")

        println("│  $casa   │ $(rpad(nac, 12)) │ $(rpad(col, 9)) │ $(rpad(mas, 8)) │ $(rpad(beb, 8)) │ $(rpad(cig, 11)) │")
    end

    println("└──────┴──────────────┴───────────┴──────────┴──────────┴─────────────┘")

    # Encontrar y destacar quién tiene el pez
    println("\n" * "="^30 * " RESPUESTA " * "="^30 )
    for casa in 1:5
        if get(casas[casa], "Mascota", "") == "Pez"
            nacionalidad = get(casas[casa], "Nacionalidad", "Desconocido")
            println("         ¡El $nacionalidad tiene el PEZ! (Casa #$casa)")
            break
        end
    end
    println("="^71)

    return casas
end

function resolver_acertijo_einstein()
    println("Resolviendo el Acertijo de Einstein...")
    println("   (También conocido como 'Zebra Puzzle')")

    formulas = formula_acertijo_einstein()
    clausulas = vcat([to_CF(f) for f in formulas]...)

    println("\n   ⏳ Generando fórmula SAT...")
    variables = reduce(union, map(vars_of, formulas));
    println("      Variables generadas: $(length(variables))")
    println("      Subfórmulas generadas: $(length(formulas))")
    println("      Cláusulas generadas: $(length(clausulas))")
    println("\n   ⏳ Ejecutando solucionador DPLL...")
    sat, sol = DPLL(clausulas);
    println("      Resultado SAT: $sat")

    casas = visual_sol_acertijo_einstein(sol)
end

resolver_acertijo_einstein()