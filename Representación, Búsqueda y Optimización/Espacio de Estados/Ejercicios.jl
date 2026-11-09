using Plots
include("Search.jl")

function ResumeSol(sol, metodo)
    if sol.status == :success
        println("-----------------------------------")
        println("La búsqueda con $metodo ha sido exitosa. El camino encontrado es\n")
        println("$(sol.path), con $(length(sol.path)) estados.\n")
        println("Durante la búsqueda, se han visitado $(sol.closedsetsize) nodos, que se han cerrado,\ny otros $(sol.opensetsize) que se han quedado abiertos.")
        println("-----------------------------------")
    else
        println("La búsqueda con $metodo no ha dado resultado")
    end
end

#= 
#################################
Problema de las 2 Jarras
#################################

Se tienen dos jarras de agua, una de 3 litros y otra de 4 litros sin escala de 
medición. Se desea obtener 2 litros de agua en la jarra de 4 litros. Las opera-
ciones válidas son: 
- llenar completamente cada una de las jarras, 
- vacíar completamente una de las jarras, 
- pasar agua de una jarra a otra (hasta que la primera se vacía o la segunda se llena). 
=#

# Los estados vendrán dados por un vector de componentes s = [j1,j2],
# donde j1 es el contenido de la Jarra 1, y j2 el de la Jarra 2

# La función sucesores calcula el conjunto de estados sucesores válidos
function sucesoresJarras(s)
    res = [
        [s[1],0], # Vaciar J2
        [0,s[2]], # Vaciar J1
        [s[1],4], # Llenar J2
        [3,s[2]]  # Llenar J1
    ]
    if s[1] > 0 && s[2] < 4
        trasvase = min(s[1], 4-s[2])
        push!(res,[s[1]-trasvase, s[2]+trasvase]) # Pasar de J1 a J2
    end
    if s[2] > 0 && s[1] < 3
        trasvase = min(s[2], 3-s[1])
        push!(res,[s[1]+trasvase, s[2]-trasvase]) # Pasar de J2 a J1
    end
    return res
end

function final(s,g)
    s[2] == 2
end

solJarras = BFS(sucesoresJarras, [0,0], false; isgoal=final)
ResumeSol(solJarras, "BFS")

#= 
#################################
Problema de las 3 Jarras
#################################

Se tienen 3 jarras de 12, 8 y 3 litros de capacidad y un grifo. Las operaciones que se 
pueden realizar con ellas son: 
- llenar cada una de las jarras de agua, 
- volcar el contenido de una en cualquier otra (hasta que una se vacía o la otra se llena), 
- o bien vaciar su contenido en el suelo. 
El objetivo es conseguir exactamente 1 litro en alguna de las jarras.
=#

function sucesores3Jarras(s)
    res = [
        [12,s[2],s[3]],
        [s[1],8,s[3]],
        [s[1],s[2],3],
        [0,s[2],s[3]],
        [s[1],0,s[3]],
        [s[1],s[2],0]
    ]
    if s[1] > 0 && s[2] < 8
        trasvase = min(s[1], 8-s[2])
        push!(res,[s[1]-trasvase, s[2]+trasvase, s[3]]) # Pasar de J1 a J2
    end
    if s[2] > 0 && s[1] < 12
        trasvase = min(s[2], 12-s[1])
        push!(res,[s[1]+trasvase, s[2]-trasvase, s[3]]) # Pasar de J2 a J1
    end
    if s[1] > 0 && s[3] < 3
        trasvase = min(s[1], 3-s[3])
        push!(res,[s[1]-trasvase, s[2], s[3]+trasvase]) # Pasar de J1 a J3
    end
    if s[3] > 0 && s[1] < 12
        trasvase = min(s[3], 12-s[1])
        push!(res,[s[1]+trasvase, s[2], s[3]-trasvase]) # Pasar de J3 a J1
    end
    if s[2] > 0 && s[3] < 3
        trasvase = min(s[2], 3-s[3])
        push!(res,[s[1], s[2]-trasvase, s[3]+trasvase]) # Pasar de J2 a J3
    end
    if s[3] > 0 && s[2] < 8
        trasvase = min(s[3], 8-s[2])
        push!(res,[s[1], s[2]+trasvase, s[3]-trasvase]) # Pasar de J3 a J2
    end
    return res
end

function final3Jarras(s,g)
    s[2] == 2
end

sol3Jarras = BFS(sucesores3Jarras, [0,0,0], false; isgoal=final3Jarras)
ResumeSol(sol3Jarras, "BFS")

#= 
#################################
Misioneros y Caníbales
#################################

Hay 3 misioneros y 3 caníbales a la orilla de un río. Tienen una canoa con capa-
cidad para dos personas como máximo. Se desea que los seis crucen el río, pero 
hay que considerar que no debe haber más caníbales que misioneros en ningún si-
tio porque entonces los caníbales se comerían a los misioneros. Además, la canoa 
siempre debe ser conducida por alguien (no puede cruzar el río sola). =#

# Estados: (misioneros_izq, canibales_izq, pos_barca)
# 	misioneros_der = 3 - misioneros_izq
# 	misioneros_izq = 3 - misioneros_der
# 	pos_barca = -1 (izquierda) | 1 (derecha)

function mover(s, mv)
    mi, ci, b = s
    m, c = mv
    return (mi + b * m, ci + b * c, -b)
end

function valido(s)
    m, c = s
    return (m >= c || m == 0) && (3 - m >= 3 - c || m == 3)
end

function sucesoresMisioneros(s)
    res = []
    # Posibles traslados (max 2 personas en la barca) en cualquier dirección
    moves = [(m, c) for m in 0:2 for c in 0:2 if 1 <= m + c <= 2]
    for mv in moves
        s1 = mover(s, mv)  # Aplicación del movimiento mv
        # Comprobación de que el estado es válido
        if valido(s1)
            push!(res, s1)
        end
    end
    return res
end
                    
solMis = BFS(sucesoresMisioneros, (3,3,-1), (0,0,1))
ResumeSol(solMis,"BFS")

#=
#################################
Ej. 9: Problema de las bolsas
#################################

Se pretende encontrar una manera de distribuir un conjunto de objetos, 
O={o1,…,on}, en bolsas usando el menor número posible de ellas. 
Cada objeto tiene un peso asociado, pi, y las bolsas tienen un límite de 
peso máximo soportado, B.

El estado será una tupla: (bolsas_actuales, objetos_pendientes)
- bolsas_actuales: Vector con el peso acumulado en cada bolsa abierta.
- objetos_pendientes: Vector con los pesos de los objetos que faltan por guardar.
=#

function sucesoresBolsas(s, B)
    bolsas, pendientes = s
    res = []
    
    # Si no hay objetos pendientes, no hay movimientos posibles (estamos en la meta)
    if isempty(pendientes)
        return res
    end
    
    # Cogemos SIEMPRE el primer objeto de la lista para evitar explosión combinatoria.
    # El orden de los factores no altera el producto final de las bolsas.
    objeto = pendientes[1]
    resto_pendientes = pendientes[2:end]
    
    # Opción 1: Intentar meter el objeto en cada una de las bolsas ya abiertas
    for i in 1:length(bolsas)
        if bolsas[i] + objeto <= B
            nuevas_bolsas = copy(bolsas)
            nuevas_bolsas[i] += objeto
            
            # Ordenamos las bolsas de mayor a menor peso para evitar estados duplicados simétricos
            # (Ej: [3, 2] es el mismo estado real que [2, 3])
            sort!(nuevas_bolsas, rev=true) 
            
            push!(res, (nuevas_bolsas, resto_pendientes))
        end
    end
    
    # Opción 2: Abrir una bolsa completamente nueva
    nuevas_bolsas_ext = copy(bolsas)
    push!(nuevas_bolsas_ext, objeto)
    sort!(nuevas_bolsas_ext, rev=true)
    push!(res, (nuevas_bolsas_ext, resto_pendientes))
    
    # Usamos unique para limpiar posibles estados repetidos generados en este paso
    return unique(res)
end

function finalBolsas(s, g)
    bolsas, pendientes = s
    return isempty(pendientes)
end

# --- EJEMPLO DE USO ---
# Límite de peso por bolsa
B = 10 
# Objetos a empaquetar
objetos = [4, 8, 1, 4, 2, 1] 

# Estado inicial: cero bolsas abiertas, todos los objetos pendientes
estado_inicial = (Int[], objetos)

# Ejecución de la búsqueda (nota cómo pasamos s -> sucesoresBolsas(s, B) para inyectar B)
solBolsas = BFS(s -> sucesoresBolsas(s, B), estado_inicial, false; isgoal=finalBolsas)

ResumeSol(solBolsas, "BFS (Bolsas)")