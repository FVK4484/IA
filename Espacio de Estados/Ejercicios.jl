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

function final3Jarras(s,g=nothing)
    s[2] == 2
end

sol3Jarras = BFS(sucesores3Jarras, [0,0,0], false; isgoal=final3Jarras)
ResumeSol(sol3Jarras, "BFS")


