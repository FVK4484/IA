using Random
using Plots 
include("Opt.jl")

#=
J1 (3 ptos). Haciendo uso de la librería de Templado Simulado, implementa 
una solución al siguiente problema (en el fichero de descarga se muestra 
un ejemplo del formato de datos que sirven de entrada):

En una empresa de transporte se recibe cada mañana una lista con los paquetes 
que deben enviarse (cada paquete tiene un peso y un destino), y la lista de 
vehículos disponibles (cada uno con una capacidad máxima de carga, coste de 
uso, y destinos accesibles). El objetivo es distribuir los paquetes en los 
vehículos de forma que se minimice el coste necesario (no tienen porqué usarse 
todos los vehículos).
=#

paquetes_ejemplo = [(10, 1), (20, 2), (15, 3), (25, 1), (30, 2), (5, 3)]
vehiculos_ejemplo = [(50, 50, [1, 2]), (40, 60, [2, 3]), (60, 120, [1, 2, 3])]

function energia(estado, paquetes, vehiculos)
    
   vehiculos_usados = Set{Int}()
   cap_usada = zeros(length(vehiculos))
   penalizacion = 0.0
   
   for (i, v) in enumerate(estado)
    peso, dest = paquetes[i]
    cap, _, dests = vehiculos[v]

    if !(dest in dests)
        penalizacion += 1000
    end

    cap_usada[v] += peso
    if cap_usada[v] > cap
        penalizacion += 500
    end
    push!(vehiculos_usados, v)
   end

   return sum(vehiculos[v][2] for v in vehiculos_usados) + penalizacion
end

function vecino(estado, paquetes, vehiculos)
   nuevo = copy(estado)
   nuevo[rand(1:length(estado))] = rand(1:length(vehiculos))
   return nuevo
end

function solucion(paquetes, vehiculos)
   s0 = rand(1:length(vehiculos), length(paquetes))
   E(s) = energia(s, paquetes, vehiculos)
   V(s) = vecino(s, paquetes, vehiculos)
   return SA(E, s0, 100, V, 0.001, 1.0, true)
end

solucion(paquetes_ejemplo, vehiculos_ejemplo)

#=
J2 (3 ptos). Haciendo uso de la librería de Templado Simulado, implementa 
una solución al siguiente problema (en el fichero de descarga se muestra 
un ejemplo del formato de datos que sirven de entrada):

Un ayuntamiento tiene que adjudicar 10 proyectos de obra mediante concurso 
público. Se han presentado al concurso 5 empresas, dando presupuestos para 
cada uno de los diez proyectos. La adjudicación debe realizarse de manera que 
a cada empresa solo se le concedan dos proyectos. Encuentra la adjudicación de 
menor precio.
=#

presupuestos = rand(100:500, 10, 5)

function energia_ayuntamiento(x, matriz_precios)
    coste = 0
    for i in 1:10
        empresa = x[i]
        coste += matriz_precios[i, empresa]
    end
    penalizacion = 0
    for j in 1:5
        proyectos_asignados = 0
        for empresa_en_proyecto in x
            if empresa_en_proyecto == j
                proyectos_asignados += 1
            end
        end
        penalizacion += abs(proyectos_asignados - 2) * 10000
    end
    return coste + penalizacion
end

function vecino_ayuntamiento(x)
    nuevo_x = copy(x)
    proyecto_al_azar = rand(1:10)
    empresa_actual = nuevo_x[proyecto_al_azar]
    nueva_empresa = rand(1:5)
    while nueva_empresa == empresa_actual
        nueva_empresa = rand(1:5)
    end
    nuevo_x[proyecto_al_azar] = nueva_empresa
    return nuevo_x
end

function solucion(matriz_precios)
   s0 = rand(1:5, 10)
   E(s) = energia_ayuntamiento(s, matriz_precios)
   V(s) = vecino_ayuntamiento(s)
   # Ejecutamos el Templado Simulado
   return SA(E, s0, 100, V, 0.001, 1.0, true)
end

mejor_adjudicacion, coste_total = solucion(presupuestos)

print("La mejor adjudicación es: $mejor_adjudicacion")
print("El coste total es: $coste_total")

#=
J3 (3 ptos). Haciendo uso de la librería de Templado Simulado, implementa 
una solución al siguiente problema (en el fichero de descarga se muestra 
un ejemplo del formato de datos que sirven de entrada):

Un ganadero tiene un rebaño de ovejas. Cada oveja tiene un peso y se vende 
por un precio prefijado. El ganadero dispone de un camión que es capaz de 
cargar un peso máximo. Su problema es seleccionar una colección de ovejas 
para llevarlas al mercado de ganado en el camión de manera que se maximice 
el precio total de las ovejas transportadas, sin superar el peso total 
soportado por el camión.
=#

# Número de ovejas y tara del camión
N = rand(1:50)
T = rand(500:1000)
# Pesos y precios de las N ovejas
pesos = rand(20:80, N)
precios = rand(50:300, N)
println("$N ovejas, Tara = $T\n  Pesos: $pesos\n  Precios: $precios")

# Un estado es una lista binaria de longitud N indicando qué oveja se vende o no

# Un estado vecino consiste en cambiar el estado de una oveja al azar
function vecino_ovj(s)
    s1 = copy(s)
    i = rand(1:N)
    s1[i] = 1 - s[i]
    s1
end

# La energía (bondad) de un estado depende de si nos pasamos o no del peso 
# total, y del precio total conseguido:
function energia_ovj(s)
    peso_total = sum(s .* pesos)
    precio_total = sum(s .* precios)
    penalizacion = peso_total > T ? (peso_total- T) : 0
    return -precio_total + 100 * penalizacion
end

s0 = rand(0:1,N);
sol, en = SA(energia_ovj, s0, 10, vecino_ovj, 0.001, 1.0, true)	
print("\nPeso final: $(sum(sol .* pesos))")
print("\nPrecio final: $(sum(sol .* precios))")

#=
J4 (3 ptos). Haciendo uso de la librería de Templado Simulado, implementa 
una solución al siguiente problema (en el fichero de descarga se muestra 
un ejemplo del formato de datos que sirven de entrada):

Dado un grafo G=(V,E) sin ciclos, donde cada arista lleva asociado un peso 
entero positivo, se trata de encontrar una partición de V en dos conjuntos 
disjuntos, V_0 y V_1 de tal manera que la suma de los pesos de las aristas
que tienen un extremo en V_0 y el otro extremo en V_1, sea máxima (problema 
max-cut) o mínima (problema min-cut, en este caso hay que poner restricciones 
para que no sea válido tomar alguno de los dos conjuntos muy pequeño).
=#

# N: Número de nodos
# M: Número de aristas
# K: Número de colores
# Generamos un grafo N-M al azar
N =rand(3:20)
M = rand(2:(N*(N-1)÷2))
nodos = collect(1:N);

todas_aristas = [(i,j) for i in 1:(N-1) for j in (i+1):N]
aristas = (shuffle(todas_aristas))[1:M]
Pesos_aristas = rand(1:100,M)

println("Grafo: N = $N nodos, M = $M aristas")
println("   Aristas:  $aristas")
println("   Pesos Aristas: $Pesos_aristas")

# Un estado es un vector binario de longitud N que indica si el nodo 
# está en V0 o en V1

# Un estado sucesor se consigue cambiando un nodo al azar de V0 a V1, 
# y viceversa
function vecino_cut(s)
    s1 = copy(s)
    i = rand(1:N)
    s1[i] = 1 - s[i]
    s1
end

# La energía de la solución se calcula como la suma de las aristas entre 
# V0 y V1
function energia_cut(s)
    total_en = 0
    for i in 1:M
        n1, n2 = aristas[i]
        if s[n1] != s[n2]
            total_en += Pesos_aristas[i]
        end
    end
    return -total_en
end

# A partir de un estado (vector 0/1), extrae los nodos de V0 y V1
function extraer_Vs(s)
    # Combinamos los índices y el estado
    R = zip(nodos,s)
    # A partir del estado, filtramos los índices que nos interesan
    V0 = [x[1] for x in R if x[2] != 0]
    V1 = [x[1] for x in R if x[2] == 0]
    # Mostramos los resultados
    println("\nEstado: $s")
    println("   V0 = $V0")
    println("   V1 = $V1")
    println("   Peso Corte =  $(-energia_cut(s))")		
    return V0, V1
end

s0 = rand(0:1,N)
extraer_Vs(s0)

sol, en = SA(energia_cut, s0, 10, vecino_cut, 0.001, 1.0, true)
extraer_Vs(sol)