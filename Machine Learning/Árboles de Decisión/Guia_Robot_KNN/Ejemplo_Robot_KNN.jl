#=====================================================================
CLONACIÃ“N DE COMPORTAMIENTO CON KNN + EXTRACCIÃ“N DE REGLAS CON ID3
=====================================================================

Un robot circular se mueve en una arena 10x10 con obstÃ¡culos y debe llegar
a una meta. Tiene 3 sensores de distancia (izquierda, frente, derecha) y
conoce el Ã¡ngulo hacia la meta. En cada paso elige una acciÃ³n:
girar a la izquierda, seguir recto o girar a la derecha.

Pasos:
  1. Un "experto" (reglas escritas a mano) genera demostraciones
     (observaciÃ³n â†’ acciÃ³n).
  2. Se aprende una polÃ­tica con kNN (aprendizaje por imitaciÃ³n).
  3. Se compara el Ã©xito del robot segÃºn k y segÃºn nÂº de demostraciones.
  4. Se "destila" la polÃ­tica kNN en un Ã¡rbol ID3 â†’ reglas interpretables.
=====================================================================#

#=
QuÃ© hace

- Mundo: un robot circular en una arena de 10Ã—10 con 4 obstÃ¡culos y una meta en 
   (9, 9). Sus observaciones son tres sensores de distancia (izquierda, frente, 
   derecha) y el Ã¡ngulo hacia la meta. Las acciones son girar a la izquierda, 
   seguir recto o girar a la derecha.
- Experto: unas pocas reglas escritas a mano genera 3000 demostraciones. Las 
   ejecuto con un 15 % de acciones aleatorias para que haya mÃ¡s variedad, pero 
   guardo siempre la acciÃ³n del experto.
- PolÃ­tica kNN: aprende de esas demostraciones con knn, normalizando con 
   ajustar_normalizador y normalizar de ML.jl.
- Reglas: etiqueto con la polÃ­tica kNN las observaciones de las demos y las de 
   30 recorridos del propio robot. Con ellas entreno un Ã¡rbol con id3_train y lo 
   muestro con show_arbol. Mido su fidelidad con matriz_confusion.
- GrÃ¡ficos: trayectorias del experto, de kNN con k=1, de kNN con k=11 y del 
   Ã¡rbol (verde es meta, rojo choque). AdemÃ¡s, Ã©xito y choques en funciÃ³n de k 
   y del nÃºmero de demos.

Resultados, con 40 inicios de prueba:

   PolÃ­tica	            Ã‰xito	   Choques
   --------             -----    -------
   Experto	            87.5 %	12.5 %
   kNN, 3 <= k <= 11	   90 %	   10 %
   kNN, k = 1           85 %	   15 %
   kNN, k = 21	         85 %	   15 %
   kNN, k = 51	         87.5 %	12.5 %
   Ãrbol ID3	         80 %	   20 %

- Datos: el Ã©xito sube con el nÃºmero de demostraciones: 32 % con 100, 60 % con 
   300, 77 % con 1000 y 90 % con 3000.
- Ãrbol: imita a kNN con un 93.5 % de fidelidad (47 hojas), pero conduce algo 
   peor que el kNN del que procede.
- Experto: el robot kNN iguala o supera al experto porque no reproduce sus 
   errores. Con 40 episodios la diferencia es pequeÃ±a y poco significativa.

Nota: Para que el Ã¡rbol quedara legible usÃ© un mÃ­nimo de 60 muestras por hoja. 
      Antes de eso el experto chocaba demasiado y endurecÃ­ sus umbrales de 
      distancia.
=#

include("../../Fundamentos Teóricos/ML.jl")
include("../../Modelos Básicos//ML_Basicos.jl")
include("../ID3.jl")
using Random, Statistics, DataFrames, Plots

# ---------------------------------------------------------------------
# Mundo
# ---------------------------------------------------------------------

const ARENA = 10.0
const OBSTACULOS = [(3.0, 3.0, 1.0), (6.0, 6.5, 1.2), (3.5, 7.5, 0.8), (7.0, 2.5, 0.9)]  # (x, y, radio)
const META = (9.0, 9.0)
const RADIO_ROBOT = 0.2
const PASO = 0.15
const GIRO = 0.2
const ALCANCE = 3.0
const ANG_SENSOR = 0.6
const ACCIONES = ["izquierda", "recto", "derecha"]

colision(x, y) = x < RADIO_ROBOT || y < RADIO_ROBOT || x > ARENA - RADIO_ROBOT ||
                 y > ARENA - RADIO_ROBOT ||
                 any(hypot(x - ox, y - oy) < r + RADIO_ROBOT for (ox, oy, r) in OBSTACULOS)

en_meta(x, y) = hypot(x - META[1], y - META[2]) < 0.5

# Distancia al primer obstÃ¡culo/pared en la direcciÃ³n Î¸ (avance por pasos)
function sensor(x, y, Î¸)
    for d in 0.1:0.1:ALCANCE
        colision(x + d * cos(Î¸), y + d * sin(Î¸)) && return d
    end
    return ALCANCE
end

angulo_relativo(a) = mod(a + Ï€, 2Ï€) - Ï€

# ObservaciÃ³n: [izquierda, frente, derecha, Ã¡ngulo a la meta]
function observar(x, y, Î¸)
    ang = angulo_relativo(atan(META[2] - y, META[1] - x) - Î¸)
    return [sensor(x, y, Î¸ + ANG_SENSOR), sensor(x, y, Î¸), sensor(x, y, Î¸ - ANG_SENSOR), ang]
end

# ---------------------------------------------------------------------
# Experto (reglas manuales)
# ---------------------------------------------------------------------

function experto(o)
    izq, fre, der, ang = o
    if fre < 1.4
        return izq > der ? "izquierda" : "derecha"
    elseif izq < 0.9
        return "derecha"
    elseif der < 0.9
        return "izquierda"
    elseif ang > 0.15
        return "izquierda"
    elseif ang < -0.15
        return "derecha"
    end
    return "recto"
end

function mover(x, y, Î¸, accion)
    Î¸ += GIRO * (accion == "izquierda" ? 1 : accion == "derecha" ? -1 : 0)
    return x + PASO * cos(Î¸), y + PASO * sin(Î¸), Î¸
end

# ---------------------------------------------------------------------
# SimulaciÃ³n
# ---------------------------------------------------------------------

function inicio_aleatorio(rng)
    while true
        x, y = rand(rng) * ARENA, rand(rng) * ARENA
        if !colision(x, y) && hypot(x - META[1], y - META[2]) > 4
            return x, y, rand(rng) * 2Ï€ - Ï€
        end
    end
end

# Ejecuta una polÃ­tica desde `inicio`. Devuelve trayectoria, resultado y,
# si `registro` es true, las parejas (observaciÃ³n, acciÃ³n del experto).
# `ruido` = probabilidad de ejecutar una acciÃ³n aleatoria (da mÃ¡s variedad a las demos).
function simular(politica, inicio; maxpasos=400, ruido=0.0, registro=false, rng=Random.default_rng())
    x, y, Î¸ = inicio
    tray = [(x, y)]
    demos = Tuple{Vector{Float64}, String}[]
    for _ in 1:maxpasos
        o = observar(x, y, Î¸)
        registro && push!(demos, (o, experto(o)))
        a = rand(rng) < ruido ? rand(rng, ACCIONES) : politica(o)
        x, y, Î¸ = mover(x, y, Î¸, a)
        push!(tray, (x, y))
        colision(x, y) && return (tray = tray, resultado = :choque, demos = demos)
        en_meta(x, y) && return (tray = tray, resultado = :meta, demos = demos)
    end
    return (tray = tray, resultado = :tiempo, demos = demos)
end

# Genera al menos n demostraciones del experto (con ruido en la ejecuciÃ³n)
function generar_demostraciones(n; ruido=0.15, seed=1)
    rng = MersenneTwister(seed)
    demos = Tuple{Vector{Float64}, String}[]
    while length(demos) < n
        append!(demos, simular(experto, inicio_aleatorio(rng); ruido, registro=true, rng).demos)
    end
    return demos[1:n]
end

# EvalÃºa una polÃ­tica en los mismos inicios de prueba
function evaluar(politica, inicios)
    res = [simular(politica, s).resultado for s in inicios]
    return (exito = count(==(:meta), res) / length(res),
            choque = count(==(:choque), res) / length(res),
            tiempo = count(==(:tiempo), res) / length(res))
end

# PolÃ­tica kNN con normalizaciÃ³n z-score de las observaciones
function entrenar_politica_knn(demos, k)
    X = reduce(vcat, [o' for (o, _) in demos])
    norm = ajustar_normalizador(X)
    datos = [(vec(normalizar(o', norm)), a) for (o, a) in demos]
    return o -> knn(k, datos, vec(normalizar(reshape(o, 1, :), norm)))
end

# ---------------------------------------------------------------------
# Experimento 1: efecto de k
# ---------------------------------------------------------------------

rng_test = MersenneTwister(2024)
inicios_test = [inicio_aleatorio(rng_test) for _ in 1:40]

println("Experto: ", evaluar(experto, inicios_test))

demos = generar_demostraciones(3000)
println("\nDemostraciones: ", length(demos), "  acciones: ",
        Dict(a => count(d -> d[2] == a, demos) for a in ACCIONES))

ks = [1, 3, 5, 11, 21, 51]
res_k = map(ks) do k
    r = evaluar(entrenar_politica_knn(demos, k), inicios_test)
    println("kNN k = ", lpad(k, 2), " â†’ Ã©xito ", round(100r.exito, digits=1), " %, choques ",
            round(100r.choque, digits=1), " %, atascado ", round(100r.tiempo, digits=1), " %")
    r
end

# ---------------------------------------------------------------------
# Experimento 2: efecto del nÂº de demostraciones (k = 5)
# ---------------------------------------------------------------------

tamaÃ±os = [100, 300, 1000, 3000]
res_n = map(tamaÃ±os) do n
    r = evaluar(entrenar_politica_knn(demos[1:n], 5), inicios_test)
    println("n = ", lpad(n, 4), " demos â†’ Ã©xito ", round(100r.exito, digits=1), " %")
    r
end

# ---------------------------------------------------------------------
# Experimento 3: destilar kNN en un Ã¡rbol ID3 (reglas)
# ---------------------------------------------------------------------

k_destilar = 5
pol_knn = entrenar_politica_knn(demos, k_destilar)

# Etiquetamos con la polÃ­tica kNN las observaciones de las demos mÃ¡s otras
# vistas por el propio robot kNN (estados que el experto no visitÃ³).
rng_dest = MersenneTwister(7)
obs_extra = reduce(vcat, [[o for (o, _) in simular(pol_knn, inicio_aleatorio(rng_dest);
                                                  ruido=0.15, registro=true, rng=rng_dest).demos]
                          for _ in 1:30])
estados = vcat([o for (o, _) in demos], obs_extra)
etiquetas = [pol_knn(o) for o in estados]

df = DataFrame(izquierda = [o[1] for o in estados], frente = [o[2] for o in estados],
               derecha = [o[3] for o in estados], angulo = [o[4] for o in estados],
               accion = etiquetas)
df_train, df_test = train_test_split(df; train_ratio = 0.8, seed = 1)

atributos = [:izquierda, :frente, :derecha, :angulo]
arbol = id3_train(df_train, atributos, :accion, 60)   # mÃ­nimo de muestras alto â†’ Ã¡rbol pequeÃ±o y legible

fidelidad = mean(ID3_predict(arbol, df_test) .== df_test.accion)
println("\nFidelidad del Ã¡rbol respecto a kNN (test): ", round(100fidelidad, digits=1), " %")
println(matriz_confusion(df_test.accion, ID3_predict(arbol, df_test)))
println(show_arbol(arbol))

pol_arbol(o) = ID3_predict(arbol, DataFrame(izquierda = o[1], frente = o[2], derecha = o[3], angulo = o[4])[1, :])
res_arbol = evaluar(pol_arbol, inicios_test)
println("Ãrbol ID3 conduciendo â†’ Ã©xito ", round(100res_arbol.exito, digits=1), " %, choques ",
        round(100res_arbol.choque, digits=1), " %")

# ---------------------------------------------------------------------
# GrÃ¡ficas
# ---------------------------------------------------------------------

function dibujar_mundo(titulo)
    p = plot(xlim = (0, ARENA), ylim = (0, ARENA), aspect_ratio = 1, title = titulo,
             legend = false, titlefontsize = 9)
    for (ox, oy, r) in OBSTACULOS
        t = range(0, 2Ï€, length = 50)
        plot!(p, ox .+ r .* cos.(t), oy .+ r .* sin.(t), seriestype = :shape, c = :gray, lÎ += 0)
    end
    scatter!(p, [META[1]], [META[2]], marker = :star5, markersize = 10, c = :gold)
    return p
end

colores_res = Dict(:meta => :seagreen, :choque => :red, :tiempo => :orange)

function panel_trayectorias(politica, titulo, inicios)
    p = dibujar_mundo(titulo)
    for s in inicios
        r = simular(politica, s)
        plot!(p, first.(r.tray), last.(r.tray), c = colores_res[r.resultado], lw = 1.2)
        scatter!(p, [s[1]], [s[2]], c = :black, markersize = 3)
    end
    return p
end

inicios_vis = inicios_test[1:8]
p_exp = panel_trayectorias(experto, "Experto", inicios_vis)
p_k1 = panel_trayectorias(entrenar_politica_knn(demos, 1), "kNN k = 1", inicios_vis)
p_k11 = panel_trayectorias(entrenar_politica_knn(demos, 11), "kNN k = 11", inicios_vis)
p_arb = panel_trayectorias(pol_arbol, "Ãrbol ID3 (destilado)", inicios_vis)

p_k = plot(ks, [100 .* [r.exito for r in res_k] 100 .* [r.choque for r in res_k]],
           marker = :circle, label = ["Ã©xito" "choque"], xscale = :log10,
           xlabel = "k", ylabel = "% de episodios", title = "Efecto de k", titlefontsize = 9)
p_n = plot(tamaÃ±os, [100 .* [r.exito for r in res_n] 100 .* [r.choque for r in res_n]],
           marker = :circle, label = ["Ã©xito" "choque"], xscale = :log10,
           xlabel = "nÂº de demostraciones", ylabel = "% de episodios",
           title = "Efecto de los datos (k = 5)", titlefontsize = 9)

fig = plot(p_exp, p_k1, p_k11, p_arb, p_k, p_n, layout = (3, 2), size = (900, 1300))
display(fig)

# ---------------------------------------------------------------------
# GIF animado: varias polÃ­ticas conduciendo desde el mismo inicio
# ---------------------------------------------------------------------

"""
    animar_robot(politicas, inicio; archivo="robot.gif", fps=20, salto=2)

Graba un GIF con las `politicas` (vector de `nombre => funciÃ³n`) ejecutadas desde el
mismo `inicio = (x, y, Î¸)`. Cada una deja su estela; al terminar se marca el resultado.
Ejemplo:
    animar_robot(["Experto" => experto, "kNN k=5" => entrenar_politica_knn(demos, 5)],
                 inicios_test[3]; archivo = "robot.gif")
"""
function animar_robot(politicas, inicio; archivo = "robot.gif", fps = 20, salto = 2)
    sims = [simular(pol, inicio) for (_, pol) in politicas]
    paleta = [:royalblue, :darkorange, :purple, :teal, :brown]
    nframes = maximum(length(s.tray) for s in sims)
    anim = @animate for f in 1:salto:nframes
        p = dibujar_mundo("Paso $f")
        for (i, ((nombre, _), s)) in enumerate(zip(politicas, sims))
            t = min(f, length(s.tray))
            plot!(p, first.(s.tray[1:t]), last.(s.tray[1:t]), c = paleta[mod1(i, 5)], lw = 1.5,
                  label = "")
            fin = t == length(s.tray)
            marca = fin ? (s.resultado == :meta ? "âœ”" : "âœ˜") : ""
            scatter!(p, [s.tray[t][1]], [s.tray[t][2]], c = paleta[mod1(i, 5)], markersize = 7,
                     label = "$nombre $marca", legend = :outertopright)
        end
        p
    end
    return gif(anim, archivo; fps = fps)
end

#=
animar_robot(politicas, inicio; archivo, fps, salto) pone varias polÃ­ticas a 
conducir a la vez desde el mismo inicio. Cada una deja su estela de color y, 
al terminar, la leyenda muestra:
   âœ” si llegÃ³ a la meta
   âœ˜ si no.
=#

# Ejemplo (descomentar):
# animar_robot(["Experto" => experto, "kNN k=1" => entrenar_politica_knn(demos, 1),
#               "kNN k=11" => entrenar_politica_knn(demos, 11)], inicios_test[4])