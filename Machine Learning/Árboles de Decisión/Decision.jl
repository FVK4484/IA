include("ID3.jl")

###########################
## ID3 Datos Categóricos ##
###########################

# Ejercicio 1
# -----------

# Vamos a trabajar con el dataset Golf, que contiene la opinión de varios 
# usuarios de un campo de Golf, indicando si la experiencia de haber jugado en 
# un día determinado ha sido agradable o no dependiendo de las condiciones 
# climáticas.

# Comenzamos por cargar el dataset en un dataframe de la forma habitual:
data_golf = CSV.read("./Practicas/datasets/Golf.csv", DataFrame, header=true)
resumen(data_golf)

# El atributo objetivo es la columna última, `Play`, y las otras se usarán como 
# atributos de decisión
atributos = propertynames(data_golf)[1:4]
objetivo = propertynames(data_golf)[end]

# Preparamos el dataframe para proporcionar los conjuntos de entrenamiento
# y test (por defecto, 80%+20%)
train_data_golf, test_data_golf = train_test_split_id3(data_golf, objetivo)

# Entrenamos el modelo (crea el árbol), usando solo el entrenamiento
id3_golf = id3_train(train_data_golf, atributos, objetivo)

# Representamos el árbol para saber qué camino siguen las decisiones:
println(show_arbol(id3_golf))
# Visualización gráfica del árbol de decisión:
plot_arbol_id3(id3_golf)

# Predecimos las respuestas sobre el conjunto de test:
true_labels = test_data_golf[:, objetivo]
predicciones = ID3_predict(id3_golf, test_data_golf)

# Y evaluamos los resultados
evaluacion_golf = evalua_modelo(true_labels, predicciones)
show_rendimiento(evaluacion_golf)
plot_MC(evaluacion_golf)

# Extrae del árbol de decisión construido las reglas de decisión que lo forman
# y escribe un programa en Julia que calcule la misma función que el árbol.

# Ejercicio 2
# -----------
# Usa los modelos categóricos de decisión (`Presion`, `Lentes`, `Examen`) para 
# realizar tareas similares.

#########################
## ID3 Datos Numéricos ##
#########################

# Ejercicio 3: Iris
# -----------------

# Carga del conjunto de datos
data_iris = CSV.read("./Practicas/datasets/iris.csv", DataFrame, header=true)
resumen(data_iris)

# El atributo objetivo es la columna última, `class`, y las otras se usarán como 
# atributos de decisión
atributos = propertynames(data_iris)[1:4]
objetivo = propertynames(data_iris)[end]

# Preparamos el dataframe para proporcionar los conjuntos de entrenamiento
# y test (por defecto, 80%+20%)
train_data_iris, test_data_iris =
    train_test_split_id3(data_iris, objetivo; train_ratio=0.66)

# Entrenamos el modelo (crea el árbol), usando solo el entrenamiento
id3_iris = id3_train(train_data_iris, atributos, objetivo)

# Representamos el árbol para saber qué camino siguen las decisiones:
plot_arbol_id3(id3_iris)

# Predecimos las respuestas sobre el conjunto de test:
true_labels = test_data_iris[:, objetivo]
predicciones = ID3_predict(id3_iris, test_data_iris)

# Y evaluamos los resultados
evaluacion_iris = evalua_modelo(true_labels, predicciones)
show_rendimiento(evaluacion_iris)
plot_MC(evaluacion_iris)

# Ejercicio 4: Wine
# -----------------

# Carga del conjunto de datos
data_wine = CSV.read("./Practicas/datasets/wine.csv", DataFrame, header=true)
resumen(data_wine)

# El atributo objetivo es la columna última, `class`, y las otras se usarán como 
# atributos de decisión
atributos = propertynames(data_wine)[2:14]
objetivo = propertynames(data_wine)[1]

# Preparamos el dataframe para proporcionar los conjuntos de entrenamiento
# y test (por defecto, 80%+20%)
train_data_wine, test_data_wine = train_test_split_id3(data_wine, objetivo)

# Entrenamos el modelo (crea el árbol), usando solo el entrenamiento
id3_wine = id3_train(train_data_wine, atributos, objetivo)

# Representamos el árbol para saber qué camino siguen las decisiones:
plot_arbol_id3(id3_wine)

# Predecimos las respuestas sobre el conjunto de test:
true_labels = test_data_wine[:, objetivo]
predicciones = ID3_predict(id3_wine, test_data_wine)

# Y evaluamos los resultados
evaluacion_wine = evalua_modelo(true_labels, predicciones)
show_rendimiento(evaluacion_wine)
plot_MC(evaluacion_wine)

#########################
## ID3 Datos Mezclados ##
#########################

# Ejercicio 5: Golf Numérico
# --------------------------

# Carga del conjunto de datos
data_golfnum = CSV.read("./Practicas/datasets/GolfNum.csv", DataFrame, header=true)
resumen(data_golfnum)

# El atributo objetivo es la columna última, `class`, y las otras se usarán como 
# atributos de decisión
atributos = propertynames(data_golfnum)[2:5]
objetivo = propertynames(data_golfnum)[6]

# Preparamos el dataframe para proporcionar los conjuntos de entrenamiento
# y test (por defecto, 80%+20%)
train_data_golfnum, test_data_golfnum =
    train_test_split_id3(data_golfnum, objetivo)

# Entrenamos el modelo (crea el árbol), usando solo el entrenamiento
id3_golfnum = id3_train(train_data_golfnum, atributos, objetivo,5)

# Representamos el árbol para saber qué camino siguen las decisiones:
plot_arbol_id3(id3_golfnum)

# Predecimos las respuestas sobre el conjunto de test:
true_labels = test_data_golfnum[:, objetivo]
predicciones = ID3_predict(id3_golfnum, test_data_golfnum)

# Y evaluamos los resultados
evaluacion_golfnum = evalua_modelo(true_labels, predicciones)
show_rendimiento(evaluacion_golfnum)
plot_MC(evaluacion_golfnum)

# Ejercicio 6: Obesity
# --------------------

# Carga del conjunto de datos
data_obesity = CSV.read("./Practicas/datasets/Obesity.csv", DataFrame, header=true)
resumen(data_obesity)

# El atributo objetivo es la columna última, `class`, y las otras se usarán como 
# atributos de decisión
atributos = propertynames(data_obesity)[1:16]
objetivo = propertynames(data_obesity)[17]

# Preparamos el dataframe para proporcionar los conjuntos de entrenamiento
# y test (por defecto, 80%+20%)
train_data_obesity, test_data_obesity =
    train_test_split_id3(data_obesity, objetivo)

# Entrenamos el modelo (crea el árbol), usando solo el entrenamiento
id3_obesity = id3_train(train_data_obesity, atributos, objetivo)

# Representamos el árbol para saber qué camino siguen las decisiones:
plot_arbol_id3(id3_obesity)

# Predecimos las respuestas sobre el conjunto de test:
true_labels = test_data_obesity[:, objetivo]
predicciones = ID3_predict(id3_obesity, test_data_obesity)

# Y evaluamos los resultados
evaluacion_obesity = evalua_modelo(true_labels, predicciones)
println(show_rendimiento(evaluacion_obesity))
plot_MC(evaluacion_obesity)