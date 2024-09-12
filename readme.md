# Trabajo de Evaluación 1: Machine Learning

Este repositorio contiene la entrega del trabajo práctico de Machine Learning, realizado por Matías Peña y Gustavo Flores.

## Enlaces de Perfiles

- **Matías Peña:**
  - Kaggle: [matiasinacap](https://www.kaggle.com/matiasinacap)
  - GitHub: [xMa7icK](https://github.com/xMa7icK)

- **Gustavo Flores:**
  - Kaggle: [gusinacap](https://www.kaggle.com/gusinacap)
  - GitHub: [Gus-Inacap](https://github.com/Gus-Inacap)

## Requisitos de Instalación

Antes de ejecutar los notebooks, asegúrarse de tener instaladas las siguientes bibliotecas:

   ```bash
   pip install gdown
   pip install scikit-learn
   pip install matplotlib
   ```

1. **gdown**: Para descargar archivos directamente desde Google Drive.
2. **scikit-learn**: Para trabajar con algoritmos de Machine Learning.
3. **matplotlib**: Para la visualización de datos.


## Descripción del Proyecto

El trabajo se basa en la resolución de varios problemas de Machine Learning, utilizando diferentes algoritmos y métricas de evaluación. A continuación, se detallan los problemas abordados:

### Problema 1: Predicción del Precio de una Casa en California

- **Descripción:** Predicción del precio de una casa utilizando características como tamaño, ubicación, etc.
- **Dataset:** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Preguntas a resolver:**
  1. ¿Qué características influyen más en el valor de una casa?
  2. ¿Cuál es la precisión del modelo?

### Respuesta a la pregunta 1: ¿Qué características influyen más en el valor de una casa?

El análisis de correlación y los coeficientes de regresión proporcionan información valiosa sobre las características que influyen más en el valor de una casa.

#### a) Correlación entre características y el valor medio de la vivienda

La correlación es una medida estadística que describe la relación lineal entre dos variables. Los resultados de correlación entre las características y el valor medio de la vivienda (`median_house_value`) muestran que la característica con mayor influencia es el **ingreso medio** de los habitantes de la casa (`median_income`), con una correlación positiva fuerte de 0.688. Esto significa que a medida que aumenta el ingreso medio, también tiende a aumentar el valor de la vivienda.

Otras características tienen una influencia considerablemente menor. Por ejemplo:
- **total_rooms** (número total de habitaciones): correlación de 0.134, una relación positiva débil.
- **housing_median_age** (edad mediana de las viviendas): correlación de 0.105, lo que sugiere una influencia mínima en el valor de la casa.
- **longitude** y **latitude** (coordenadas geográficas): presentan correlaciones negativas (-0.045 y -0.144 respectivamente), lo que indica que la ubicación geográfica también afecta, pero con una relación negativa.

#### b) Coeficientes de regresión lineal

Los coeficientes de regresión lineal proporcionan una medida más precisa de cómo las características afectan el valor de una casa en términos absolutos. El coeficiente más alto corresponde al **ingreso medio** (`median_income`), con un valor de 40,538, lo que indica que por cada unidad adicional de ingreso medio, el valor de la vivienda aumenta en aproximadamente 40,538 dólares. Este coeficiente refuerza la importancia del ingreso medio como el factor más influyente.

Además, la **longitud** y **latitud** también muestran coeficientes considerables, -42,632 y -42,450 respectivamente, lo que sugiere que las viviendas ubicadas más al norte y más al este tienen valores menores, una indicación de la importancia de la ubicación geográfica en la determinación del precio.

#### c) Métricas de evaluación del modelo

El modelo de regresión lineal obtuvo un **puntaje R² de 0.623**, lo que significa que el 62.3% de la variabilidad en el valor de las casas puede ser explicada por las características incluidas en el modelo. Aunque este puntaje no es extremadamente alto, indica que el modelo tiene una capacidad moderada para predecir los precios de las casas basándose en las características seleccionadas.

Las otras métricas del modelo son:
- **Error Absoluto Medio (MAE)**: $51,338
- **Error Cuadrático Medio (MSE)**: $4,953,936,521
- **Raíz del Error Cuadrático Medio (RMSE)**: $70,384

Estas métricas muestran que, aunque el modelo es útil para detectar tendencias generales, aún tiene una cantidad considerable de error en las predicciones.

#### Conclusión

De acuerdo con los datos proporcionados, la característica que más influye en el valor de una casa en California es el **ingreso medio** de los habitantes. Las demás características, como la **edad de la vivienda**, el **número total de habitaciones** y la **ubicación geográfica** (longitud y latitud), también tienen influencia, pero en menor medida. Estas conclusiones se derivan tanto del análisis de correlación como de los coeficientes del modelo de regresión lineal.


### Problema 2: Clasificación de Correos Electrónicos en Spam/No Spam

- **Descripción:** Clasificación de correos electrónicos en spam o no spam basándose en su contenido.
- **Dataset:** [Spambase](https://archive.ics.uci.edu/dataset/94/spambase)
- **Preguntas a resolver:**
  1. Justificar el modelo utilizado.
  2. ¿Qué características afectan más en que un correo sea Spam?

#### Preguntas a Resolver:

##### 1. Justificación del Modelo Utilizado:

El modelo seleccionado para este problema fue **Random Forest**, una técnica de clasificación basada en la construcción de múltiples árboles de decisión y la agregación de sus resultados. Este modelo es particularmente útil para problemas de clasificación debido a su capacidad de manejar características no lineales y correlaciones complejas. Además, Random Forest ofrece la ventaja de poder evaluar la importancia relativa de cada característica en la clasificación final.

En este caso, Random Forest mostró un desempeño adecuado con una **precisión global del 82%** y un **F1-Score** promedio ponderado de 0.82, como se observa en la matriz de confusión:

El modelo también mostró una **tasa de error del 18%**. Estos resultados sugieren que Random Forest es un modelo robusto para la clasificación de correos electrónicos, equilibrando precisión y capacidad de generalización.

### 2. ¿Qué características afectan más en que un correo sea Spam?

El modelo de Random Forest permite identificar la importancia de cada característica en la clasificación. Las características más relevantes, según sus valores de importancia, son las siguientes:

| Característica               | Importancia  |
|------------------------------|--------------|
| **capital_run_length_total**  | 0.125967     |
| **capital_run_length_longest**| 0.095602     |
| **char_freq_!**               | 0.078916     |
| **word_freq_remove**          | 0.072008     |
| **capital_run_length_average**| 0.057135     |

A continuación, se describen las características más importantes:

- **capital_run_length_total**: Representa el número total de letras mayúsculas consecutivas en el correo electrónico. Los correos electrónicos que contienen largas secuencias de letras mayúsculas tienden a ser spam, lo que explica su alta importancia en el modelo.

- **capital_run_length_longest**: Se refiere a la longitud de la secuencia ininterrumpida más larga de letras mayúsculas. Los correos spam a menudo contienen palabras o frases enteras en mayúsculas, lo que contribuye a esta alta relevancia.

- **char_freq_!**: Es la frecuencia del carácter de exclamación en el correo electrónico. Los correos spam suelen emplear este tipo de símbolos para captar la atención del destinatario, lo que hace que esta característica sea clave.

- **word_freq_remove**: Representa la frecuencia de la palabra "remove". Es común que los correos spam incluyan instrucciones para "eliminar" o "remover" suscripciones, lo que explica la importancia de esta palabra.

- **capital_run_length_average**: Es el promedio de la longitud de secuencias de letras mayúsculas. Este comportamiento también está asociado con correos no deseados, que a menudo utilizan texto en mayúsculas.

### Conclusión:

El modelo Random Forest es una buena elección para la clasificación de correos electrónicos en spam o no spam. Las características que más influyen en la clasificación incluyen la frecuencia de letras mayúsculas y símbolos especiales como el signo de exclamación, que son típicos de correos electrónicos no deseados.

### Problema 3: Recomendación de Películas

- **Descripción:** Recomendación de películas basadas en similitudes con otros usuarios o productos.
- **Dataset:** [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=movie.csv)
- **Preguntas a resolver:**
  1. ¿Qué película recomendarían si se quiere ver una película de terror?
  2. ¿Qué película recomendarían si la última vista fue "Toy Story"?

#### Preguntas a Resolver:

##### 1. ¿Qué película recomendarían si se quiere ver una película de terror?

Para una recomendación basada en el género de **terror**, podemos analizar las películas disponibles en el dataset que incluyen el género "Horror". A continuación, se muestran algunas recomendaciones basadas en títulos con ese género:

| movieId | Título                                  | Géneros                            |
|---------|-----------------------------------------|------------------------------------|
| 12      | Dracula: Dead and Loving It (1995)      | Comedy, Horror                     |
| 22      | Copycat (1995)                          | Crime, Drama, Horror, Mystery      |
| 70      | From Dusk Till Dawn (1996)              | Action, Comedy, Horror, Thriller   |
| 92      | Mary Reilly (1996)                      | Drama, Horror, Thriller            |
| 93      | Vampire in Brooklyn (1995)              | Comedy, Horror, Romance            |

Entre estas opciones, si se busca una película **de terror clásico**, una recomendación podría ser **"From Dusk Till Dawn (1996)"**, ya que mezcla elementos de acción y comedia junto con el género de terror, lo que la convierte en una película popular y entretenida en ese género.

##### 2. ¿Qué película recomendarían si la última vista fue "Toy Story"?

Si el último filme visto fue **Toy Story**, podemos recomendar películas similares basándonos en géneros como **aventura, niños, y fantasía**. A continuación, algunas películas relacionadas con esos géneros:

| movieId | Título                                | Géneros                         |
|---------|---------------------------------------|---------------------------------|
| 2       | Jumanji (1995)                        | Adventure, Children, Fantasy    |
| 3       | Grumpier Old Men (1995)               | Comedy, Romance                 |
| 4       | Waiting to Exhale (1995)              | Comedy, Drama, Romance          |
| 5       | Father of the Bride Part II (1995)    | Comedy                          |
| 6       | Heat (1995)                           | Action, Crime, Thriller         |

Una recomendación similar a **Toy Story**, que también es apta para toda la familia y contiene elementos de aventura y fantasía, sería **"Jumanji (1995)"**. Esta película tiene un enfoque similar en cuanto a la mezcla de aventura y fantasía, y es adecuada para un público infantil y familiar.

### Problema 4: Detección de Fraude en Transacciones Bancarias

- **Descripción:** Detección de transacciones fraudulentas basándose en patrones históricos.
- **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Preguntas a resolver:**
  1. ¿Qué kernel sería más adecuado para abordar este problema? Justificar.
  2. Comparar las métricas de Precisión, Puntaje F1, Recall y Exactitud.

#### Preguntas a Resolver:

##### 1. ¿Qué kernel sería más adecuado para abordar este problema? Justificación.

Existen varios tipos de kernels que se pueden utilizar en los modelos de **Máquinas de Soporte Vectorial (SVM)**. Los más comunes son el **kernel RBF (Radial Basis Function)** y el **kernel polinomial**. Basado en los resultados obtenidos, el kernel **polinomial** parece ser más adecuado para este problema, dado que logra un mejor balance entre **Precisión (precision)**, **Puntaje F1** y **Recall**, especialmente en la detección de las transacciones fraudulentas (clase 1).

Comparando ambos kernels:
- El **kernel RBF** tiene una precisión casi perfecta (0.97), pero un **recall** bajo (0.62), lo que significa que aunque el modelo clasifica correctamente la mayoría de las transacciones fraudulentas detectadas, se le escapan muchas.
- El **kernel polinomial**, por otro lado, logra una mayor **recall** (0.72) y un **Puntaje F1** mejorado (0.81), lo que indica que es más eficaz en la identificación de fraudes.

En problemas de fraude financiero, es crucial maximizar el **recall**, ya que el principal objetivo es detectar el mayor número de transacciones fraudulentas posible, aunque ello implique una ligera pérdida de precisión. Por lo tanto, el kernel **polinomial** es más adecuado debido a su capacidad para capturar más fraudes.

##### 2. Comparar las métricas de Precisión, Puntaje F1, Recall y Exactitud.

A continuación se presenta una comparación entre los resultados obtenidos utilizando el **kernel RBF** y el **kernel polinomial**:

| Métrica            | Kernel RBF | Kernel Polinomial |
|--------------------|------------|-------------------|
| **Precisión (accuracy)**  | 0.9993     | 0.9994            |
| **Puntaje F1**      | 0.7578     | 0.8114            |
| **Recall**          | 0.6224     | 0.7245            |

- **Precisión (accuracy)**: Ambas configuraciones muestran una precisión cercana al 100%, pero dado el desbalance de clases, esta métrica no es la más relevante en este contexto.
  
- **Puntaje F1**: El kernel polinomial tiene un mejor puntaje F1 (0.81 frente a 0.76 del RBF), lo que indica que logra un mejor balance entre **precisión** y **recall**. Esto es crucial en problemas donde la clase minoritaria (fraude) es la de mayor interés.
  
- **Recall**: El **recall** con el kernel polinomial (0.72) es significativamente mejor que con el kernel RBF (0.62). Esto implica que el modelo con kernel polinomial detecta un mayor porcentaje de fraudes, lo cual es fundamental en problemas de detección de fraude.

#### Conclusión:

- El **kernel polinomial** es más adecuado para este problema, ya que logra un mejor **recall** y **Puntaje F1**, lo que indica que es más efectivo en la detección de fraudes.
- Aunque ambos modelos muestran una alta precisión global, el kernel polinomial es preferible porque captura más fraudes, lo que lo hace más útil en la detección de transacciones fraudulentas, donde minimizar los falsos negativos es crucial.

 
## Problema Extra: Clasificación con Perceptrón

**Descripción**: Clasificación de dos tipos de flores usando un perceptrón simple con el dataset Iris de Scikit-learn, simplificado para dos clases y dos características.

- **Dataset**: Iris Dataset (incluido en `scikit-learn`)

### Resultados

Este problema aborda la clasificación de dos tipos de flores del dataset Iris utilizando un perceptrón simple. Se ha simplificado el conjunto de datos para usar solo dos clases: Setosa y Versicolor, y dos características: longitud y ancho del sépalo.
Visualización del Modelo:

En la imagen proporcionada, se observa cómo el perceptrón separa ambas clases de flores (Setosa y Versicolor) con una frontera de decisión lineal. El gráfico muestra:

    Puntos rojos: Representan la clase Setosa.
    Cruces azules: Representan la clase Versicolor.
    La frontera de decisión es la línea que separa el área coloreada en azul y rojo, que define las regiones donde el perceptrón clasifica las flores como Setosa (azul) o Versicolor (rojo).
El perceptrón simple es capaz de realizar una clasificación binaria con una separación lineal entre las dos clases de flores, proporcionando una solución sencilla pero efectiva para este tipo de problemas de clasificación linealmente separables. La frontera de decisión claramente divide las clases basándose en las características seleccionadas (longitud y ancho del sépalo).

## Bibliografia 

-Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3).
-Breiman, L. (2001). Random forests. Machine Learning, 45(1).
