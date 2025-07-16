
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Título
st.title("Análisis de la Encuesta Nacional de Inclusión Financiera (ENIF) 2024")
st.header("Crédito Formal y Endeudamiento Informal")

# Introducción
st.markdown("""
La Encuesta Nacional de Inclusión Financiera (ENIF) 2024 proporciona información relevante sobre los hábitos financieros de la población mexicana. 
Este análisis se enfoca en las secciones 4.6 (Uso de crédito formal) y 4.8 (Endeudamiento informal), con el objetivo de identificar patrones relevantes en los comportamientos financieros de las personas.""")


#Datos

st.header("Base de datos")

st.markdown("""
La información utilizada en este análisis proviene de la Encuesta Nacional de Inclusión Financiera (ENIF) 2024, disponible en el portal del INEGI [link](https://www.inegi.org.mx/programas/enif/2024/#microdatos).

La base de datos contiene 13,502 observaciones y 398 variables. En este estudio se utilizan sólo las preguntas relacionadas con uso de crédito formal y endeudamiento informal, así como la pregunta sobre si la persona lleva o no un registro de sus gastos. Para el análisis de la información, se realiza un proceso de limpieza de la base de datos, con el objetivo de filtrar solo la información de interés.
""")

#Cargamos datos 

df = pd.read_csv("base_limpia.csv")
st.header("Visualización de datos")
st.dataframe(df.head())

st.markdown("""
Cada columna de la base de datos está asociada a una pregunta de la encuesta. La codificación de cada columna la puedes encontrar [aquí](https://drive.google.com/file/d/1WfhWUPQ-9Ymyb5kVvhp2WqwE-TSLbhhS/view?usp=drive_link).
""")


# Análisis Descriptivo
st.header("Análisis Descriptivo")

st.markdown("""
Se realiza un análisis exploratorio y descriptivo de las variables seleccionadas con el fin de visualizar la distribución de respuestas y detectar posibles patrones preliminares en el comportamiento financiero de los encuestados.

""")

#Variables seleccionadas
variables_46 = ['P4_6_1', 'P4_6_2', 'P4_6_3', 'P4_6_4', 'P4_6_5', 'P4_6_6']
variables_48 = ['P4_8_1', 'P4_8_2', 'P4_8_3', 'P4_8_4', 'P4_8_5', 'P4_8_6']
variable_estudio = 'P4_1'
todas_las_vars = variables_46 + variables_48 + [variable_estudio]


# Variables con su significado
descripcion_preguntas = {
    'P4_6_1': '¿Considera cuidadosamente si puede pagar algo antes de comprarlo?',
    'P4_6_2': '¿Paga sus cuentas a tiempo (tarjeta de crédito, servicios, crédito, etcétera)?',
    'P4_6_3': '¿Prefiere gastar dinero que ahorrarlo para el futuro?',
    'P4_6_4': '¿Se pone metas económicas a largo plazo y se esfuerza por alcanzarlas?',
    'P4_6_5': '¿El manejo de sus ingresos y gastos controla su vida?',
    'P4_6_6': '¿Le sobra dinero a fin de mes?',
    'P4_8_1': 'Suele pensar en el presente sin preocuparse por el futuro',
    'P4_8_2': 'El dinero está para gastarse',
    'P4_8_3': 'Mantiene una revisión detallada del manejo de su dinero',
    'P4_8_4': 'Siente que tendrá las cosas que desea',
    'P4_8_5': 'Le alcanza bien el dinero para cubrir sus gastos',
    'P4_8_6': 'Se siente tranquilo(a) de que su dinero sea suficiente',
    'P4_1': '¿Lleva un presupuesto o registro de sus ingresos y gastos?'
}

# Paleta de colores
pastel_colors = plt.get_cmap('Set2')

# Función para graficar
def graficar_pie_claro(df, variable, descripcion):
    valores = df[variable].value_counts(normalize=True).sort_index()
    etiquetas = valores.index.astype(str)
    tamanos = valores.values
    colores = pastel_colors(np.linspace(0, 1, len(valores)))

    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(
        tamanos,
        labels=etiquetas,
        autopct='%1.1f%%',
        colors=colores,
        startangle=90,
        textprops=dict(color="black", fontsize=10)
    )
    ax.set_title(descripcion, fontsize=11)
    ax.axis('equal')
    plt.tight_layout()
    return fig


# Selectbox para elegir variable
variable_seleccionada = st.selectbox(
    "Selecciona la pregunta que deseas visualizar:",
    todas_las_vars,
    format_func=lambda x: descripcion_preguntas[x]
)

# Mostrar la gráfica elegida
fig = graficar_pie_claro(df, variable_seleccionada, descripcion_preguntas[variable_seleccionada])
st.pyplot(fig)

# Generación de modelo predictivo 


variables_46 = ['P4_6_1', 'P4_6_2', 'P4_6_3', 'P4_6_4', 'P4_6_5', 'P4_6_6']
variables_48 = ['P4_8_1', 'P4_8_2', 'P4_8_3', 'P4_8_4', 'P4_8_5', 'P4_8_6']
variable_estudio = 'P4_1'
variables_interes = variables_46 + variables_48 + [variable_estudio]

df = df.astype(str)
X = df.drop(columns=variable_estudio)
y = df[variable_estudio].map({'Sí': 1, 'No': 0})
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# ------------------------------------------------------------
# Predicciones y métricas
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

matriz = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# ------------------------------------------------------------
# Streamlit Visualización
st.header("Resultados del Modelo de Regresión Logística")

st.markdown("""
Se utilizó un modelo de regresión logística para predecir si una persona lleva o no un registro de sus gastos, 
en función de sus respuestas sobre crédito formal y endeudamiento informal.
""")


st.subheader("Matriz de Confusión")
st.write(pd.DataFrame(matriz, index=["Real: No", "Real: Sí"], columns=["Predicho: No", "Predicho: Sí"]))

st.subheader("Reporte de Clasificación")
st.text(reporte)

st.subheader("Curva ROC")
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax)
st.pyplot(fig)




st.header("Análisis de resultados")


st.markdown("""
 Se observa que el desempeño general del clasificador es moderado. El valor de exactitud (accuracy) alcanzado fue de aproximadamente 75.5% , lo cual indica que el modelo acierta en tres de cada cuatro observaciones al predecir si una persona lleva o no un registro de sus ingresos y gastos.
            
Sin embargo, al analizar con mayor detalle la matriz de confusión y las métricas por clase, se identifica un desequilibrio importante en el desempeño del modelo entre las dos clases. Por un lado, la especificidad es muy alta (97.2%), lo que significa que el modelo identifica correctamente la mayoría de los casos negativos, es decir, a las personas que no llevan un presupuesto. Por otro lado, el recall o sensibilidad para los casos positivos (personas que sí llevan un presupuesto) es considerablemente baja (11.8%), lo cual implica que el modelo falla en detectar a la mayoría de estas personas.
            
La precisión positiva, es decir, la proporción de predicciones positivas que realmente son correctas, fue de aproximadamente 59.2%. Aunque este valor es razonable, su utilidad se ve limitada debido al bajo nivel de recall, lo que indica que el modelo predice pocos positivos y que, aunque la mayoría son correctos, está dejando pasar muchos casos reales sin detectar. Todo lo anterior se complementa con un valor bajo del f1 score para la clase positiva (20%) y 
un f1 score alto  para la clase negativa (86%).   

Por último, La curva ROC muestra que el modelo tiene un desempeño razonablemente bueno para distinguir entre personas que llevan o no llevan un registro de sus gastos. El área bajo la curva (AUC = 0.75) indica que, en promedio, el modelo acierta un 75% de las veces al asignar mayor probabilidad a los casos positivos frente a los negativos. Aunque el desempeño es aceptable, aún hay margen de mejora, especialmente para detectar correctamente a quienes sí llevan un presupuesto.  
""")
