import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# 1. Configuración de la página web
st.set_page_config(page_title="IA Reciclaje", page_icon="♻️", layout="centered")

st.title("♻️ Clasificador de Reciclaje con IA")
st.write("Sube una imagen o toma una foto con tu cámara para clasificar si el objeto es Papel/Cartón, Vidrio o Plástico.")

# 2. Cargar el modelo (Usamos caché para que no se cargue cada vez que interactúas con la web)
@st.cache_resource
def cargar_modelo():
    ruta_modelo = os.path.join("converted_savedmodel", "model.savedmodel")
    ruta_labels = os.path.join("converted_savedmodel", "labels.txt")
    
    modelo = tf.saved_model.load(ruta_modelo)
    
    with open(ruta_labels, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
        
    return modelo, labels

modelo, labels = cargar_modelo()
infer = modelo.signatures["serving_default"]

# 3. Interfaz de usuario para ingresar la imagen
opcion = st.radio("Elige el método de entrada:", ("Usar Cámara", "Subir Imagen"))

imagen_cargada = None

if opcion == "Usar Cámara":
    imagen_cargada = st.camera_input("Toma una foto del objeto")
else:
    imagen_cargada = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])

# 4. Procesamiento e Inferencia (Tu código adaptado)
if imagen_cargada is not None:
    # Mostrar la imagen en la web
    image = Image.open(imagen_cargada)
    st.image(image, caption="Imagen a analizar", use_column_width=True)
    
    # Convertir formato de Pillow a Numpy array para OpenCV
    img_array = np.array(image)
    
    # Si la imagen viene de la web con canal Alfa (RGBA), la pasamos a RGB
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    # Preprocesamiento idéntico al de tu Notebook
    img_resized = cv2.resize(img_array, (224, 224))
    img_float = img_resized.astype(np.float32)
    img_normalized = (img_float / 127.5) - 1
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    tensor = tf.convert_to_tensor(img_expanded)
    
    with st.spinner("Analizando..."):
        # Realizar la predicción
        # Nota: Usamos 'sequential_5_input' basándonos en la salida de tu notebook
        resultado = infer(sequential_5_input=tensor)
        
        # Extraemos la capa de salida dinámicamente (en tu código era sequential_7)
        output_layer = list(resultado.keys())[0]
        predicciones = resultado[output_layer].numpy()
        
        indice = np.argmax(predicciones)
        confianza = predicciones[0][indice] * 100
        clase_detectada = labels[indice]
    
    # 5. Mostrar resultados en la interfaz
    st.divider()
    st.subheader("Resultado:")
    
    # Colorear el resultado según el material para mejor diseño visual
    if "Papel" in clase_detectada or "Carton" in clase_detectada:
        st.info(f"📦 **Clase:** {clase_detectada} \n\n📊 **Confianza:** {confianza:.2f}%")
    elif "Vidrio" in clase_detectada:
        st.success(f"🍾 **Clase:** {clase_detectada} \n\n📊 **Confianza:** {confianza:.2f}%")
    elif "Plástico" in clase_detectada or "Plástico" in clase_detectada:
        st.warning(f"🥤 **Clase:** {clase_detectada} \n\n📊 **Confianza:** {confianza:.2f}%")
    else:
       st.write(f"**Clase:** {clase_detectada}) | **Confianza:** {confianza:.2f}%")