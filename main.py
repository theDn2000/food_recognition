import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
""" # Mostrar imagenes con pyplot
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(15, 15))

carpeta = 'content/manzanas'
imagenes = os.listdir(carpeta)

for i, nombreimg in enumerate(imagenes[:25]):
  plt.subplot(5, 5, i+1)
  imagen = mpimg.imread(carpeta + '/' + nombreimg)
  plt.imshow(imagen)

plt.show() """


""" # Copiar imagenes que subimos a carpetas del dataset
# Limitar a máximo 270 imagenes (mínimo de imágenes subidas de un mismo elemento)

carpeta_fuente = 'content/platanos'
carpeta_destino = 'content/dataset/platano'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimg in enumerate(imagenes):
  if i < 270:
    # Copia de la carpeta fuente a la de destino
    shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_destino + '/' + nombreimg) """











# Aumento de datos con ImageDataGenerator

# Creat el dataset generador
datagen = ImageDataGenerator(
    rescale = 1. / 225,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    validation_split = 0.2 # 20 % para pruebas

)

# Generadores para sets de entrenamiento y pruebas
data_gen_entrenamiento = datagen.flow_from_directory('content/dataset', target_size=(224, 224),
                                                     batch_size=32, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory('content/dataset', target_size=(224, 224),
                                                     batch_size=32, shuffle=True, subset='validation')

# Imprimir 10 imagenes del generador de entrenamiento
for imagen, etiqueta in data_gen_entrenamiento:
  for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i])
  break
plt.show()



import tensorflow as tf
import tf_keras
import tensorflow_hub as hub 

#url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3"
url = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224, 224, 3))


# Congelar el modelo descargado
mobilenetv2.trainable = False


modelo = tf_keras.Sequential([
    mobilenetv2,
    tf_keras.layers.Dense(3, activation='softmax')
])


modelo.summary()



# Compilar
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Entrenar el modelo
EPOCAS = 2

historial = modelo.fit(
    data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas
)