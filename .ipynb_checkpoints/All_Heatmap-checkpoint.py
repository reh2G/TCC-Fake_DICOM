# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, jaccard_score, f1_score, precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
from sklearn import preprocessing
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Lambda, GlobalAveragePooling2D, Lambda
from keras.src.legacy.preprocessing.image import ImageDataGenerator

import random
import re
import glob
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import gc

# %%
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {gpus}")
else:
    print("Nenhuma GPU encontrada.")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Desativa XLA

# %%
# nome = 'PACS'
nome = 'Yildirim'
#nome = 'Todos'

imgs_path = 'Bases/Dataset' + nome

model_path = "Results/test-CNN-espectro-ResNet50 new results (0.1%)/model_Yildirim_fold_1.keras"

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

EPOCHS = 5
BATCH_SIZE = 32

HIGH_FREQ_THRESHOLD = 0.1


# %%
def get_next_filename(output_folder, base_name, type):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    image_index = 0
    
    while True:
        output_filename = f"{base_name}_{image_index}.{type}"
        output_path = os.path.join(output_folder, output_filename)
        
        if not os.path.exists(output_path):
            return output_filename
            
        image_index += 1


# %%
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else 0


# %%
def sort_files_numerically(file_paths):
    return sorted(file_paths, key=lambda x: extract_number(x))


# %%
def read_dataset(path, jpg, png):
    print(f'Reading dataset...\n')
    
    img_type = []
    images = []
    image_paths = []

    if jpg:
        img_type.append('*.jpg')
    if png:
        img_type.append('*.png')

# Normal

    normal_path = os.path.join(path, 'Train', 'Normal')
    print(f'Reading Normal images from: {normal_path}')
    
    for img_type_pattern in img_type:
        img_paths = glob.glob(os.path.join(normal_path, img_type_pattern))
        img_paths = sort_files_numerically(img_paths)
        
        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            image_paths.append(img_path)

# Kidney

    kidney_stone_path = os.path.join(path, 'Train', 'Kidney_stone')
    print(f'Reading Kidney_stone images from: {kidney_stone_path}')
    
    for img_type_pattern in img_type:
        img_paths = glob.glob(os.path.join(kidney_stone_path, img_type_pattern))
        img_paths = sort_files_numerically(img_paths)
        
        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            image_paths.append(img_path)

# ...
    
    images = np.array(images)
    image_paths = np.array(image_paths)

    return images, image_paths


# %%
all_X, all_image_paths = read_dataset(path=imgs_path, jpg=True, png=True)


# %%
def freq_spec(image, threshold, add_noise):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    if add_noise:
        amplification_factor = 0.5
        rows, cols = image.shape
        noise_size = int(np.sqrt(threshold * rows * cols))

        # Selecionar aleatoriamente um dos quatro cantos
        corner = np.random.randint(0, 4)

        # Aplicar ruído no canto selecionado
        if corner == 0:  # Canto superior esquerdo
            fshift[:noise_size, :noise_size] = 1
        elif corner == 1:  # Canto superior direito
            fshift[:noise_size, cols - noise_size:] = 1
        elif corner == 2:  # Canto inferior esquerdo
            fshift[rows - noise_size:, :noise_size] = 1
        else:  # Canto inferior direito
            fshift[rows - noise_size:, cols - noise_size:] = 1

    magnitude_spectrum_high = 20 * np.log(np.abs(fshift) + 1)

    return magnitude_spectrum_high


# %%
all_specs = []
all_labels = []

half = len(all_X) // 2
for i, img in enumerate(all_X):
    if i < half:
        all_specs.append(freq_spec(img, HIGH_FREQ_THRESHOLD/100, add_noise=False))
        all_labels.append(0)
    else:
        all_specs.append(freq_spec(img, HIGH_FREQ_THRESHOLD/100, add_noise=True))
        all_labels.append(1)

all_specs = np.array(all_specs)
all_labels = np.array(all_labels)

# %%
print("Quantidade das imagens:", all_specs.shape)
print("Exemplo dos labels (False = original, True = com ruído):", all_labels)

# %%
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        all_specs, all_labels, all_image_paths,
        test_size=TEST_SIZE, 
        stratify=all_labels,
    )


# %%
def plot_specs(specs, title):
    total_images = len(specs)
    imgs_per_figure = 100
    cols = 5
    rows = (imgs_per_figure + cols - 1) // cols

    for start in range(0, total_images, imgs_per_figure):
        plt.figure(figsize=(cols * 3, rows * 3))
        end = min(start + imgs_per_figure, total_images)
        
        for i in range(start, end):
            plt.subplot(rows, cols, i - start + 1)
            plt.imshow(specs[i], cmap='gray')
            plt.title(f"{title} - freq spec: img {i+1}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()


# %%
# Plotar todos os espectros de teste
plot_specs(X_test, title="test")


# %%
#pausa de teste

# %%
# Função que replica a operação da camada Lambda
def expand_channels(x):
    return tf.stack([x[..., 0]] * 3, axis=-1)

# Carregue o modelo com ambos os objetos personalizados
model = load_model(
    model_path,
    custom_objects={'tf': tf, '<lambda>': expand_channels},
    safe_mode=False
)

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Lambda):
        layer.function = expand_channels

for layer in model.layers:
    layer.trainable = False

# %%
model.summary()

# %%
#pausa de teste

# %%
from matplotlib import cm


# %%
def generate_heatmap(model, sample_image):
    sample_image_exp = np.expand_dims(sample_image, axis=0)

    # Obtem as ativações da última camada convolucional
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer('conv5_block3_3_conv').output)
    activations = intermediate_model.predict(sample_image_exp)
    activations = tf.convert_to_tensor(activations)

    predictions = model.predict(sample_image_exp)

    with tf.GradientTape() as tape:
        # Cria um modelo que nos dê a saída do modelo e as ativações da última camada convolucional
        iterate = Model([model.input], [model.output, model.get_layer('conv5_block3_3_conv').output])
        model_out, last_conv_layer = iterate(sample_image_exp)
        
        # Pegua a saída da classe predita
        class_out = model_out[:, np.argmax(model_out[0])]
        
        # Registra as operações a serem diferenciadas
        tape.watch(last_conv_layer)
        
        # Calcula os gradientes
        grads = tape.gradient(class_out, last_conv_layer)
        
        # Mensagem de depuração para verificar gradientes
#       print("Gradients calculated:", grads)

    # Verifica se os gradientes foram calculados corretamente
    if grads is None:
        raise ValueError('Gradients could not be computed. Check the model and layer names.')
    
    # Média dos gradientes sobre as dimensões espaciais
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Mensagem de depuração para verificar pooled_grads
#   print("Pooled gradients:", pooled_grads)
    
    # Regularização dos gradientes
    pooled_grads = tf.where(pooled_grads == 0, tf.ones_like(pooled_grads) * 1e-10, pooled_grads)
    
    # Multiplique as ativações pelos gradientes ponderados e tire a média
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer[0]), axis=-1)
    
    # Mensagem de depuração para verificar heatmap inicial
#   print("Initial heatmap:", heatmap)

    # Normaliza o heatmap se possível
    min_value = np.min(heatmap)
    
    # Normaliza o heatmap se possível
    max_value = np.max(heatmap)

    # Aplica ReLU para garantir que os valores sejam não-negativos
    heatmap = (heatmap - min_value) / (max_value - min_value)
    heatmap = np.asarray(heatmap)
    heatmap = (heatmap - 1) * (-1)

    # Mensagem de depuração para verificar heatmap normalizado
#   print("Normalized heatmap:", heatmap)
    
    # Redimensiona o heatmap para o tamanho da imagem de entrada
    heatmap_resized = cv2.resize(heatmap, (sample_image.shape[1], sample_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Aplica um mapa de cores
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)
    
    # Cria um canal alfa a partir do heatmap redimensionado
    alpha_channel = np.uint8(heatmap_resized)
    heatmap_colored_with_alpha = np.dstack((heatmap_colored, alpha_channel))
    
    # Converte a imagem original para uint8 e RGBA
    sample_image_uint8 = np.uint8(255 * np.squeeze(sample_image))
    image_rgb = cv2.cvtColor(sample_image_uint8, cv2.COLOR_GRAY2RGB)
    image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    
    # Combina a imagem original com o heatmap
    alpha_factor = alpha_channel / 255.0
    for c in range(0, 3):
        image_rgba[..., c] = image_rgba[..., c] * (1 - alpha_factor) + heatmap_colored[..., c] * alpha_factor
    
    return image_rgba


# %%
# Gerar o mapa de calor
output_folder = 'AnyHeatmap'
type = 'png'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def heatmap_models(model, image_normalized, nome):
    base_name = 'heatmap_' + nome
    output_filename = get_next_filename(output_folder, base_name, type)
    output_path = os.path.join(output_folder, output_filename)
    
    heatmap_image = generate_heatmap(model, image_normalized)
    plt.figure(figsize=(10, 5))
    
# Imagem original
    plt.subplot(1, 2, 1)
    plt.imshow(image_normalized.squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
# Mapa de calor com predição
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_image)
    plt.title(f'ResNet50 {nome}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# %%
for idx in tqdm(range(int(len(X_test)/2)), desc="Gerando Mapas de Ativação"):
    # Pré-processamento
    image = X_test[idx].astype('float32')
    image = np.expand_dims(image, axis=-1)
    
    # Gerar heatmap
    try:
        heatmap_image = generate_heatmap(model, image)
        
        # Plotar e salvar
        plt.figure(figsize=(10, 5))
        
        # Imagem Original
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Original - Img {idx}')
        plt.axis('off')
        
        # Mapa de Calor
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap_image)
        plt.title(f'Heatmap - Classe: {y_test[idx]}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'heatmap_{nome}_test_{idx}.png'))
        plt.close()
    
    except Exception as e:
        print(f"Erro na imagem {idx}: {str(e)}")

# %%
# GC collect
tf.keras.backend.clear_session()
gc.collect()

# %%
