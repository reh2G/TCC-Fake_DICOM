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
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Lambda, GlobalAveragePooling2D, Conv2D
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

# %%
# nome = 'PACS'
nome = 'Yildirim'
#nome = 'Todos'

imgs_path = 'Bases/Dataset' + nome

output_dir = "Results_Streamlit/CNN-MobileNet (0.1%)/"
os.makedirs(output_dir, exist_ok=True)

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

EPOCHS = 75
BATCH_SIZE = 32
SIGMA_BLUR = 0.8

HIGH_FREQ_THRESHOLD = 0.1
NUM_FOLD = 2


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
output_dir_shift = "All_fshift"
os.makedirs(output_dir_shift, exist_ok=True)

output_dir_spectrum = "All_freqs"
os.makedirs(output_dir_spectrum, exist_ok=True)

def freq_spec(image, threshold, add_noise):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    if add_noise:
        rows, cols = image.shape
        noise_size = int(np.sqrt(threshold * rows * cols))
        sigma = SIGMA_BLUR

        def gaussian_blur(size, sigma):
            ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
            return kernel / np.sum(kernel)

        blur_kernel = gaussian_blur(noise_size, sigma)

        def blur_patch(spectrum, r_start, c_start):
            patch = spectrum[r_start:r_start+noise_size, c_start:c_start+noise_size]

            if patch.shape != blur_kernel.shape:
                return
            real_blurred = cv2.filter2D(np.real(patch), -1, blur_kernel)
            imag_blurred = cv2.filter2D(np.imag(patch), -1, blur_kernel)
            spectrum[r_start:r_start+noise_size, c_start:c_start+noise_size] = real_blurred + 1j * imag_blurred

        corner = np.random.randint(0, 4)
        if corner == 0:
            blur_patch(fshift, 0, 0)
        elif corner == 1:
            blur_patch(fshift, 0, cols - noise_size)
        elif corner == 2:
            blur_patch(fshift, rows - noise_size, 0)
        else:
            blur_patch(fshift, rows - noise_size, cols - noise_size)

    magnitude_spectrum_high = 20 * np.log(np.abs(fshift) + 1)

    return magnitude_spectrum_high


# %%
all_specs = []
all_labels = []

for i, img in enumerate(all_X):
    all_specs.append(freq_spec(img, HIGH_FREQ_THRESHOLD/100.0, add_noise=False))
    all_labels.append(0)

for i, img in enumerate(all_X):
    all_specs.append(freq_spec(img, HIGH_FREQ_THRESHOLD/100.0, add_noise=True))
    all_labels.append(1)

all_image_paths = np.concatenate([all_image_paths, all_image_paths])

all_image_paths = np.array(all_image_paths)
all_specs = np.array(all_specs)
all_labels = np.array(all_labels)

# %%
print("Quantidade das imagens:", all_specs.shape)
print("Quantidade dos paths de imagens:", all_image_paths.shape)
print("Exemplo dos labels (False = original, True = com ruído):", all_labels)

# %%
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        all_specs, all_labels, all_image_paths,
        test_size=TEST_SIZE, 
        stratify=all_labels,
#       random_state=53
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
#plot_specs(X_test, title="test")

# %%
#pausa de teste

# %% [markdown]
# # Função nomeada para conversão de canais
# def expand_channels(x):
#     return tf.stack([x[..., 0]]*3, axis=-1)
#
# # Define as layers do modelo MobileNet
# def model_mobilenet():
#     inputs = Input(shape=(224, 224, 1))
#
#     x = Lambda(expand_channels, name='channel_expander')(inputs)
#     
#     x = tf.keras.applications.mobilenet.preprocess_input(x)
#     base_model = tf.keras.applications.MobileNet(
#         weights='imagenet',
#         include_top=False,
#         input_tensor=x
#     )
#     
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(150, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
#     x = Dropout(0.25)(x)
#     x = Dense(100, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
#     x = Dropout(0.25)(x)
#     outputs = Dense(2, activation='softmax')(x)
#     
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     
#     return model

# %%
# Define as layers do modelo MobileNet
def model_mobilenet():
    inputs = Input(shape=(224, 224, 1))

    # Substitui a Lambda por Conv2D para expansão de canais
    x = Conv2D(
        filters=3,                          # 3 canais de saída
        kernel_size=1,                      # Kernel 1x1
        padding='same',                     # Mantém as dimensões
        use_bias=False,                     # Sem bias
        trainable=False,                    # Congela os pesos
        kernel_initializer='ones',          # Inicializa com pesos 1
        name='channel_expander'
    )(inputs)
    
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    base_model = tf.keras.applications.MobileNet(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(150, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(100, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# %%
acc = []
jacc = []
f1 = []
prec = []
rec = []

# Configurar K-Fold com random_state fixo para reprodutibilidade
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=53)
kfold = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True)

# Verificar quais folds já foram completados
completed_folds = []
for f in range(1, NUM_FOLD + 1):
    model_path = output_dir + f'model_{nome}_fold_{f}.keras'
    if os.path.exists(model_path):
        completed_folds.append(f)
print(f"Folds concluídos: {completed_folds}")

fold_no = 1
histories = []
metrics = []

for train_idx, val_idx in kfold.split(X_train, y_train):
# Pular folds já concluídos
    if fold_no in completed_folds:
        print(f"\nPulando fold {fold_no} (já concluído)")
        fold_no += 1
        continue

    print(f'\nTreinando Fold {fold_no}/{NUM_FOLD}')

# Split dos dados
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

# Pré-processamento final
    X_train_fold = np.expand_dims(X_train_fold, axis=-1)
    X_val_fold = np.expand_dims(X_val_fold, axis=-1)
    y_train_fold_cat = to_categorical(y_train_fold, 2)
    y_val_fold_cat = to_categorical(y_val_fold, 2)

# Criar novo modelo para cada fold
    model = model_mobilenet()

# Checkpoint com nome do fold
    checkpoint_filepath = output_dir + f'model_{nome}_fold_{fold_no}.keras'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

# Calcula tempo (start)
    start_time = time.time()
    
# Treinar modelo
    history = model.fit(
        X_train_fold, y_train_fold_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_fold, y_val_fold_cat),
        callbacks=callbacks,
        verbose=1
    )

# Calcula tempo (end)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"\nO modelo demorou {training_time:.2f} segundos para treinar.")

# Coletar métricas e salvar modelo
    predictions = model.predict(X_val_fold)
    y_pred = np.argmax(predictions, axis=1)
    
    metrics.append({
        'fold': fold_no,
        'report': classification_report(y_val_fold, y_pred, output_dict=True, zero_division=0),
        'matrix': confusion_matrix(y_val_fold, y_pred)
    })
    
# Métricas de classificação (por fold)
    acc.append(accuracy_score(y_val_fold, y_pred))
    jacc.append(jaccard_score(y_val_fold, y_pred))
    f1.append(f1_score(y_val_fold, y_pred))
    prec.append(precision_score(y_val_fold, y_pred))
    rec.append(recall_score(y_val_fold, y_pred))

    print("Accuracy: "+ str(np.mean(acc)) + "+- " + str(np.std(acc)))
    print("Jaccard: "+ str(np.mean(jacc)) + "+- " + str(np.std(jacc)))
    print("Dice: "+ str(np.mean(f1)) + "+- " + str(np.std(f1)))
    print("Precision: "+ str(np.mean(prec)) + "+- " + str(np.std(prec)))
    print("Recall: "+ str(np.mean(rec)) + "+- " + str(np.std(rec)))

# Salvar métricas em um arquivo .txt
    metrics_filename = os.path.join(output_dir, f'metrics_{nome}_fold_{fold_no}.txt')
    with open(metrics_filename, 'w') as f:
        f.write(f"Fold {fold_no} Metrics:\n")
        f.write(f"Accuracy: {acc[-1]}\n")
        f.write(f"Jaccard Score: {jacc[-1]}\n")
        f.write(f"F1 Score: {f1[-1]}\n")
        f.write(f"Precision: {prec[-1]}\n")
        f.write(f"Recall: {rec[-1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_val_fold, y_pred, zero_division=0))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_val_fold, y_pred)))

# Limpeza de memória
    del model
    tf.keras.backend.clear_session()
    gc.collect()

# Salvar checkpoint a cada 3 folds
    if fold_no % 3 == 0:
        print(f"\nCheckpoint: Folds {fold_no-2}-{fold_no} concluídos")

    fold_no += 1

# %%
