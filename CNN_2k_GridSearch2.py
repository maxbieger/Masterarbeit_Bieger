#from tensorflow.keras.models import Sequential
from keras.models import Sequential 
from keras.layers import Conv2D                              # Convolutional layer
from keras.layers import MaxPooling2D                        # Max pooling layer 
from keras.layers import Flatten                             # Layer used to flatten 2D arrays for fully-connected layers.
from keras.layers import Dense                               # This layer adds fully-connected layers to the neural network.
from keras.layers import Dropout                             # This serves to prevent overfitting by dropping out a random set of activations.
from keras.layers import BatchNormalization                  # This is used to normalize the activations of the neurons.
from keras.layers import Activation                          # Layer for activation functions
from keras.callbacks import EarlyStopping, ModelCheckpoint   # Classes used to save weights and stop training when improvements reach a limit
from keras.layers import Rescaling                           # This layer rescales pixel values
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from plotly.subplots import make_subplots
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
seed = 123
paper_color = '#EEF6FF'
bg_color = '#EEF6FF'
from sklearn.model_selection import ParameterGrid
import os
from datetime import datetime
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Multiply
from itertools import product
from keras.layers import Conv2D, Multiply, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import os

# Funktion zur Anzeige und Speicherung der Konvolutionsfilter
def visualize_and_save_conv_filters(model, save_dir):
    # Iteriere über alle Layer des Modells
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):  # Prüfe, ob es sich um eine Convolutional Layer handelt
            filters, biases = layer.get_weights()  # Extrahiere Filter und Biases
            num_filters = filters.shape[-1]  # Anzahl der Filter
            filter_size = filters.shape[:2]  # Größe der Filter (z.B., 3x3)

            print(f"Layer {i} - {layer.name}: {num_filters} Filter mit Größe {filter_size}")

            # Normiere die Filter für Visualisierung
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)

            # Visualisierung der Filter
            cols = 8  # Anzahl der Spalten (Anzahl der Filter pro Zeile)
            rows = num_filters // cols + (num_filters % cols > 0)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = axes.flatten()

            for j in range(num_filters):
                ax = axes[j]
                filter_img = filters[..., j]  # Wähle Filter j
                ax.imshow(filter_img[:, :, 0], cmap="viridis")
                ax.axis("off")
            plt.tight_layout()

            # Speicher den Plot
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{layer.name}_filters.png")
            plt.savefig(save_path)
            print(f"Konvolutionsmatrix gespeichert unter: {save_path}")
            plt.show()

param_grid = {
    'learning_rate': [0.001],#0.05 beste ergebnisse als 0.01, 0.2, ab 0.05 nur noch 65%, 0.001 =88
    'epochs_list': [4],#da viele bilde, wenig epochen
    'layer1': [10],#Erst schicht viel ist gut, Zu viele layer = overfitting
    'layer2': [9,8,7],
    'layer3': [5,4,3],
    'Denselayer': [5,4,3],#Denselayer 4>6>3>5
    'Augmentierung':[True]# Augmentierung ein/aus schalten
}
DatensatzName= "Dataset_complete_neu"
Ergebnisse_pfad = r'master\CNN\ErgebnisseCNN.txt'
model_save_path = r'master\CNN'
batch_size= 8
Early_stopping_patience = 5
plot = True
gpu = False
Show_Augmentation_on_train = False

# Loading training, testing, and validation directories
# wird für das Training verwendet
train_dir = fr'master\CNN\{DatensatzName}\Train'
# wird accuracy berechnung des modells wären des trainings verwendet
val_dir = fr'master\CNN\{DatensatzName}\Validation'
# wird am ende des Progrmms verwendet um das fertige modell mit neuen daten zu testen
test_dir = fr'master\CNN\{DatensatzName}\Test'


def InErgebnisseDateiSichern(text):
    try:
        with open(Ergebnisse_pfad, 'a', encoding='utf-8') as datei:
            datei.write(text + '\n')
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

aktuelle_zeit = datetime.now()
InErgebnisseDateiSichern("--------Neuer Programm start:-----"+aktuelle_zeit.strftime("%d.%m.%Y")+"_"+aktuelle_zeit.strftime("%H:%M:%S"))
print('Setup complete.','DatensatzName')

# Verzeichnisse für jede Klasse im Trainingsdatensatz
train_healthy_dir = os.path.join(train_dir, 'Healthy')
train_dry_dir = os.path.join(train_dir, 'Dry')


# Giving names to each directory
directories = {
    train_dir: 'Train',
    test_dir: 'Test',
    val_dir: 'Validation'
    }

# Naming subfolders
subfolders = ['Healthy', 'Dry']

print('\n* * * * * Number of files in each folder * * * * *')

# Counting the total of pictures inside each subfolder and directory
for dir, name in directories.items():
    total = 0
    for sub in subfolders:
        path = os.path.join(dir, sub)
        num_files = len([f for f in os.listdir(path) if os.path.join(path, f)])
        total += num_files
        print(f'{name}/{sub}: {num_files}')
    print(f'  Total: {total}')
    print("-" * 80)

unique_dimensions = set()

for dir, name in directories.items():
    for sub in subfolders:
        folder_path = os.path.join(dir, sub)
        
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            with Image.open(image_path) as img:
                unique_dimensions.add(img.size)
                
if not len(unique_dimensions) == 1:
    print(f"\nFound {len(unique_dimensions)} unique image dimensions: {unique_dimensions}")


# Checking if all the images in the dataset have the same dimensions
dims_counts = defaultdict(int)

for dir, name in directories.items():
    for sub in subfolders:
        folder_path = os.path.join(dir, sub)
        
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            with Image.open(image_path) as img:
                dims_counts[img.size] += 1
                
for dimension, count in dims_counts.items():
    print(f"\nDimension {dimension}: {count} images")


# Checking images dtype
all_uint8 = True
all_in_range = True

for dir, name in directories.items():
    for sub in subfolders:
        folder_path = os.path.join(dir, sub)
        
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            with Image.open(image_path) as img:
                img_array = np.array(img)
                
            if img_array.dtype == 'uint8':
                all_uint8 = False
            
            if img_array.min() < 0 or img_array.max() > 255:
                all_in_range = False
                
if not all_uint8:
    print(" - Not all images are of data type uint8\n")
    
if not all_in_range:
    print(" - Not all images have the same pixel values from 0 to 255")

#Configuring GPU
if gpu:
    print('GPUS Tensor')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print('\nDynamischer Ram')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")  
            print('\nGPU Found! Using GPU...')
        except RuntimeError as e:
            print(e)
    else:
        strategy = tf.distribute.get_strategy()
        print('Number of replicas:', strategy.num_replicas_in_sync)

# Augmentation da die bilder alle aus einer richtung 
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
        factor = (-.25, .3),
        fill_mode = 'reflect',
        interpolation = 'bilinear',
        seed = seed),
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),  # Zufällige Spiegelung
    ]
)

# Augmentierung NUR auf das Trainingsset anwenden
train = tf.keras.utils.image_dataset_from_directory(
    train_dir,  
    labels='inferred', 
    label_mode='categorical',
    class_names=['Healthy', 'Dry'],
    batch_size=batch_size,    
    image_size=(192, 256), 
    shuffle=True,  
    seed=seed,  
    validation_split=0,  
    crop_to_aspect_ratio=True
)#Augmentierung wird später vorgenommen

def show_images_from_dataset(dataset, class_names, num_images):
    # Lade eine Batch aus dem Dataset
    for images, labels in dataset.take(1):  # Nur die erste Batch nehmen
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)  # 3x3 Raster
            plt.imshow(images[i].numpy().astype("uint8"))  # Bild anzeigen
            plt.title(class_names[tf.argmax(labels[i]).numpy()])  # Label anzeigen
            plt.axis("off")
        plt.show()

if Show_Augmentation_on_train:
    show_images_from_dataset(train, class_names=['Healthy', 'Dry'], num_images=batch_size)


# Reskalierung auf das augmentierte Trainingsset anwenden
scaler = Rescaling(1./255)
train = train.map(lambda x, y: (scaler(x), y))

# Test- und Validierungsset ohne Augmentierung reskalieren
test = tf.keras.utils.image_dataset_from_directory(
    test_dir,  
    labels='inferred', 
    label_mode='categorical',
    class_names=['Healthy', 'Dry'],
    batch_size=batch_size,    
    image_size=(192, 256), 
    shuffle=True,  
    seed=seed,  
    validation_split=0,  
    crop_to_aspect_ratio=True
).map(lambda x, y: (scaler(x), y))  # Nur Reskalierung

validation = tf.keras.utils.image_dataset_from_directory(
    val_dir,  
    labels='inferred', 
    label_mode='categorical',
    class_names=['Healthy', 'Dry'],
    batch_size=batch_size,    
    image_size=(192, 256), 
    shuffle=True,  
    seed=seed,  
    validation_split=0,  
    crop_to_aspect_ratio=True
).map(lambda x, y: (scaler(x), y))  # Nur Reskalierung

# Initialisierung der Minimal- und Maximalwerte
min_value = float('inf')
max_value = -float('inf')

# Checking minimum and maximum pixel values in the Validation dataset
min_value = float('inf')
max_value = -float('inf')
try:
    for img, label in validation:
        batch_min = tf.reduce_min(img)
        batch_max = tf.reduce_max(img)
        
        min_value = min(min_value, batch_min.numpy())
        max_value = max(max_value, batch_max.numpy())
except Exception as e:
    print("An error occurred: Line 343")
    
if min_value !=0 or max_value!=1:
    print('\nWrong pixel values')
    print('Minimum pixel value in the Validation dataset', min_value)
    print('Maximum pixel value in the Validation dataset', max_value)


#Zähle die Wiederholungen
Anz_combis=len(list(product(*list(param_grid.values()))))
grid = ParameterGrid(param_grid)
best_model = None
best_accuracy = 0
combi_counter=1

def spatial_attention(x):
        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        return Multiply()([x, attention])

for params in grid:
    print(f"\n---\nKombinationen: {combi_counter}/{Anz_combis}")
    print(f"Testing with params: {params}")
    combi_counter+=1
    InErgebnisseDateiSichern(f"Testing with params: {params}")

    # Spatial Attention Mechanism
    # CNN Architektur
    model = Sequential()

    model.add(Conv2D(params['layer1'], (3, 3), padding='same', input_shape=(192, 256, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(params['layer2'], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(params['layer3'], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Fully-Connected Layers
    model.add(Flatten())

    attention_output = spatial_attention(model.layers[-2].output)  # Verwende den letzten Layer
    attention_data = Flatten()(attention_output)
    model = tf.keras.Model(inputs=model.input, outputs=attention_data)

    model = tf.keras.Model(inputs=model.input, outputs=Dense(2, activation='softmax')(attention_data))
    
    # Modellkompilierung
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
              loss='categorical_crossentropy', metrics=['accuracy'])

    # Defining an Early Stopping and Model Checkpoints
    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                              patience = Early_stopping_patience, mode = 'max',
                              restore_best_weights = True)

    checkpoint = ModelCheckpoint(os.path.join(os.path.dirname(__file__), 'best_model_checkpoint.keras'),
                            monitor = 'val_accuracy',
                            save_best_only = True)

    train_temp = train
    if params['Augmentierung']:
        train_temp = train_temp.map(lambda x, y: (augmentation(x), y))
    # Training and Testing Model
    tf.get_logger().setLevel('ERROR')
    try:
        history = model.fit(
            train_temp, epochs = params['epochs_list'],
            validation_data = validation,
            callbacks = [early_stopping, checkpoint])
    except Exception as e:
        print("An error occurred: ", e)


    # print('\nValidation Loss: ', val_loss)
    # print('\nValidation Accuracy: ', np.round(val_acc * 100,2), '%') 
    # preds = model.predict(validation)  # Running model on the validation dataset
    # val_loss, val_acc = model.evaluate(validation) # Obtaining Loss and Accuracy on the val dataset
    # print('\nTrain Loss: ', train_loss)
    # print('\nTrain Accuracy: ', np.round(train_acc * 100,2), '%')
    # preds = model.predict(train)  # Running model on the validation dataset
    # train_loss, train_acc = model.evaluate(train)
    
    preds = model.predict(test)  # Running model on the test dataset
    test_loss, test_acc = model.evaluate(test)
    print('\nTest Loss: ', test_loss)
    print('Test Accuracy: ', np.round(test_acc * 100,2), '%')
    InErgebnisseDateiSichern(f"Test Loss: {test_loss}")
    InErgebnisseDateiSichern(f"Test Accuracy: {np.round(test_acc * 100,2)} %")
          
          
    aktuelle_zeit = datetime.now()
    # Speichere das beste Modell
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model
        best_history=history
        best_params = params
        print(f"New best model with accuracy: {best_accuracy}")
        InErgebnisseDateiSichern(f"New best model with accuracy: {best_accuracy}")
        

# # Speicherort definieren
# filter_save_dir = os.path.join(model_save_path, "conv_filters")
# # Rufe die Funktion auf
# visualize_and_save_conv_filters(best_model, filter_save_dir)


# Beste Ergebnisse ausgeben
model.save(os.path.join(os.path.dirname(__file__), f'best_model{aktuelle_zeit.strftime("%d.%m.%Y")}_{aktuelle_zeit.strftime("%H-%M-%S")}.h5'))
print('\nErgebnisse:')
print(f"Best Test Accuracy: {best_accuracy}")
InErgebnisseDateiSichern(f"Best Test Accuracy: {best_accuracy}")
aktuelle_zeit = datetime.now()
InErgebnisseDateiSichern("Ende des Programms. = "+aktuelle_zeit.strftime("%d.%m.%Y")+"_"+aktuelle_zeit.strftime("%H:%M:%S")+"\n\n")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if plot:
    # Creating subplot
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['<b>Best Loss Over Epochs</b>', '<b>Best Accuracy Over Epochs</b>'],
        horizontal_spacing=0.2
    )

    # Loss over epochs
    train_loss = go.Scatter(
        x=list(range(len(best_history.history['loss']))),
        y=best_history.history['loss'],
        mode='lines',
        line=dict(color='rgba(0, 67, 162, .75)', width=4.75),
        name='Training',
        showlegend=False
    )

    val_loss = go.Scatter(
        x=list(range(len(best_history.history['val_loss']))),
        y=best_history.history['val_loss'],
        mode='lines',
        line=dict(color='rgba(255, 132, 0, .75)', width=4.75),
        name='Validation',
        showlegend=False
    )

    fig.add_trace(train_loss, row=1, col=1)
    fig.add_trace(val_loss, row=1, col=1)

    # Accuracy over epochs
    train_acc = go.Scatter(
        x=list(range(len(best_history.history['accuracy']))),
        y=best_history.history['accuracy'],
        mode='lines',
        line=dict(color='rgba(0, 67, 162, .75)', width=4.75),
        name='Training',
        showlegend=True
    )

    val_acc = go.Scatter(
        x=list(range(len(best_history.history['val_accuracy']))),
        y=best_history.history['val_accuracy'],
        mode='lines',
        line=dict(color='rgba(255, 132, 0, .75)', width=4.75),
        name='Validation',
        showlegend=True
    )

    fig.add_trace(train_acc, row=1, col=2)
    fig.add_trace(val_acc, row=1, col=2)

    # Updating layout
    fig.update_layout(
        title={'text': f"Best Params: {best_params}<br>Test Acc: {np.round(best_accuracy * 100, 2)}%",'x': 0.025, 'xanchor': 'left', 'font': {'size': 14}},
        margin=dict(t=100),
        plot_bgcolor=bg_color, paper_bgcolor=paper_color,
        height=500, width=1000,
        showlegend=True
    )

    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=2)

    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)

    # Showing figure
    fig.show()
    # Save the figure
    fig.write_html(os.path.join(model_save_path, f'best_model_plot_{aktuelle_zeit.strftime("%d_%m_%Y_%H_%M_%S")}.html'))

