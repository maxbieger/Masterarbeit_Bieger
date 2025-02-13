import tensorflow as tf
import os
from keras.applications import ResNet50, ResNet101
from keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, 
                          Multiply, Reshape, Input)
from itertools import product
from sklearn.model_selection import ParameterGrid
from keras.layers import Rescaling 
import numpy as np
seed = 123
from datetime import datetime

Ergebnisse_pfad = r'master\ResNet\Ergebnisse.txt'
data_dir = r'master\ResNet\Dataset_ResNet_mini'
train_dir, val_dir, test_dir = [os.path.join(data_dir, d) for d in ["Train", "Validation", "Test"]]

# Hyperparameter
batch_size = 32
img_size = (192, 256)
early_stopping_patience = 3
plot=True

aktuelle_zeit = datetime.now()
model_save_path = fr'master\ResNet\ResNet_model_{aktuelle_zeit.strftime("%d.%m.%Y")+"_"+aktuelle_zeit.strftime("%H:%M:%S")}.h5'

param_grid = {
    'learning_rate': [0.001],
    'epochs_list': [6],
    'layer1': [4],
    'Attention':[True],
    'minus_layeranzahl': [0,-30,-2],
    'layer2': [128],
    #'layer3': [4]
}

def InErgebnisseDateiSichern(text):
    try:
        with open(Ergebnisse_pfad, 'a', encoding='utf-8') as datei:
            datei.write(text + '\n')
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

InErgebnisseDateiSichern("--------Neuer Programmstart:--------")

# Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(0.2)
])

scaler = Rescaling(1./255)

#daten
train_generator = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
).map(lambda x, y: (scaler(x), y)).map(lambda x, y: (augmentation(x), y))

val_generator = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
).map(lambda x, y: (scaler(x), y))

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical", shuffle=False
).map(lambda x, y: (scaler(x), y))

# Squeeze-and-Excitation Block
def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([input_tensor, se])

# Parameter-Grid
Anz_combis = len(list(product(*list(param_grid.values()))))
grid = ParameterGrid(param_grid)
best_model = None
best_accuracy = 0
combi_counter = 1

for params in grid:
    print(f"\n---\nKombinationen: {combi_counter}/{Anz_combis}")
    print(f"Testing with params: {params}")
    combi_counter += 1
    InErgebnisseDateiSichern(f"Testing with params: {params}")

    #alternativ: EfficientNetB0
    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(192, 256, 3))

    # trainierbare Schichten
    for layer in base_model.layers[params['minus_layeranzahl']:]:
        layer.trainable = True

    x = base_model.output
    if params['Attention']: #Attention Layer
        x = squeeze_excite_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(params['layer1'], kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    from keras.activations import relu
    x = relu(x) 
    x = Dropout(0.5)(x)

    x = Dense(params['layer2'], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # x = Dense(params['layer3'], activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    
    output = Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Optimizer 
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=early_stopping_patience, mode='max', restore_best_weights=False)
    # Todo süäter saubere Lösung finden

    #early_stopping = EarlyStopping(monitor='val_accuracy', patience=early_stopping_patience, mode='max', restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

    # Training
    history = model.fit(train_generator, validation_data=val_generator, epochs=params['epochs_list'])#callbacks=[early_stopping, checkpoint]
    
    # Wandelt `history.history` in ein JSON-kompatibles Format um
    #history.history = {k: [float(x) for x in v] for k, v in history.history.items()}
    
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')
    InErgebnisseDateiSichern(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

    
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model
        best_history = history
        best_params = params
        print(f"New best model with accuracy: {best_accuracy}")
        InErgebnisseDateiSichern(f"New best model with accuracy: {best_accuracy}")


import plotly.graph_objects as go
from plotly.subplots import make_subplots

if plot:
    paper_color = '#EEF6FF'
    bg_color = '#EEF6FF'
    # Creating subplot
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['<b>Best Loss Over Epochs</b>', '<b>Best Accuracy Over Epochs</b>'],
        horizontal_spacing=0.2
    )


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

    fig.show()
    fig.write_html(os.path.join(model_save_path, f'best_model_plot.html'))

