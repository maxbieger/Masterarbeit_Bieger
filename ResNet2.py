import tensorflow as tf
import os
from keras.applications import ResNet101
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, 
                          Multiply, Reshape, Conv2D, Add, Activation, Input)
from keras.regularizers import l2
import numpy as np
import tensorflow as tf
seed = 123
from datetime import datetime
from keras.layers import Rescaling 

Ergebnisse_pfad = r'master\ResNet\Ergebnisse.txt'
data_dir = r'master\ResNet\Dataset_ResNet_mini'
train_dir, val_dir, test_dir = [os.path.join(data_dir, d) for d in ["Train", "Validation", "Test"]]
model_save_path = r'master\ResNet'

# Hyperparameter
batch_size = 32
img_size = (192, 256)
early_stopping_patience = 3
plot=True

param_grid = {
    'learning_rate': [0.001],
    'epochen': [6],
    'layer1': [8,64],
    'layer2': [4,8],
    'early_stopping_patience': [3]
}

#Dynamische anpasst
optimizer = Adam(learning_rate=params['learning_rate'], decay=1e-4)  # Simuliert AdamW

def InErgebnisseDateiSichern(text):
    try:
        with open(Ergebnisse_pfad, 'a', encoding='utf-8') as datei:
            datei.write(text + '\n')
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

aktuelle_zeit = datetime.now()
InErgebnisseDateiSichern(f"--------Neuer Programmstart:ResNet2--------{aktuelle_zeit.strftime("%d.%m.%Y")+"_"+aktuelle_zeit.strftime("%H:%M:%S")}")

# Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomFlip("horizontal_and_vertical")
    #,tf.keras.layers.RandomZoom(0.2)
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


# SE-Block verbessert & CBAM als Alternative
# def squeeze_excite_block(input_tensor, ratio=8):
#     filters = input_tensor.shape[-1]
#     se = GlobalAveragePooling2D()(input_tensor)
#     se = Dense(filters // ratio, activation="relu")(se)
#     se = Dense(filters, activation="sigmoid")(se)
#     se = Reshape((1, 1, filters))(se)
#     return Multiply()([input_tensor, se])

def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = Dense(filters // ratio, activation="relu")(input_tensor)
    se = Dense(filters, activation="sigmoid")(se)
    return Multiply()([input_tensor, se])

def cbam_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    
    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalAveragePooling2D()(input_tensor)
    shared_dense = Dense(filters // ratio, activation="relu")
    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)
    channel_attention = Dense(filters, activation="sigmoid")(Add()([avg_out, max_out]))
    
    # Spatial Attention
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_attention = Conv2D(1, (3,3), padding="same", activation="sigmoid")(concat)

    return Multiply()([input_tensor, channel_attention[:, None, None, :]]) * spatial_attention



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

    # Lade Basis-ResNet101 (mit eingefrorenen frühen Schichten)
    base_model = ResNet101(weights="imagenet", include_top=False, input_shape=(192, 256, 3))
    for layer in base_model.layers[:150]:  # Nur letzte 50 Schichten trainierbar
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Feature-Maps auf globale Merkmale reduzieren

    # Dense-Schichten mit BatchNorm & Dropout
    x = Dense(params['layer1'], kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(params['layer2'], kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.4)(x)

    # SE-Block direkt vor dem Output-Layer
    x = squeeze_excite_block(x)

    # Output-Layer mit Softmax für Klassifikation
    output = Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=params['leraning_rate']), 
                loss="categorical_crossentropy", 
                metrics=["accuracy"])


    # Callbacks für Early Stopping & Model-Speicherung
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=params['early_stopping_patience'], restore_best_weights=True)
    checkpoint = ModelCheckpoint("master\ResNet\checkpoint_ResNet.h5", save_best_only=True, monitor="val_loss", mode="min")

    # Trainieren
    history = model.fit(train_generator, validation_data=val_generator, epochs=params['epochen'], callbacks=[early_stopping, checkpoint])

    # Evaluieren
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_history = history
            best_params = params
            print(f"New best model with accuracy: {best_accuracy}")
            InErgebnisseDateiSichern(f"New best model with accuracy: {best_accuracy}")

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
        title={'text': f"Best Params: {best_params}<br>Test Acc: {np.round(test_acc * 100, 2)}%",'x': 0.025, 'xanchor': 'left', 'font': {'size': 14}},
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
    aktuelle_zeit = datetime.now()
    fig.write_html(os.path.join(model_save_path, f'ResNet_model_{aktuelle_zeit.strftime("%d.%m.%Y")+"_"+aktuelle_zeit.strftime("%H:%M:%S")}.html'))

