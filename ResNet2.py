from keras.layers import Rescaling
from keras.layers import (GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
                          Multiply, Reshape, Conv2D, Add, Activation, Input)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
from keras.applications import ResNet101, ResNet50, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, decay=1e-4)  # Simuliert AdamW
seed = 123


Ergebnisse_pfad = r'master\ResNet\Ergebnisse.txt'
data_dir = r'master\ResNet\Dataset_ResNet_a'
train_dir, val_dir, test_dir = [os.path.join(data_dir, d) for d in [
    "Train", "Validation", "Test"]]


# Abbruchbedingungen: Die Zahlen-Kombination muss groß genug sein: 
# Epochen=3 => (Layer1=32 Layer2=16) geht
# Epochen=3 => (Layer1=32 Layer2=4) geht nicht
# Epochen=10 => (Layer1=8 Layer2=4) geht nicht
# Epochen=20 => (Layer1=8 Layer2=4) geht nicht
# Epochen=20 => (Layer1=8 Layer2=8) geht 
# Epochen=10 => (Layer1=32 Layer2=32) geht
# Epochen=100 => (Layer1=6 Layer2=4) geht nicht
# Epochen=200 => (Layer1=6 Layer2=4) geht nicht
# Epochen=200 => (Layer1=6 Layer2=6) geht nicht
# Epochen=100 => (Layer1=8 Layer2=6) geht nicht
# Epochen=100 => (Layer1=10 Layer2=6) geht nicht
# Epochen=100 => (Layer1=10 Layer2=8) geht
# Epochen=5 => (Layer1=16 Layer2=8) geht
# Epochen=5 => (Layer1=12 Layer2=4) geht nicht
# Epochen=15 => (Layer1=12 Layer2=4) geht nicht
# Epochen=5 => (Layer1=16 Layer2=8) 170, geht 
# Epochen=5 => (Layer1=16 Layer2=8) 170, 50V2, geht über 50% acc
# Epochen=5 => (Layer1=16 Layer2=8) 200, 50V2, cbam_block, l2(0.05), Dropout hoch, geht nicht
# Epochen=5 => (Layer1=16 Layer2=8) 200, 50V2, l2(0.05), Dropout hoch, geht 61% acc
# Epochen=10 => (Layer1=10 Layer2=8) 185, 50V2, l2(0.05), Dropout hoch, ResNet_a, 71%
# Epochen=10 => (Layer1=10 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 84%
# Epochen=5 => (Layer1=10 Layer2=8) 170, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 64%
# Epochen=10 => (Layer1=12 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 78%
# # Epochen=10 => (Layer1=12 Layer2=8) 185, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 77%
# Epochen=50 => (Layer1=10 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 94%
# Epochen=50 => (Layer1=10 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a", 71%
#Epochen=50 => (Layer1=10 Layer2=8) 182, 50V2, l2(0.04), Dropout 0.5/0.4, ResNet_a, 73%
# Epochen=50 => (Layer1=8 Layer2=8) 185, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 90%
# Epochen=60 => (Layer1=8 Layer2=8) 187, 50V2, l2(0.1), Dropout 0.5/0.5, ResNet_a, 89%
# Epochen=60 => (Layer1=8 Layer2=8) 190, 50V2, l2(0.1), Dropout 0.5/0.5, ResNet_a, 79% weniger overfitting"
InBearbeitung= "Epochen=60 => (Layer1=8 Layer2=8) 185, 50V2, l2(0.1), Dropout 0.5/0.5, ResNet_a, noch machen"
#Test dauer = 1min
epochen = 6
layer1 = 8
layer2 = 8
frozen_layers = 185

# Hyperparameter: Notiz: Grid Search funktioniert für Resnet nicht
batch_size = 16
img_size = (256,192)
early_stopping_patience = 20
plot = True
learning_rate = 0.001

model_save_path = fr'master\ResNet'

def InErgebnisseDateiSichern(text):
    try:
        with open(Ergebnisse_pfad, 'a', encoding='utf-8') as datei:
            datei.write(text + '\n')
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

aktuelle_zeit = datetime.now()
InErgebnisseDateiSichern(f"\nResNet2--------Neuer Programmstart:--------{aktuelle_zeit.strftime('%d-%m-%Y')}_{aktuelle_zeit.strftime('%H-%M-%S')}")
InErgebnisseDateiSichern(f"Learning Rate: {learning_rate}, epochen: {epochen}, layer1: {layer1}, layer2: {layer2}, batch_size: {batch_size}, early_stopping_patience: {early_stopping_patience},")
InErgebnisseDateiSichern(InBearbeitung)

# Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(0.1)
])
scaler = Rescaling(1./255)

AUTOTUNE = tf.data.AUTOTUNE# augmentation(scaler(x))

train_generator = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
).map(lambda x, y: (augmentation(scaler(x)), y)) \
  .cache() \
  .prefetch(buffer_size=AUTOTUNE)

val_generator = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
).map(lambda x, y: (scaler(x), y)) \
  .cache() \
  .prefetch(buffer_size=AUTOTUNE)

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical", shuffle=False
).map(lambda x, y: (scaler(x), y)) \
  .cache() \
  .prefetch(buffer_size=AUTOTUNE)

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
    channel_attention = Dense(filters, activation="sigmoid")(
        Add()([avg_out, max_out]))

    # Spatial Attention
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_attention = Conv2D(
        1, (3, 3), padding="same", activation="sigmoid")(concat)
    return Multiply()([input_tensor, channel_attention[:, None, None, :]]) * spatial_attention


# Lade Basis-ResNet101 (mit eingefrorenen frühen Schichten)
base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=( 256,192, 3))
for layer in base_model.layers[:frozen_layers]:  # Nur letzte 50 Schichten trainierbar
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Feature-Maps auf globale Merkmale reduzieren
# Dense-Schichten mit BatchNorm & Dropout
x = Dense(layer1, kernel_regularizer=l2(0.1))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = Dropout(0.5)(x)
x = Dense(layer2, kernel_regularizer=l2(0.1))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = Dropout(0.5)(x)

# SE-Block direkt vor dem Output-Layer
x = squeeze_excite_block(x)
#x = cbam_block(x)

# Output-Layer mit Softmax für Klassifikation
output = Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
# AdamW statt Adam für bessere Generalisierung
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss="categorical_crossentropy",  # One-Hot-Encoded Labels
              metrics=["accuracy"])
# Callbacks für Early Stopping & Model-Speicherung
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=early_stopping_patience, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
#TODO
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor="val_loss",  # oder "val_accuracy" je nach Ziel
#     factor=0.5,  # Reduziert die Lernrate um 50%
#     patience=2,  # Nach 2 Epochen ohne Verbesserung wird reduziert
#     min_lr=1e-6,  # Mindestgrenze für die Lernrate
#     verbose=1  # Zeigt Infos über Änderungen in der Konsole
# )

# Trainieren
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochen,
    callbacks=[early_stopping, checkpoint]#, reduce_lr
)

# Evaluieren
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
InErgebnisseDateiSichern(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

aktuelle_zeit = datetime.now()
path=os.path.join(model_save_path, f'best_model_acc{test_acc:.3f}_{aktuelle_zeit.strftime("%d.%m.%Y")}_{aktuelle_zeit.strftime("%H-%M-%S")}.h5')
model.save(path)

# .h5 löschen
if os.path.exists("best_model.h5"):
    os.remove("best_model.h5")

if plot:
    paper_color = '#EEF6FF'
    bg_color = '#EEF6FF'
    # Creating subplot
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['<b>Best Loss Over Epochs</b>',
                        '<b>Best Accuracy Over Epochs</b>'],
        horizontal_spacing=0.2
    )

    train_loss = go.Scatter(
        x=list(range(len(history.history['loss']))),
        y=history.history['loss'],
        mode='lines',
        line=dict(color='rgba(0, 67, 162, .75)', width=4.75),
        name='Training',
        showlegend=False
    )

    val_loss = go.Scatter(
        x=list(range(len(history.history['val_loss']))),
        y=history.history['val_loss'],
        mode='lines',
        line=dict(color='rgba(255, 132, 0, .75)', width=4.75),
        name='Validation',
        showlegend=False
    )

    fig.add_trace(train_loss, row=1, col=1)
    fig.add_trace(val_loss, row=1, col=1)

    train_acc = go.Scatter(
        x=list(range(len(history.history['accuracy']))),
        y=history.history['accuracy'],
        mode='lines',
        line=dict(color='rgba(0, 67, 162, .75)', width=4.75),
        name='Training',
        showlegend=True
    )

    val_acc = go.Scatter(
        x=list(range(len(history.history['val_accuracy']))),
        y=history.history['val_accuracy'],
        mode='lines',
        line=dict(color='rgba(255, 132, 0, .75)', width=4.75),
        name='Validation',
        showlegend=True
    )

    fig.add_trace(train_acc, row=1, col=2)
    fig.add_trace(val_acc, row=1, col=2)

    fig.update_layout(
        title={'text': f"Learning Rate: {learning_rate}, epochen: {epochen}, layer1: {layer1}, layer2: {layer2}, batch_size: {batch_size}, e.s.patience: {early_stopping_patience} <br>Test Acc: {np.round(test_acc * 100, 2)}%",
               'x': 0.025, 'xanchor': 'left', 'font': {'size': 14}},
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
    fig.write_html(os.path.join(
        model_save_path, fr'ResNet_model_{aktuelle_zeit.strftime("%d-%m-%Y")+"_"+aktuelle_zeit.strftime("%H-%M-%S")}_Acc_{np.round(test_acc * 100, 2)}.html'))
