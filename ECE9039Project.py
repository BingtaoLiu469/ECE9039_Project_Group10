import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers,models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
import keras_tuner as kt
from pathlib import Path
import os.path
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import wget
url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py'
file = wget.download(url)

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, pred_and_plot

Image_Dataset = "Drug Vision/Data Combined"
Image_Directory = Path(Image_Dataset)
Filepaths = list(Image_Directory.glob(r'**/*.jpg'))
Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Filepaths))
Filepaths = pd.Series(Filepaths, name='Filepath').astype(str)
Labels = pd.Series(Labels, name='Label')
# Concatenate together
Image_Dataframe = pd.concat([Filepaths, Labels], axis=1)
Image_Dataframe.head(10)
Train_Dataframe, Test_Dataframe = train_test_split(Image_Dataframe, test_size=0.2, shuffle=True, random_state=1)
Train_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input, validation_split=0.2)
Test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

# Split the data into three categories.
Train_Images = Train_generator.flow_from_dataframe(
    dataframe=Train_Dataframe,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=30,
    shuffle=True,
    seed=1,
    subset='training'
)

Validation_Images = Train_generator.flow_from_dataframe(
    dataframe=Train_Dataframe,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=30,
    shuffle=True,
    seed=1,
    subset='validation'
)

Test_Images = Test_generator.flow_from_dataframe(
    dataframe=Test_Dataframe,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=30,
    shuffle=False
)

# Resize and rescale input Layer
Resize_Rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(250,250),
  layers.experimental.preprocessing.Rescaling(1./255),
])

# Load the pretained model
Pretrained_Model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
Pretrained_Model.trainable = False

# Create checkpoint callback
Checkpoint_Path = "Drugs_classification_model_checkpoint"
Checkpoint_Callback = ModelCheckpoint(Checkpoint_Path,
                                      save_weights_only=True,
                                      monitor="val_accuracy",
                                      save_best_only=True)

Early_Stopping = EarlyStopping(monitor = "val_loss",
                               patience = 3,
                               restore_best_weights = True)

#Define model using hyperparameter tunning

def model_builder(hp):
    model = keras.Sequential()
    model.add(Pretrained_Model)
    #Tune the number of units in the first Dense layer
    #Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    #Three hidden layers
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    #Dropout for prevent overfitting
    model.add(keras.layers.Dropout(hp.Choice('Dropout_Rate', values=[0.1, 0.2])))
    model.add(keras.layers.Dense(10))
    #Tune the learning rate for he optimizer
    #Choose an optimal value between 1e-2 and 1e-5
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4, 1e-5])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
tuner=kt.RandomSearch(hypermodel = model_builder, objective = 'val_accuracy', max_trials = 50,
                   executions_per_trial=2, project_name = "ECE9039project")
tuner.search_space_summary()
tuner.search(Train_Images, epochs=10, validation_data=(Validation_Images))
bestmodel=tuner.get_best_models(num_models=1)[0]
bestmodel.summary()
bestmodel.save("ECE9039BEST")

Fit_Model = bestmodel.fit(
    Train_Images,
    validation_data=Validation_Images,
    epochs=100,
    callbacks=[
        Early_Stopping,
        create_tensorboard_callback("training_logs", 
                                    "drug_classification"),
        Checkpoint_Callback,
    ]
)
bestmodel.save("FinalModel")

results = model.evaluate(Test_Images, verbose=0)
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

plot_loss_curves(Fit_Model)

# Predict test_images
pred = model.predict(Test_Images)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (Train_Images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

# Result
print(f'The first 5 predictions: {pred[:5]}')

y_test = list(Test_Dataframe.Label)
print(classification_report(y_test, pred))

report = classification_report(y_test, pred, output_dict=True)
report_dataframe = pd.DataFrame(report).transpose()
report_dataframe

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(15, 7), text_size=10, norm=False, savefig=False): 
  
  # Create the confustion matrix
  conM = confusion_matrix(y_true, y_pred)
  conM_norm = conM.astype("float") / conM.sum(axis=1)[:, np.newaxis] # normalize
  n_classes = conM.shape[0] # number of classes

  # Plot
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(conM, cmap=plt.conM.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(conM.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes), 
         xticklabels=labels,
         yticklabels=labels)
  
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  plt.xticks(rotation=90, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set threshold
  threshold = (conM.max() + conM.min()) / 2.

  # Text on cell
  for i, j in itertools.product(range(conM.shape[0]), range(conM.shape[1])):
    if norm:
      plt.text(j, i, f"{conM[i, j]} ({conM_normm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if conM[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{conM[i, j]}",
              horizontalalignment="center",
              color="white" if conM[i, j] > threshold else "black",
              size=text_size)

  # Save the figure
  if savefig:
    fig.savefig("ConfusionMatrixResult.png")

make_confusion_matrix(y_test, pred, list(labels.values()))