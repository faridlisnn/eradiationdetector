# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 00:23:52 2021

@author: Farid
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.4f}'.format

dataset = pd.read_csv('data_farid_scaled.csv', sep=",")
# Random reindex
dataset = dataset.reindex(np.random.permutation(dataset.index))
# Describe dataset
dataset.describe()

# Define the input feature
input_feature = dataset[["contrast","dissimilarity","homogeneity",
     "energy","correlation","ASM","SNR","Pixel Mean"]]
# Define the label/target
input_target = dataset[["HVL"]]

# Configure training, validation, and test dataset
# Training : validation : test = 70% : 15% : 15%
amount_training = 0.7 * len(dataset)
amount_training = int(np.around(amount_training))
amount_validation_and_test = int(len(dataset)) - amount_training 

training_dataset = input_feature.head(amount_training)
training_target = input_target.head(amount_training)

validation_and_test_dataset = input_feature.tail(amount_validation_and_test)
validation_and_test_target = input_target.tail(amount_validation_and_test)

amount_validation = 0.5 * len(validation_and_test_dataset)
amount_validation = int(np.around(amount_validation))
amount_test = int(len(validation_and_test_dataset)) - amount_validation 

validation_dataset = validation_and_test_dataset.head(amount_validation)
validation_target = validation_and_test_target.head(amount_validation)

test_dataset = validation_and_test_dataset.tail(amount_test)
test_target = validation_and_test_target.tail(amount_test)

# Save training, validation, and test dataset
#training_dataset.to_csv('Training_feature.csv', index = True)
#training_target.to_csv('Training_target.csv', index = True)
#validation_dataset.to_csv('Validation_feature.csv', index = True)
#validation_target.to_csv('Validation_target.csv', index = True)
#test_dataset.to_csv('Test_feature.csv', index = True)
#test_target.to_csv('Test_target.csv', index = True)

# Assemble Layers into a Model
l0 = tf.keras.layers.Dense(units=3, input_shape=[8], activation='relu')
l1 = tf.keras.layers.Dense(units=4, activation='relu')
l2 = tf.keras.layers.Dense(units=1, activation='linear')
model = tf.keras.Sequential([l0,l1,l2])

# Compile the Model
optimizer = tf.keras.optimizers.Adam(lr=0.01)
loss = tf.keras.losses.MeanAbsoluteError()
rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse')
mape = tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[rmse,mape])
model.summary()

# Train the Model
batch_size = 36
epochs = 5000
early_stopping = tf.keras.callbacks.EarlyStopping(patience=300)
filepath = "my_checkpoint.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
   filepath=filepath, save_best_only=True, verbose=True)

# Train the model with early stopping and model checkpoint
history = model.fit(training_dataset, training_target, epochs=epochs, verbose=True, 
                    steps_per_epoch=int(amount_training/batch_size),
                    validation_data=(validation_dataset, validation_target),
                    validation_steps=int(amount_validation/batch_size),
                    callbacks=[early_stopping, model_checkpoint])

print("Finished training the model")
print("These are the layer variables: {}\n{}".format(l0.get_weights(),l1.get_weights()))

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.title('Training and Validation '+string)
    # plt.savefig('xx.png')
    plt.show()
    
# Display Training and Validation RMSE, MAPE and Loss(MAE) 
plot_graphs(history, "rmse")
plot_graphs(history, "loss")
plot_graphs(history, "mape")

# Load The Saved Model
model.load_weights(filepath)

# Predict Values
predict = model.predict(test_dataset)
print(predict, test_target[["HVL"]])

# Model Testing
test_loss, test_rmse, test_mape = model.evaluate(test_dataset, test_target, steps=amount_test/batch_size)
print('MAE on test dataset:', test_loss)
print('MAPE on test dataset:', test_mape)
print('RMSE on test dataset:', test_rmse)