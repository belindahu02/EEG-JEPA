import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
from tensorflow.keras import layers

from backbones import *
from data_loader import *
from transformations_tf import *

def cohen_kappa_score(y_true, y_pred, num_classes):
    """
    Calculate Cohen's Kappa score manually
    """
    # Convert predictions to class labels if needed
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
    
    # Calculate observed accuracy
    n = np.sum(confusion_matrix)
    po = np.trace(confusion_matrix) / n
    
    # Calculate expected accuracy
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)
    
    # Calculate kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    
    return kappa

def trainer(num_classes):
  frame_size   = 30
  BATCH_SIZE = 8
  AUTO = tf.data.AUTOTUNE
  path = "/app/data/musicid"
  
  # Select only the specified number of users (classes)
  all_users = list(range(1,21)) #All users for dataset 2
  users_to_use = all_users[:num_classes]  # Select first num_classes users
  print(f"Using {num_classes} classes (users): {users_to_use}")
  
  folder_train = ["TrainingSet"]
  folder_val = ["TestingSet"]
  folder_test = ["TestingSet_secret"]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_to_use, folders=folder_train, frame_size=30)
  print("training samples : ", x_train.shape[0])
  
  # Debug: Check if data exists
  if x_train.shape[0] == 0:
    print(f"WARNING: No training data loaded for users {users_to_use}")
    print(f"Expected path format: {path}/TrainingSet/user#_fav_session#.csv")
    print(f"Checking if directory exists: {os.path.exists(os.path.join(path, folder_train[0]))}")
    # Try to list what's actually in the directory
    train_dir = os.path.join(path, folder_train[0])
    if os.path.exists(train_dir):
      files = os.listdir(train_dir)
      print(f"Files in {train_dir}: {files[:10] if len(files) > 10 else files}")  # Show first 10 files
    else:
      print(f"Directory does not exist: {train_dir}")
  
  x_val, y_val, sessions_val = data_load_origin(path, users=users_to_use, folders=folder_val, frame_size=30)
  print("validation samples : ", x_val.shape[0])
  
  x_test, y_test, sessions_test = data_load_origin(path, users=users_to_use, folders=folder_test, frame_size=30)
  print("testing samples : ", x_test.shape[0])
  
  # Check if data was loaded successfully
  if x_train.shape[0] == 0 or x_val.shape[0] == 0 or x_test.shape[0] == 0:
    print(f"ERROR: No data loaded for {num_classes} classes. Check that data files exist for users {users_to_use}")
    print(f"Sessions found - train: {sessions_train}, val: {sessions_val}, test: {sessions_test}")
    return 0.0, 0.0
  
  classes, counts  = np.unique(y_train, return_counts=True)
  num_classes_actual = len(classes)
  print("number of classes : ", num_classes_actual)
  if num_classes_actual > 0:
    print("minimum samples per user : ", min(counts))
  
  x_train, x_val, x_test = norma(x_train, x_val, x_test)
  print("x_train", x_train.shape)
  print("x_val", x_val.shape)
  print("x_test", x_test.shape)
  
  # Use all available samples instead of limiting
  print("Using all available training samples : ", x_train.shape[0])
  classes, counts  = np.unique(y_train, return_counts=True)
  print("samples per class:", counts)
  
  SEED = 34
  ds_x = tf.data.Dataset.from_tensor_slices(x_train)
  #ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
  ds_x = (
      ds_x.shuffle(1024, seed=SEED)
      .map(tf_magwarp, num_parallel_calls=AUTO)
      .batch(BATCH_SIZE)
      .prefetch(AUTO)
  )
  
  ds_y = tf.data.Dataset.from_tensor_slices(y_train)
  ds_y = (
      ds_y.shuffle(1024, seed=SEED)
      .batch(BATCH_SIZE)
      .prefetch(AUTO)
  )
  ssl_ds = tf.data.Dataset.zip((ds_x, ds_y))
  
  val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
  
  ks = 3
  con =3
  inputs = Input(shape=(frame_size, x_train.shape[-1]))
  x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs) 
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling1D(pool_size=4, strides=4)(x)
  x = Dropout(rate=0.1)(x)
  x = resnetblock_final(x, CR=32*con, KS=ks)
  x = Flatten()(x)
  x = Dense(256, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  outputs = Dense(num_classes_actual, activation='softmax')(x)
  resnettssd = Model(inputs, outputs)
  #resnettssd.summary()
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000)# 0.0001, 0.9, 100000
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  #optimizer = tf.keras.optimizers.Adam()
  resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
  history = resnettssd.fit(ssl_ds, validation_data=val_ds, epochs=100, callbacks=callback, batch_size=BATCH_SIZE)
  
  results = resnettssd.evaluate(x_test,y_test)
  test_acc = results[1]
  print("test acc:", results[1])
  
  #Calculating kappa score using custom implementation
  y_pred = resnettssd.predict(x_test)
  kappa_score = cohen_kappa_score(y_test, y_pred, num_classes_actual)
  print('kappa score: ', kappa_score)
  
  return test_acc, kappa_score
