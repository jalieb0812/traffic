
First attempt (model 1):

    model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
                        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    tf.layers.Dense(128, activation = 'relu'),

    tf.keras.layers.Dense(NUM_CATEGORIES, activation=softmax)

    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

1 convolutional layer:
  filters: 32
  kernels: 3x3
  activation: relu

2 pooling layer:
  using: MaxPooling2D

  pool size:2,2

3 Flatten

4 1 of 2 dense layers:

  function count: 128
  activation: relu

Output layer (2 of 2 dense layers):
  outputs: NUM_CATEGORIES
  activation: softmax

compiled:
  optimizer="adam",
  loss="categorical_crossentropy",
  metrics=["accuracy"]


results: for model 1
Epoch 1/10
500/500 [==============================] - 3s 6ms/step - loss: 6.1545 - accuracy: 0.6048
Epoch 2/10
500/500 [==============================] - 3s 6ms/step - loss: 0.3966 - accuracy: 0.8978
Epoch 3/10
500/500 [==============================] - 3s 6ms/step - loss: 0.2167 - accuracy: 0.9456
Epoch 4/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1777 - accuracy: 0.9580
Epoch 5/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1406 - accuracy: 0.9671
Epoch 6/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1366 - accuracy: 0.9677
Epoch 7/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1510 - accuracy: 0.9663
Epoch 8/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1365 - accuracy: 0.9697
Epoch 9/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1506 - accuracy: 0.9686
Epoch 10/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1502 - accuracy: 0.9693
333/333 - 1s - loss: 0.3984 - accuracy: 0.9437

2nd time I ran model1:

Epoch 1/10
500/500 [==============================] - 3s 6ms/step - loss: 7.4803 - accuracy: 0.5821
Epoch 2/10
500/500 [==============================] - 3s 6ms/step - loss: 0.4423 - accuracy: 0.8910
Epoch 3/10
500/500 [==============================] - 3s 6ms/step - loss: 0.2525 - accuracy: 0.9366
Epoch 4/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1529 - accuracy: 0.9623
Epoch 5/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1728 - accuracy: 0.9573
Epoch 6/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1497 - accuracy: 0.9642
Epoch 7/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1321 - accuracy: 0.9690
Epoch 8/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1727 - accuracy: 0.9615
Epoch 9/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1521 - accuracy: 0.9697
Epoch 10/10
500/500 [==============================] - 3s 6ms/step - loss: 0.1228 - accuracy: 0.9747
333/333 - 1s - loss: 0.5622 - accuracy: 0.9197

1st model 4th try:
Epoch 10/10
500/500 [==============================] - 3s 6ms/step - loss: 0.0913 - accuracy: 0.9799
333/333 - 1s - loss: 0.5656 - accuracy: 0.9367


the first model i had one convulted layer with 32 filters using 3,3 kernal using relu as activation method . Then I did maxpooling with 2,2. Then I flattened. Then I used a Dense layer with 128 units, with activation relu. Finally, I had the output layer, with 43 units (one for each category), with activation Softmax (turning into probability distribution).
I ran this model 4 times with the following results:

Time 1: step - loss: 0.1502 - accuracy: 0.9693
333/333 - 1s - loss: 0.3984 - accuracy: 0.9437

Time 2: step - loss: 0.1228 - accuracy: 0.9747
333/333 - 1s - loss: 0.5622 - accuracy: 0.9197

Time 3: step - loss: 0.0913 - accuracy: 0.9799
333/333 - 1s - loss: 0.5656 - accuracy: 0.9367

Time 4:  step - loss: 0.1309 - accuracy: 0.9682
333/333 - 1s - loss: 0.4086 - accuracy: 0.9320
