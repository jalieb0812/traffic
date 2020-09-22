
The first model, which I'll call the "Base Model",
had one convulted layer with 32 filters using 3,3 layer size and using relu as activation method . Then I did maxpooling with 2,2 pool sizinf. Then I flattened. Then I used a Dense layer with 128 units, with activation relu. Finally, I had the output layer, with 43 units (one for each category), with activation Softmax (turning into probability distribution).
Base Model looked like this:

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

I ran this model 5 times (results copied below). I am struck by the variability in accuracy between of the eventual testing models:from  0.8873 (time 5) to 0.9437 (Time 1).
There was also a clear overfitting issue with testing models being less accurate than
the final training models. I tried adding dropout of 0.5 and 0.25 which dealt with overfitting
but decreased the accuracy of the program significantly by around half. Twice I ran the model
but removed the pooling layer and noticed a significant decrease in accuracy.
Increasing the epochs to 20 did not result in more acurate results.

Next, using the same amount and order of layers as Base Model, I ran a number
of different models changing:

Number of filters on convuluted layer to 64:
Size of pooling layers to 4,4 and 2,2; and
Pool layers to 3,3 and 1,1.

I tried many combinations of the above changes and noticed the following:
Increasing the amount of filters alone did not alter the accuracy of the program.
using a pooling size of 1,1 and convenlutional layers 4,4, with 64 filters,  resulted in the most
accurate models but there was significant overfitting by around 9%.
with training model accuracy: 0.9808 and testing accuracy: 0.9079. Of course,
the model took around 3 times longer to train

I also tried dropout layers of .25, .33 and .5. While the overfitting issues was resolved,
the accuracy of the program declined significantly.


MODEL 2: Next, I added additional identical convolutional and pooling layers to the Base MODEL
like so (all other parameters staying the same as the base model):

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
                    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
                    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Flatten(),
tf.layers.Dense(128, activation = 'relu'),
tf.keras.layers.Dense(NUM_CATEGORIES, activation=softmax)
])

I got final training accuracy of 0.9803
and a testing accuracy of 0.9562, slightly better than than without
the additional layers (See model 2 training data below).


Model 3: I then added a third convenlutional layer this time with 64 filters and
a third identical pooling layer, with the intent of testing a pooling and convoluted layer
for each of the 3 dimensions of dimensions. The results were not that different from
model 2.

Model 4: In model 4 I added another hidden dense layer prior to the output layer,
this time with 256 units (see model and data below). The accuracy of the training
or testing models was not changed much. It did seem that having the densest
layer last provided the best results. train accuracy: 0.9633
test: accuracy of 0.9568


I then used the layering of model 4 but with different number filters and sizes
for convolutional layers, pool sizes, and dropout %.

First I attempted to determine the optimal dropout value trying .25, .33 and .5 with the
following results

 0.33: accuracy-Tr: 0.9242  accuracy-TE: 0.9557

 0.25: accuracyTR 0.9482 accuracy-TE: 0.9752

 0.50:  accuracyTR: .9271 Accuracy-TE: 0.9684

 I then deiced, perhaps in error, to use 0.25 dropout as it provided the best results.


 Then I altered the size of convolutional layers and found 2,2
 resulted in accuracyTR: 0.9166 and accuracy-TE: 0.9404.

From there I determined that decreasing the convenlutional layers alone
would not improve results.


 Next I altered the pooling size to 1,1 and tried various convulational size layers,
Some of the results are below. What I found was using pooling sizes of 1,1
actually led to more accurate results (probably b/c increased data),though
the model training time increased significantly.  I also found that I could
increase the dropout to 0.50 without decreasing the accuracy of the testing
models if the pooling size was 1,1. Ultimately, model 4 with layers 3,3
pooling 1,1(dropout .50) produced the best testing model, which I call "BESTFINALMODEL33":

The first time I ran the "BEST FINAL MODEL33" I got results:
accuracyTR: 0.9787
accuracy-TE: 0.9772;

The 2nd Time I ran "BEST FINAL MODEL33" I got results:
 accuracyTR: 0.9821
 accuracy-TE:: 0.9780


  ------- using 1,1 pooling size -----
 Using 4,4 layers and 32 filters, with pooling size 1,1, with a dropout of 0.33
 returned results accuracy-TR: 0.9817 accuracy-TE: 0.9535.

 using 2, 2 layers with 1,1 pooling i got: accuracyTR: 0.9826 accuracy-TE: 0.9687

 layers 3,3 pooling 1,1(dropout .33):  accuracyTR: 0.9816 accuracy-TE: 0.9516

 layers 3,3 pooling 1,1(dropout .50): accuracyTR: 0.9787 accuracy-TE: 0.9772

 (2nd time: accuracyTR: 0.9821 accuracy-TE:: 0.9780


The code which appears in get_model() herein is the BESTFINALMODEL33.

This was fun. Thank you.

Data and models below.
--------------------------------------------------------------------------------




BASE MODEL RESULTS:

Time 1: best train accuracy: 0.9693
Test accuracy: 0.9437

Time 2: best train :accuracy: 0.9747
333/333 - 1s - loss: 0.5622 - accuracy: 0.9197

Time 3: best train :accuracy: 0.9799
333/333 - 1s - loss: 0.5656 - accuracy: 0.9367

Time 4:  best train : accuracy: 0.9682
333/333 - 1s - loss: 0.4086 - accuracy: 0.9320

Time 5:
Epoch 1/10
500/500 [==============================] - 3s 5ms/step - loss: 6.8973 - accuracy: 0.1558
Epoch 2/10
500/500 [==============================] - 3s 5ms/step - loss: 2.0804 - accuracy: 0.4633
Epoch 3/10
500/500 [==============================] - 3s 5ms/step - loss: 1.3112 - accuracy: 0.6450
Epoch 4/10
500/500 [==============================] - 3s 6ms/step - loss: 0.9408 - accuracy: 0.7372
Epoch 5/10
500/500 [==============================] - 3s 6ms/step - loss: 0.7317 - accuracy: 0.7845
Epoch 6/10
500/500 [==============================] - 3s 5ms/step - loss: 0.5558 - accuracy: 0.8356
Epoch 7/10
500/500 [==============================] - 3s 6ms/step - loss: 0.4363 - accuracy: 0.8733
Epoch 8/10
500/500 [==============================] - 3s 6ms/step - loss: 0.3439 - accuracy: 0.9011
Epoch 9/10
500/500 [==============================] - 3s 5ms/step - loss: 0.2925 - accuracy: 0.9170
Epoch 10/10
500/500 [==============================] - 3s 5ms/step - loss: 0.2823 - accuracy: 0.9201
333/333 - 1s - loss: 0.5258 - accuracy: 0.8873
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 29, 29, 32)        416
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0
_________________________________________________________________
dense (Dense)                (None, 128)               802944
_________________________________________________________________
dense_1 (Dense)              (None, 43)                5547
=================================================================
Total params: 808,907
Trainable params: 808,907
Non-trainable params: 0

MODEL 2 Results:

1st result for model 2:

Epoch 1/10
500/500 [==============================] - 5s 11ms/step - loss: 2.0986 - accuracy: 0.5796
Epoch 2/10
500/500 [==============================] - 5s 10ms/step - loss: 0.4270 - accuracy: 0.8819
Epoch 3/10
500/500 [==============================] - 5s 10ms/step - loss: 0.2397 - accuracy: 0.9332
Epoch 4/10
500/500 [==============================] - 6s 11ms/step - loss: 0.1650 - accuracy: 0.9549
Epoch 5/10
500/500 [==============================] - 6s 11ms/step - loss: 0.1197 - accuracy: 0.9665
Epoch 6/10
500/500 [==============================] - 5s 11ms/step - loss: 0.1532 - accuracy: 0.9608
Epoch 7/10
500/500 [==============================] - 5s 11ms/step - loss: 0.1312 - accuracy: 0.9655
Epoch 8/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0949 - accuracy: 0.9725
Epoch 9/10
500/500 [==============================] - 5s 11ms/step - loss: 0.1462 - accuracy: 0.9647
Epoch 10/10
500/500 [==============================] - 5s 10ms/step - loss: 0.0853 - accuracy: 0.9778
333/333 - 1s - loss: 0.3193 - accuracy: 0.9440


2nd time I ran model 2 with summary:

Epoch 10/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0714 - accuracy: 0.9803
333/333 - 1s - loss: 0.2457 - accuracy: 0.9562
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 32)          0
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0
_________________________________________________________________
dense (Dense)                (None, 128)               147584
_________________________________________________________________
dense_1 (Dense)              (None, 43)                5547
=================================================================
Total params: 163,275
Trainable params: 163,275
Non-trainable params: 0



Model 4:
model = tf.keras.models.Sequential([

tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu',
                    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2,2)),



tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
                    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',
                    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation = 'relu'),

tf.keras.layers.Dense(256, activation = 'relu'),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

])
Model 4 Data:

Epoch 1/10
500/500 [==============================] - 6s 12ms/step - loss: 1.8794 - accuracy: 0.5282
Epoch 2/10
500/500 [==============================] - 6s 12ms/step - loss: 0.5379 - accuracy: 0.8495
Epoch 3/10
500/500 [==============================] - 6s 13ms/step - loss: 0.2943 - accuracy: 0.9194
Epoch 4/10
500/500 [==============================] - 6s 12ms/step - loss: 0.2533 - accuracy: 0.9312
Epoch 5/10
500/500 [==============================] - 6s 13ms/step - loss: 0.1698 - accuracy: 0.9530
Epoch 6/10
500/500 [==============================] - 6s 13ms/step - loss: 0.1800 - accuracy: 0.9526
Epoch 7/10
500/500 [==============================] - 6s 13ms/step - loss: 0.1418 - accuracy: 0.9603
Epoch 8/10
500/500 [==============================] - 6s 13ms/step - loss: 0.1398 - accuracy: 0.9627
Epoch 9/10
500/500 [==============================] - 6s 13ms/step - loss: 0.1164 - accuracy: 0.9683
Epoch 10/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1400 - accuracy: 0.9633
333/333 - 1s - loss: 0.1851 - accuracy: 0.9568
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 2, 2, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               32896     
_________________________________________________________________
dense_1 (Dense)              (None, 256)               33024     
_________________________________________________________________
dense_2 (Dense)              (None, 43)                11051     
=================================================================
Total params: 105,611
Trainable params: 105,611
Non-trainable params: 0
______________

BEST FINAL MODEL33 Results for Model 4 with 3,3 layers pooling 1,1 and dropout 0.50 (the best model):

Epoch 1/10
500/500 [==============================] - 39s 77ms/step - loss: 2.4850 - accuracy: 0.6565
Epoch 2/10
500/500 [==============================] - 38s 76ms/step - loss: 0.3538 - accuracy: 0.9077
Epoch 3/10
500/500 [==============================] - 38s 76ms/step - loss: 0.2207 - accuracy: 0.9424
Epoch 4/10
500/500 [==============================] - 38s 76ms/step - loss: 0.1542 - accuracy: 0.9589
Epoch 5/10
500/500 [==============================] - 38s 76ms/step - loss: 0.2094 - accuracy: 0.9490
Epoch 6/10
500/500 [==============================] - 38s 76ms/step - loss: 0.1059 - accuracy: 0.9725
Epoch 7/10
500/500 [==============================] - 38s 77ms/step - loss: 0.1317 - accuracy: 0.9695
Epoch 8/10
500/500 [==============================] - 38s 77ms/step - loss: 0.0993 - accuracy: 0.9772
Epoch 9/10
500/500 [==============================] - 38s 76ms/step - loss: 0.1250 - accuracy: 0.9715
Epoch 10/10
500/500 [==============================] - 38s 76ms/step - loss: 0.0891 - accuracy: 0.9787
333/333 - 3s - loss: 0.1106 - accuracy: 0.9772
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 28, 28, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 26, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 26, 26, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 24, 24, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 36864)             0         
_________________________________________________________________
dropout (Dropout)            (None, 36864)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               4718720   
_________________________________________________________________
dense_1 (Dense)              (None, 256)               33024     
_________________________________________________________________
dense_2 (Dense)              (None, 43)                11051     
=================================================================
Total params: 4,791,435
Trainable params: 4,791,435
Non-trainable params: 0
