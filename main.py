import warnings
from Utils import utils
import tensorflow as tf
import os
import datetime

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# configuration
use_checkpoint = False
is_prediction = False

# variables
epochs = 5
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)  # Adam
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate) # SGD
logs_path = './Logs/5-0.01-Adam/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_size = 0.8
batch_size = 32
# path
train_path = os.path.join("./Datasets/", 'train')
test_path = os.path.join("./Datasets/", 'test')
# image size
img_width = 224
img_height = 224
img_channel = 3
classes = 2
# checkpoint
checkpoint_path = os.path.join("./Generator/", 'checkpoint.h5')
checkpoint_best_path = os.path.join("./Generator/", 'checkpoint_best.h5')
output_weight = os.path.join("./Generator/", 'output_weight.h5')

predict_image = './Generated/prediction.png'
label_names = {'cat': 0, 'dog': 1}
label_key = ['cat', 'dog']
# callbacks
# log_dir = './Logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = utils.get_callbacks(checkpoint_path=checkpoint_path, checkpoint_best_path=checkpoint_best_path,
                                logs_path=logs_path)

# data
data = utils.read_dataset(train_path)
train_size = int(len(data) * train_size)
train = data[:train_size]  # train dataset
val = data[train_size:]  # test dataset in training
test = utils.read_dataset(test_path, shuffle=False)  # test dataset after training

input_shape = (img_width, img_height, img_channel)

base_model = tf.keras.applications.vgg19.VGG19(
    include_top=None, weights='imagenet',
    input_shape=input_shape,
    classifier_activation='softmax'
)

# base_model.summary()

base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

#  def __init__(self, dataset, shuffle, classes, width, height, channels, batch_size, is_train=True):
train_loader = utils.DataGenerator(dataset=train,

                                   shuffle=True,
                                   classes=classes,
                                   width=img_width,
                                   height=img_height,
                                   channels=img_channel,
                                   batch_size=batch_size)
val_loader = utils.DataGenerator(dataset=val,
                                 shuffle=True,
                                 classes=classes,
                                 width=img_width,
                                 height=img_height,
                                 channels=img_channel,
                                 batch_size=batch_size)
test_loader = utils.DataGenerator(dataset=test,
                                  shuffle=False,
                                  classes=classes,
                                  width=img_width,
                                  height=img_height,
                                  channels=img_channel,
                                  batch_size=1,
                                  is_train=False)

if is_prediction:
    model.load_weights(output_weight)
    class_predict = []
    predictions = model.predict(test_loader)
    predictions = tf.argmax(predictions, 1).numpy()
    # print(predictions)
    print(predictions)
    for prediction in predictions:
        # prediction = int(tf.argmax(predictions, 1))

        if prediction == 1:
            class_predict.append('Dog')
            print('Dog')
        else:
            class_predict.append('Cat')
            print('Cat')
else:
    # train
    if use_checkpoint:
        # restore_weight = './Generator/checkpoint.h5'
        print(f"Restoring model weights from: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    else:
        model_json = model.to_json()
        with open('./Generator/graph.json', "w") as json_file:
            json_file.write(model_json)
    history = model.fit(train_loader,
                        epochs=epochs,
                        # steps_per_epoch=2,
                        # validation_steps=2,
                        validation_data=val_loader,
                        callbacks=callbacks)
    model.save_weights(output_weight)
# train_loader = initialization.DataGenerator(config_yaml, train, shuffle=True)
# val_loader = initialization.DataGenerator(config_yaml, val, shuffle=True)

# model = Models(config=config_yaml).network()
# trainer = trainer.Trainer(config=config_yaml, train_size=len(train), model=model, train_loader=train_loader,
#                           val_loader=val_loader)
# trainer.train()
#
# # Prediction accuracy
# predict = predictor.Predictor(config_yaml, test)
# class_predict = predict.predict()
# show_some_image_prediction(test, class_predict, path_to_save=config_yaml['dataset']['predict_image'])
