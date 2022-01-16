import tensorflow as tf
import os
from Utils import utils

test_path = os.path.join("./Datasets/", 'test1')
graph_path = './Generator/graph.json'
# image size
img_width = 224
img_height = 224
img_channel = 3
classes = 2
learning_rate = 0.001
batch_size = 16

checkpoint_path = os.path.join("./Generator/", 'checkpoint.h5')
checkpoint_best_path = os.path.join("./Generator/", 'checkpoint_best.h5')

# input_shape = (img_width, img_height, img_channel)
# base_model = tf.keras.applications.vgg19.VGG19(
#     include_top=None, weights='imagenet',
#     input_shape=input_shape,
#     classifier_activation='softmax'
# )
#
# # base_model.summary()
#
# base_model.trainable = False
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(classes, activation='softmax')
#
# model = tf.keras.Sequential([
#     base_model,
#     global_average_layer,
#     prediction_layer
# ])
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer=optimizer,
#               loss=tf.keras.losses.sparse_categorical_crossentropy,
#               metrics=["accuracy"])


test = utils.read_dataset(test_path, shuffle=False)
test_loader = utils.DataGenerator(dataset=test,
                                  shuffle=False,
                                  classes=classes,
                                  width=img_width,
                                  height=img_height,
                                  channels=img_channel,
                                  batch_size=batch_size,
                                  is_train=False)


# model.load_weights(checkpoint_best_path)

json_file = open(graph_path, 'r')
load_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(load_json)
model.load_weights(checkpoint_best_path)

class_predict = []
predictions = model.predict(test_loader, batch_size=None)
for prediction in predictions:
    if prediction >= 0.5:
        class_predict.append('Dog')
        print('Dog')
    else:
        class_predict.append('Cat')
        print('Cat')

print(class_predict)