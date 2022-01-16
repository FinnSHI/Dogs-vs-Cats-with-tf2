import tensorflow as tf



class Predictor:
    def __init__(self, config, test_path, model_weight_path, img_width, img_height, img):
        self.config = config
        self.test_path = test_path
        self.model_weight_path = model_weight_path
        self.width = config['dataset']['width']
        self.height = config['dataset']['height']
        self.model = self.load_model()
        self.test_loader = DataGenerator(self.config, test_path, shuffle=False, is_train=False)

    def load_model(self):
        json_file = open(self.graph_path, 'r')
        load_json = json_file.read()
        json_file.close()

        model = tf.keras.models.model_from_json(load_json)
        model.load_weights(self.model_weight)
        return model

    def predict(self):
        class_predict = []
        predictions = self.model.predict(self.test_loader, batch_size=None)
        for prediction in predictions:
            if prediction >= 0.5:
                class_predict.append('Dog')
            else:
                class_predict.append('Cat')
        return class_predict

