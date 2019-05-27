from keras.layers import Conv2D, Dense, Flatten

# Variable sharing in Keras by same object, so we can do it like so
class DQNCNNModelKeras:
    def __init__(self, output_size):
        # original architecture
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        self.model_in = Conv2D(32, kernel_size=3, strides=(1, 1), activation="relu")
        self.model_hidden = [Conv2D(64, kernel_size=2, strides=(2, 2), activation="relu"),
                             Conv2D(64, kernel_size=2, strides=(1, 1), activation="relu"),
                             Flatten(), Dense(264, activation="relu")]
        self.model_out = Dense(output_size)

    def __call__(self, ob_placeholder):
        x = self.model_in(ob_placeholder)
        for layer in self.model_hidden:
            x = layer(x)
        return self.model_out(x)

# Variable sharing in Keras by same object, so we can do it like so
class DQNFCModelKeras:
    def __init__(self, output_size):
        # original architecture
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        self.model_in = Dense(64, activation="relu")
        self.model_hidden = [Dense(64, activation="relu")]
        self.model_out = Dense(output_size)

    def __call__(self, ob_placeholder):
        x = self.model_in(ob_placeholder)
        for layer in self.model_hidden:
            x = layer(x)
        return self.model_out(x)
