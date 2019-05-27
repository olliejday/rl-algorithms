from keras.layers import Input, Dense
from keras import Model
import keras.backend as keras_backend
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    inp1 = Input((2,))
    x1 = Dense(2)(inp1)
    x1 = Dense(1)(x1)

    m1 = Model(inp1, x1)

    m1.compile(loss='mse', optimizer='adam')

    inp2 = Input((2,))
    x2 = Dense(2)(inp2)
    x2 = Dense(1)(x2)

    m2 = Model(inp2, x2)


    def fn():
        m2.set_weights(m1.get_weights())

    x = np.array([[1, 1]])

    sess = keras_backend.get_session()
    sess.run(tf.global_variables_initializer())
    m1.summary()

    print("m1, ", m1.predict(x))
    print("m1 sess, ", sess.run(x1, feed_dict={inp1: x}))
    print("m2, ", m2.predict(x))
    print("m2 sess, ", sess.run(x2, feed_dict={inp2: x}))

    fn()
    m1.fit(x, np.array([[1]]), verbose=0)
    fn()

    print("m1, ", m1.predict(x))
    print("m1 sess, ", sess.run(x1, feed_dict={inp1: x}))
    print("m2, ", m2.predict(x))
    print("m2 sess, ", sess.run(x2, feed_dict={inp2: x}))