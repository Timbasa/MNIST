import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

batch_size = 64
nb_classes = 10
epoch = 12
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
pool_size = (2, 2)


def model_withoutdropout():
    model_without = Sequential()
    model_without.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding="valid"))
    model_without.add(Conv2D(32, (3, 3), activation='relu'))
    model_without.add(MaxPool2D(pool_size=pool_size))
    model_without.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_without.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_without.add(MaxPool2D(pool_size=(2, 2)))
    model_without.add(Flatten())
    model_without.add(Dense(128, activation='relu'))
    model_without.add(Dense(nb_classes, activation='softmax'))
    model_without.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.RMSprop(),
                              metrics=['accuracy'])
    model_without.summary()
    return model_without


def model_withdropout():
    model_with = Sequential()
    model_with.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding="valid"))
    model_with.add(Conv2D(32, (3, 3), activation='relu'))
    model_with.add(MaxPool2D(pool_size=pool_size))
    model_with.add(Dropout(0.2))
    model_with.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_with.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_with.add(MaxPool2D(pool_size=(2, 2)))
    model_with.add(Flatten())
    model_with.add(Dense(128, activation='relu'))
    model_with.add(Dropout(0.25))
    model_with.add(Dense(nb_classes, activation='softmax'))
    model_with.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.RMSprop(),
                              metrics=['accuracy'])
    model_with.summary()
    return model_with


def plot_history(history_withDropout_1,history_withDropout_2,history_withDropout_3,history_withoutDrop_1,history_withoutDrop_2,history_withoutDrop_3):
    plt.plot(history_withDropout_1.history['val_loss'], 'b-')
    plt.plot(history_withoutDrop_1.history['val_loss'], 'r--')
    plt.plot(history_withDropout_2.history['val_loss'], 'b-')
    plt.plot(history_withDropout_3.history['val_loss'], 'b-')
    plt.plot(history_withoutDrop_2.history['val_loss'], 'r--')
    plt.plot(history_withoutDrop_3.history['val_loss'], 'r--')
    plt.title('model loss about whether use dropout or not')
    plt.xlabel('epoch')
    plt.ylabel('validation loss')
    plt.grid()
    plt.legend(['withDropout', 'withoutDropout'], loc=1)
    plt.show()

def main():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    withdropout_1 = model_withdropout()
    history_withdropout_1 = withdropout_1.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                                                verbose=1, validation_data=(X_test, Y_test))
    withdropout_2 = model_withdropout()
    history_withdropout_2 = withdropout_2.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                                                verbose=1, validation_data=(X_test, Y_test))
    withdropout_3 = model_withdropout()
    history_withdropout_3 = withdropout_3.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                                                verbose=1, validation_data=(X_test, Y_test))

    withoutdropout_1 = model_withoutdropout()
    history_withoutdropout_1 = withoutdropout_1.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                                                      verbose=1, validation_data=(X_test, Y_test))

    withoutdropout_2 = model_withoutdropout()
    history_withoutdropout_2 = withoutdropout_2.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                                                      verbose=1, validation_data=(X_test, Y_test))

    withoutdropout_3 = model_withoutdropout()
    history_withoutdropout_3 = withoutdropout_3.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                                                      verbose=1, validation_data=(X_test, Y_test))

    plot_history(history_withdropout_1, history_withdropout_2, history_withdropout_3, history_withoutdropout_1, history_withoutdropout_2, history_withoutdropout_3)


if __name__ == "__main__":
    main()