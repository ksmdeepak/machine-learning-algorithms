
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches

        # TODO: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length
        w, h = 28, 28
        self.train_x=train_x.reshape(train_x.shape[0],28,28,1)
        self.test_x=test_x.reshape(test_x.shape[0],28,28,1)

        # TODO: one hot encoding for train_y and test_y
        num_classes = 10

        w, h = num_classes, len(train_y)
        self.train_y = [[0 for x in range(w)] for y in range(h)]
        h = len(test_y)
        self.test_y = [[0 for x in range(w)] for y in range(h)]

        for (count,y) in enumerate(train_y):
            for column in range(0,num_classes):
                if y==column:
                    self.train_y[count][column] = 1
                else:
                    self.train_y[count][column] = 0
        for (count,y) in enumerate(test_y):
            for column in range(0,num_classes):
                if y==column:
                    self.test_y[count][column] = 1
                else:
                    self.test_y[count][column] = 0
        # self.train_y = np_utils.to_categorical(train_y)
        # self.test_y = np_utils.to_categorical(test_y)
        # num_classes = self.test_y.shape[1]
        # print(num_classes)
        # TODO: build you CNN model
        self.model = Sequential()
        self.model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64,(3, 3),activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(num_classes,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x,self.train_y,self.batch_size,self.epoches)
        pass

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")


    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)