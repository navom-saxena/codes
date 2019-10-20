from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential

# create a directory and 2 sub directory of test and train,
# create subdirectories of different classification classes

# parent directory path
parent_directory = '/Users/navomsaxena/codes/src/main/resources/machinelearning/'

# Initialising the CNN. Creating a sequential CNN classifier which uses neural network.
classifier = Sequential()

# Step 1 - Convolution. Adding convolution layer, it runs feature detector of size (3,3) and creates feature map.
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling. Pooling layer calculates max pool of matrix(2,2) and creates smaller matrix without losing feature.
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer and thereby pooling for improving efficiency.
# Other way is to add another fully connected layer.
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening. Creating flattened vector of pooled matrix.
classifier.add(Flatten())

# Step 4 - Full connection. ANN which is fully connected and using relu for hidden layer and sigmoid for output layer.
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN. Adding backpropagation, adam is type of Stochastic gradient, crossentropy is loss function.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
# image generator to modify image for given sizes. Test data image generator is also updated to rescale with same value.
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

test_image_generator = ImageDataGenerator(rescale=1. / 255)

training_set = train_image_generator.flow_from_directory(parent_directory + 'dataset/training_set',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')

test_set = test_image_generator.flow_from_directory(parent_directory + 'dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

# fitting training dataset to classifier and for validation - test dataset.
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)
