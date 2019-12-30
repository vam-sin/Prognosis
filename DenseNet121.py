import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.models import Sequential, Input, Model
import keras.backend as K

# classifier = keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=14)

# Parameters
height = 224
width = 224
depth = 3
nb_filters = 16

def DenseBlock(x, num, nb_filters, name):
	with K.name_scope(name):
		for i in range(num):
			x = Conv2D(nb_filters, (1,1), padding = 'same', activation = 'relu')(x)
			x = Conv2D(nb_filters, (3,3), padding = 'same', activation = 'relu')(x)

	return x

def TransitionBlock(x, nb_filters, name):
	with K.name_scope(name):
		x = Conv2D(nb_filters, (1,1), padding = 'same', activation = 'relu')(x)
		x = MaxPooling2D((2,2), 2)(x)

	return x

def DenseNet121(nb_filters):
	input_shape = (height, width, depth)
	chanDim = -1
	inputs = Input(shape = input_shape)

	x = Conv2D(nb_filters, (7,7), padding = 'same', activation = 'relu')(inputs)
	x = MaxPooling2D((3,3), 2, padding = 'same')(x)

	# First Set
	x = DenseBlock(x, 6, nb_filters, 'Dense1')
	x = TransitionBlock(x, nb_filters, 'Trans1')

	# Second Set
	x = DenseBlock(x, 12, nb_filters, 'Dense2')
	x = TransitionBlock(x, nb_filters, 'Trans2')

	# Third Set
	x = DenseBlock(x, 24, nb_filters, 'Dense3')
	x = TransitionBlock(x, nb_filters, 'Trans3')

	# Fourth Set
	x = DenseBlock(x, 16, nb_filters, 'Dense4')

	x = GlobalAveragePooling2D()(x)
	# x = Flatten()(x)
	x = Dense(1, activation = 'sigmoid')(x)

	model = Model(inputs, x, name = "densenet121")

	model.compile('adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	model.summary()

	return model

model = DenseNet121(nb_filters)


