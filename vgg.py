from keras.applications import VGG16
import os

################# model ######################
conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (128, 128, 3))
conv_base.summary()

from keras import models, layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
conv_base.trainable = False
model.summary()

#################### train #####################
base_dir = r'dir\Einstein'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'v')

from keras import optimizers
model.compile(loss='binary_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(88,88),
    batch_size=20,
    class_mode='binary')
    
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(88, 88),
    batch_size=20,
    class_mode='binary')
history = model.fit_generator(train_generator,steps_per_epoch=128,epochs=20,
                              validation_data=validation_generator,validation_steps=50)

#保存模型
model.save('EM.h5')

#################### test #####################
from keras.preprocessing import image
from keras import models
import numpy as np
img = image.load_img(r'dir\Einstein\EM.jpg',target_size=(128,128,3))
img = np.array(img)
img = img/255
model = models.load_model(r'dir\Einstein\EMvgg_1.h5')
img = img.reshape(1,128,128,3)
pre = model.predict(img)
print('预测结果：',pre)
pre = model.predict_classes(img)
print('预测结果：',pre)


# 预测结果： [[0.00213499]]
# 预测结果： [[0]]