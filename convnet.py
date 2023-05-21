import os
from keras import layers
from keras import models
################# model ######################
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
      input_shape=(88, 88,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
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
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

from keras.preprocessing import image
import matplotlib.image as mpimg
from keras import models
import numpy as np
img = image.load_img(r'dir\Einstein\EM.jpg',target_size=(88,88,3))
img = np.array(img)
img = img/255
model = models.load_model(r'dir\Einstein\EM.h5')
img = img.reshape(1,88,88,3)
pre = model.predict(img)
print('预测结果：'，pre)

# -----------------------------------
# 预测结果： [[0.00376787]]
