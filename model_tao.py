from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import load_model
import os
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_url = r'D:\AI_PROJECT\CNN_FACE\dataset\test_images'
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_url = r'D:\AI_PROJECT\CNN_FACE\dataset\img_val'

train_dataset = train_datagen.flow_from_directory(
    train_url,
    target_size=(150, 150),
    batch_size=128,
    class_mode='categorical'
)

validation_dataset = validation_datagen.flow_from_directory(
    validation_url,
    target_size=(150, 150),
    batch_size=128,
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
# Điều chỉnh số đơn vị trong lớp cuối cùng để phù hợp với số lượng nhãn trong dữ liệu của bạn
model.add(Dense(4, activation='softmax'))

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_dataset,
    batch_size=128,
    epochs=50,
    verbose=1,
    validation_data=validation_dataset
)

model.save(r'D:\AI_PROJECT\CNN_FACE\model\Final_test.h5')

model_Final = load_model(r'D:\AI_PROJECT\CNN_FACE\model\Final_test.h5')

# Đánh giá mô hình trên tập validation
score = model_Final.evaluate(validation_dataset, verbose=1)
print('Sai số: ', score[0])
print('Độ chính xác: ', score[1])

output_folder = r'D:\AI_PROJECT\CNN_FACE\img_Final_test'
plt.figure(figsize=(12, 6))
# Vẽ đồ thị accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Biểu đồ độ chính xác ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(output_folder, 'chinhxac.png'))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Biểu đồ tổn thất')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'saiso.png'))
plt.show()