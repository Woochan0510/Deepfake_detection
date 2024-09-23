import numpy as np
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 경로 설정
real_dir = 'real_dir'
fake_dir = 'fake_dir'

# 이미지와 레이블 불러오기
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # 이미지 크기 조정
            images.append(img)
            labels.append(label)
    return np.array(images, dtype=np.float32), np.array(labels)

real_images, real_labels = load_images_from_folder(real_dir, 0)  # 실제 영상: 레이블 0
fake_images, fake_labels = load_images_from_folder(fake_dir, 1)  # 합성 영상: 레이블 1

# 데이터 병합 및 전처리
X = np.concatenate([real_images + fake_images], axis=0)
y = np.concatenate([real_labels + fake_labels], axis=0)

# 데이터 정규화
X = X / 255.0

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test)
)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 학습 과정 시각화
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 새로운 이미지 예측 함수
def predict_image(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("이미지를 불러올수 없습니다.")
        return None
    
    img = cv2.resize(img, (128,128))
    # 데이터 정규화 (0과 1사이 값으로 만듬)
    img = img.astype('float32') / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction >= 0.5:
        print("딥페이크 이미지 입니다.")
    else:
        print("실제 이미지 입니다.")
    
    return prediction

image_path = 'test_fake_img.jpg'

predict_image(model, image_path)
