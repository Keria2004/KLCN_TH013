import os
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ==== Cấu hình ====
DATA_DIR = './datasets/train'  # Thay đổi đường dẫn đến thư mục dữ liệu của bạn
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
NUM_CLASSES = 10  # bạn có 10 cảm xúc
MODEL_PATH = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

# ==== 1. Load mô hình gốc ====
base_model = load_model(MODEL_PATH, compile=False)

# ==== 2. Tạo output head mới ====
x = base_model.layers[-3].output  # lấy đầu ra trước softmax cũ
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=output)

# ==== 3. Chuẩn bị Data ====
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ==== 4. Callback ====
checkpoint = ModelCheckpoint(
    'fine_tuned_best.hdf5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)

# ==== 5. Phase 1 - Train Head (Freeze backbone) ====
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print("🔹 Phase 1: Train phần đầu ra (backbone bị đóng băng)...")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# ==== 6. Phase 2 - Fine-tune sâu (Mở 20 tầng cuối) ====
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🔹 Phase 2: Fine-tune các tầng cuối...")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# ==== 7. Lưu model cuối ====
model.save('./trained_models/emotion_models/fine_tuned_miniXCEPTION_final.hdf5')
print("✅ Huấn luyện hoàn tất! Model đã được lưu: fine_tuned_miniXCEPTION_final.hdf5")

# ==== 8. Biểu đồ Loss / Accuracy ====
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
