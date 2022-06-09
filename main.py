import os
import random

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras import backend as K
import matplotlib.pyplot as pyplt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imgaug import augmenters as iau

#Объявление глобальных массивов, в которых хранятся все рисунки и классы

allimages = []
classlabels = []

#Цикл для заполнения изображений из дата-сета

for n in range(0, 40):
    dir = os.path.join('train', str(n)) #доступ к train
    imageDirs = os.listdir(dir) #список, содержащий имена файлов и директорий в каталоге
    for im in imageDirs:
        img = Image.open(dir + '/' + im) #проход по каждой папке
        img = img.resize((30, 30)) #уменьшение картинки до определенного размера
        img = img_to_array(img) #превращение картинки в массив
        allimages.append(img) #добавление байт в массив картинок
        classlabels.append(n) #добавление к классам

allimages = np.array(allimages) #создание массива allimages
classlabels = np.array(classlabels) #создание массива classlabels

print(allimages.shape, classlabels.shape) #количество данных(размерность массивов)

X_trn, X_tst, Y_trn, Y_tst = train_test_split(allimages, classlabels, test_size=0.2, random_state=42) #вычислить случайное разбиение на обучающие и тестовые наборы 20\80
print(X_trn.shape, X_tst.shape, Y_trn.shape, Y_tst.shape)

#Функция для подсчета картинок в классах
def num_imgs_in_class(classlabels):
    number = {}
    for n in classlabels:
        if n in number:
            number[n] += 1
        else:
            number[n] = 1
    return number

samp_distrib1 = num_imgs_in_class (Y_trn)

#Вывод представления структуры дата-сета на экран

def graph(num_classes):
    pyplt.bar(range(len(num_classes)), sorted(list(num_classes.values())), align='center')
    pyplt.xticks(range(len(num_classes)), sorted(list(num_classes.keys())), rotation=90, fontsize=7)
    pyplt.show()


graph(samp_distrib1)


#Функции для расширения дата-сета и выравнивание минимального числа элементов в каждом классе

def aug_imgs(imgs, prob):
    augments = iau.SomeOf((2, 4),
                      [
                          iau.Crop(px=(0, 4)),  # обрезка изображений с каждой стороны от 0 до 4 пикселей (выбирается случайным образом)
                          iau.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}), # увеличивает или уменьшает изображение с заданным коэффициентом(больше 1 удаляют изображение, меньше 1 — приближают)
                          iau.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}), # перемещение каждой точки изображения на заданное расстояние
                          iau.Affine(rotate=(-45, 45)),  # поворот на -45 до +45 градусов
                          iau.Affine(shear=(-10, 10))  # сдвиг от -10 до +10 градусов
                      ])
    sequen = iau.Sequential([iau.Sometimes(prob, augments)])
    ret_value = sequen.augment_images(imgs)
    return ret_value


def augmentate(imgs, classlabels):
    min_im = 500
    clss = num_imgs_in_class(classlabels)
    for i in range(len(clss)): #цикл для выравнивания классов под min_im
        if (clss[i] < min_im):
            add_n = min_im - clss[i] #считаем разницу между min_im и clss[i]
            images_for_aug = []
            labels_for_aug = []
            for j in range(add_n): #заполняем до минимума из расширенного дата-сета
                im_index = random.choice(np.where(classlabels == i)[0]) #выбор дополнения с таким же индексом
                images_for_aug.append(imgs[im_index]) #добавляем в конец массива imgs
                labels_for_aug.append(classlabels[im_index]) ##добавляем в конец массива classlabels
            aug_class = aug_imgs(images_for_aug, 1)
            aug_class_np = np.array(aug_class) #создание массива aug_class
            aug_labels_np = np.array(labels_for_aug) #создание массива labels_for_aug
            newimages = np.concatenate((imgs, aug_class_np), axis=0) #соединяем массивы вдоль оси 0 imgs и aug_class_np
            newlabels = np.concatenate((classlabels, aug_labels_np), axis=0) #соединяем массивы вдоль оси 0 classlabels и aug_labels_np
    return (newimages, newlabels)


X_trn, Y_trn = augmentate(X_trn, Y_trn)

print(X_trn.shape, X_tst.shape, Y_trn.shape, Y_tst.shape)

samples_distrib2 = num_imgs_in_class(Y_trn)
graph(samples_distrib2)

Y_trn = to_categorical(Y_trn, 43) #Преобразует вектор класса (целые числа) в двоичную классную матрицу
Y_tst = to_categorical(Y_tst, 43)


#Класс для нейронной сети

class NeuroNet:
  @staticmethod
  def build(wid, hei, dep, clss):
    mod = Sequential()
    inShape = (hei, wid, dep)
    if K.image_data_format() == 'channels_first':
      inShape = (dep, hei, wid)
    mod = Sequential()
    mod.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=inShape))
    mod.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    mod.add(MaxPool2D(pool_size=(2, 2)))
    mod.add(Dropout(rate=0.25))
    mod.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    mod.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    mod.add(MaxPool2D(pool_size=(2, 2)))
    mod.add(Dropout(rate=0.25))
    mod.add(Flatten())
    mod.add(Dense(500, activation='relu'))
    mod.add(Dropout(rate=0.5))
    mod.add(Dense(clss, activation='softmax'))
    return mod


#Обучение сети в 25 эпох

ep = 25
mod = NeuroNet.build(wid=30, hei=30, dep=3, clss=43)
mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #подготовка модели к работе
history = mod.fit(X_trn, Y_trn, batch_size=64, validation_data=(X_tst, Y_tst), epochs=ep) #обучение и тестирование(batch_size - ограничение примеров количество)


#Для отображения результатов

pyplt.style.use("ggplot")
pyplt.figure()
pyplt.plot(np.arange(0, ep), history.history["loss"], label="train_loss")
pyplt.plot(np.arange(0, ep), history.history["val_loss"], label="val_loss")
pyplt.plot(np.arange(0, ep), history.history["accuracy"], label="train_acc")
pyplt.plot(np.arange(0, ep), history.history["val_accuracy"], label="val_acc")
pyplt.title("Training Loss and Accuracy")
pyplt.xlabel("Epoch")
pyplt.ylabel("Loss/Accuracy")
pyplt.legend(loc="lower left")
pyplt.show()


#Проверка результатов обучения программы

Y_tst = pd.read_csv('Test.csv') #считывание файла Test.csv
classlabels = Y_tst["ClassId"].values #получение значений для ClassId
ims = Y_tst["Path"].values #получение значений для Path



newimages=[]

for imagefile in ims:
    im = Image.open(imagefile)
    im = im.resize((30,30))
    newimages.append(img_to_array(im))

X_tst=np.array(newimages)
recognized = mod.predict(X_tst)
classes_x=np.argmax(recognized, axis=1)

print(accuracy_score(classlabels, classes_x))
