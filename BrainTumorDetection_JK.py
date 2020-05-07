import numpy as np
import pandas as pd
import os
import cv2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sn


def CNN_Model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=128, 
                     kernel_size=(5,5), 
                     border_mode='valid',
                     input_shape=(64, 64, 3), 
                     data_format='channels_last', 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Converting the Images into flattened images
BrainTumorTrue = os.listdir('C:/Users/storm/Documents/Github/BrainTumorDetection/brain-mri-images-for-brain-tumor-detection/yes')
BrainTumorFalse = os.listdir('C:/Users/storm/Documents/Github/BrainTumorDetection/brain-mri-images-for-brain-tumor-detection/no')

target1 = np.full((1,155), 1)
target2 = np.full((1, 98), 0)
target = np.concatenate((target1,target2), axis = 1)
X_data = []

for file in BrainTumorTrue:
    img = cv2.imread('C:/Users/storm/Documents/Github/BrainTumorDetection/brain-mri-images-for-brain-tumor-detection/yes/'+file)
    face = cv2.resize(img, (64, 64))
    (b, g, r) = cv2.split(face) 
    img = cv2.merge([r,g,b])
    X_data.append(img)
    
for file in BrainTumorFalse:
    #face = misc.imread('../input/brain_tumor_dataset/yes/'+file)
    img = cv2.imread('C:/Users/storm/Documents/Github/BrainTumorDetection/brain-mri-images-for-brain-tumor-detection/no/'+file)
    face = cv2.resize(img, (64, 64))
    (b, g, r)=cv2.split(face) 
    img=cv2.merge([r,g,b])
    X_data.append(img)

X = np.squeeze(X_data)

#ploting the amount difference between number of positive and negative brain tumor cases
'''
df = pd.DataFrame(Counter(target[0]), index = ['0'])

plt.bar(0, df[1])
plt.bar(1, df[0])
plt.xticks(range(2), ["Positive", "Negative"])
plt.xlabel("Brain Tumor Presence")
plt.ylabel("Count")
plt.show()
'''

target = target.reshape(253,1)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

y_test = y_test.flatten()
y_true = y_test.tolist()

'''
#plot difference between training and test data
trainy = y_train.flatten()
true_y = trainy.tolist()

yt = y_true.count(0)

df = pd.DataFrame({"Positive" : [true_y.count(1), y_true.count(1)],
                   "Negative" : [true_y.count(0), y_true.count(0)]}, index=["Training Set", "Test Set"])

df.plot.bar(rot=0)
plt.xlabel("Datasets")
plt.ylabel("Count")
plt.show()
'''

target = np_utils.to_categorical(y_train)   
target = target.reshape((169, 2))

num_classes = target.shape[1]

cnn = CNN_Model(num_classes)

history = cnn.fit(X_train, target, epochs=10)

'''
#Plot Accuracy Score
plt.plot(np.linspace(0,10,10), history.history['accuracy'])
plt.xlabel("EPOCH")
plt.ylabel("Accuracy Score")
plt.show()



#Plot Loss Score
plt.plot(np.linspace(0,10,10), history.history['loss'])
plt.xlabel("EPOCH")
plt.ylabel("Loss Score")
plt.show()
'''

pred = cnn.predict(X_test)
yp = np.argmax(pred, 1)
accuracy_score = accuracy_score(y_test, yp, normalize=True, sample_weight=None)
recall_score = recall_score(y_test, yp, sample_weight=None)
print(accuracy_score)
print(recall_score)
confMat = confusion_matrix(y_true, yp)

df_cm = pd.DataFrame(confMat, index = ["Negative", "Positive"],
                  columns = ["Negative", "Positive"])
plt.figure(figsize = (2,2))
sn.heatmap(df_cm, annot=True)
