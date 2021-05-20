import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print("Shape of train dataset:",x_train.shape)
print("Shape of test dataset:",x_test.shape)

x_train[0]

plt.matshow(x_train[2])
print(y_train[2])

#first five elements in the train dataset
y_train[:5]

fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()
for i in range(9):
    _, shu = cv.threshold(x_train[i], 30, 200, cv.THRESH_BINARY)
    axes[i].imshow(np.reshape(x_train[i], (28,28)), cmap="Greys")
plt.show()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train,epochs=3)

loss,accuracy = model.evaluate(x_test, y_test)
print("loss:",loss)
print("accuracy:",accuracy)

y_predicted = model.predict(x_test)
y_predicted[0]

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
plt.matshow(x_test[0])
print("Predicted value using model is:",np.argmax(y_predicted[0]))

#this is to be done because labels are whole numbers but 
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm

#visualizung confusion matrix
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel('Truth')

img=cv.imread(r"/content/sample_data/num7.png")[:,:,0]
img=np.invert(np.array([img]))
prediction=model.predict(img)
print("The Predicted result using model is ",np.argmax(prediction))
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()

