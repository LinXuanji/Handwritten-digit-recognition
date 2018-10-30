import scipy.misc as sm
import matplotlib.pyplot as plt
import numpy
import lnnlib

# load the model
weightih = lnnlib.load("NNmodelinputw.csv")
weightho = lnnlib.load("NNmodelhiddenw.csv")

# load the image
img_array = sm.imread("target/number.png", flatten=True)

# Inverse selection the image and reshape the image matrix to 28x28
img_data = 255.0 - img_array.reshape(784)
img_show = 255.0 - img_array.reshape((28, 28))
input_data = (img_data / 255.0 * 0.99) + 0.01

# recognize the image
output = lnnlib.transfer(weightih, weightho, input_data)
label = numpy.argmax(output)

# display output
plt.imshow(img_show, cmap='Greys', interpolation='None')
plt.title("identification completed, target is:" + str(label))
plt.show()

print("identification completed, target is:" + str(label))