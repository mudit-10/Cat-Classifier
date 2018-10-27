import numpy as np
import h5py

def print_dimensions(m_train, m_test, num_px, train_set_x_orig, train_set_y, test_set_x_orig, test_set_y):
	print ("Dataset dimensions:")
	print ("Number of training examples: m_train = " + str(m_train))
	print ("Number of testing examples: m_test = " + str(m_test))
	print ("Height/Width of each image: num_px = " + str(num_px))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_set_x shape: " + str(train_set_x_orig.shape))
	print ("train_set_y shape: " + str(train_set_y.shape))
	print ("test_set_x shape: " + str(test_set_x_orig.shape))
	print ("test_set_y shape: " + str(test_set_y.shape))

#Retrieving the Data
train_dataset = h5py.File('/home/mudit/ML_Projects/Cat_Detection/Cat_Detection_Dataset/train_catvnoncat.h5','r')
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('/home/mudit/ML_Projects/Cat_Detection/Cat_Detection_Dataset/test_catvnoncat.h5','r')
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])

classes = np.array(test_dataset['list_classes'][:])

train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

# Processing the Data
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print_dimensions(m_train, m_test, num_px, train_set_x_orig, train_set_y, test_set_x_orig, test_set_y)

# Reshape the training and test examples ( Flattening so that 4D --> 2D)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T # -1 performs merging of the other 3 dimensions
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# Standardising the Data Set
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#Defining the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(train_set_x.T,train_set_y.T.ravel())

print("Dimensions of Weights (1 X (64*64*3) ): ",lr.coef_.shape)
print("Coefficient Values: ",lr.coef_)
print("Bias/Intercept Value: ",lr.intercept_)

#Now Predicting
Y_prediction = lr.predict(test_set_x.T)
abs_error = np.abs( Y_prediction - test_set_y )
mean_error = np.mean ( abs_error )

# Appending format to a string means fill in that value with numbers in append
print("Test Accuracy: {} %".format(100 * (1 - mean_error)) )

#Checking for own image
import scipy
#from PIL import Image
from scipy import ndimage

my_image_filename = "cat.png"   # change this to the name of your image file 

# We preprocess the image to fit the LR algorithm.
fname = "/home/mudit/ML_Projects/Cat_Detection/" + my_image_filename
image = np.array(scipy.ndimage.imread(fname, flatten=False, mode = 'RGB'))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T

my_predicted_image = lr.predict(my_image.T)

# plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", the algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


