import numpy as np
import h5py
# from matplotlib import *

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

# Standardising the Data Set
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

n_x = train_set_x_flatten.shape[0]
n_y = 1;

#DEFINING THE NEURAL NETWORK

nn_layers = [n_x,20,7,5,n_y]

#  FUNCTION: initialise_parameters
def initialise_parameters(layer_dims):
 
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
 
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
 
            # unit tests
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
 
    return parameters

#  FUNCTION: linear_forward
 
def linear_forward(A, W, b):
 
    Z = np.dot(W, A) + b
 
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
 
    return Z, cache

#  FUNCTION: sigmoid
def sigmoid(Z):
 
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
 
    A = np.maximum(0,Z)
 
    assert(A.shape == Z.shape)
 
    cache = Z
    return A, cache

# FUNCTION: linear_activation_forward
 
def linear_activation_forward(A_prev, W, b, activation):
 
    Z, linear_cache = linear_forward(A_prev, W, b)
 
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
 
    elif activation == "relu":
        A, activation_cache = relu(Z)
 
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
 
    return A, cache

#  FUNCTION: L_model_forward
 
def L_model_forward(X, parameters):
 
    caches = []
    A = X 									  # Activation from first layer (A0) is the input itself
    L = len(parameters) // 2                  # number of layers in the neural network
 
    # Implement [LINEAR -&amp;gt; RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        w_l = parameters['W' + str(l)]
        b_l = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, w_l, b_l, activation = "relu")
        caches.append(cache)
 
    # Implement LINEAR -&amp;gt; SIGMOID. Add "cache" to the "caches" list.
    w_L = parameters['W' + str(L)]
    b_L = parameters['b' + str(L)]
    Yhat, cache = linear_activation_forward(A, w_L, b_L, activation = "sigmoid")
    caches.append(cache)
 
    assert(Yhat.shape == (1,X.shape[1]))
 
    return Yhat, caches

#  FUNCTION: Computing Cross Entropy Cost
def compute_cost(Yhat, Y):
 
    m = Y.shape[1]
 
    # Compute loss from AL and Y.
    logprobs = np.dot(Y, np.log(Yhat).T) + np.dot((1-Y), np.log(1-Yhat).T)
 
    cost = (-1./m) * logprobs 
 
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
 
    return cost

#  FUNCTION: linear_backward
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]
 
    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
 
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
 
    return dA_prev, dW, db

def sigmoid_backward(dA, cache):
 
    Z = cache
 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s) #Because
 
    assert (dZ.shape == Z.shape)
 
    return dZ

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

#  FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

#  FUNCTION: L_model_backward
 
def L_model_backward(Yhat, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = Yhat.shape[1]
    Y = Y.reshape(Yhat.shape) # after this line, Y is the same shape as AL
 
    # Initializing the backpropagation
    dAL = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat)) # derivative of cost with respect to AL
 
    # Lth layer (SIGMOID -&amp;gt; LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
 
    for l in reversed(range(L-1)):
        # lth layer: (RELU -&amp;gt; LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
 
    return grads

#  FUNCTION: update_parameters
 
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network
 
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

# EVERYTHING TOGETHER IN A NN

#  FUNCTION: L_layer_model
 
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
 
    costs = []                         # keep track of cost
 
    # Parameters initialization.
    parameters = initialise_parameters(layers_dims)
 
    # Loop (gradient descent)
    for i in range(0, num_iterations):
 
        # Forward propagation: [LINEAR -&amp;gt; RELU]*(L-1) -&amp;gt; LINEAR -&amp;gt; SIGMOID.
        AL, caches = L_model_forward(X, parameters)
 
        # Compute cost.
        cost = compute_cost(AL, Y)
 
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
 
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
 
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
 
    return parameters

np.random.seed(1)
fit_params = L_layer_model(train_set_x, train_set_y, nn_layers, num_iterations = 2500, print_cost = True)

def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
 
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
 
    # convert probs to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
 
    # print results
    print("Accuracy: "  + str(np.sum((p == y)/m)))
 
    return p

pred_train = predict(train_set_x, train_set_y, fit_params)
pred_test = predict(test_set_x, test_set_y, fit_params)

#Checking for own image
import scipy
#from PIL import Image
from scipy import ndimage

my_image_filename = "my_image.png"   # change this to the name of your image file 

# We preprocess the image to fit the LR algorithm.
fname = "/home/mudit/ML_Projects/Cat_Detection/" + my_image_filename
image = np.array(scipy.ndimage.imread(fname, flatten=False, mode = 'RGB'))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T

# Remember that my_image contains the processed image, previously prepared

my_predicted_image = predict(my_image, [0], fit_params)

# plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", the algorithm predicts a \"" + 
	classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")