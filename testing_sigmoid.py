# %% 1 
# Package imports 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import matplotlib 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
# Display plots inline and change default figure size 
#matplotlib inline 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
 
# %% 2 
np.random.seed(3) 
X, y = sklearn.datasets.make_moons(200, noise=0.20) 
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.viridis) 
 
# %% 3 
# Train the logistic rgeression classifier 
clf = sklearn.linear_model.LogisticRegressionCV() 
clf.fit(X, y) 
 
# %% 4 
# Helper function to plot a decision boundary. 
# If you don't fully understand this function don't worry, it just generates the contour plot below. 
def plot_decision_boundary(pred_func): 
    #Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    #Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis) 
    plt.contour(xx, yy, Z, colors='k', linewidths=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis) 
 
# %% 12 
# Plot the decision boundary 
#plot_decision_boundary(lambda x: clf.predict(x)) 
#plt.title("Logistic Regression") 
 
# %% 15 
num_examples = len(X) # training set size 
nn_input_dim = 2 # input layer dimensionality 
nn_output_dim = 2 # output layer dimensionality 
 
# Gradient descent parameters (I picked these by hand) 
epsilon = 0.01 # learning rate for gradient descent 
reg_lambda = 0.01 # regularization strength 

#def sigmoid(x):
    #return 1 / (1 + np.exp(-x))

#def sigmoid_derivative(x):
    #return x * (1 - x)

# %% 7 
# Helper function to evaluate the total loss on the dataset 
def calculate_loss(model): 
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    # Forward propagation to calculate our predictions 
    z1 = X.dot(W1) + b1 
    a1 = sigmoid(z1) 
    z2 = a1.dot(W2) + b2 
    exp_scores = np.exp(z2) 
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    # Calculating the loss 
    corect_logprobs = -np.log(probs[range(num_examples), y]) 
    data_loss = np.sum(corect_logprobs) 
    # Add regulatization term to loss (optional) 
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) 
    return 1./num_examples * data_loss 
 
# %% 8 
# Helper function to predict an output (0 or 1) 
def predict(model, x): 
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    # Forward propagation 
    z1 = x.dot(W1) + b1 
    a1 = sigmoid(z1) 
    z2 = a1.dot(W2) + b2 
    exp_scores = np.exp(z2) 
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    return np.argmax(probs, axis=1) 

def sigmoid(x):
    return 1/ (1 + np.exp(np.exp(np.clip(-x, -709, 709))))
 
# %% 16 
# This function learns parameters for the neural network and returns the model. 
# - nn_hdim: Number of nodes in the hidden layer 
# - num_passes: Number of passes through the training data for gradient descent 
# - print_loss: If True, print the loss every 1000 iterations 
def build_model(nn_hdim, num_passes=20000, print_loss=False): 
 
    # Initialize the parameters to random values. We need to learn these. 
    np.random.seed(0) 
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) 
    b1 = np.zeros((1, nn_hdim)) 
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim) 
    b2 = np.zeros((1, nn_output_dim)) 
 
    # This is what we return at the end 
    model = {} 
 
    # Gradient descent. For each batch... 
    for i in range(0, num_passes): 
 
        # Forward propagation 
        z1 = X.dot(W1) + b1 
        a1 = np.tanh(z1) 
        z2 = a1.dot(W2) + b2 
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
 
        # Backpropagation 
        delta3 = probs 
        delta3[range(num_examples), y] -= 1 
        dW2 = (a1.T).dot(delta3) 
        db2 = np.sum(delta3, axis=0, keepdims=True) 
        delta2 = delta3.dot(W2.T) * (a1*(1-a1))
        dW1 = np.dot(X.T, delta2) 
        db1 = np.sum(delta2, axis=0) 
 
        # Add regularization terms (b1 and b2 don't have regularization terms) 
        dW2 += reg_lambda * W2 
        dW1 += reg_lambda * W1 
 
        # Gradient descent parameter update 
        W1 += -epsilon * dW1 
        b1 += -epsilon * db1 
        W2 += -epsilon * dW2 
        b2 += -epsilon * db2 
 
        # Assign new parameters to the model 
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
 
        # Optionally print the loss. 
        # This is expensive because it uses the whole dataset, so we don't want to do it too often. 
        if print_loss and i % 1000 == 0: 
          print("Loss after iteration %i: %f" %(i, calculate_loss(model))) 
    
   
    return model 

plt.figure(figsize=(16, 32)) 
hidden_layer_dimensions = [1, 2, 3, 4, 5] 
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5,2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model=build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
    predictions=predict(model, X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    print(f"Hidden Layer size {nn_hdim}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
   
plt.show()
 
# %% 17 
# Build a model with a 3-dimensional hidden layer 
#model = build_model(3, print_loss=True) 
 
# Plot the decision boundary 


 
# %% 14 
plt.figure(figsize=(16, 32)) 
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50] 
for i, nn_hdim in enumerate(hidden_layer_dimensions): 
    plt.subplot(5, 2, i+1) 
    plt.title('Hidden Layer size %d' % nn_hdim) 
    model = build_model(nn_hdim) 
    plot_decision_boundary(lambda x: predict(model, x)) 
    
predictions = predict(model, X)
loss = calculate_loss(model, X, y)
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)

# Print the results
print("Loss:", loss)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
plt.show()
