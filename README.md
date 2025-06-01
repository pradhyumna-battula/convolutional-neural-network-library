Using this library which I created I trained on the mnist image databaseand got 95% test accuracy.
The weights are saved in .npy format in the form of a numpy array which can be loaded using load_weights() function on the network class.
If you want to use the weights, first create a network with the architecture given above, use the load_weights() function.
To get the output, use np.argmax(net.networkForward(input)[-1]).
The input shuld have the shape of (28,28,1) and the values shuld be from 0 to 1.
