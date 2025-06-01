Using this library which I created I trained on the mnist image databaseand got 95% test accuracy.
the weights are saved in .npy format in the form of a numpy array which can be loaded using load_weights() function on the network class.
If you want to use the weights, first create a network with the architecture given above, use the load_weights() function.
to get the output, use np.argmax(net.networkForward(input)[-1]).
the input shuld have the shape of (28,28,1) and the values shuld be from 0 to 1
if you want to know what each function does, paste the code in chatGPT and ask your doubt(im too lazy to do it myself lol)
