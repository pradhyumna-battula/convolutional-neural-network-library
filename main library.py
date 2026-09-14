import numpy as np 
import scipy.signal as sp
import random 

class Layer:
    def __init__(self):
        self.trainable = False

    def forward(self,layerInput: np.ndarray) -> np.ndarray:
        return layerInput

    def backwards(self, delta_layerOutput: np.ndarray, layerInput: np.ndarray) -> np.ndarray:
        return delta_layerOutput

class DenseLayer(Layer):
    def __init__(self, inputs: int, outputs: int):
        self.trainable = True
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.randn(self.outputs,self.inputs) * np.sqrt(1 / self.inputs)
        self.bias = np.zeros((self.outputs,1))

    def forward(self, layerInput: np.ndarray) -> np.ndarray:
        return np.dot(self.weights, layerInput) + self.bias

    def backwards(self, delta_layerOutput: np.ndarray, layerInput: np.ndarray) -> dict:
        delta_bias = delta_layerOutput
        delta_weight = np.dot(delta_layerOutput, layerInput.transpose())
        delta_layerInput = np.dot(self.weights.transpose(), delta_layerOutput)
        return {
            "delta bias"        : delta_bias,
            "delta weight"      : delta_weight,
            "delta layerInput"  : delta_layerInput
        }

class ConvolutionalLayer(Layer):
    def __init__(self, inp_shape: tuple, k_size: int, depth: int):
        self.trainable    = True
        self.input_shape  = inp_shape               #input shape = width, height, depth
        self.input_depth  = inp_shape[2]
        self.input_width  = inp_shape[0]
        self.input_height = inp_shape[1]
        self.output_depth = depth
        self.kernel_shape = (depth, k_size, k_size, inp_shape[2])
        self.output_shape = (inp_shape[0] - k_size + 1, inp_shape[1] - k_size + 1, depth)
        self.weights = np.random.randn(*self.kernel_shape) * np.sqrt(1 / (k_size * k_size * inp_shape[2]))
        self.bias = np.zeros(self.output_shape)

    def forward(self, layerInput: np.ndarray) -> np.ndarray:
        output1 = np.copy(self.bias)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                output1[...,i] += sp.correlate2d(layerInput[:,:,j], self.weights[i,:,:,j], 'valid')
        return output1

    def backwards(self, delta_layerOutput: np.ndarray, layerInput: np.ndarray) -> dict:
        delta_weight = np.zeros(self.kernel_shape)
        delta_bias = delta_layerOutput
        delta_layerInput = np.zeros_like(layerInput)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                delta_weight[i,:,:,j] = sp.correlate2d(layerInput[:,:,j], delta_layerOutput[:,:,i], 'valid')
                delta_layerInput[:,:,j] += sp.convolve2d(delta_layerOutput[:,:,i], self.weights[i,:,:,j], 'full')
        return {
            "delta bias"        : delta_bias,
            "delta weight"      : delta_weight,
            "delta layerInput"  : delta_layerInput
        }

class LeakyReluLayer(Layer):
    def __init__(self):
        self.trainable = False

    def forward(self, layerInput: np.ndarray) -> np.ndarray:
        return np.where(layerInput < 0, layerInput * 0.1, layerInput)

    def backwards(self,  delta_layerOutput: np.ndarray, layerInput) -> np.ndarray:
        return np.where(layerInput < 0, delta_layerOutput * 0.1, delta_layerOutput)

class SigmoidLayer(Layer):
    def __init__(self):
        self.trainable = False

    def forward(self, layerInput: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-layerInput))

    def backwards(self, delta_layerOutput: np.ndarray, layerInput: np.ndarray) -> np.ndarray:
        tmp = self.forward(layerInput)
        return delta_layerOutput * tmp * (1 - tmp)

class SoftmaxEntropyLayer(Layer):
    def __init__(self):
        self.trainable = False

    def forward(self, layerInput: np.ndarray) -> np.ndarray:
        tmp = np.exp(layerInput - np.max(layerInput))
        return tmp / np.sum(tmp)

    def backwards(self, delta_layerOutput: np.ndarray, layerInput: np.ndarray) -> np.ndarray:
            return delta_layerOutput

class FlattenLayer(Layer):
    def __init__(self):
        self.trainable = False
    
    def forward(self, layerInput: np.ndarray) -> np.ndarray:
        return layerInput.reshape((layerInput.size, 1))

    def backwards(self, delta_layerOutput: np.ndarray, layerInput: np.ndarray) -> np.ndarray:
        return delta_layerOutput.reshape(layerInput.shape)

class Loss:
    def forward(self, networkOutput: np.ndarray, trueOutput: np.ndarray) -> float:
        return ((networkOutput - trueOutput)**2).mean()

    def backwards(self, networkOutput: np.ndarray, trueOutput: np.ndarray) -> np.ndarray:
        return 2 * (networkOutput - trueOutput)

class SparseCategoricalCrossEntropy(Loss):
    def forward(self, networkOutput: np.ndarray, trueOutput: int) -> float:
        networkOutput = np.clip(networkOutput, 1e-9, 1)
        return -np.log(networkOutput[trueOutput, 0])
    
    def backwards(self, networkOutput: np.ndarray, trueOutput: int) -> np.ndarray:
        tmp = np.copy(networkOutput)
        tmp[trueOutput, 0] -= 1
        return tmp

class Network:
    def __init__(self, layers: list, outputLoss: Loss):
        self.layers = layers
        self.outputLoss = outputLoss
        self.trainablity = []
        for i in range(len(layers)):
            self.trainablity.append(layers[i].trainable)

    def networkForward(self, networkInput: np.ndarray) -> list:
        z = [np.copy(networkInput)]
        for i in range(len(self.layers)):
            z.append(self.layers[i].forward(z[-1]))
        return z

    def lossLandscape(self, data_couple) -> dict:
        z = self.networkForward(data_couple[0])
        return {
            "Z" : z,
            "L" : self.outputLoss.forward(z[-1], data_couple[1])
        }

    def gradient(self, data_couple):
        cash = self.lossLandscape(data_couple)
        grad = []
        z = cash['Z']
        delta_loss = self.outputLoss.backwards(z[-1], data_couple[1])
        delta = [delta_loss]
        for i in range(len(self.layers)-1,-1,-1):
            if self.trainablity[i]:
                delta_vars = self.layers[i].backwards(delta_layerOutput = delta[-1], layerInput = z[i])
                delta.append(delta_vars["delta layerInput"])
                grad.append((delta_vars["delta weight"], delta_vars["delta bias"]))
            elif self.trainablity[i] == False:
                delta.append(self.layers[i].backwards(delta_layerOutput = delta[-1], layerInput = z[i]))
                grad.append("not trainable")
        grad = grad[::-1]
        delta = delta[::-1]
        return grad

    def set_tData(self, tdata: list):
        self.tData = tdata

    def create_batches(self, batch_size: int) -> list:
        data = self.tData.copy()
        random.shuffle(data)

        batches = [
            data[i:i+batch_size] for i in range(0,len(data),batch_size)
        ]

        return batches

    def train_batch(self, batch: list, learn_rate: float = 0.1):
        acc = []
        for i in range(len(self.layers)):       #sets zero
            if self.trainablity[i]:
                acc.append([np.zeros_like(self.layers[i].weights),np.zeros_like(self.layers[i].bias)])
            else:
                acc.append("not trainable")
        
        for i in range(len(batch)):             #sums all of the grads
            grad = self.gradient(batch[i])
            for j in range(len(self.layers)):
                if self.trainablity[j]:
                    acc[j][0] += grad[j][0]
                    acc[j][1] += grad[j][1]

        for i in range(len(self.layers)):
            if self.trainablity[i]:
                self.layers[i].weights -= acc[i][0] * learn_rate / len(batch)
                self.layers[i].bias -= acc[i][1] * learn_rate / len(batch)
                
    def train_epoch(self, batch_size: int, learn_rate: float = 0.1):
        batches = self.create_batches(batch_size)
        for batch in batches:
            self.train_batch(batch, learn_rate)

    def avg_loss(self):
        l = 0
        for i in range(len(self.tData)//a):
            l += self.lossLandscape(self.tData[i])["L"]
        return l / (len(self.tData)//a)
    
    def accuracy(self):
        l = 0
        for i in range(len(self.tData)//a):
            if np.argmax(self.networkForward(self.tData[i][0])[-1]) == self.tData[i][1]:
                l += 1
        return l * 100 / (len(self.tData)//a)

    def save_weights(self):
        l = []
        for i in range(len(self.layers)):
            if self.trainablity[i]:
                l.append((self.layers[i].weights, self.layers[i].bias))
            else: l.append("not trainable")
        np.save("weights.npy", np.array(l, dtype = object))

    def load_weights(self, weights: np.ndarray):
        for i in range(len(self.layers)):
            if self.trainablity[i]:
                self.layers[i].weights = weights[i][0]
                self.layers[i].bias = weights[i][1]

