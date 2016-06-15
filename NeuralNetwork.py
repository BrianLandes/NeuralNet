import math, random
import numpy as np

P = 1

def activation(value):
    #Map the output to a curve between 0 and 1
    #output = 1/(1+e^(-a/p))
    try:
        return 1/(1+math.exp(-value/P))
    except OverflowError:
        return 1/(1+math.exp(700/P))


class Neuron(object):

    def reset(self):
        return

    def adjustWeights(self, targetOutput, EW,\
       learningRate = None, useMomentum = None ):
        return

class OutputNeuron(Neuron):

    def __init__(self, outputCount, bias = None, learningRate = 0.3, useMomentum = False ):
        self.outputCount = outputCount
        self.inputs = list()
        self.weights = list()
        self.learningRate = learningRate
        self.useMomentum = useMomentum
        if bias is None:
            self.bias = random.random() * 1.0
        else:
            self.bias = bias
        self.processed = False
        self.rightReceived = 0
        self.errorRate = 0
        self.outputValue = 0
        self.changeMomentums = list()

    def getOutput(self):
        #If we've already processed it this time around then just skip the work and return
        #the answer
        if self.processed:
            return self.outputValue
        #Do some work, get the values from the inputs times by their respective weights
        #added all up
        totalSum = 0
        for i in range(len(self.inputs)):
            totalSum += self.inputs[i].getOutput() * self.weights[i]
        #Subtract the bias
        totalSum -= self.bias
        #Save the outputValue after putting it between 0 and 1
        self.outputValue = activation(totalSum)
        self.processed = True
        return self.outputValue


    def adjustWeights(self, targetOutput, EW,\
            learningRate = None, useMomentum = None ):
        #If this is an output layer neuron targetOutput will be a value and EW will be 0
        # if this is a hidden layer neuron targetOutput will be 0 and EW will be
        # one of the downstream connected neuron's weight times error rate
        #Only if we've processed it
        if learningRate is not None:
            self.learningRate = learningRate
        if useMomentum is not None:
            self.useMomentum = useMomentum
        if self.processed:
            runAdjustment = False
            #If this is an output layer neuron
            if self.outputCount == 0:
                runAdjustment = True
                self.errorRate = (targetOutput - self.outputValue) * \
                                 self.outputValue * ( 1 - self.outputValue )
            else:
                # if this is a hidden layer neuron
                # add the weighted error rate
                self.errorRate += EW
                # count on up
                self.rightReceived += 1
                # if that's all the downstream connected neurons that we're waiting for
                if self.rightReceived == self.outputCount:
                    runAdjustment = True
                    #calculate our actual error rate
                    self.errorRate *= self.outputValue * ( 1 - self.outputValue )

            if runAdjustment:
                for i in range(len(self.inputs)):
                    # Adjust the weight for each input based on its weight and output
                    if self.useMomentum:
                        self.changeMomentums[i] += self.inputs[i].getOutput() *\
                                       self.learningRate * self.errorRate
                        self.changeMomentums[i] /= 2.0
                        self.weights[i] += self.changeMomentums[i]
                    else:
                        self.weights[i] += self.inputs[i].getOutput() *\
                            self.learningRate * self.errorRate
                    # Then adjust the weight on up
                    self.inputs[i].adjustWeights( 0, self.weights[i] * self.errorRate,\
                        learningRate = learningRate, useMomentum = useMomentum )

        return

    def reset(self):
        if self.processed:
            self.processed = False
            self.rightReceived = 0
            self.errorRate = 0
            self.outputValue = 0
            for i in self.inputs:
                i.reset()

class InputNeuron(Neuron):

    def __init__(self ):
        self.inputValue = 0

    def getOutput(self):
        #return activation( self.inputValue )
        return self.inputValue


class Network(object):
    def __init__(self, inputs, outputs, hiddenLayerMakeup, structure = None, learningRate = 0.3, useMomentum = False ):
        self.inputs = inputs
        self.outputs = outputs
        self.inputNeurons = list()
        #self.outputNeurons = list()
        self.allNeurons = list()
        self.tempLayer = list()
        self.lastLayer = list()
        self.hiddenLayerMakeup = hiddenLayerMakeup
        #Create input layer
        for i in range(inputs):
            newNeuron = InputNeuron( )
            self.inputNeurons.append( newNeuron )
            self.allNeurons.append( newNeuron )
            self.lastLayer.append( newNeuron )
        #Create each hidden layer
        id = 0
        for i in range(len(hiddenLayerMakeup)):
            outputCount = outputs
            if i < len(hiddenLayerMakeup)-1:
                outputCount = hiddenLayerMakeup[i+1]

            for j in range(hiddenLayerMakeup[i]):
                s = None
                if structure is not None:
                    s = structure[id]
                self.createNeuron( outputCount, structure = s,\
                    learningRate = learningRate, useMomentum = useMomentum )
                id += 1
            self.lastLayer = self.tempLayer
            self.tempLayer = list()

        #Create the output layer
        for i in range(outputs):
            s = None
            if structure is not None:
                s = structure[id]
            self.createNeuron( 0, structure = s )
            id += 1
        self.outputNeurons = self.tempLayer


    def createNeuron(self, outputCount, structure = None, learningRate = 0.3, useMomentum = False ):
        b = None
        if structure is not None:
            b = structure[0] # float
        newNeuron = OutputNeuron( outputCount, bias = b,\
            learningRate = learningRate, useMomentum = useMomentum )
        if structure is not None:
            newNeuron.weights = structure[1] # list
        for n in self.lastLayer:
            newNeuron.inputs.append( n )
            if structure is None:
                newNeuron.weights.append( random.random() * 1.0 + 0.0000000000001 )
            newNeuron.changeMomentums.append( 0.0 )
        self.tempLayer.append( newNeuron )
        self.allNeurons.append( newNeuron )

    def getOutput(self, inputs):
        if len(inputs) != len(self.inputNeurons):
            raise NameError('Inputs not the same, expected ' + str(len(self.inputNeurons))\
                + ' but got ' +str(len(inputs))  )
        #Give the input neurons their inputs
        for i in range(len(inputs)):
            self.inputNeurons[i].inputValue = inputs[i]
#            self.inputNeurons[i].inputValue = activation( inputs[i] )

        #Reset the neurons
        for outputNeuron in self.outputNeurons:
            outputNeuron.reset()

        #Fill the output list and return it
        outputList = list()
        for outputNeuron in self.outputNeurons:
            outputList.append( outputNeuron.getOutput() )

        return outputList

    def adjustWeights( self, targetOutputs,\
            learningRate = None, useMomentum = None ):

        if len(targetOutputs) != len(self.outputNeurons):
            raise NameError('Outputs not the same, expected ' + str(len(self.outputNeurons))\
                + ' but got ' +str(len(targetOutputs))  )

        # make sure our targetOutputs are all between 0 and 1
        if np.less(targetOutputs,0.0).any():
            raise NameError('Outputs cannot be below 0.0 ')
        if np.greater(targetOutputs,1.0).any():
            raise NameError('Outputs cannot be above 1.0 ')
##        for i in range( len( targetOutputs) ):
##            if targetOutputs[i] > 1:
##                targetOutputs[i] = 1
##            elif targetOutputs[i] < 0:
##                targetOutputs[i] = 0
        for i in range( len( self.outputNeurons ) ):
            self.outputNeurons[i].adjustWeights( targetOutputs[i], 0,\
                learningRate = learningRate, useMomentum = useMomentum )

    def printStructure(self):
        print( 'Inputs:', self.inputs )
        print( 'Hidden Layer Makeup:', self.hiddenLayerMakeup )
        print( 'Outputs:', self.outputs )
        print( 'Structure:' )
        print ('[' )
        for neuron in self.allNeurons:
            if neuron in self.inputNeurons:
                continue
            print ('[', neuron.bias, ',', neuron.weights,'  ],' )
        print (']' )
        print()

if __name__ is '__main__':
    import time
    starttime = time.time()
    network = Network(2,1,[ 2 ] )
    results = list()
    count = 0
    while True:
        count += 1
        a = random.randint(0,1)
        b = random.randint(0,1)
        outputs = network.getOutput( [ a, b] )
        if len(results)>=500:
            del results[0]
        if (a or b) and not (a and b):
            network.adjustWeights( [1] )
            if outputs[0]>0.5:
                results.append( True )
            else:
                results.append( False )
        else:
            network.adjustWeights( [0] )
            if outputs[0]<0.5:
                results.append( True )
            else:
                results.append( False )
        ratio = 0.0
        for result in results:
            if result:
                ratio += 1.0
        ratio /= len(results)
        print( 'Accuracy: %s' % ratio )
        if len(results)>90 and ratio==1.0:
            break;
    print( count )
    secs = time.time() - starttime
    print( 'Per iteration time: ', secs/count )
#    network.printStructure()
#
#
#    network = Network(2,1,[ 3, 2 ], structure = [
#                [ 0.6633567588312668 ,
#                [-1.5375843295941822, 5.874506983837322]   ],
#                [ 0.4120877206601512 ,
#                [4.8660870490681996, -1.2138872657105046]   ],
#                [ 1.4782880551917223 ,
#                [5.693297909529692, 4.938349753203479]   ],
#                [ 0.4721888376272143 ,
#                [6.556709105012925, 1.0470135495294646, -5.415906557801467]   ],
#                [ 1.3969295933872528 ,
#                [1.7701308884840654, -4.126382784320171, 3.712795339141683]   ],
#                [ 0.17927038692452002 ,
#                [-6.498125924286288, 7.297829204670113]   ],
#                ]  )
#    results = list()
#    count = 0
#    while True:
#        count += 1
#        a = random.randint(0,1)
#        b = random.randint(0,1)
#        outputs = network.getOutput( [ a, b] )
#        if len(results)>=500:
#            del results[0]
#        if (a or b) and not (a and b):
#            network.adjustWeights( [1] )
#            if outputs[0]>0.5:
#                results.append( True )
#            else:
#                results.append( False )
#        else:
#            network.adjustWeights( [0] )
#            if outputs[0]<0.5:
#                results.append( True )
#            else:
#                results.append( False )
#        ratio = 0.0
#        for result in results:
#            if result:
#                ratio += 1.0
#        ratio /= len(results)
#        print( ratio )
#        if len(results)>90 and ratio==1.0:
#            break;
#    print( count )
#
#    network.printStructure()
