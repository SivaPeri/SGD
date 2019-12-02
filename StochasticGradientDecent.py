import numpy as np
import math

I = np.array([[0.4, -0.7],[0.3,-0.5],[0.6,0.1],[0.2,0.4],[0.1,-0.2]])
W1 = np.array([[0.1, 0.4],[-0.2,0.2]])
W2 = np.array([0.2, -0.5])
output = np.array([0.1,0.05,0.3,0.25,0.12])
learningrate = 0.6
count = 0
NumberOfLoops = output.size

while (count < NumberOfLoops):
    print("******************************************* ")
    print("Count: ",count)
    #step3
    W1Transpose=W1.transpose()
    IW1=np.matmul(W1Transpose,I[count])

    #step4
    reluarray1 = (1/(1+(np.exp((-1*IW1[0])))))
    reluarray2 = (1/(1+(math.exp((-1*IW1[1])))))

    relu1=np.array([reluarray1,reluarray2])
    #print("relu1",relu1)
    
    #step5
    W2Transpose = W2.transpose()
    #print("W2Transpose",W2Transpose)
    W2relu1 = np.matmul(W2Transpose,relu1)

    #step6
    relu2 = (1/(1+(np.exp((-1*W2relu1)))))

    #step7
    error = math.pow((output[count] - relu2),2)
    print("Error: ",error)

    #step8 lets adjust weight
    d=(output[count] - relu2)*relu2*(1-relu2)


    Y = np.multiply(relu1,d)
    #print("Y:", Y)
    
    #step9
    deltaW = np.multiply(Y,learningrate)
    #print("deltaW:", deltaW)
    
    #step10
    e = np.multiply(W2,d)
    #print("e:", e)

    #step11
    d1Sub1=e[0] * relu1[0] * (1-relu1[0])
    d1Sub2=e[1] * relu1[1] * (1-relu1[1])

    d1 = np.array([d1Sub1,d1Sub2])
    #print("d1:", d1)
    
    #step12
    #IRow = I.reshape(1,2)
    #d1Column = d1.reshape(2,1)
    #print(IRow)
    #print(d1Column)
    #X1 = IRow.dot(d1Column)
    #print(X1) => Not working
    
    #step12 - workaround
    X11=I[count][0] * d1[0]
    X12=I[count][0] * d1[1]
    X21=I[count][1] * d1[0]
    X22=I[count][1] * d1[1]

    X = np.array([[X11,X12],[X21,X22]])
    #print("X:", X)

    #step13
    deltaV = np.multiply(X,learningrate)
    #print("deltaV:", deltaV)

    #step14
    NewW1 = W1 + deltaV
    NewW2 = W2 + deltaW
    print("NewW1: ",NewW1)
    print("NewW2: ",NewW2)

    #step15
    W1 = NewW1
    W2 = NewW2

    #step16
    #Reiterate
    count +=1




