from NN import NeuronsPC as n
import numpy as np

netshape234 = (4,6,6,4)
import random
import pyximport; pyximport.setup

class f:
    def __init__(self):
        pass

    def matAcreator(self,inx):
        degree = len(inx)
        templst = []
        for item in inx:
            temp1 = [item[0]**n for n in range(degree)]
            templst.append(temp1)
        return np.array(templst)

    def createinputs(self,rangex,rangey):#return an array
        lstx = [(random.random() * rangex)-rangex/2 for n in range(2)]
        lsty = [(random.random() * rangey) - rangey/2 for n in range(2)]
        return np.array(lstx+lsty).reshape((4,1))

    def createlabels(self,inx,iny,conditions):
        A = self.matAcreator(inx)
        B = np.array(iny).reshape((2,1))
        C = np.linalg.solve(A,B)
        e=C[0][0]
        labels = [0,0,0,0]
        if e > conditions[0]:
            labels[0] = np.float64(1.0)
        elif e > conditions[1]:
            labels[1] = np.float64(1.0)
        elif e > conditions[2]:
            labels[2] = np.float64(1.0)
        else:
            labels[3] = np.float64(1.0)
        return np.array(labels).reshape((4,1))

    def createinputs_labels(self,inputconds,labelconds):
        inputs = self.createinputs(*inputconds)
        labels = self.createlabels(inputs[0:2],inputs[2:4],labelconds)
        return inputs,labels

def randfloatrange(a,b):
    return random.random() * (b-a) + a

class g:
    def __init__(self):
        pass
    def generatec(self,want,labelconds):
        if want == 3:
            c = randfloatrange(labelconds[2], labelconds[2] + (labelconds[2]-labelconds[1])*10)
        elif want == 2:
            c = randfloatrange(labelconds[1],labelconds[2])
        elif want == 1:
            c = randfloatrange(labelconds[0],labelconds[1])
        elif want == 0:
            c = randfloatrange(labelconds[0] - (labelconds[1]-labelconds[0])*10, labelconds[0])
        else:
            print('ya fucked up')
            print(10/0)
        return c

    def create_labels(self,want,labelconds):
        labels = np.array([[np.float64(0.0) for n1 in range(4)]]).reshape((4,1))
        labels[want] = np.float64(1.0)
        c = 0
        c = self.generatec(want,labelconds)
        return c, labels

    def createinputlabel(self,want,labelconds,rangex):
        c, labels = self.create_labels(want,labelconds)
        m = randfloatrange(-100,100)
        inx = [randfloatrange(rangex[0],rangex[1]) for i1 in range(2)]
        iny = [inx[i] * m + c for i in range(2)]
        inp = np.array(inx+iny).reshape((4,1))
        return inp, labels

def addlist(lst1,lst2):
    for i in range(len(lst1)):
        lst1[i] += lst2[i]
    return lst1


def doshit(trains):
    tcost = 0
    for i in range(trains):
        tcost += tnetr.temptrain(G,rangex,labelconds123,10,100)
    return tcost




tnetr = n.NN(netshape234, 'npzcity/tnetwbr.npz', np.float64(1))

inputconds123 = (1000,1000)
labelconds123 = (100,0,-100)
rangex = (-100,100)
G = g()
tcost = 0
a = 1
costs = []

print('hello shit')


for i in range(25):
    costs.append(doshit(1000))
print(costs)

with open('tempvalnotes.txt','w') as data:
    data.write(str(costs))




'''
rtrains0t1000    =  0.7550527442675256
rtrains1000t2000 =  0.7552939670655324
rtrains2000t27000,  0.7544692177249194
                    0.7544374809275549
                    0.753.9431885591684
                    0.754.2188887313822
                    0.753.9808132819627
                    0.753.8487730388493
                    0.754.5149108870647
                    0.754.6515056926601
                    0.754.2199520449635
                    0.754.580422469184
                    0.754.5729508828185
                    0.753.9851056450531
                    0.753.8648449980426
                    0.754.1275717353103
                    0.753.5243027003052
                    0.753.5243027003052
                    0.754.2056053146277
                    0.754.5649384979073
                    0.754.5620242524225
                    0.754.1571150646218
                    0.754.274487151161
                    0.755.120911713357
                    0.754.8011853034653
                    0.753.4148673443234
                    0.755.1335721439257
rtrains27000 =      
                    
                    
                    







'''