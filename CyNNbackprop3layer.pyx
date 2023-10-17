# cython: profile=True
cimport numpy as np
import numpy as np
import copy
from numpy import asarray as asr
from random import randint as random1
import random
import random
from cython cimport boundscheck,wraparound,initializedcheck,nonecheck, nogil, profile
from cpython cimport array
from cython.parallel cimport prange, parallel
import concurrent.futures
import graphviz
import os
from time import time
from scipy.special import expit
import math
from matplotlib import pyplot as plt

#profile : True
#hello matery
#hello

ctypedef Py_ssize_t ind
ctypedef double[:,:] mv
ctypedef np.float64_t flt


def imshow(array):
   plt.imshow(array.reshape((28,28)),cmap='gray',vmin=0,vmax=1)
   plt.show()




cdef extern from "math.h" nogil:
   double exp(double num)
   double fabs(double num)


cdef double[:,:] expitt(double[:,:] arr, double[:,:] out):
   cdef ind i
   for i in range(arr.shape[0]):
      out[i,0] = 1./(1.+exp(-arr[i,0]))
   return out[:,:]



@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline void arrdexpitfromexpitIP(double[:,:] expitT, double[:,:] out) nogil:
   cdef ind row
   for row in range(expitT.shape[0]):
      out[row,0] = expitT[row,0] * (1. - expitT[row,0])


@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] matmul(double[:,:] arr1, double[:,:] mat1,double[:,:] out) nogil:
   cdef Py_ssize_t row, column
   cdef int rows, columns
   cdef double s
   for row in range(mat1.shape[0]):
      s = 0.0
      for column in range(mat1.shape[1]):
         s += arr1[column,0] * mat1[row,column]
      out[row,0] = s
   return  out[:,:]


@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] arrsqukeepsgn(double[:,:] arr, double[:,:] out):
   cdef ind i
   for i in range(arr.shape[0]):
      out[i,0] = arr[i,0] * fabs(arr[i,0])
   return out[:,:]





@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] arradd(double[:,:] arr1, double[:,:] arr2, double[:,:] out) nogil:
   cdef ind row
   for row in range(arr1.shape[0]):
      out[row,0] = arr1[row,0] + arr2[row,0]
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] arrsub(double[:,:] arr1, double[:,:] arr2, mv out) nogil:
   cdef ind row
   for row in range(arr1.shape[0]):
      out[row,0] = arr1[row,0] - arr2[row,0]
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] arrmul(double[:,:] arr1, double[:,:] arr2, mv out) nogil:
   cdef ind row
   for row in range(arr1.shape[0]):
      out[row,0] = arr1[row,0] * arr2[row,0]
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] arrmulsca(double[:,:] arr1, double scalar, mv out) nogil:
   cdef ind row
   for row in range(arr1.shape[0]):
      out[row,0] = arr1[row,0] * scalar
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double arrsum(double[:,:] arr1) nogil:
   cdef ind row
   cdef double sum = 0.0
   for row in range(arr1.shape[0]):
      sum += arr1[row,0]
   return sum

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] dcopymv(double[:,:] arr1):
   cdef double[:,:] out1 = np.zeros((arr1.shape[0],arr1.shape[1]))
   cdef ind row, column
   for row in range(arr1.shape[0]):
      for column in range(arr1.shape[1]):
         out1[row,column] += arr1[row,column]
   return out1[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] arrsqu(double[:,:] arr1, mv out) nogil:
   cdef ind row
   for row in range(arr1.shape[0]):
      out[row,0] = arr1[row,0] * arr1[row,0]
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] matmulsca(double[:,:] arr1, double scalar, mv out) nogil:
   cdef ind row, column
   for row in range(arr1.shape[0]):
      for column in range(arr1.shape[1]):
         out[row,column] = arr1[row,column] * scalar
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] mataddsca(double[:,:] arr1, double scalar, mv out) nogil:
   cdef ind row, column
   for row in range(arr1.shape[0]):
      for column in range(arr1.shape[1]):
         out[row,column] = arr1[row,column] + scalar
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline void errortowchaDip(mv D, mv sigvals, mv outm):
   cdef ind row, column
   cdef double s
   for row in range(outm.shape[0]):
      s = D[row,0]
      for column in range(outm.shape[1]):
         outm[row,column] = s * sigvals[column,0]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline double[:,:] reversematmul(mv arr, mv weights, mv out):
   cdef ind row, column
   cdef double s
   for row in range(weights.shape[1]):
      s = 0.0
      for column in range(weights.shape[0]):
         s += arr[column,0] * weights[column,row]
      out[row,0] = s
   return out[:,:]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline void matadd2to1ip(mv mat1, mv mat2):
   cdef ind row, column
   for row in range(mat1.shape[0]):
      for column in range(mat1.shape[1]):
         mat1[row,column] = mat1[row,column] + mat2[row,column]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@nonecheck(False)
cdef inline void setto0(mv m):
   cdef ind row, column
   for row in range(m.shape[0]):
      for column in range(m.shape[1]):
         m[row,column] = 0.0




class NNgraph:

   def __init__(self,netshape,name):
      self.graph = graphviz.Graph(name=name,format='png', filename=name,directory=os.getcwd(),strict=True)
      self.shape = netshape
      #initialise nodes with nothing but names
      for layer in range(len(self.shape)):
         for node in range(self.shape[layer]):
            str1 = str(layer) + ' ' + str(node)
            self.graph.node(str1)


      #initialise edges with nothing but start end
      for layer in range(len(self.shape)-1):
         for snode in range(self.shape[layer]):
            for enode in range(self.shape[layer+1]):
               str1 = str(layer) + ' ' + str(snode)
               str2 = str(layer+1) + ' ' + str(enode)
               self.graph.edge(str1,str2)










cdef class NN:
   cdef public graph
   cdef:
      double deriv_multiplier
      mv weights0, weights1, weights2, weights3,biases0, biases1, biases2, biases3, values1, values2, values3, sigvals0, sigvals1, sigvals2, sigvals3
      mv wcha0, wcha1, wcha2, wcha3, dexpitv0, dexpitv1, dexpitv2, dexpitv3, labels
      mv error0, error1, error2, error3, D0, D1, D2, D3
      mv out0, out1, out2, out3, outm1, outm2, outm3
      mv sw0,sw1, sw2, sw3, sb0, sb1, sb2, sb3
      mv bcha0, bcha1, bcha2, bcha3
      public ind netshapec[3]
      public ind matshapec[3][2]
      list pweights, pbiases, matshapes
      double[:,:,:] testinputsnormal, testlabelsnormal, traininginputs, traininglabels
      tuple netshape
      int[:] index_list
      int batchsize, currentbatch
      double[:] cost
      list failedIndexes
      public np.ndarray correctArr
      public int currentSet
      double mdecay


   def setgraph(self,graph):
      self.graph = graph

   def getFailures(self):
      return self.failedIndexes




   def __init__(self, netshape, file, deriv_multiplier, loadmnist=True, batchsize=10, mDecay=0.9):
      self.deriv_multiplier = deriv_multiplier
      self.biases0 = np.ones((1,1))
      self.weights0 = np.ones((1,1))
      self.mdecay = mDecay
      def weights_matrix_shapes(netshape):
         shapes = [(0, 0)]
         for i in range(1, len(netshape)):
            shapes.append((netshape[i], netshape[i-1]))
         return shapes

      self.netshape = netshape
      file1 = file.split()
      if file1[0] == 'ones':
         self.currentSet = 0
         self.correctArr = np.zeros((1000))
         self.pweights = [[]] * 3
         self.pbiases = [[]] * 3
         matshapes = weights_matrix_shapes(netshape)
         self.matshapes = matshapes
         self.out0 = np.ones((netshape[0],1))
         self.out1 = np.ones((self.netshape[1],1))
         self.out2 = np.ones((self.netshape[2],1))
         self.outm1 = np.ones((self.matshapes[1]))
         self.outm2 = np.ones((self.matshapes[2]))
         wmatshapeslist = [list(item) for item in matshapes]
         self.weights0 = np.ones((1,1))
         self.weights1 = np.ones(matshapes[1])
         self.weights2 = np.ones(matshapes[2])
         self.biases1 = np.ones((netshape[1],1))
         self.biases2 = np.ones((netshape[2],1))
      elif file1[0] == 'rand':
         self.currentSet = 0
         self.correctArr = np.zeros((1000))
         lowbound = float(file1[1])
         highbound= float(file1[2])
         matshapes = weights_matrix_shapes(netshape)
         self.matshapes = matshapes
         self.out0 = np.ones((netshape[0],1))
         self.out1 = np.ones((self.netshape[1],1))
         self.out2 = np.ones((self.netshape[2],1))
         self.outm1 = np.ones((self.matshapes[1]))
         self.outm2 = np.ones((self.matshapes[2]))
         wmatshapeslist = [list(item) for item in matshapes]
         print(wmatshapeslist)
         self.matshapec = wmatshapeslist
         range1 = highbound - lowbound
         self.weights0 = np.ones((1,1))
         self.weights1 = mataddsca(matmulsca(np.random.random(matshapes[1]),range1,self.outm1),lowbound,self.outm1)
         self.weights2 = mataddsca(matmulsca(np.random.random(matshapes[2]),range1,self.outm2),lowbound,self.outm2)
         self.biases1 = np.zeros((netshape[1],1))
         self.biases2 = np.zeros((netshape[2],1))
      else:
         self.read_weights_biases_from_file(file)
         matshape = weights_matrix_shapes(self.netshape)
         self.matshapes = matshape
         self.out0 = np.ones((netshape[0],1))
         self.out1 = np.ones((self.netshape[1],1))
         self.out2 = np.ones((self.netshape[2],1))
         self.outm1 = np.ones((self.matshapes[1]))
         self.outm2 = np.ones((self.matshapes[2]))

         listmatshape = [list(item) for item in matshape]
         self.matshapec = listmatshape
      self.error0 = np.zeros((self.netshape[0],1))
      self.error1 = np.zeros((self.netshape[1],1))
      self.error2 = np.zeros((self.netshape[2],1))
      self.D0 = np.zeros((self.netshape[0],1))
      self.D1 = np.zeros((self.netshape[1],1))
      self.D2 = np.zeros((self.netshape[2],1))


      cdef list tmpnetshape = list(self.netshape)
      self.netshapec = tmpnetshape
      self.weights0 = np.ones((1,1))
      self.biases0 = np.ones((1,1))
      self.wcha2 = np.zeros((self.netshape[2],self.netshape[1]))
      self.wcha1 = np.zeros((self.netshape[1],self.netshape[0]))
      self.wcha0 = np.zeros((0,0))

      self.bcha2 = np.zeros((self.netshape[2],1))
      self.bcha1 = np.zeros((self.netshape[1],1))

      self.values1 = np.zeros((self.netshape[1],1))
      self.values2 = np.zeros((self.netshape[2],1))
      self.sigvals1 = np.zeros((self.netshape[1],1))
      self.sigvals2 = np.zeros((self.netshape[2],1))
      self.dexpitv1 = np.zeros((self.netshape[1],1))
      self.dexpitv2 = np.zeros((self.netshape[2],1))

      #at some point measure the speed of inplace vs not in place, this could be quite important

      self.batchsize = batchsize
      with np.load('npzcity/mnist.npz') as data:
         self.traininginputs = data['training_images']
         self.traininglabels = data['training_labels']
         self.testinputsnormal = data['test_images']
         self.testlabelsnormal = data['test_labels']

      tmp = np.linspace(0,49999,50000,dtype=np.int32)
      self.index_list = tmp
      del tmp
      self.cost = np.zeros((50000//self.batchsize))
      self.failedIndexes = []






   cpdef double computevaluesandderivs(self,np.ndarray[flt,ndim=2] inputs, np.ndarray[flt,ndim=2] labels):
      self.labels = labels
      self.sigvals0 = inputs
      self.values1[:,:] = arradd(matmul(self.sigvals0, self.weights1,self.out1), self.biases1, self.out1)
      self.sigvals1[:,:] = expitt(self.values1,self.out1)
      arrdexpitfromexpitIP(self.sigvals1,self.dexpitv1)
      self.values2[:,:] = arradd(matmul(self.sigvals1, self.weights2,self.out2), self.biases2, self.out2)
      self.sigvals2[:,:] = expitt(self.values2,self.out2)
      arrdexpitfromexpitIP(self.sigvals2,self.dexpitv2)

      self.error2[:,:] = arrsub(self.labels,self.sigvals2,self.out2)

      #self.error2[:,:] = arrsqukeepsgn(self.error2,self.out2)

      self.error2[:,:] = arrmulsca(self.error2,self.deriv_multiplier, self.out2)
      self.D2[:,:] = arrmul(self.error2, self.dexpitv2,self.out2)
      self.error1[:,:] = reversematmul(self.D2,self.weights2,self.out1)
      self.D1[:,:] = arrmul(self.error1,self.dexpitv1,self.out1)

      errortowchaDip(self.D2,self.sigvals1,self.wcha2)
      errortowchaDip(self.D1,self.sigvals0,self.wcha1)
      self.bcha2[:,:] = arradd(self.bcha2,self.D2,self.out2)
      self.bcha1[:,:] = arradd(self.bcha1,self.D1,self.out1)
      return self.calccost()


   cdef double mvcomputevaluesandderivs(self, mv inputs, mv labels):
      self.labels = labels
      self.sigvals0 = inputs
      self.values1[:,:] = arradd(matmul(self.sigvals0, self.weights1,self.out1), self.biases1, self.out1)
      self.sigvals1[:,:] = expitt(self.values1,self.out1)
      arrdexpitfromexpitIP(self.sigvals1,self.dexpitv1)
      self.values2[:,:] = arradd(matmul(self.sigvals1, self.weights2,self.out2), self.biases2, self.out2)
      self.sigvals2[:,:] = expitt(self.values2,self.out2)
      arrdexpitfromexpitIP(self.sigvals2,self.dexpitv2)


      self.error2[:,:] = arrsub(self.labels,self.sigvals2,self.out2)

      #self.error2[:,:] = arrsqukeepsgn(self.error2,self.out2)

      self.error2[:,:] = arrmulsca(self.error2,self.deriv_multiplier, self.out2)
      self.D2[:,:] = arrmul(self.error2, self.dexpitv2,self.out2)
      self.error1[:,:] = reversematmul(self.D2,self.weights2,self.out1)
      self.D1[:,:] = arrmul(self.error1,self.dexpitv1,self.out1)

      errortowchaDip(self.D2,self.sigvals1,self.wcha2)
      errortowchaDip(self.D1,self.sigvals0,self.wcha1)
      self.bcha2[:,:] = arradd(self.bcha2,self.D2,self.out2)
      self.bcha1[:,:] = arradd(self.bcha1,self.D1,self.out1)
      return self.calccost()













   def randomise_indexes(self):
      available_indexes = list(range(50000))
      shuffled_indexes = []
      tmpi = 0
      for i in range(50000):
         tmpi = random1(0,50000-i-1)
         self.index_list[i] = available_indexes[tmpi]
         del available_indexes[tmpi]


   def derandomise_indexes(self):
      a = np.linspace(0,50000-1,50000,dtype=np.int32)
      self.index_list = a
      print(self.index_list[0])
      return self.index_list




   #runbatch
   '''cpdef runbatch(self, list lstin,list lstlabel):
      cdef double tcost = 0.0
      cdef ind inputs
      for inputs in range(len(lstlabel)):
         tcost += self.computevaluesandderivs(lstin[inputs],lstlabel[inputs])
      self.applychanges()
      return tcost/len(lstlabel)'''


   cpdef public runbatchmnist(self, int batch):
      cdef double tcost = 0.
      cdef ind inputs
      for index in range(batch * self.batchsize, (batch + 1) * self.batchsize):
         tcost += self.mvcomputevaluesandderivs(self.traininginputs[self.index_list[index]],self.traininglabels[self.index_list[index]])
      self.applychanges()
      #self.applychangesmomentum()
      return tcost / self.batchsize


   cpdef mnistfinishset(self):
      totalbatches = 50000//self.batchsize

      self.randomise_indexes()

      cdef double sum = 0
      cdef ind batch
      for batch in range(self.currentbatch, totalbatches):
         sum += self.runbatchmnist(batch)
      self.correctArr[self.currentSet] = self.testCorrect()
      self.currentSet += 1
      if self.currentSet >= self.correctArr.shape[0]:
         pass



      return sum/50_000






   def getcost(self):
      return np.asarray(self.cost)

   #temptrainfunc
   '''def temptrainfunc(self,func,bs,batches,finalcostval=50):
      start = time()
      batchs = bs * batches
      #self.deriv_multiplier /= bs THIS MIGHT BE USEFUL AT SOME POINT
      cost = np.zeros(batches)
      tmp = 0.
      for batch in range(batchs):
         tmp += self.computevaluesandderivs(*func())
         if batch%1==0:
             self.applychangesmomentum()
             cost[batch//bs] = tmp#/bs
             tmp = 0.
      return cost'''




   #temptrain
   '''def temptrain(self, func, batchsize, batches, finalcostval=50):
      start = time()
      cost = np.zeros((batches))
      finalav = 0.
      for batch in range(batches):
         for datapoint in range(batchsize):
            cost[batch] += self.computevaluesandderivs(*func())
         cost[batch] /= batchsize
         self.applychanges()







      finalav = sum(cost[-finalcostval:])/finalcostval

      print(str(time()-start) + ' seconds:',finalav )
      return cost'''




   cpdef applychanges(self):
      self.biases1[:,:] = arradd(self.biases1, self.bcha1, self.out1)
      self.biases2[:,:] = arradd(self.biases2, self.bcha2, self.out2)
      matadd2to1ip(self.weights1,self.wcha1)
      matadd2to1ip(self.weights2,self.wcha2)
      setto0(self.bcha2)
      setto0(self.bcha1)
      setto0(self.wcha2)
      setto0(self.wcha1)


   cpdef setChangesTo0(self):
      setto0(self.bcha2)
      setto0(self.bcha1)
      setto0(self.wcha2)
      setto0(self.wcha1)




   cpdef testIndex(self,index,bool=False, show=False, erase=False):
      cost = self.mvcomputevaluesandderivs(self.testinputsnormal[index],self.testlabelsnormal[index])
      if erase: self.setChangesTo0()
      cdef ind i
      if bool:
         return cost
      else:
         highesti = -1
         highestv = -1.
         ideal = -1
         for i in range(0,10):
            if self.sigvals2[i,0] > highestv:
               highestv = self.sigvals2[i,0]
               highesti = i
            if self.testlabelsnormal[index,i,0] == 1.0:
               ideal = i
         if highesti == ideal:
            return 1
         else:
            self.failedIndexes.append(index)
            return 0



   cpdef testCorrect(self,cost=False):

      self.failedIndexes = []
      cdef ind i
      cdef int sum = 0

      for i in range(10_000):
         sum += self.testIndex(i,cost)

      self.setChangesTo0()


      return float(sum)/10_000.






















   cpdef applychangesmomentum(self):
      self.biases1[:,:] = arradd(self.biases1, self.bcha1, self.out1)
      self.biases2[:,:] = arradd(self.biases2, self.bcha2, self.out2)
      matadd2to1ip(self.weights1,self.wcha1)
      matadd2to1ip(self.weights2,self.wcha2)

      self.bcha1[:,:] = matmulsca(self.bcha1,self.mdecay,self.out1)
      self.bcha2[:,:] = matmulsca(self.bcha2,self.mdecay,self.out2)
      self.wcha1[:,:] = matmulsca(self.wcha1,self.mdecay,self.outm1)
      self.wcha2[:,:] = matmulsca(self.wcha2,self.mdecay,self.outm2)












   # problem, when value is high and you want it to be negative, it changes very slowly because of the dexpit term being very small

   # hh(False)
   (False)
   (False)
   (False)
   cpdef double calccost(self):
      cdef ind row
      cdef double sum = 0.
      for row in range(self.netshapec[2]):
         sum += (self.sigvals2[row, 0] - self.labels[row, 0]) * (self.sigvals2[row, 0] - self.labels[row, 0])
      return sum

   def weights_matrix_shapes(self):
      shapes = [(0, 0)]
      for i in range(1, 3):
         shapes.append((self.netshape[i], self.netshape[i - 1]))
      return shapes

   def weights_biases_matrix_shapes(self):
      shapes = [[(0, 0), (0, 0)]]
      for i in range(1, 3):
         shapes.append([(self.netshape[i], self.netshape[i - 1]), (self.netshape[i], 1)])
      return shapes

   def read_weights_biases_from_file(self, file):
      weights123 = []
      biases123 = []
      with np.load(file) as data:
         length = len(data.files)
         for i in range(0, length // 2):
            weights123.append(data['w' + str(i)])
            biases123.append(data['b' + str(i)])
         self.correctArr = data['accuracy']

      self.currentSet = 0
      while self.correctArr[self.currentSet] != 0. :
         self.currentSet += 1

      self.weights1 = weights123[1]
      self.weights2 = weights123[2]
      self.biases1 = biases123[1]
      self.biases2 = biases123[2]
      tempnetshape = [self.weights1.shape[1], self.biases1.shape[0], self.biases2.shape[0]]
      self.netshape = tuple(tempnetshape)

   def saveToFile(self,file):
      sw0 = dcopymv(self.weights0)
      sw1 = dcopymv(self.weights1)
      sw2 = dcopymv(self.weights2)
      sb0 = dcopymv(self.biases0)
      sb1 = dcopymv(self.biases1)
      sb2 = dcopymv(self.biases2)
      history = copy.deepcopy(self.correctArr)

      string1 = 'np.savez("npzcity/' + file + '"'


      for i in range(3):
         string1 += ' , w' + str(i) + '=sw' + str(i)
         string1 += ' , b' + str(i) + '=sb' + str(i)



      string1 += ' , accuracy=history'
      string1 += ')'
      exec(string1)


   def correct(self):
      return self.correctArr[:self.currentSet]


   def getcost(self):
      return np.asarray(self.cost)
   def getsigcor(self):
      return [np.asarray(self.sigvals2),np.asarray(self.labels)]
   def getweights(self):
      return [np.asarray(self.weights0),np.asarray(self.weights1),np.asarray(self.weights2)]
   def getbiases(self):
      return [np.asarray(self.biases0),np.asarray(self.biases1),np.asarray(self.biases2)]
   def getallvaluesvaldexpsigerror(self):
      lst0 = [asr(self.sigvals0)]
      lst1 = [asr(self.values1),asr(self.dexpitv1),asr(self.sigvals1),asr(self.error1)]
      lst2 = [asr(self.values2),asr(self.dexpitv2),asr(self.sigvals2),asr(self.error2)]
      return [lst0,lst1,lst2]
   def getdmult(self):
      return self.deriv_multiplier
   def setdmult(self,new):
      self.deriv_multiplier = new












def randfloatrange(a,b):
   return random.random() * (b-a) + a













