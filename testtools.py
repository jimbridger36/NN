import numpy as np
from numpy import asarray as asr
from matplotlib import pyplot as plt
from random import randint


def pbl(lst, start='start'):
	print(start)
	for item in lst:
		print(item, '\n')
	print('stop')


def p(lst, title=''):
	plt.plot(lst, label=title)
	plt.title(title)
	plt.show()


netshape = (2, 2, 1)

i1 = -1
def func():
	return inp2, cor2


def xor2det():
	inputs = np.zeros((2, 1))
	outputs = np.zeros((1, 1))
	global i1
	i1 += 1
	op = i1 % 4
	if op == 0:
		inputs[0, 0] = 1.0
		outputs[0, 0] = 1.0
	elif op == 1:
		inputs[1, 0] = 1.0
		outputs[0, 0] = 1.0
	elif op == 2:
		inputs[0, 0] = 1.0
		inputs[1, 0] = 1.0
	return inputs, outputs


i = np.array([0.,1.]).reshape((2,1))
a = np.array([[1.]]).reshape((1,1))
def ia():
	return i,a


def xor2():
	inputs = np.zeros((2, 1))
	outputs = np.zeros((1, 1))
	outputs[0, 0] = randint(0, 1)
	ranint = randint(0, 1)
	if outputs[0, 0] == 1.0 and ranint == 1:
		inputs[0, 0] = 1.0
	elif outputs[0, 0] == 1.0 and ranint == 0:
		inputs[1, 0] = 1.0
	elif ranint == 1:
		inputs[0, 0] = 1.0
		inputs[1, 0] = 1.
	return inputs, outputs


def rolav(lst, av=100):
	nlist = np.zeros((len(lst)))
	for i in range(av, len(lst)):
		nlist[i] = np.sum(lst[i - av:i]) / av
	return nlist


def single():
	lst = []
	global i1
	i1 = -1
	for i in range(4):
		c = tnet.computevaluesandderivs(*xor2det())
		lst.append([c, tnet.sigvals2[0, 0], tnet.labels[0, 0], str(tnet.sigvals0[0, 0]), str(tnet.sigvals0[1, 0])])
	return lst

def rs(arr,size=-10):
	if size == -10:
		size = arr.shape[0] * 10
	tmp = np.zeros((size))
	tmp[:arr.shape[0]] = arr
	return tmp

if True:
	inp2 = np.array([0.7, 0.2]).reshape((2, 1))
	cor2 = np.array([[0.4]])

	f = xor2det
	u = [None] * 4
	v = [None] * 4
	for i in range(4):
		tmp = xor2det()
		u[i] = tmp[0]
		v[i] = tmp[1]



def nptrue(array):
	for row in range(array.shape[0]):
		for column in range(array.shape[1]):
			if (not array[row,column]):
				print(array[row,column])
				return False
	return True

def np1true(array):
	for row in range(array.shape[0]):
		if (not array[row]):
			print(array[row])
			return False
	return True

def average(array):
	return sum(array)/len(array)


def compwcha(cnet,pnet):
	temp = True
	if nptrue(asr(cnet.wcha1) == pnet.wcha[1]):
		pass
	else:
		print('wcha1 is different')
		print(asr(cnet.wcha1))
		print(asr(pnet.wcha[1]))
		temp = False
	if nptrue(asr(cnet.wcha2) == pnet.wcha[2]):
		pass
	else:
		print('wcha2 is different')
		print(asr(cnet.wcha2))
		print(asr(pnet.wcha[2]))
		temp = False
	return temp


def compw(cnet,pnet):
	temp = True
	if nptrue(asr(cnet.weights1) == pnet.weights[1]):
		pass
	else:
		print('weights1 is different')
		print(asr(cnet.weights1))
		print(asr(pnet.weights[1]))
		temp = False
	if nptrue(asr(cnet.weights2) == pnet.weights[2]):
		pass
	else:
		print('weights2 is different')
		print(asr(cnet.weights2))
		print(asr(pnet.weights[2]))
		temp = False
	return temp





def tempcompare(func,batchsize,numbatches,cnet,pnet):
	ccost = np.zeros(numbatches)
	pcost = np.zeros(numbatches)

	'''    def temptrainfunc(self,func,bs,batches):
        start = time()
        self.deriv_multiplier /= bs
        cost = np.zeros(batches)
        for batch in range(batches):
            for example in range(bs):
                cost[batch] += self.compute_values(*func())
                self.backprop()
            cost[batch] /= bs
            self.applychanges()
        self.deriv_multiplier *= bs
        print(str(time() - start) + ' seconds:', sum(cost[-bs:])/bs)
        return cost'''

	for batch in range(numbatches):
		temp = func()
		cc = cnet.computevaluesandderivs(*temp)
		cp = pnet.compute_values(*temp)
		ccost[batch] = cc
		pcost[batch] = cp
		pnet.backprop()
		if not compwcha(cnet,pnet):
			print('on batch ' + str(batch) + ' when wcha error occured')
			if not cc == cp:
				print('on batch ' + str(batch) + ' when cost error occured')
			return ccost,pcost
		if not cc == cp:
			print('on batch ' + str(batch) + ' when cost error occured')
			return ccost,pcost
		if not compw(cnet,pnet):
			print('on batch ' + str(batch) + ' when wcha error occured')
			if not cc == cp:
				print('on batch ' + str(batch) + ' when cost error occured')
			return ccost,pcost
		cnet.applychanges()
		pnet.applychanges()
	return ccost, pcost











