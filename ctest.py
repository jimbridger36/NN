import CyNNbackprop3layer as n
import numpy as np
from matplotlib import pyplot as plt
from time import time
from NN.testtools import p, asr, rolav
from PIL.Image import fromarray as frarray

#timer code
if True:
	Time = 0
	time_start = 0
	from time import time as tim
	def timer(start=False):
		global Time
		global time_start
		if start:
			time_start = tim()
			return None
		else:
			tmp = tim() - time_start
			Time += tmp
			time_start = 0
			return tmp



def imshow(array):
	plt.imshow(asr(array).reshape((28,28)),cmap='gray',vmin=0,vmax=1)
	plt.show()




def fromarray(array):
	array = array.reshape((28,28))
	array = (array * 255).astype(np.uint8)
	img = frarray(array)
	img.show()

i1=-1
i = np.zeros((784,1))
a = np.zeros((10,1))






netshape = (2,2,1)
#tnet = n.NNtmp(netshape,'py221.npz', 1.)
#net = N.NNtmp(netshape,'py221.npz', 1.)
rnet = n.NN((784,30,10),'npzcity/wb784_30_10.npz', 1.,mDecay=0.)








with np.load('npzcity/mnist.npz') as data:
	ti = data['training_images']
	tl = data['training_labels']
	tti = data['test_images']
	ttl = data['test_labels']

h = sum(ti)
g = sum(tl)








sets = 3
nsets = 3

def tryout(dmult,maxsets = 10):
	threshold = 0.9
	net = n.NN((1,1,1),'npzcity/h.npz',dmult)
	tmp = 0.
	net.mnistfinishset(); net.mnistfinishset()
	while tmp < threshold and net.currentSet < maxsets:
		net.mnistfinishset()
		tmp = np.sum(net.correct()[-4:]) / 4
		print(net.currentSet, tmp)
	if net.currentSet == maxsets and tmp < threshold: print(str(dmult) + ' didnt work: got to ' + str(tmp) + ' in ' + str(maxsets) + ' sets')
	plt.plot(rolav(net.correct(),3)[3:])
	plt.title(str(dmult))
	plt.show()
	return net.currentSet


def map(arr,func):
	for i in range(arr.shape[0]):
		arr[i] = func(arr[i])
	return arr


'''
for value in map(np.linspace(0,2,2),math.exp):
	tryout(float(value))

'''


d = {
	0.7 : 10, # to 0.9
	# retry 1.
	0.3: 9

}




start = time()




'''
for i in range(sets):
	rnet.mnistfinishset()
'''
p(rnet.correct(),title=str(rnet.getdmult()))


#print(str((time() - start)/sets) + ' per set. total: ' + str(time()-start))
