import sys
import statistics 


import numpy as np
from numpy import linalg as LA

from scipy.sparse import csgraph
from scipy.linalg import fractional_matrix_power
np.set_printoptions(threshold=np.inf)


def vector_norm(input_matrix):
	A = np.array(input_matrix,dtype='d')
	shape = np.shape(A)
	m = shape[0]
	n = shape[1]
	print(A[:,0])
	print(np.std(A[:,0]))
	for i in range(n):
		current_column = A[:,i]
		vector_mean = np.mean(current_column)
		vector_std = np.std(current_column)
		A[:,i] = (A[:,i] - vector_mean) / vector_std
	return A

def compute_dnsqrt(input_matrix):
	vec = np.sum(input_matrix, axis=1)
	vec = np.sqrt(vec)
	vec = np.divide(1,vec)
	vec = np.diag(np.diag(np.diag(vec)))
	#dnsqrt = np.diag(np.diag(   fractional_matrix_power(np.sum(input_matrix, axis=1),-0.5 )      ))
	return vec




def spectural_clustering_based_classifier(A):
	# Normalize software metrics.

	A = vector_norm(A)

	# Construct the weighted adjacency matrix W.
	W = A.dot(np.transpose(A))
	# Set all negative values to zero.
	W[W < 0] = 0
	# Set the self-similarity to zero.
	np.fill_diagonal(W,0)
	# Construct the symmetric Laplacian matrix
	dnsqrt = compute_dnsqrt(W)
	Lsym = csgraph.laplacian(W, normed=True)
	# Perform the eigendecomposition
	eigenvalue, eigenvector = LA.eigh(Lsym)
	print(eigenvalue)
	# Pick up the second smallest eigenvector
	v1 = eigenvector[1]
	v1 = dnsqrt.dot(v1)
	v1 = v1 / LA.norm(v1,2)
	# Divide the data set into two clusters
	defect_proneness = (v1 > 0)
	# Label the defective and clean clusters
	rs = np.sum(A, axis=1)

	if (np.mean(rs[v1 > 0]) <  np.mean(rs[v1 < 0])):
		defect_proneness = (v1<0)

	return defect_proneness


with open('test_data.txt','r') as f:
    C = [[int(num) for num in line.split(' ')] for line in f]

#print(len(C[1]))
#print(spectural_clustering_based_classifier(C))
result = spectural_clustering_based_classifier(C)
for i in range(len(result)):
	print(i+1, end=' ')
	print(result[i])



