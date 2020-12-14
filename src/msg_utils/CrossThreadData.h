
#ifndef CROSSTHREADDATA_H
#define CROSSTHREADDATA_H

// Assuming node number is N, we would use cross_data=CrossThreadData[N*N], cross_data[i+j*node_num] stores the ID of fired neurons on node i to be sent to node j and cross_data[j+j*node_num] stores allow the IDs received by node j
struct CrossThreadData {
	// Max number of fired neurons
	int _maxNNum;
	// Acutal number of fired neurons
	int _firedNNum;
	// IDs of fired neurons
	int *_firedNIdxs;
};

// Assuming node number is N. Parameter[i*node_num+j] stores corresponding paramter on node i to be sent to node[j+j*node_num] stores allow the IDs received by node j
struct CrossThreadDataGPU {
	// Max number of fired neurons
	int *_maxNum;
	// Acutal number of fired neurons
	int *_firedNum;
	// IDs of fired neurons
	int **_firedArrays;
};

#endif // CROSSTHREADDATA_H

