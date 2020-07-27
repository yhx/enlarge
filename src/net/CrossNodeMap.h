
#ifndef CROSSNODEMAP_H
#define CROSSNODEMAP_H

struct CrossNodeMap {
	// ID of neurons on this node to index in this map 
	// index = _idx2index[id]
	int *_idx2index;
	// idx in this map to ID of shadow neurons on node j
	// id = _crossnode_index2idx[index * node_num + j], -1 means no shadow neuron on node j
	int *_crossnodeIndex2idx;
	// _cross_size = node_num * number_of_the_neurons_on_this_node_which_have_crossnode_connections
	int _crossSize;
	int _num;
};

// Assuming node number is N, we would use cross_data=CrossNodeData[N*N], cross_data[i+j*node_num] stores the ID of fired neurons on node i to be sent to node j and cross_data[j+j*node_num] stores allow the IDs received by node j
struct CrossNodeData {
	// Max number of fired neurons
	int _maxNNum;
	// Acutal number of fired neurons
	int _firedNNum;
	// IDs of fired neurons
	int *_firedNIdxs;
};

// Assuming node number is N. Parameter[i*node_num+j] stores corresponding paramter on node i to be sent to node[j+j*node_num] stores allow the IDs received by node j
struct CrossNodeDataGPU {
	// Max number of fired neurons
	int *_maxNum;
	// Acutal number of fired neurons
	int *_firedNum;
	// IDs of fired neurons
	int **_firedArrays;
};

int sendMap(CrossNodeMap * network, int dest, int tag, MPI_Comm comm);
CrossNodeMap * recvMap(int src, int tag, MPI_Comm comm);


#endif // CROSSNODEMAP_H
