

#include "../utils/helper_c.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "CrossNodeMap.h"

CrossNodeMap * to_gpu(CrossNodeMap *cpu)
{
	CrossNodeMap *ret = malloc_c<CrossNodeMap>();
	ret->_num = cpu->_num;
	ret->_crossSize = cpu->_crossSize;
	ret->_idx2index = copyToGPU(cpu->_idx2index, cpu->_num);
	if (cpu->_crossSize > 0) {
		ret->_crossnodeIndex2idx = copyToGPU(cpu->_crossnodeIndex2idx, cpu->_crossSize);
	} else {
		ret->_crossnodeIndex2idx = NULL;
	}

	return ret;
}
