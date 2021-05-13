
#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "IzhikevichData.h"

void updateIzhikevich(Connection *connection, void *_data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int firedTableCap, int num, int offset, int time)
{
	// IzhikevichData *data = (IzhikevichData *)_data;
	// int currentIdx = time % (connection->maxDelay+1);
	// for (int nid=0; nid<num; nid++) {
	// }
}

