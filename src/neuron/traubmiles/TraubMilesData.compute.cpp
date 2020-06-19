
#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "TraubMilesData.h"

void updateTraubMiles(Connection *connection, void *_data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int offset, int time)
{
	TraubMilesData *data = (TraubMilesData*)_data;
	int currentIdx = time % (connection->maxDelay+1);
	for (int nid=0; nid<num; nid++) {
	}
}

