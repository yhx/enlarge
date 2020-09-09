
#include "../../src/utils/FileOp.h"
#include "../../src/net/DistriNetwork.h"

int main(int arc, char **argv)
{
	FILE *f1 = openFile(argv[1], "r+");
	FILE *f2 = openFile(argv[2], "r+");
	DistriNetwork * n1 = loadDistriNet(f1);
	DistriNetwork * n2 = loadDistriNet(f2);
	closeFile(f1);
	closeFile(f2);

	bool equal = compareDistriNet(n1, n2);
	if (equal) {
		printf("Same\n");
	} else {
		printf("Diff\n");
	}

	return 0;
}
