
#include <string.h>
#include "../../msg_utils/helper/helper_c.h"
#include "../../src/net/DistriNetwork.h"

using std::string;

int main(int arc, char **argv)
{
	DistriNetwork * n1 = loadDistriNet(string(argv[1]));
	DistriNetwork * n2 = loadDistriNet(string(argv[2]));

	bool equal = compareDistriNet(n1, n2);
	if (equal) {
		printf("Same\n");
	} else {
		printf("Diff\n");
	}

	return 0;
}
