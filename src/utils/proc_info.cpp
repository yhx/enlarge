
#include "proc_info.h"


const int VMSIZE_LINE = 16;
const int PROCESS_ITEM = 14;

int get_proc_info(proc_info *info, pid_t pid)
{
	char file_name[64] = {0};
	if (pid > 0) {
		sprintf(file_name, "/proc/%d/status", pid);
	} else {
		sprintf(file_name, "/proc/self/status");
	}
	FILE *fd = fopen(file_name, "r");

	if (!fd) {
		return 0;
	}

	char buff[2048] = {0};
	
	for (int i=0; i<VMSIZE_LINE; i++) {
		fgets(buff, 2048, fd);
	}
	fscanf(fd, "%s %ld", buff, &(info->mem_used));
	// printf("%s\n", buff);

	return 0;
}
