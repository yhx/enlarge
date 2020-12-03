
#ifndef PROC_INFO_H
#define PROC_INFO_H

#include <stdio.h>
#include <unistd.h>

typedef struct {
	size_t mem_used;
	size_t vm_used;
} proc_info;

int get_proc_info(proc_info *info, pid_t pid = -1);

#endif //PROC_INFO_H
