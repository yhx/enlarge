/* This program is writen by qp09.
 * usually just for fun.
 * Mon March 14 2016
 */
#if defined(_WIN32)
#include <direct.h>

// #elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
// #include <unistd.h>
// #include <sys/resource.h>
// 
// #if defined(__APPLE__) && defined(__MACH__)
// #include <mach/mach.h>
// 
// #elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
// #include <fcntl.h>
// #include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include<sys/stat.h>
#include<sys/types.h>

// #endif
#else
#error "Cannot include file for an unknown OS."
#endif

#include <errno.h>
#include <iostream>

#include "../third_party/util/mem.h"
#include "utils.h"
#include "proc_info.h"

void print_mem(const char *info)
{
	// Proc_info pinfo;
	// Get_proc_info(&pinfo);
	// printf("%s, MEM used: %lfGB\n", info, static_cast<double>(pinfo.mem_used/1024.0/1024.0));
	size_t mem_used = getCurrentRSS();
	printf("%s, MEM used: %lfGB\n", info, static_cast<double>(mem_used/1024.0/1024.0/1024.0));
}

double realRandom(double range)
{
	long f = rand();
	return ((double)f/RAND_MAX)*range;
}

//int id2idx(ID* array, int num, ID id) {
//	for (int i=0; i<num; i++) {
//		if (array[i] == id) {
//			return i;
//		}
//	}
//	printf("ERROR: Cannot find ID!!!\n");
//	return 0;
//}

int getIndex(Type *array, int size, Type type)
{
	for (int i=0; i<size; i++) {
		if (array[i] == type) {
			return i;
		}
	}

	//printf("ERROR: Cannot find type %d !!!\n", type);
	return -1;
}

int getType(int *array, int size, int index)
{
	for (int i=0; i<size; i++) {
		if (array[i+1] > index) {
			return i;
		}
	}

	//printf("ERROR: Cannot find index %d !!!\n", index);
	return -1;
}

int getOffset(int *array, int size, int index)
{
	for (int i=0; i<size; i++) {
		if (array[i+1] > index) {
			return (index - array[i]);
		}
	}

	//printf("ERROR: Cannot find index %d !!!\n", index);
	return -1;
}

Json::Value testValue(Json::Value value, unsigned int idx)
{
	if (value.type() == Json::nullValue) {
		return 0;
	}

	if (value.type() == Json::arrayValue) {
		if (idx < value.size()) {
			return value[idx];
		} else {
			std::cout  << "Not enough parameters:" << value << "@" << idx << std::endl;
		}
	} 

	return value;
}

real *loadArray(const char *filename, int size)
{
	real *res = (real*)malloc(sizeof(real) * size);
	FILE *logFile = fopen(filename, "rb+");
	if (logFile == NULL) {
		printf("ERROR: Open file %s failed\n", filename);
		return res;
	}
	fread_c(res, size, logFile);

	fflush(logFile);
	fclose(logFile);

	return res;
}

int saveArray(const char *filename, real *array, int size)
{
	FILE *logFile = fopen(filename, "wb+");
	if (logFile == NULL) {
		printf("ERROR: Open file %s failed\n", filename);
		return -1;
	}
	fwrite(array, sizeof(real), size, logFile);
	fflush(logFile);
	fclose(logFile);

	return 0;
}

void log_array_noendl(FILE *f, int *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%d ", array[i]);
	}
}

void log_array_noendl(FILE *f, unsigned int *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%u ", array[i]);
	}
}

void log_array_noendl(FILE *f, long *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%ld ", array[i]);
	}
}

void log_array_noendl(FILE *f, unsigned long *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%lu ", array[i]);
	}
}

void log_array_noendl(FILE *f, long long *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%lld ", array[i]);
	}
}

void log_array_noendl(FILE *f, unsigned long long *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%llu ", array[i]);
	}
}

void log_array_noendl(FILE *f, float *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%.10lf ", array[i]);
	}
}

void log_array_noendl(FILE *f, double *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%.10lf ", array[i]);
	}
}

void system_c(const char *cmd)
{
	int status = system(cmd);
	
	if (-1 == status) {
		printf("%s: system error\n", cmd);
	} else {
		// printf("%s: exit status value = [0x%x]\n", cmd, status);
		if (WIFEXITED(status))
		{
			if (0 != WEXITSTATUS(status))
			{
				printf("%s: run shell script fail, script exit code: %d\n", cmd, WEXITSTATUS(status));
			}
		}
		else
		{
			printf("%s: exit status = [%d]\n", cmd, WEXITSTATUS(status));
		}
	}
}

void mkdir(const char *dir) 
{
#if defined(_WIN32)
	int status = _mkdir(dir);
	if (-1 == status) {
		printf("mkdir %s error %d\n", dir, errno);
	}
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
	int status = 0;
	struct stat st = {0};

	if (stat(dir, &st) == -1) {
		status = mkdir(dir, 0777);
	}
	if (-1 == status) {
		printf("mkdir %s error %d\n", dir, errno);
	}
#else
#error "Cannot define mkdir() for an unknown OS."
#endif
}
