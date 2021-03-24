
#ifndef HELPER_C_H
#define HELPER_C_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

inline FILE * fopen_c(const char *filename, const char *mode) 
{
	FILE * file = fopen(filename, mode);
	if (file == NULL) {
		printf("ERROR: Open file %s failed\n", filename);
	}
	assert(file != NULL);
	return file;
}

inline int fclose_c(FILE *file) 
{
	fflush(file);
	return fclose(file);
}

template<typename T>
void fread_c(T *ptr, size_t memb, FILE *stream) 
{
	size_t ret = fread(ptr, sizeof(T), memb, stream);
	assert(ret == memb);
}

template<typename T>
void fwrite_c(const T *ptr, size_t memb, FILE *stream) 
{
	size_t ret = fwrite(ptr, sizeof(T), memb, stream);
	assert(ret == memb);
}

template<typename T>
T * malloc_c(size_t size = 1)
{
	T *ret = NULL;
	ret = static_cast<T*>(malloc(sizeof(T) * (size)));
	assert(ret != NULL);
	memset(ret, 0, sizeof(T) * (size));
	return ret;
}

template<typename T>
void  free_c(T *array)
{
	free(array);
	array = NULL;
}

template<typename T>
void  memset_c(T *array, int c, size_t size = 1)
{
	memset(array, c, sizeof(T) * (size));
}

#endif // HELPER_C_H
