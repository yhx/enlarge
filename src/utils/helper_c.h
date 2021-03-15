
#ifndef HELPER_C_H
#define HELPER_C_H

#include <stdio.h>
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

inline void fread_c(void *ptr, size_t size, size_t memb, FILE *stream) 
{
	size_t ret = fread(ptr, size, memb, stream);
	assert(ret == memb);
}

inline void fwrite_c(const void *ptr, size_t size, size_t memb, FILE *stream) 
{
	size_t ret = fwrite(ptr, size, memb, stream);
	assert(ret == memb);
}

template<typename T>
T * malloc_c(size_t size)
{
	T *ret = NULL;
	ret = static_cast<T*>(malloc(sizeof(T) * (size)));
	assert(ret != NULL);
	memset(ret, 0, sizeof(T) * (size));
	return ret;
}

#endif // HELPER_C_H
