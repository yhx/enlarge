
#ifndef FILEOP_H
#define FILEOP_H

#include <stdio.h>
#include <iostream>
#include <assert.h>

inline FILE * openFile(const char *filename, const char *mode) 
{
	FILE * file = fopen(filename, mode);
	if (file == NULL) {
		printf("ERROR: Open file %s failed\n", filename);
	}
	assert(file != NULL);
	return file;
}

inline int closeFile(FILE *file) 
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

#endif // FILEOP_H
