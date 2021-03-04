/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <typeinfo>

#include "../third_party/json/json.h"
#include "type.h"

static bool rand_seed_inited = false;

double realRandom(double range);

int getIndex(Type *array, int size, Type type);
int getType(int *array, int size, int index);
int getOffset(int *array, int size, int index);

Json::Value testValue(Json::Value value, unsigned int idx);

real *loadArray(const char *filename, int size);
int saveArray(const char *filename, real *array, int size);

void print_mem(const char *info);

template<typename T>
bool compareArray(T *a, T *b, int size) 
{
	bool equal = true;
	for (int i=0; i<size; i++) {
		equal = equal && (a[i] == b[i]);
	}
	return equal;
}

inline int upzero_else_set_one(int value) {
	if (value > 0) {
		return value;
	}

	return 1;
}

template<typename T>
T *getRandomArray(T range, size_t size) {
	if (!rand_seed_inited) {
		srand(time(NULL));
		rand_seed_inited = true;
	}

	T *res = new T[size];
	for (size_t i=0; i<size; i++) {
		res[i] = static_cast<T>(realRandom(static_cast<double>(range)));
	}

	return res;
}

template<typename T>
T *getConstArray(T value, size_t size)
{
	T *res = new T[size];
	for (size_t i=0; i<size; i++) {
		res[i] = value;
	}

	return res;
}

template<typename T>
void delArray(T *value)
{
	delete[] value;
}

template<typename T>
bool isEqualArray(T const & a, T const & b, size_t size)
{
	for (size_t i=0; i<size; i++) {
		if (fabs(a[i] - b[i]) > 1e-10) {
			return false;
		}
	}
	return true;
}

template<typename T>
void log_array(FILE *f, T *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		if (typeid(T) == typeid(int)) {
			fprintf(f, "%d ", array[i]);
		} else if (typeid(T) == typeid(float)) {
			fprintf(f, "%.10lf \t", array[i]);
		} else if (typeid(T) == typeid(double)) {
			fprintf(f, "%.10lf \t", array[i]);
		} else {
			fprintf(f, "Unsupported type\n");
			break;
		}
	}
	fprintf(f, "\n");
}

template<typename T>
void log_array_noendl(FILE *f, T *array, size_t size)
{
	for (size_t i=0; i<size; i++) {
		if (typeid(T) == typeid(int)) {
			fprintf(f, "%d ", array[i]);
		} else if (typeid(T) == typeid(float)) {
			fprintf(f, "%.10lf \t", array[i]);
		} else if (typeid(T) == typeid(double)) {
			fprintf(f, "%.10lf \t", array[i]);
		} else {
			fprintf(f, "Unsupported type\n");
		}
	}
}

template<typename T>
void del_content_vec(T &vec)
{
	for (auto iter = vec.begin(); iter != vec.end();  iter++) {
		if (!(*iter)) {
			continue;
		}
		delete (*iter);
	}
	vec.clear();
}

template<typename T>
void del_content_map(T &map)
{
	for (auto iter = map.begin(); iter != map.end();  iter++) {
		if (!(iter->second)) {
			continue;
		}
		delete (iter->second);
	}
	map.clear();
}

#endif /* UTILS_H */
