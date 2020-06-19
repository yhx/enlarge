

#include <Python.h>

wchar_t **argv2argv_w(char **argv, int argc);
void free_argv_w(wchar_t **argv, int argc);

void printPyErr();

PyObject * PyModule(const char * moduleName);

PyObject * PyFunc(PyObject * module, const char * funcName);

PyObject * PyCall(PyObject * func, PyObject *args);

PyObject * PyTuple(const char * types, ...);
