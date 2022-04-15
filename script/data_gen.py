
def myhash(v):
    return abs(hash(v))

class DataGen():
    def __init__(self, name, parent, parameters):
        self.name = name
        self.parent = parent.capitalize()
        self.parameters = parameters
        self.tab = '\t'
        self.bname = name
        self.parameters = self.create_parameters(parameters)

    def write_line(self,op, tab_num, line):
        op.write('{}{}\n'.format(self.tab * tab_num, line))

    def create_parameters(self,parameters):
        dict_parameters = {}
        for key in parameters:
            if parameters[key] in dict_parameters:
                dict_parameters[parameters[key]].append(key)
            else:
                dict_parameters[parameters[key]] = []
                dict_parameters[parameters[key]].append(key)
        return dict_parameters

    def creat_c(self, filename, filetype, path="", util_path="../utils/"):
        op_c = open('{}/{}{}.cpp'.format(path, self.name,filename), "w+")

        self.write_line(op_c, 0, '#include <assert.h>')
        self.write_line(op_c, 0, '#include "{}helper_c.h"'.format(util_path))
        self.write_line(op_c, 0, '#include "{}.h"'.format(self.bname))
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, '{}::{}()'.format(self.bname, self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, '_is_view = false;')
        self.write_line(op_c, 1, '_num = 0;')
        self.write_line(op_c, 1, '_gpu_array = NULL;')
        if filetype == "model":
            self.write_line(op_c, 1, '_gpu = NULL;')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, '{} = NULL;'.format(value))
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 0, '{}::{}(size_t num) '.format(self.bname,self.bname) + '{')
        self.write_line(op_c, 1, 'alloc(num);')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '{}::~{}() '.format(self.bname,self.bname)+'{')
        self.write_line(op_c, 1, 'if (_num > 0 && !_is_view) {')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 2, 'delete [] {};'.format(value))
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 1, '}')
        self.write_line(op_c, 0, '#ifdef USE_GPU')
        self.write_line(op_c, 1, 'free_gpu();')
        self.write_line(op_c, 0, '#else')
        if filetype == "model":
            self.write_line(op_c, 1, 'assert(!_gpu && !_gpu_array);')
        else:
            self.write_line(op_c, 1, 'assert(!_gpu);')
        self.write_line(op_c, 0, '#endif')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, 'void {}::alloc(size_t num)'.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, '_is_view = false;')
        self.write_line(op_c, 1, '_num = num;')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, '{} = new uinteger_t[num]();'.format(value))
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 1, '_gpu_array = NULL;')
        if filetype == "model":
            self.write_line(op_c, 1, '_gpu = NULL;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, 'int {}::save(FILE *f, size_t num)'.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, 'if (num <=0 || num > _num) {')
        self.write_line(op_c, 2, 'num = _num;')
        self.write_line(op_c, 1, '}')
        self.write_line(op_c, 1, 'fwrite_c(&num, 1, f);')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, 'fwrite_c(&{}, _num, f);'.format(value))
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 1, 'return 0;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, 'int {}::load(FILE *f)'.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, 'size_t num;')
        self.write_line(op_c, 1, 'fread_c(&num, 1, f);')
        self.write_line(op_c, 1, 'if (_num != 0) {')
        self.write_line(op_c, 2, r'printf("Reconstruct Data is not supported!\n");')
        self.write_line(op_c, 2, 'return -1;')
        self.write_line(op_c, 1, '}')
        self.write_line(op_c, 1, 'alloc(num);')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, 'fread_c({}, num, f);'.format(value))
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, '#ifdef USE_MPI')
        self.write_line(op_c, 0, 'int {}::send(int dest, int tag, MPI_Comm comm, int offset, size_t num) '.format(self.bname) + '{')
        self.write_line(op_c, 1, 'if (offset >= _num || offset + num > _num) {')
        self.write_line(op_c, 2, r'printf("Wrong offset %d and num %d\n", offset, num);')
        self.write_line(op_c, 2, 'return -1;')
        self.write_line(op_c, 1, '}')
        self.write_line(op_c, 1, 'if (num <= 0) {')
        self.write_line(op_c, 2, 'num = _num - offset;')
        self.write_line(op_c, 1, '}')
        self.write_line(op_c, 1, 'ret = MPI_Send(&(num), 1, MPI_SIZE_T, dest, tag, comm);')
        self.write_line(op_c, 1, 'assert(ret == MPI_SUCCESS);')
        num = 1
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, 'ret = MPI_Send({} + offset, num, MPI_{}, dest, tag+{}, comm);'.format(value,key.upper(),num))
                num += 1
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 1, 'assert(ret == MPI_SUCCESS);')
        self.write_line(op_c, 1, 'return 0;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, 'int {}::recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset) '.format(self.bname) + '{')
        self.write_line(op_c, 1, 'size_t num = 0;')
        self.write_line(op_c, 1, 'MPI_Status status;')
        self.write_line(op_c, 1, 'ret = MPI_Recv(&(_num), 1, MPI_SIZE_T, src, tag, comm, &status);')
        self.write_line(op_c, 1, 'assert(ret==MPI_SUCCESS);')
        self.write_line(op_c, 1, 'if (_num == 0) {')
        self.write_line(op_c, 2, 'alloc(offset + num);')
        self.write_line(op_c, 1, '}')
        self.write_line(op_c, 1, 'if (_num < offset + num) {')
        self.write_line(op_c, 2, r'printf("Wrong offset %d and num %d\n", offset, num);')
        self.write_line(op_c, 2, 'return -1;')
        self.write_line(op_c, 1, '}')
        num = 1
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, 'ret = MPI_Recv({} + offset, num, MPI_{}, src, tag+{}, comm, &status);'.format(value,key.upper(),num))
                num += 1
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 1, 'assert(ret==MPI_SUCCESS);')
        self.write_line(op_c, 1, 'return 0;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, '#else')
        self.write_line(op_c, 0, 'int {}::send() '.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, r'printf("MPI not enabled!\n");')
        self.write_line(op_c, 1, 'return -1;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, 'int {}::recv() '.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, r'printf("MPI not enabled!\n");')
        self.write_line(op_c, 1, 'return -1;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '#endif')
        self.write_line(op_c, 0, '\n')

        self.write_line(op_c, 0, 'bool {}::is_equal({} *p, size_t *shuffle1, size_t *shuffle2) '.format(self.bname, self.parent))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, 'Data *d = dynamic_cast<Data *>(p)')
        self.write_line(op_c, 1, 'bool ret = _num == d.num;')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_c, 1, 'ret = ret && isEqualArray({}, d->{}, _num, shuffle1, shuffle2);'.format(value, value))
            self.write_line(op_c, 0, '\n')
        self.write_line(op_c, 1, 'return ret;')
        self.write_line(op_c, 0, '}')

        self.write_line(op_c, 0, '#ifndef USE_GPU')
        self.write_line(op_c, 0, 'void {}::free_gpu'.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, r'printf("GPU not enabled!\n");')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, 'int {}::to_gpu()'.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, r'printf("GPU not enabled!\n");')
        self.write_line(op_c, 1, 'return -1;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, 'int {}::from_gpu()'.format(self.bname))
        self.write_line(op_c, 0, '{')
        self.write_line(op_c, 1, r'printf("GPU not enabled!\n");')
        self.write_line(op_c, 1, 'return -1;')
        self.write_line(op_c, 0, '}')
        self.write_line(op_c, 0, '#endif')
        op_c.close()




    def creat_h(self, filename, filetype, path, util_path="../base/"):
        op_h = open('{}/{}{}.h'.format(path, self.name, filename), "w+")
        self.write_line(op_h, 0, '#ifndef {}_H'.format(self.name.upper()))
        self.write_line(op_h, 0, '#define {}_H'.format(self.name.upper()))
        self.write_line(op_h, 0, '#include "{}constant.h"'.format(util_path))
        self.write_line(op_h, 0, '\n')

        self.write_line(op_h, 0, 'class {}::{} '.format(self.bname, self.parent)+'{')
        self.write_line(op_h, 0, 'public:')
        self.write_line(op_h, 1, '{}();'.format(self.bname))
        self.write_line(op_h, 1, '{}(size_t num);'.format(self.bname))
        self.write_line(op_h, 1, '~{}();'.format(self.bname))
        self.write_line(op_h, 0, '\n')

        self.write_line(op_h, 1, 'virtual int save() override;')
        self.write_line(op_h, 1, 'virtual int load() override;')
        self.write_line(op_h, 0, '\n')

        self.write_line(op_h, 1, 'virtual int send(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0, size_t num=0) override;')
        self.write_line(op_h, 1, 'virtual int recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0) override;')
        self.write_line(op_h, 0, '\n')

        self.write_line(op_h, 1, 'virtual int to_gpu() override;')
        self.write_line(op_h, 1, 'virtual int fetch() override;')
        self.write_line(op_h, 1, 'template<typename T>')
        self.write_line(op_h, 1, 'virtual bool is_equal({} *p, T *shuffle1=NULL, T *shuffle2=NULL) override;'.format(self.parent))
        self.write_line(op_h, 1, 'bool _is_view;')
        self.write_line(op_h, 1, 'size_t _num;')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_h, 1, 'uinteger_t *{};'.format(value))
            self.write_line(op_h, 0, '\n')
        if filetype == "model":
            self.write_line(op_h, 1, '{} * _gpu;'.format(self.bname))
        self.write_line(op_h, 1, '{} * _gpu_aray;'.format(self.bname))
        self.write_line(op_h, 0, 'protected:')
        self.write_line(op_h, 1, 'int alloc(size_t num);')
        self.write_line(op_h, 1, 'void free_gpu();')
        self.write_line(op_h, 0, '};')
        self.write_line(op_h, 0, '#endif')
        op_h.close()

    def creat_cu(self, filename, filetype, path="", util_path="../gpu_utils/"):
        op_cu = open('{}/{}{}.cu'.format(path, self.name, filename), "w+")
        self.write_line(op_cu, 0, '#include <assert.h>')
        self.write_line(op_cu, 0, '#include "{}helper_gpu.h"'.format(util_path))
        self.write_line(op_cu, 0, '#include "{}.h"'.format(self.bname))
        self.write_line(op_cu, 0, '\n')
        self.write_line(op_cu, 0, 'void {}::free_gpu()'.format(self.bname))
        self.write_line(op_cu, 0, '{')
        self.write_line(op_cu, 1, 'if (_gpu_array && num > 0) {')
        self.write_line(op_cu, 2, 'gpuFree(_gpu_array->data);')
        self.write_line(op_cu, 1, '}')
        if filetype == "model":
            self.write_line(op_cu, 1, '_gpu->_num = 0;')
        self.write_line(op_cu, 1, 'delete _gpu_array;')
        self.write_line(op_cu, 1, '_gpu_array = NULL;')
        if filetype == "model":
            self.write_line(op_cu, 1, 'if (_gpu) {')
            self.write_line(op_cu, 2, 'gpuFree(_gpu);')
            self.write_line(op_cu, 1, '}')
            self.write_line(op_cu, 1, '_gpu = NULL;')
            self.write_line(op_cu, 0, '}')



        self.write_line(op_cu, 0, 'int to_gpu() ')
        self.write_line(op_cu, 0, '{')
        self.write_line(op_cu, 1, 'if (!_gpu_array) {')
        self.write_line(op_cu, 2, '_gpu_array = new {}();'.format(self.bname))
        self.write_line(op_cu, 2, '_gpu_array->_is_view = _is_view;')
        self.write_line(op_cu, 2, '_gpu_array->_num = _num;')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_cu, 2, '_gpu_array->{} = copyToGPU(_data, num);'.format(value))
            self.write_line(op_cu, 0, '\n')
        if filetype == "model":
            self.write_line(op_cu, 0, '_gpu = copyToGPU(_gpu_array, 1);')
        self.write_line(op_cu, 1, '} else {')
        self.write_line(op_cu, 2, 'assert(_gpu_array->_num == _num);')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_cu, 2, 'copyToGPU(_gpu_array->{}, {}, num);'.format(value, value))
            self.write_line(op_cu, 0, '\n')
        self.write_line(op_cu, 1, '}')
        self.write_line(op_cu, 1, 'return 0;')
        self.write_line(op_cu, 0, '}')
        self.write_line(op_cu, 0, '\n')

        self.write_line(op_cu, 0, 'int from_gpu()')
        self.write_line(op_cu, 0, '{')
        self.write_line(op_cu, 1, 'if (!_gpu_array) {')
        self.write_line(op_cu, 2, r'printf("No Data on GPU!\n");')
        self.write_line(op_cu, 2, 'return -1;')
        self.write_line(op_cu, 1, '}')
        for key in self.parameters:
            for value in self.parameters[key]:
                self.write_line(op_cu, 1, 'copyFromGPU({}, _gpu_array->{}, num);'.format(value, value))
            self.write_line(op_cu, 0, '\n')
        self.write_line(op_cu, 1, 'return 0;')
        self.write_line(op_cu, 0, '}')

        op_cu.close()



if __name__ == '__main__':
    datagen = DataGen('CrossMap','Data',{"_idx2index":"integer_t *",
        "_crossnodeIndex2idx":"integer_t *", "_crossSize":"size_t", "_num":"size_t"})
    datagen.creat_c('', "model1", path='./test')
    datagen.creat_h('', "model1", path='./test')
    datagen.creat_cu('', "model1", path='./test')





