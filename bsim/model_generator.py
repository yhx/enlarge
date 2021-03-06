
from generator import *
from data_generator import Data

C_TYPE_SORT = {
    'char' : 0,
    'unsigned char' : 1,
    'short' : 2,
    'unsigned short' : 3,
    'int' : 4,
    'unsigned int' : 5,
    'float' : 6,
    'long' : 7,
    'unsigned long': 8,
    'long long' : 9,
    'unsigned long long': 10,
    'double' : 11,
    'long double' : 12
}

def mycap(v):
    if len(v) < 1:
        return v
    return v[0].upper() + v[1:]

def myhash(v):
    if v in C_TYPE_SORT:
        return C_TYPE_SORT[v]
    else:
        return len(C_TYPE_SORT) + abs(hash(v))

class Model(object):
    def __init__(self, name, parameters, variables = None, path='./', pre='',
                 post='', type_='', compute='', headers=[], cu_headers=[]):
        self.name = mycap(name);
        self.classname = "{}{}{}{}".format(mycap(pre), mycap(name), mycap(post), mycap(type_))
        self.type_ = mycap(type_)
        self.pre = pre
        self.post = post
        self.path = path
        self.compute = compute
        self.headers = headers
        self.cu_headers = cu_headers
        self.parameters = {k:parameters[k] for k in sorted(parameters.keys(), key= lambda x:myhash(x), reverse = False)}
        self.variables = {k:variables[k] for k in sorted(variables.keys(), key= lambda x:myhash(x), reverse = False)} if variables else self.parameters
        if not os.path.exists(path):
            os.mkdir(path)

    def generate(self):
        self.generate_h()
        self.generate_c()
        data = Data(name=self.name, parameters=self.variables, path=self.path,
                    pre=self.pre, post = self.post+"Data",
                    compute=self.compute, headers=['../../utils/type.h', '../../utils/BlockSize.h'], cu_headers=['../../third_party/cuda/helper_cuda.h'])
        data.generate()

        # self.generate_cu()
        # self.generate_mpi()

    def generate_h(self):
        h = HGenerator("{}/{}.h".format(self.path, self.classname))

        h.include_std("stdio.h")
        h.blank_line()
        h.include("../../interface/Neuron.h")
        h.blank_line()

        # for i in self.headers:
        #     h.include(i)
        # h.blank_line()

        h.struct(self.classname, self.type_, 0)
        h.line_no_end("public:", tab=0)
        h.print_("{}(".format(self.classname))
        for k in self.parameters:
            for v in self.parameters[k]:
                h.print_("{} {}={}, ".format(k, v, '1e-4' if (('delay' in v) or ('tau' in v)) else '0'))
        h.backspace(2)
        h.line(")", tab=0)
        h.line("{}(const {} &templ)".format(self.classname, self.classname))
        h.line("~{}()".format(self.classname))
        h.blank_line()

        h.line("virtual Type getType() const override")
        h.line("virtual int hardCopy(void * data, int idx, int base, const SimInfo &info) override")
        if self.type_ == "Neuron":
            h.line("virtual Synapse * createSynapse(real weight, real delay, SpikeType type, real tau) override;") 
        h.blank_line()

        h.line("const static Type type;")
        h.blank_line()

        h.line_no_end("protected:", tab=0)
        for k in self.parameters:
            for v in self.parameters[k]:
                if self.type_ != 'Synapse' or (v!='weight' and v!='delay'):
                    h.line("{} _{}".format(k, v))
        h.struct_end()

        h.close()
        return 0

    def generate_c(self):
        c = CGenerator("{}/{}.cpp".format(self.path, self.classname))
        c.include_std("math.h")
        c.blank_line()
        c.include("{}.h".format(self.classname))
        c.include("{}Data.h".format(self.name))
        c.blank_line()

        c.line("const Type {}::type = {};".format(self.classname, mycap(self.name)), tab=0);
        c.blank_line()

        c.print_("{}::{}(".format(self.classname, self.classname))
        for k in self.parameters:
            for v in self.parameters[k]:
                c.print_("{} {}, ".format(k, v))
        c.backspace(2)
        c.print_("):")
        if self.type_ == 'Synapse':
            c.print_(" Synapse(0, weight, delay)")
        else:
            c.print_(" {}()".format(self.type_))
        for k in self.parameters:
            for v in self.parameters[k]:
                if self.type_ != 'Synapse' or (v!='weight' and v!='delay'):
                    c.print_(", _{}({})".format(v, v))
        c.blank_line()
        c.open_brace()
        c.close_brace()
        c.blank_line()

        c.func_start("{}::{}(const {} &templ)".format(self.classname, self.classname, self.classname))
        for k in self.parameters:
            for v in self.parameters[k]:
                c.line("this->_{} = templ._{}".format(v, v))
        c.func_end()
        c.func_start("{}::~{}()".format(self.classname, self.classname))
        c.func_end()

        c.func_start("Type {}::getType() const".format(self.classname))
        c.func_end("type")
        c.blank_line()

        if self.type_ == "Neuron":
            c.func_start("Synapse * {}::createSynapse(real weight, real delay, SpikeType type, real tau)".format(self.classname)) 
            c.line(r'printf("Not implemented!\n")')
            c.func_end("NULL");
            c.blank_line()

        c.func_start("int {}::hardCopy(void * data, int idx, int base, const SimInfo &info)".format(self.classname))
        c.line("{}Data *p = ({}Data *)data".format(self.name, self.name))
        # c.line("real dt = info.dt")
        if self.type_ == 'Synapse':
            c.line("int delay_steps = static_cast<int>(round(_delay/dt))")
            c.line("real weight = this->_weight")
            c.line("assert(fabs(_tau_syn) > ZERO)")
            c.line_no_end("if (fabs(_tau_syn) > ZERO) {")
            c.line("real c1 = exp(-(_delay-dt*delay_steps)/_tau_syn)", tab=2)
            c.line("weight = weight * c1", tab=2)
            c.line_no_end("}")
        c.blank_line()
        c.line("setID(idx+base)")
        c.blank_line()

        for k in self.parameters:
            for v in self.parameters[k]:
                c.line("p->p{}[idx] = this->_{}".format(mycap(v), v))
        c.func_end("1")
        c.close()

        return 0

    def generate_compute(self):
        lines = self.compute.split()
        c = CGenerator("{}/{}Data.compute.cpp".format(self.path, self.classname))
        c.func_start("void update{}(Connection *connection, void *_data, real * currentE, real *currentI, int *firedTable, int *firedTablesfiredTableSizes, int num, int offset, int time)".format(self.name))
        c.line("{} *data = ({}*)_data".format(self.classname, self.classname));
        c.line("int currentIdx = time % connection->maxDelay+1)")
        c.for_start("int nid=0; nid<num; nid++")
        c.line("int gnid = offset + nid")
        c.if_start("data->pRefracStep[nid] <= 0")
        for i in lines:
            c.line(i);
        c.line("bool fired = data->pV_m[nid] >= data->pV_thresh[nid]")
        c.if_start("fired")
        c.line("firedTable[firedTableSizes[currentIdx] + gFiredTableCap * currentIdx] =  gnid")
        c.line("firedTableSizes[currentIdx]++")
        c.line("data->pRefracStep[nid] = data->pRefracTime[nid] - 1")
        c.line("data->pV_m[nid] = data->pV_reset[nid]")
        c.else()
        c.line("data->pI_e[nid] += currentE[gnid]")
        c.line("data->pI_i[nid] += currentI[gnid]")
        c.if_end()
        c.else()
        c.line("currentE[gnid] = 0")
        c.line("currentI[gnid] = 0")
        c.line("data->pRefracStep[nid]--")
        c.if_end()
        c.for_end()
        c.func_end();
        c.close()


        cu = CUDAGenerator("{}/{}Data.kernel.cu".format(self.path, self.classname))
        cu.close()


if __name__ == '__main__':
    izh = Model('Izhikevich', {'real': ['v', 'u', 'a', 'b', 'c', 'd']},
                path='../src/neuron/izhikevich/', pre='', post='',
                type_='Neuron')
    # izh.generate()

    traub = Model('TraubMiles', 
                  {'real': ['gNa', 'ENa', 'gK', 'EK', 'gl', 'El', 'C', "V", "m", "h", "n", "tau", "E"]},
                  {'real': ['gNa', 'ENa', 'gK', 'EK', 'gl', 'El', 'C', "V", "m", "h", "n", "decay", "E"]},
                  path='../src/neuron/traubmiles/', pre='', post='',
                  type_='Neuron')
    traub.generate()

    pw = Model('PiecewiseSTDP', 
                  {'real': ['tLrn', 'tChng', 'tDecay', 'tPunish10', 'tPunish01', 'gMax', 'gMid', "gSlope", "tauShift", "gSyn0", 'g', 'gRaw', "tau", 'E']},
                  {'real': ['lim0', 'lim1', 'slope0', 'slope1', 'off0', 'off1', 'off2', 'g', 'gRaw', "tau", 'E']},
                  path='../src/neuron/piecewiseSTDP/', pre='', post='',
                  type_='Synapse')

