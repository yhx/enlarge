
import os
from ctypes import *
import importlib

from bsim.cudamemop import cudamemops
from bsim.data import Data
from bsim.generator import CUDAGenerator, CGenerator, PyGenerator


class Connection(Data):
    def __init__(self, debug=False):
        self.delay_start = []
        self.delay_num = []
        self.rev_delay_start = []
        self.rev_delay_num = []
        self.rev_map2sid = []

        self.dir = os.path.dirname(__file__)
        self.debug = debug
        self._so = None

    def so(self):
        if not self._so:
            self.compile_()
        return self._so

    def to_c(self):
        assert len(self.delay_start) == len(self.delay_num) and \
               len(self.rev_delay_start) == len(self.rev_delay_num)
        self._generate_py()

        self.c_type = importlib.import_module(
            'bsim.py_code.cconnection'.format(len(self.delay_start), len(self.rev_map2sid))
            ).CConnection
        c = self.c_type()
        c.n_len = len(self.delay_start)
        c.r_n_len = len(self.rev_delay_start)
        c.s_len = len(self.rev_map2sid)
        c.delay_start = (c_int * len(self.delay_start))(*(self.delay_start))
        c.delay_num = (c_int * len(self.delay_num))(*(self.delay_num))

        c.rev_delay_start = (c_int * len(self.rev_delay_start))(*(self.rev_delay_start))
        c.rev_delay_num = (c_int * len(self.rev_delay_num))(*(self.rev_delay_num))
        c.rev_map2sid = (c_int * len(self.rev_map2sid))(*(self.rev_map2sid))

        return c

    def to_gpu(self):
        c_data = self.to_c()
        gpu_data = self.so().to_gpu_connection(pointer(c_data))

        if self.debug:
            print("\nPython GPU Pointer: %s" % hex(cast(gpu_data, c_void_p).value))

        return gpu_data

    def from_gpu(self, gpu, only_struct=True):
        """
        :param gpu:  must be of type POINTER(self.cconnection)
        :param only_struct:
        :return:
        """
        cpu = self.so().from_gpu_connection(gpu)
        c = cast(cpu, POINTER(self.c_type * 1)).contents[0]

        if self.debug:
            print("\nPython CPU Pointer: %s" % hex(cast(cpu, c_void_p).value))
            print("Python CPU n_len: %s r_n_len: %s s_len: %s\n" % (int(c.n_len), int(c.r_n_len), int(c.s_len)))

        if not only_struct:
            c.delay_start = cudamemops.from_gpu_int(c.delay_start, len(self.delay_start))
            c.delay_num = cudamemops.from_gpu_int(c.delay_num, len(self.delay_num))
            c.rev_delay_start = cudamemops.from_gpu_int(c.rev_delay_start, len(self.rev_delay_start))
            c.rev_delay_num = cudamemops.from_gpu_int(c.rev_delay_num, len(self.rev_delay_num))
            c.rev_map2sid = cudamemops.from_gpu_int(c.rev_map2sid, len(self.rev_map2sid))

        return c

    def compile_(self):
        self._generate_h()
        self._generate_data_cu()

        if CUDAGenerator.compile_(
                src='{}/c_code/connection.data.cu'.format(self.dir),
                output='{}/c_so/connection.data.so'.format(self.dir)
        ):
            self._so = cdll.LoadLibrary('{}/c_so/connection.data.so'.format(self.dir))
            self._so.to_gpu_connection.restype = POINTER(self.c_type)
            self._so.from_gpu_connection.restype = POINTER(self.c_type)
        else:
            self._so = None
            raise EnvironmentError('Compile file connection.data.so failed')

    def _generate_h(self):
        h_gen = CGenerator("%s/c_code/connection.h" % self.dir)

        h_gen.blank_line(2)
        h_gen.if_define('connection.h')
        h_gen.blank_line(2)

        h_gen.struct("CConnection")
        h_gen.line("int *delay_start")
        h_gen.line("int *delay_num")
        h_gen.line("int *rev_delay_start")
        h_gen.line("int *rev_delay_num")
        h_gen.line("int *rev_map2sid")
        h_gen.line("int n_len")
        h_gen.line("int r_n_len")
        h_gen.line("int s_len")

        h_gen.line("}")
        h_gen.blank_line()

        h_gen.line_no_end('extern "C" {', 0)
        h_gen.line("CConnection * to_gpu_connection(CConnection *cpu)")
        h_gen.line("CConnection * from_gpu_connection(CConnection *gpu)")
        h_gen.close_brace()
        h_gen.blank_line()

        h_gen.end_if_define('connection.h')

        h_gen.close()

    def _generate_data_cu(self):
        cu_gen = CUDAGenerator('{}/c_code/connection.data.cu'.format(self.dir))

        cu_gen.blank_line(2)
        if self.debug:
            cu_gen.include_std('stdio.h')

        cu_gen.include_std('stdlib.h')
        cu_gen.blank_line()
        cu_gen.include('helper_cuda.h')
        cu_gen.include('connection.h')
        cu_gen.blank_line(2)

        cu_gen.line_no_end("CConnection * to_gpu_connection(CConnection *cpu)", 0)
        cu_gen.open_brace()
        cu_gen.line('CConnection * gpu = (CConnection*)malloc(sizeof(CConnection))')
        cu_gen.line('gpu->n_len = cpu->n_len')
        cu_gen.line('gpu->r_n_len = cpu->r_n_len')
        cu_gen.line('gpu->s_len = cpu->s_len')

        cu_gen.to_gpu(ret='gpu->delay_start', cpu='cpu->delay_start', type_='int', num='cpu->n_len')
        cu_gen.to_gpu(ret='gpu->delay_num', cpu='cpu->delay_num', type_='int', num='cpu->n_len')
        cu_gen.to_gpu(ret='gpu->rev_delay_start', cpu='cpu->rev_delay_start', type_='int', num='cpu->r_n_len')
        cu_gen.to_gpu(ret='gpu->rev_delay_num', cpu='cpu->rev_delay_num', type_='int', num='cpu->r_n_len')
        cu_gen.to_gpu(ret='gpu->rev_map2sid', cpu='cpu->rev_map2sid', type_='int', num='cpu->s_len')

        cu_gen.line('CConnection * ret = NULL')
        cu_gen.to_gpu(ret='ret', cpu='gpu', type_='CConnection')

        if self.debug:
            cu_gen.line(line=r'printf("GPU CConnection Pointer: %p\n", ret)')
            cu_gen.line(line=r'printf("GPU n_len: %d r_n_len: %d s_len: %d\n", gpu->n_len, gpu->r_n_len, gpu->s_len)')

        cu_gen.line('return ret')
        cu_gen.close_brace()
        cu_gen.blank_line()

        cu_gen.line_no_end("CConnection * from_gpu_connection(CConnection *gpu)", 0)
        cu_gen.open_brace()
        cu_gen.from_gpu(gpu='gpu', ret='ret', type_='CConnection')

        if self.debug:
            cu_gen.line(line=r'printf("CPU CConnection Pointer: %p\n", ret)')
            cu_gen.line(line=r'printf("CPU n_len: %d r_n_len: %d s_len: %d\n", ret->n_len, ret->r_n_len, ret->s_len)')

        cu_gen.line('return ret')
        cu_gen.close_brace()
        cu_gen.blank_line()

        cu_gen.close()
        return

    def _generate_py(self):
        py_gen = PyGenerator('{}/py_code/cconnection.py'.format(self.dir))

        py_gen.blank_line()
        py_gen.import_("*", "ctypes")
        py_gen.blank_line(2)
        py_gen.class_("CConnection", "Structure")
        py_gen.line("_fields_ = [")
        py_gen.line('("delay_start", POINTER(c_int)),', 2)
        py_gen.line('("delay_num", POINTER(c_int)),', 2)
        py_gen.line('("rev_delay_start", POINTER(c_int)),', 2)
        py_gen.line('("rev_delay_num", POINTER(c_int)),', 2)
        py_gen.line('("rev_map2sid", POINTER(c_int)),', 2)
        py_gen.line('("n_len", c_int),', 2)
        py_gen.line('("r_n_len", c_int),', 2)
        py_gen.line('("s_len", c_int)', 2)
        py_gen.line("]")
        py_gen.blank_line()

        py_gen.close()
        return


# class CConnection(Structure):
#    _fields_ = [
#        ("delay_start", POINTER(c_int)),
#        ("delay_num", POINTER(c_int)),
#        ("rev_delay_start", POINTER(c_int)),
#        ("rev_delay_num", POINTER(c_int)),
#        ("rev_map2sid", POINTER(c_int)),
#        ("n_length", c_int),
#        ("s_length", c_int)
#    ]

