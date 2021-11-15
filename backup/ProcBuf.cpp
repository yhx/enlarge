
#include <string>

#include "../helper/helper_c.h"
#include "../helper/helper_array.h"
#include "../helper/helper_parallel.h"
#include "ProcBuf.h"

using std::string;
using std::to_string;

ProcBuf::ProcBuf(CrossSpike **cs, int proc_rank, int proc_num, int thread_num, int min_delay)
{
	_cs = cs;
	_proc_rank = proc_rank;
	_proc_num = proc_num;
	_thread_num = thread_num;
	_min_delay = min_delay;

	_rdata_size = malloc_c<size_t>(_thread_num + 1);
	_sdata_size = malloc_c<size_t>(_thread_num + 1);

	_rdata_size[0] = 0;
	_sdata_size[0] = 0;

	for (int i=0; i<thread_num; i++) {
		_rdata_size[i+1] = _rdata_size[i] + cs[i]->_recv_offset[cs[i]->_proc_num];
		_sdata_size[i+1] = _sdata_size[i] + cs[i]->_send_offset[cs[i]->_proc_num];
	}

	_recv_offset = malloc_c<integer_t>(_proc_num);
	_send_offset = malloc_c<integer_t>(_proc_num);
	_recv_offset[0] = 0;
	_send_offset[0] = 0;
	for (int p=0; p<_proc_num-1; p++) {
		_recv_offset[p+1] = _recv_offset[p];
		_send_offset[p+1] = _send_offset[p];
		for (int t=0; t<_thread_num; t++) {
			int idx = p * _thread_num + t;
			for (int k=0; k<_thread_num; k++) {
				_recv_offset[p+1] += cs[k]->_recv_offset[idx+1] - cs[k]->_recv_offset[idx];
				_send_offset[p+1] += cs[k]->_send_offset[idx+1] - cs[k]->_send_offset[idx];
			}
		}
	}

	_data_offset = malloc_c<integer_t>(_proc_num * _thread_num * _thread_num);
	for (int p=0; p<_proc_num; p++) {
		// _data_offset[p*_thread_num*_thread_num] = _send_offset[p];
		_data_offset[p*_thread_num*_thread_num] = 0;
	}

	_rdata_offset = malloc_c<integer_t>(_proc_num * _thread_num);
	_sdata_offset = malloc_c<integer_t>(_proc_num * _thread_num);


	int size = _thread_num * _proc_num * _thread_num * (_min_delay +  1);

	_recv_start = malloc_c<integer_t>(size);
	_send_start = malloc_c<integer_t>(size);

	_recv_num = malloc_c<integer_t>(_proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);

	_recv_data = malloc_c<nid_t>(_rdata_size[thread_num]);
	_send_data = malloc_c<nid_t>(_sdata_size[thread_num]);
}

ProcBuf::~ProcBuf()
{
	_cs = NULL;

	_rdata_size = free_c(_rdata_size);
	_sdata_size = free_c(_sdata_size);

	_recv_start = free_c(_recv_start);
	_send_start = free_c(_send_start);

	_recv_offset = free_c(_recv_offset);
	_send_offset = free_c(_send_offset);

	_rdata_offset = free_c(_recv_offset);
	_sdata_offset = free_c(_send_offset);

	_recv_num = free_c(_recv_num);
	_send_num = free_c(_send_num);

	_recv_data = free_c(_recv_data);
	_send_data = free_c(_send_data);
}

int ProcBuf::update_cpu(const int &thread_id, const int &time, pthread_barrier_t *barrier)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay - 1) {
		// msg start info
		CrossSpike *cst = _cs[thread_id];

		pthread_barrier_wait(barrier);
		if (thread_id == 0) {
			for (int i=0; i<_thread_num; i++) {
				int size = _thread_num * (_min_delay+1);
				// _recv_start [s_t, s_p, d_t]
				MPI_Alltoall(_cs[i]->_send_start, size, MPI_INTEGER_T, _recv_start + i * _proc_num * size, size, MPI_INTEGER_T, MPI_COMM_WORLD);
			}
		}
		// calc thread offset
		// _data_offset [d_p, d_t, s_t]
		int bk_size = _proc_num / _thread_num;
		for (int p=0; p<bk_size; p++) {
			int pid = bk_size * thread_id + p;
			for (int d_t=0; d_t<_thread_num; d_t++) {
				int d_idx = pid * _thread_num + d_t;
				for (int s_t=0; s_t<_thread_num; s_t++) {
					// int s_idx = _proc_rank * _thread_num + s_t;
					int idx = d_idx * _thread_num + s_t;
					if (d_t != _thread_num - 1 || s_t != _thread_num -1) { 
						_data_offset[idx+1] = _data_offset[idx] + (_cs[s_t]->_send_start[d_idx*(_min_delay+1) + _min_delay] - _cs[s_t]->_send_start[d_idx*(_min_delay+1)]);
					} else {
						_send_num[pid] = _data_offset[idx] - _data_offset[pid*_thread_num*_thread_num] + (_cs[s_t]->_send_start[d_idx*(_min_delay+1) + _min_delay] - _cs[s_t]->_send_start[d_idx*(_min_delay+1)]);
					}	
				}
				_sdata_offset[d_idx] = _data_offset[d_idx * _thread_num];
			}
		}

		pthread_barrier_wait(barrier);
		// msg thread offset
		if (thread_id == 0) {
			MPI_Alltoall(_sdata_offset, _thread_num , MPI_INTEGER_T, _rdata_offset, _thread_num, MPI_INTEGER_T, MPI_COMM_WORLD);
		}
		// fetch data
		for (int p=0; p<_proc_num; p++) {
			for (int d_t=0; d_t<_thread_num; d_t++) {
				int d_idx = p * _thread_num + d_t;
				int idx = d_idx * _thread_num + thread_id;
				// memcpy(_send_data + _send_offset[p] + _data_offset[idx], cst->_send_data + cst->_send_offset[d_idx] + cst->_send_start[d_idx*(_min_delay+1)], sizeof(nid_t)*(cst->_send_start[d_idx*(_min_delay+1) + _min_delay] - cst->_send_start[d_idx*(_min_delay+1)]));
				memcpy_c(_send_data + _send_offset[p] + _data_offset[idx], cst->_send_data + cst->_send_offset[d_idx] + cst->_send_start[d_idx*(_min_delay+1)], cst->_send_start[d_idx*(_min_delay+1) + _min_delay] - cst->_send_start[d_idx*(_min_delay+1)]);
			}
		}
		pthread_barrier_wait(barrier);
		// calc recv_num
		for (int p=0; p<bk_size; p++) {
			int s_p = bk_size * thread_id + p;
			int idx = s_p * _thread_num + _thread_num - 1;
			_recv_num[s_p] = _rdata_offset[idx];
			int idx_d = idx * (_min_delay+1);
			for (int s_t=0; s_t<_thread_num; s_t++) {
			    int idx_t = idx_d + s_t * _proc_num * _thread_num * (_min_delay+1);
				_recv_num[s_p] += _recv_start[idx_t + _min_delay] - _recv_start[idx_t];
			}
		}
		// msg data
		pthread_barrier_wait(barrier);
		if (thread_id == 0) {
#if 1
			for (int i=0; i<_thread_num; i++) {
				mpi_print_array(_cs[i]->_send_start, _proc_num*_thread_num*(_min_delay+1), _proc_rank, _proc_num, (string("cs ")+to_string(_proc_rank)+"_"+to_string(i)+" send_start:").c_str());
			}
			for (int i=0; i<_thread_num; i++) {
				mpi_print_array(_cs[i]->_send_offset, _proc_num*_thread_num+1, _proc_rank, _proc_num, (string("cs ")+to_string(_proc_rank)+"_"+to_string(i)+" send_offset:").c_str());
			}
			for (int i=0; i<_thread_num; i++) {
				mpi_print_array(_cs[i]->_send_data, _cs[i]->_send_offset[_proc_num*_thread_num], _proc_rank, _proc_num, (string("cs ")+to_string(_proc_rank)+"_"+to_string(i)+" send_data").c_str());
			}
			mpi_print_array(_recv_start, _thread_num*_proc_num*_thread_num*(_min_delay+1), _proc_rank, _proc_num, (string("recv_start ")+to_string(_proc_rank)+":").c_str());
			mpi_print_array(_data_offset, _thread_num*_thread_num*_proc_num, _proc_rank, _proc_num, (string("data_offset ")+to_string(_proc_rank)+":").c_str());
			mpi_print_array(_sdata_offset, _thread_num*_proc_num, _proc_rank, _proc_num, (string("sdata_offset ")+to_string(_proc_rank)+":").c_str());
			mpi_print_array(_rdata_offset, _thread_num*_proc_num, _proc_rank, _proc_num, (string("rdata_offset ")+to_string(_proc_rank)+":").c_str());
			mpi_print_array(_send_num, _proc_num, _proc_rank, _proc_num, (string("Proc send num ")+to_string(_proc_rank)+":").c_str());
			mpi_print_array(_recv_num, _proc_num, _proc_rank, _proc_num, (string("Proc recv num ")+to_string(_proc_rank)+":").c_str());
			mpi_print_array(_send_data, _send_offset[_proc_num-1]+_send_num[_proc_num-1], _proc_rank, _proc_num, (string("Send ")+to_string(_proc_rank)+":").c_str());
#endif
			int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_NID_T, MPI_COMM_WORLD);
			assert(ret == MPI_SUCCESS);
			mpi_print_array(_recv_data, _recv_offset[_proc_num-1]+_recv_num[_proc_num-1], _proc_rank, _proc_num, (string("Proc recv ")+to_string(_proc_rank)+":").c_str());
		}
	} else {
		_cs[thread_id]->update_cpu(time);
	}
	pthread_barrier_wait(barrier);

	return 0;
}

int ProcBuf::upload_cpu(const int &thread_id, nid_t *tables, nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay -1) {
// #ifdef PROF
// 		double ts = 0, te = 0;
// 		
// 		ts = MPI_Wtime();
// #endif
// #ifdef PROF
// 		te = MPI_Wtime();
// 		_gpu_wait += te - ts;
// #endif

#if 0
// #ifdef ASYNC
#ifdef PROF
		ts = MPI_Wtime();
#endif 
		MPI_Status status_t;
		int ret = MPI_Wait(&_request, &status_t);
		assert(ret == MPI_SUCCESS);
#ifdef PROF
		te = MPI_Wtime();
		_cpu_wait_gpu += te - ts;
#endif
#endif

		for (int d=0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int s_p = 0; s_p<_proc_num; s_p++) {
				for (int s_t = 0; s_t<_thread_num; s_t++) {
					int idx = s_p * _thread_num + thread_id;
					integer_t *start_t = _recv_start + s_t * _thread_num * _proc_num * (_min_delay+1);
					int start = start_t[idx*(_min_delay+1)+d];
					int end = start_t[idx*(_min_delay+1)+d+1];
					int num = end - start;
					if (num > 0) {
						assert(table_sizes[delay_idx] + num <= table_cap);
						memcpy(tables + table_cap*delay_idx + table_sizes[delay_idx], _recv_data + _recv_offset[s_p] + _rdata_offset[idx] + start, num);
						table_sizes[delay_idx] += num;
					}
				}
			}
		}

		{ // Reset
			// memset(_cs[thread_id]->_recv_start, 0, _min_delay * _proc_num + _proc_num);
			memset(_cs[thread_id]->_send_start, 0, _min_delay * _proc_num + _proc_num);

			memset_c(_recv_num, 0, _proc_num);
			memset_c(_send_num, 0, _proc_num);
		}
	}

	return 0;
}
