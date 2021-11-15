#ifndef PROCBUF_H
#define PROCBUF_H

#include "CrossSpike.h"

class ProcBuf {
public:
	ProcBuf(CrossSpike **cs, int proc_rank, int proc_num, int thread_num, int min_delay);
	~ProcBuf();

	int update_cpu(const int &thread_id, const int &time, pthread_barrier_t *barrier);

	int upload_cpu(const int &thread_id, nid_t *tables, nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time);

	int update_gpu(const int &thread_id, const int &time, pthread_barrier_t *barrier);

	int upload_gpu(const int &thread_id, nid_t *tables, nsize_t *table_sizes, nsize_t *c_table_sizes, const size_t &table_cap, const int &max_delay, const int &time, const int &grid, const int &block);

	// Cap _thread_num * _proc_num * _thread_num * (_min_delay + 1)
	integer_t *_recv_start;
	integer_t *_send_start;

	// Cap _proc_num + 1
	integer_t *_recv_offset;
	integer_t *_send_offset;

	// instance level offset view for sender
    // Cap _proc_num * dst_thread_num * src_thread_num;
	integer_t *_data_offset;

	// instance level offset view for receiver
	// Cap _proc_num * dst_thread_num
	integer_t *_rdata_offset;
	integer_t *_sdata_offset;


	// Cap _proc_num
	integer_t *_recv_num;
	integer_t *_send_num;

	// Cap _rdata_size[_thread_num]
	nid_t *_recv_data;
	// Cap _sdata_size[_thread_num]
	nid_t *_send_data;

	// Cap _thread_num
	CrossSpike **_cs;

	// Cap _thread_num + 1;
	size_t *_rdata_size;
	size_t *_sdata_size;

	int _proc_rank;
	int _proc_num;
	int _thread_num;
	int _min_delay;
};

#endif // PROCBUF_H
