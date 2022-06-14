# Process Buffer

## 1. MultiLevelSimulator.cu大循环

**总共** `network->_simCycle`次循环。

### 1.1 初始化

**首先调用** `update_time<<<1, 1>>>(g_buffer->_fired_sizes, max_delay, time)`来让GPU的第0个线程来初始化。在这一步中，令 `currentIdx = time % (max_delay + 1)`，令 `gActiveTableSize = 0`，令 `firedTableSizes[currentIdx] = 0`。置零fire_table

### 1.2 神经元计算

**对于每一种类型的所有神经元，调用cudaUpdateLIF或者是cudaUpdateIAF来更新神经元状态。**

```
for (int i=0; i<nTypeNum; i++) {
assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppNeurons[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
}
```

**调用参数解释：**

* `c_pNetGPU->ppConnections[i]`：
* `c_pNetGPU->ppNeurons[i]`
* `g_buffer->_data`
* `g_buffer->_fire_table`
* `g_buffer->_fired_sizes`
* `allNeuronNum`
* `c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i]`
* `c_pNetGPU->pNeuronNums[i]`
* `time`：global时间步
* `&updateSize[c_pNetGPU->pNTypes[i]]`

### 1.3 `fetch_gpu`将接受到的spike进行分配

**每一个线程thread_id都调用** `fetch_gpu`:

```
pbuf->fetch_gpu(thread_id, cm, g_buffer->_fire_table, g_buffer->_fired_sizes, g_buffer->_fire_table_cap, max_delay, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
```

**参数说明：**

* `thread_id`
* `cm`
* `g_buffer->_fire_table`
* `g_buffer->_fired_sizes`
* `g_buffer->_fire_table_cap`
* `max_delay`
* `time`
* `(allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE`
* `MAX_BLOCK_SIZE`

**进一步调用的是** `CrossSpike`对象 `_cs`中每一个线程对应的 `fetch_gpu`方法：

```
inline int fetch_gpu(const int &thread_id, const CrossMap *cm, const nid_t *tables, const nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time, const int &grid, const int &block) {
return _cs[thread_id]->fetch_gpu(cm, tables, table_sizes, table_cap, _proc_num * _thread_num, max_delay, time, grid, block);
}
```

**进一步，调用的函数是****`fetch_kernel`**：

```
int CrossSpike::fetch_gpu(const CrossMap *map, const nid_t *tables, const nsize_t *table_sizes, const size_t &table_cap, const int &proc_num, const int &max_delay, const int &time, const int &grid, const int &block)
{
int delay_idx = time % (max_delay + 1);
int curr_delay = time % _min_delay;
fetch_kernel<<<grid, block>>>(_gpu_array->_send_data, _gpu_array->_send_offset, _gpu_array->_send_start, map->_gpu_array->_idx2index, map->_gpu_array->_index2ridx, tables, table_sizes, table_cap, proc_num, delay_idx, _min_delay, curr_delay);
return 0;
}
```

**调用参数：**

* `_gpu_array->_send_data`
* `_gpu_array->_send_offset`
* `_gpu_array->_send_start`
* `map->_gpu_array->_idx2index`
* `map->_gpu_array->_index2ridx`
* `tables`
* `table_sizes`
* `table_cap`
* `proc_num`：进程数
* `delay_idx`
* `_min_delay`
* `curr_delay`

**这个函数实际调用的是** `CrossSpike.cu.h`。这个函数的伪代码如下：

```
int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 当前gpu的线程号
for (int proc = 0; proc < proc_num; proc++) {  // 对每一个进程i，其实就是每一个node
    for (size_t i = 0; i < fired_size; i += blockDim.x * gridDim.x) { 
        size_t idx = i + tid;  // 获取当前的要处理的fired_stable中的第idx项
        if (idx < fired_size) {  // 如果idx没有超过fired_size
            // 当前处理的是第delay_idx个延迟中的第idx个fire的neuron id，即nid
            TID nid = static_cast<TID>(fired_table[fired_cap*delay_idx + idx]);
            integer_t tmp = idx2index[nid];  // 获取局部虚拟id
            if (tmp >= 0) {
                integer_t map_nid = index2ridx[tmp*proc_num + proc];  // 当前局部虚拟id在
                if (map_nid >= 0) {
                    size_t test_loc = static_cast<size_t>(atomicAdd(const_cast<int*>(&cross_cnt), 1));// 从当前进程要发出的总的脉冲数累加1得到这个map_nid要放置的位置
                    if (test_loc < MAX_BLOCK_SIZE) {// 如果超过了BLOCK_SIZE，则丢弃这个发放，这样不会出现数组溢出的情况
                        cross_neuron_id[test_loc] = static_cast<TID>(map_nid);  // 讲map_nid存在cross_neuron_id中的test_loc位置
                    }
                }
            }
        }
        线程同步

        if (cross_cnt > 0) {
            int idx_t = proc * (min_delay + 1) + curr_delay + 1;
            merge2array(cross_neuron_id, cross_cnt, data, &(num[idx_t]), offset[proc]);  // 将大小为cross_cnt的cross_neuron_id拷贝进入data中的第num[idx_t]+offset[proc]位置处
            if (threadIdx.x == 0) {  // 只有thread ID为0的线程将cross_cnt置为0
                cross_cnt = 0;
            }
        }
        线程同步
    }
    线程同步
}
```

**其中，比较重要的两个数据结构是：**

```
// ID of neurons on this node to index in this map 
// index = _idx2index[id]
integer_t *_idx2index;
```

**和**

```
// idx in this map to ID of shadow neurons on node j
// id = _index2ridx[index * node_num + j], -1 means no shadow neuron on node j
// index2rid的访问方式为index2ridx[index][j]
// index2rid的大小为_index2rid[max(index) + 1][node_num]
// 同一个源的rid存在一起，如果有8 node的话，那么就是(max(index) + 1) * node_num这么大
integer_t *_index2ridx;
```

**示意图如下所示：**

![image-20220426154422303](file://E:\projects\enlarge\document\images\image-20220426154422303.png?lastModify=1654822997)

```
merge2array(DATA *src, const SIZE size, DATA *dst, SIZE * dst_size, const SIZE dst_offset) 
```

**用于在GPU中将大小为size的src数组拷贝进入dst数组的dst_offset+dst_size的位置处。**

**fetch_kernel最终得到的就是存有fire信息的data数组。开始存放的位置是** `_send_start[idx_t]+_send_offset[proc]`。

### 1.4 update_gpu

**用于从GPU中将神经元发放数据** `_send_start`和 `_send_data`拷贝进入CPU中：

```
int CrossSpike::update_gpu(const int &time)
{
int curr_delay = time % _min_delay;
if (curr_delay >= _min_delay -1) {
COPYFROMGPU(_send_start, _gpu_array->_send_start, _proc_num * (_min_delay + 1));
if (_send_offset[_proc_num] > 0) {
COPYFROMGPU(_send_data, _gpu_array->_send_data, _send_offset[_proc_num]);
}
msg_gpu();
} else {
cudaDeviceSynchronize();
update_kernel<<<1, _proc_num>>>(_gpu_array->_send_start, _proc_num, _min_delay, curr_delay);
}

return 0;
}
```

**update_kernel用于更新一下start。**

### 1.5 突触计算

```
cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
```

### 1.6 upload_gpu

**将突触计算的结果重新发给对应的shadow neuron。**

**其主要的调用过程如下：**

```
for (int d=0; d < _min_delay; d++) {
int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);  // 计算当前的delay_idx在fire_table中的实际位置
for (int s_p = 0; s_p<_proc_num; s_p++) {  // 对于每一个源进程
int idx = s_p * _thread_num + thread_id;  // 源进程中的thread_id进程
for (int s_t = 0; s_t<_thread_num; s_t++) {
            // 获取目标线程的start_t中的位置
int idx_t = idx * _thread_num + s_t;
integer_t *start_t = _recv_start + s_t * _thread_num * _proc_num * (_min_delay+1);
int start = start_t[idx*(_min_delay+1)+d];
int end = start_t[idx*(_min_delay+1)+d+1];
int num = end - start;
if (num > 0) {
assert(c_table_sizes[delay_idx] + num <= table_cap);
                // 将对应的数据拷贝进入GPU中
COPYTOGPU(tables + table_cap*delay_idx + c_table_sizes[delay_idx], _recv_data + _recv_offset[s_p] + _data_r_offset[idx_t] + start, num);
                // 更新fire_table的size
c_table_sizes[delay_idx] += num;
}
}
}
}
```

**MultiLevelSimulator.cpp中的283行：**

**pthread_t *thread_ids = malloc_c<pthread_t>(thread_num);**

# You Can Do Any Kind of Atomic Read-Modify-Write Operation

**Atomic read-modify-write operations – or “RMWs” – are more sophisticated than **[atomic loads and stores](http://preshing.com/20130618/atomic-vs-non-atomic-operations). They let you read from a variable in shared memory and simultaneously write a different value in its place. In the C++11 atomic library, all of the following functions perform an RMW:

```
std::atomic<>::fetch_add()
std::atomic<>::fetch_sub()
std::atomic<>::fetch_and()
std::atomic<>::fetch_or()
std::atomic<>::fetch_xor()
std::atomic<>::exchange()
std::atomic<>::compare_exchange_strong()
std::atomic<>::compare_exchange_weak()
```

`fetch_add`, for example, reads from a shared variable, adds another value to it, and writes the result back – all in one indivisible step. You can accomplish the same thing using a mutex, but a mutex-based version wouldn’t be [lock-free](http://preshing.com/20120612/an-introduction-to-lock-free-programming). RMW operations, on the other hand, are designed to be lock-free. They’ll take advantage of lock-free CPU instructions whenever possible, such as `ldrex`/`strex` on ARMv7.

**A novice programmer might look at the above list of functions and ask, “Why does C++11 offer so few RMW operations? Why is there an atomic **`fetch_add`, but no atomic `fetch_multiply`, no `fetch_divide` and no `fetch_shift_left`?” There are two reasons:

1. **Because there is very little need for those RMW operations in practice. Try not to get the wrong impression of how RMWs are used. You can’t write safe multithreaded code by taking a single-threaded algorithm and turning each step into an RMW.**
2. **Because if you do need those operations, you can easily implement them yourself. As the title says, you can do any kind of RMW operation!**

## Compare-and-Swap: The Mother of All RMWs

**Out of all the available RMW operations in C++11, the only one that is absolutely essential is **`compare_exchange_weak`. Every other RMW operation can be implemented using that one. It takes a minimum of two arguments:

```
shared.compare_exchange_weak(T& expected, T desired, ...);
```

**This function attempts to store the **`desired` value to `shared`, but only if the current value of `shared` matches `expected`. It returns `true` if successful. If it fails, it loads the current value of `shared` back into `expected`, which despite its name, is an in/out parameter. This is called a **compare-and-swap** operation, and it all happens in one atomic, indivisible step.

![img](https://preshing.com/images/compare-exchange.png)

**So, suppose you really need an atomic **`fetch_multiply` operation, though I can’t imagine why. Here’s one way to implement it:

```
uint32_t fetch_multiply(std::atomic<uint32_t>& shared, uint32_t multiplier)
{
    uint32_t oldValue = shared.load();
    while (!shared.compare_exchange_weak(oldValue, oldValue * multiplier))
    {
    }
    return oldValue;
}
```

**This is known as a compare-and-swap loop, or ****CAS loop**. The function repeatedly tries to exchange `oldValue` with `oldValue * multiplier` until it succeeds. If no concurrent modifications happen in other threads, `compare_exchange_weak` will usually succeed on the first try. On the other hand, if `shared` is concurrently modified by another thread, it’s totally possible for its value to change between the call to `load` and the call to `compare_exchange_weak`, causing the compare-and-swap operation to fail. In that case, `oldValue` will be updated with the most recent value of `shared`, and the loop will try again.

![img](https://preshing.com/images/fetch-multiply-timeline.png)

**The above implementation of **`fetch_multiply` is both atomic and lock-free. It’s atomic even though the CAS loop may take an indeterminate number of tries, because when the loop finally does modify `shared`, it does so atomically. It’s lock-free because if a single iteration of the CAS loop fails, it’s usually because some other thread modified `shared` successfully. That last statement hinges on the assumption that `compare_exchange_weak` actually compiles to lock-free machine code – more on that below. It also ignores the fact that `compare_exchange_weak` can [fail spuriously](http://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange) on certain platforms, but that’s a rare event.

## You Can Combine Several Steps Into One RMW

`fetch_multiply` just replaces the value of `shared` with a multiple of the same value. What if we want to perform a more elaborate kind of RMW? Can we still make the operation atomic *and* lock-free? Sure we can. To offer a somewhat convoluted example, here’s a function that loads a shared variable, decrements the value if odd, divides it in half if even, and stores the result back only if it’s greater than or equal to 10, all in a single atomic, lock-free operation:

```
uint32_t atomicDecrementOrHalveWithLimit(std::atomic<uint32_t>& shared)
{
    uint32_t oldValue = shared.load();
    uint32_t newValue;
    do
    {
        if (oldValue % 2 == 1)
            newValue = oldValue - 1;
        else
            newValue = oldValue / 2;
        if (newValue < 10)
            break;
    }
    while (!shared.compare_exchange_weak(oldValue, newValue));
    return oldValue;
}
```

**It’s the same idea as before: If **`compare_exchange_weak` fails – usually due to a modification performed by another thread – `oldValue` is updated with a more recent value, and the loop tries again. If, during any attempt, we find that `newValue` is less than 10, the CAS loop terminates early, effectively turning the RMW operation into a no-op.

**The point is that you can put anything inside the CAS loop. Think of the body of the CAS loop as a critical section. Normally, we protect a critical section using a mutex. With a CAS loop, we simply retry the entire transaction until it succeeds.**

**This is obviously a synthetic example. A more practical example can be seen in the **[`AutoResetEvent`](https://github.com/preshing/cpp11-on-multicore/blob/master/common/autoresetevent.h) class described in my [earlier post about semaphores](http://preshing.com/20150316/semaphores-are-surprisingly-versatile). It uses a CAS loop with multiple steps to atomically increment a shared variable up to a limit of 1.

## You Can Combine Several Variables Into One RMW

**So far, we’ve only looked at examples that perform an atomic operation on a single shared variable. What if we want to perform an atomic operation on multiple variables? Normally, we’d protect those variables using a mutex:**

```
std::mutex mutex;
uint32_t x;
uint32_t y;

void atomicFibonacciStep()
{
    std::lock_guard<std::mutex> lock(mutex);
    int t = y;
    y = x + y;
    x = t;
}
```

**This mutex-based approach is atomic, but obviously not lock-free. That **[may very well be good enough](http://preshing.com/20111118/locks-arent-slow-lock-contention-is), but for the sake of illustration, let’s go ahead and convert it to a CAS loop just like the other examples. `std::atomic<>` is a template, so we can actually pack both shared variables into a `struct` and apply the same pattern as before:

```
struct Terms
{
    uint32_t x;
    uint32_t y;
};

std::atomic<Terms> terms;

void atomicFibonacciStep()
{
    Terms oldTerms = terms.load();
    Terms newTerms;
    do
    {
        newTerms.x = oldTerms.y;
        newTerms.y = oldTerms.x + oldTerms.y;
    }
    while (!terms.compare_exchange_weak(oldTerms, newTerms));
}
```

**Is ***this* operation lock-free? Now we’re venturing into dicey territory. As I wrote at the start, C++11 atomic operations are designed take advantage of lock-free CPU instructions “whenever possible” – admittedly a loose definition. In this case, we’ve wrapped `std::atomic<>` around a struct, `Terms`. Let’s see how GCC 4.9.2 compiles it for x64:

![img](https://preshing.com/images/atomic-terms-rmw.png)

**We got lucky. The compiler was clever enough to see that **`Terms` fits inside a single 64-bit register, and implemented `compare_exchange_weak` using `lock cmpxchg`. The compiled code is lock-free.

**This brings up an interesting point: In general, the C++11 standard does ***not* guarantee that atomic operations will be lock-free. There are simply too many CPU architectures to support and too many ways to specialize the `std::atomic<>` template. You need to [check with your compiler](http://en.cppreference.com/w/cpp/atomic/atomic/is_lock_free) to make absolutely sure. In practice, though, it’s pretty safe to assume that atomic operations are lock-free when all of the following conditions are true:

1. **The compiler is a recent version MSVC, GCC or Clang.**
2. **The target processor is x86, x64 or ARMv7 (and possibly others).**
3. **The atomic type is **`std::atomic<uint32_t>`, `std::atomic<uint64_t>` or `std::atomic<T*>` for some type `T`.

**As a personal preference, I like to hang my hat on that third point, and limit myself to specializations of the **`std::atomic<>` template that use explicit integer or pointer types. The [safe bitfield technique](http://preshing.com/20150324/safe-bitfields-in-cpp) I described in the previous post gives us a convenient way to rewrite the above function using an explicit integer specialization, `std::atomic<uint64_t>`:

```
BEGIN_BITFIELD_TYPE(Terms, uint64_t)
    ADD_BITFIELD_MEMBER(x, 0, 32)
    ADD_BITFIELD_MEMBER(y, 32, 32)
END_BITFIELD_TYPE()

std::atomic<uint64_t> terms;

void atomicFibonacciStep()
{
    Terms oldTerms = terms.load();
    Terms newTerms;
    do
    {
        newTerms.x = oldTerms.y;
        newTerms.y = (uint32_t) (oldTerms.x + oldTerms.y);
    }
    while (!terms.compare_exchange_weak(oldTerms, newTerms));
}
```

**Some real-world examples where we pack several values into an atomic bitfield include:**

* **Implementing tagged pointers as a **[workaround for the ABA problem](http://en.wikipedia.org/wiki/ABA_problem#Tagged_state_reference).
* **Implementing a lightweight read-write lock, which I touched upon briefly **[in a previous post](http://preshing.com/20150316/semaphores-are-surprisingly-versatile).

**In general, any time you have a small amount of data protected by a mutex, and you can pack that data entirely into a 32- or 64-bit integer type, you can always convert your mutex-based operations into lock-free RMW operations, no matter what those operations actually do! That’s the principle I exploited in my **[Semaphores are Surprisingly Versatile](http://preshing.com/20150316/semaphores-are-surprisingly-versatile) post, to implement a bunch of lightweight synchronization primitives.

**Of course, this technique is not unique to the C++11 atomic library. I’m just using C++11 atomics because they’re quite widely available now, and compiler support is pretty good. You can implement a custom RMW operation using any library that exposes a compare-and-swap function, such as **[Win32](https://msdn.microsoft.com/en-us/library/ttk2z1ws.aspx), the [Mach kernel API](https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man3/OSAtomicCompareAndSwap32.3.html), the [Linux kernel API](http://lxr.free-electrons.com/ident?i=atomic_cmpxchg), [GCC atomic builtins](https://gcc.gnu.org/onlinedocs/gcc-4.9.2/gcc/_005f_005fatomic-Builtins.html) or [Mintomic](http://mintomic.github.io/lock-free/atomics/). In the interest of brevity, I didn’t discuss memory ordering concerns in this post, but it’s critical to consider the guarantees made by your atomic library. In particular, if your custom RMW operation is intended to pass non-atomic information between threads, then at a minimum, you should ensure that there is the equivalent of a [*synchronizes-with*](http://preshing.com/20130823/the-synchronizes-with-relation) relationship somewhere.

## 数据结构设计

**子网络数量：subnet_num**

**进程数（MPI节点数）：proc_num**

### 1. CrossSpike

`_recv_offset`

`_send_offset`

**大小为子网络数量加一：subnet_num + 1**

**每一个元素的含义是**

`_recv_start`

`_send_start`

`_recv_num`

`_send_num`

### 2. ProcBuf

`_rdata_size`

`_sdata_size`

`_recv_offset`

`_send_offset`

`_data_offset`

### 3. Simulator中属性

#### 3.1. Network* _network

1. **_node_num为总的子网络数subnet_num**
2. **_neurons：std::map<Type, Neuron *>。第一维存储神经元类型，第二维存储每个类型的神经元数据数组.**
3. **_populations：std::vector<Population *>。存储所有的population，即神经元群。每次createPopulation都会将一个指向num个神经元的指针加入进去。所以**_**populations的大小即为实际代码中加入的population的数量**
4. `map<Type, vector<map<unsigned int, vector<ID>>>> _conn_n2s;`

   * `_conn_n2s[src.type()][src.id()][delay].push_back(syn);`
     ** 根据Type获得'神经元到突触的连接'。通过源神经元和delay获得对应的突触前提：已经获得了源神经元的类型、源神经元的id、突触延迟->得到这个源神经元的所有突触**
   * ```
     _conn_n2s[src.type()][src.id()][delay].push_back(syn);
     _conn_s2n[syn.type()][syn.id()] = dst;
     ```
5. `map<Type, vector<ID>> _conn_s2n`

   ```
   _conn_s2n[syn.type()][syn.id()] = dst;
   根据Type获取对应的突触连接ID的向量
   通过突触类型得到突触连接的目的节点
   前提：已经获得了突触的类型和突触的id的信息->得到突触连接的目的神经元
   ```
6. `_idx2node`:
   `std::map<Type, std::vector<int>>`
   **存储每个类型中的所有神经元应该被放在哪个节点中**
   `_idx2node[IAF][12]=1`：IAF类型的第12个神经元存放在node 1中
   **同理，synapse也是相同的存储方式**
7. **n2s_rev：存储的是神经元类型为T的第i个神经元，以它为目标节点的输入突触的id的集合**
   `std::map<Type, std::vector<std::vector<ID>>>`
   **第一维：神经元类型**
   **第二维：该类型的神经元个数**
   **第三维：以第二维指定的神经元为目的节点的所有突触ID的集合**

   ```
   for (auto ti = _conn_s2n.begin(); ti != _conn_s2n.end(); ti++) {
       for (size_t idx = 0; idx < ti->second.size(); idx++) {
           ID &t = ti->second[idx];  // 突触连接的目标神经元ID
           n2s_rev[t.type()][t.id()].push_back(ID(ti->first, idx));
       }
   }
   ```
8. ```
   _neuron_nums[subnet_num][type_num]：每个subnet中类型为T的神经元数量的总和 std::map<int, std::map<Type, size_t>>

   _synapse_nums[subnet_num][type_num]: 同上

   _crossnodeNeuronsSend:大小为subnet_num。如果当前神经元id所在的subnet和以id为目的的突触的源神经元所在subnet不同，那么就会放入这个set中。这是因为跨subnet的突触是存放在源神经元所在的subnet中的。本质是统计当前子网络有哪些神经元它的。本质是存储了当前子网络中需要发送给其它子网络的神经元的id的集合。这个id是从0开始编号的。

   _crossnodeNeuronsRecv:大小为subnet_num。存放突触的起始神经元的id，这些神经元通过某个突触会连接到非源神经元所在的subnet中。即这些神经元的发放需要发送到其它子网络中，因为它有连向其它自网络的边。代表当前节点需要接收的来自其它子网络的神经元id编号的集合。

   _buffer_offsets:
   ```
9. **构建网络时使用的临时变量**

   ```
   类型说明：
   typedef std::map<int, std::map<Type, size_t>> CrossTypeInfo_t
   typedef std::map<int, size_t>

   临时变量：
   CrossTypeInfo_t type_offset;
   type_offset[subnet_id][neuron_type]的偏移量，大小为这个节点上的神经元类型

   CrossTypeInfo_t neuron_offset;
   type_offset[subnet_id][neuron_type]的偏移量（每个差值为对应类型的神经元数量），大小为这个节点上的神经元类型

   CrossTypeInfo_t synapse_offset;
   同上


   CrossTypeInfo_t neuron_count;
   CrossTypeInfo_t synapse_count;
   这俩一开始都是初始化为0，因为有shadow neuron的加入所以后续才会计算它的大小

   CrossTypeInfo_t n2s_count;

   CrossInfo_t cross_idx; // idx for cross-node neuron
   CrossInfo_t node_n_offset; // neuron offset for each node
   ```

   **这里主要是调用4个arange函数：**

   * **arrangeNet**
   * **arrangeNeuron**
   * **arrangeLocal**
   * **arrangeCross**
10. 

#### 3.2. DistriNetwork* _all_nets：大小为subnet_num的数组

```
_all_nets[i]._simCycle = sim_cycle;

_all_nets[i]._nodeIdx = i;

_all_nets[i]._nodeNum = part_num;  // 大小为subnet_num的数组

_all_nets[i]._dt = _dt;
```

**DistriNetwork **_network_data：第一维有subnet_num大小，第二维分别指向一个子网络即**_**network_data[i]为第i个子网络**

**只有process_id==0的进程才有全部网络的信息**

**CrossNodeData **_data：第一维有subnet_num大小，第二维分别指向一个子网络即**_**network_data[i]为第i个子网络的数据**

### 4. GNetwork

```
ret->pNTypes = (Type *)malloc(sizeof(Type)*nTypeNum);
pNTypes: 大小为神经元类型总数

ret->pSTypes = (Type *)malloc(sizeof(Type)*sTypeNum);
pSTypes：大小为突触类型总数

ret->pNeuronNums = (size_t*)malloc(sizeof(size_t)*(nTypeNum + 1));
pNeuronNums：每种类型的神经元的数量+1（+1的原因是最后一位存储的是所有类型神经元的总数）

ret->pSynapseNums = (size_t*)malloc(sizeof(size_t)*(sTypeNum + 1));
pSynapseNums：每种类型的突触的数量+1

bufferOffsets[type_num][neuron_num + 1]：buffer数组的offset，大小为神经元类型+1。含义是这种类型的神经元它的buffer在总buffer中的偏移量

ppNeurons**：二维数组。第一维为神经元类型数，第二维为神经元的数据类型，如IafData
ppSynapses**：二维数组。第一维为突触类型数

ppConnections（Connection**）：二维数组。第一维为突触类型数


ret->pNeuronNums[0] = 0;
ret->pSynapseNums[0] = 0;



```

### 5. CrossNodeMap

```

```
