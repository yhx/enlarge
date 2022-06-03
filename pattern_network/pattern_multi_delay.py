import nest
import sys
import matplotlib.pyplot as plt
import time
import numpy as np

LOG = 0

def build_network(dt, n, depth):
    # nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": dt,
        # 'total_num_virtual_procs': 448
    })

    neurons = []
    sds = []
    # neurons.append('')

    if LOG:
        # vm = nest.Create('voltmeter')
        vms = []
        for i in range(depth):
            vm = nest.Create('voltmeter')
            vms.append(vm)
        nest.SetStatus(vm, "withtime", True)

    for i in range(depth):  # 只会创建3个神经元
        neuron = nest.Create('iaf_psc_exp', n)
        
        if i == 0:
            nest.SetStatus(neuron, "I_e", 37600.0)
        else:
            nest.SetStatus(neuron, "I_e", 376.0)

        if LOG:
            sd = nest.Create('spike_detector')
            nest.Connect(vms[i], neuron)
            nest.Connect(neuron, sd)
            sds.append(sd)

        neurons.append(neuron)
        # print(nest.GetStatus(neuron))
    
    if LOG:
        multimeters = []
        for i in range(depth):
            multimeter = nest.Create("multimeter")
            nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m", "weighted_spikes_ex", "I_syn_ex", "weighted_spikes_in"]})
            # print(nest.GetStatus(multimeter))
            nest.Connect(multimeter, neurons[i])
            multimeters.append(multimeter)

    scale = 1e3
    w1_2 = 2.4 * scale
    w2_3 = 2.9 * scale
    delay_scale = 1
    
    # delays = []
    # for i in range(n * n):
    #     delays.append(dt * (i % 10) * delay_scale)
    
    delays = []
    for i in range(n):
        bri = []
        for j in range(n):
            bri.append(dt * ((j * n + i) % 10 + 2) * delay_scale)
        delays.append(bri)
    
    nest.Connect(neurons[0], neurons[1], syn_spec={"weight": w1_2 / n, "delay": delays})
    nest.Connect(neurons[1], neurons[2], syn_spec={"weight": w2_3 / n, "delay": delays})
    print("############CONNECTION#############")
    conns = nest.GetConnections(neurons[0], neurons[1])
    print(nest.GetStatus(conns))
    conns = nest.GetConnections(neurons[1], neurons[2])
    print(nest.GetStatus(conns))
    print("###################################")
    if LOG:
        return vms, sds, multimeters, neurons

if __name__ == '__main__':
    depth = 3  # 只有三个神经元组成的前向网络
    n = int(sys.argv[1])
    LOG = int(sys.argv[2])
    dt = 1.

    if LOG:
        vms, sds, multimeters, neurons = build_network(dt, n, depth)
    else:
        build_network(dt, n, depth)

    t1 = time.time()
    nest.Simulate(1000.0)
    t2 = time.time()

    print('Rank {0:d} total time: {1:.2f} seconds'.format(nest.Rank(), t2 - t1))

    if LOG:
        total_spike = 0
        with open('./tmp/rate.nest.{0}.log'.format(nest.Rank()), 'w+') as f:
            for i in range(depth):
                f.write("{0}'s Number of spikes: {1}\n".format(i + 1, nest.GetStatus(sds[i], "n_events")[0]))
                total_spike += nest.GetStatus(sds[i], "n_events")[0]
        # print("{0}'s Number of spikes: {1}".format(1, nest.GetStatus(sd, "n_events")[0]))
        # dmm = nest.GetStatus(multimeters[0])[0]
        # print("dmm ", dmm)
        # I_syn_ex = dmm["events"]["I_syn_ex"]
        # weighted_spikes_ex = dmm["events"]["weighted_spikes_ex"]
        # # I_syn_ex = dmm["events"]["I_syn_ex"]
        # weighted_spikes_in = dmm["events"]["weighted_spikes_in"]
        # V_m = dmm["events"]["V_m"]
        # ts = dmm["events"]["times"]

        # print('spike detector:')
        # print(nest.GetStatus(sds[0]))

        # print('ts的大小', ts.shape)

        # print(weighted_spikes_ex)

        spikes_time = []
        for i in range(depth):
            sd_status = nest.GetStatus(sds[i])[0]
            senders = sd_status['events']['senders']
            times = sd_status['events']['times']
            for j in range(n):
                bri = []
                for k in range(len(senders)):
                    # print(senders[k] )
                    if nest.GetStatus(neurons[i])[j]['local_id'] == senders[k]:
                        # print(times[k])
                        bri.append(times[k])
                spikes_time.append(bri)
        
        # for i in range(len(neurons)):
        #     print(nest.GetStatus(neurons[i]))

        # 输出spike时间
        with open('./tmp/spike_time.nest.{0}.log'.format(nest.Rank()), 'w') as f:
            f.write('')
        with open('./tmp/spike_time.nest.{0}.log'.format(nest.Rank()), 'w+') as f:
            for i in range(len(spikes_time)):
                for j in range(len(spikes_time[i])):
                    f.write(str(int(spikes_time[i][j])) + ' ')
                f.write('\n')
        
        # with open('./tmp/weighted_spikes_ex.nest.{0}.log'.format(nest.Rank()), 'w+') as f:
        weighted_spikes_exs = None
        for i in range(depth):
            for j in range(n):
                # print('################')
                # print(nest.GetStatus(multimeters[i]))
                # print('################')
                dmm = nest.GetStatus(multimeters[i])[0]
                # print('切片后:', weighted_spikes_ex.shape)
                weighted_spikes_ex = dmm["events"]["weighted_spikes_ex"][j:dmm["events"]["weighted_spikes_ex"].shape[0]:n]
                # print('切片后:', weighted_spikes_ex.shape)
                weighted_spikes_ex = weighted_spikes_ex.reshape((-1, 1))
                if weighted_spikes_exs is None:
                    print(weighted_spikes_ex.shape)
                    weighted_spikes_exs = weighted_spikes_ex
                else:
                    weighted_spikes_exs = np.concatenate((weighted_spikes_exs, weighted_spikes_ex), axis=1)
        print(weighted_spikes_exs.shape)
        np.savetxt('./tmp/weighted_spikes_ex.nest.{0}.log'.format(nest.Rank()), weighted_spikes_exs, fmt='%d')

        with open('./tmp/total_rate.nest.{0}.log'.format(nest.Rank()), 'w+') as f:
            # f.write('TOTAL SPIKE NUMBER: {0}\n'.format(total_spike))
            f.write(str(total_spike))

