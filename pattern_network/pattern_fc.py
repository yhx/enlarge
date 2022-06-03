import nest
import sys
import matplotlib.pyplot as plt
import time


def build_network(dt, n, pop_num):
    # nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": dt,
        'total_num_virtual_procs': 448
    })

    neurons = []
    sds = []

    if LOG:
        vm = nest.Create('voltmeter')
        nest.SetStatus(vm, "withtime", True)

    for _ in range(pop_num):  # 只会创建3个神经元
        neuron = nest.Create('iaf_psc_exp', n)
        nest.SetStatus(neuron, "I_e", 376.0)

        if LOG:
            sd = nest.Create('spike_detector')
            sds.append(sd)

            nest.Connect(vm, neuron)
            nest.Connect(neuron, sd)

        neurons.append(neuron)
    
    if LOG:
        multimeter = nest.Create("multimeter")
        nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m", "weighted_spikes_ex", "I_syn_ex", "weighted_spikes_in"]})
        nest.Connect(multimeter, neurons[2])

    w = 3.5 * 1e3 / n / (pop_num - 1)
    delay_scale = 2

    for i in range(pop_num):
        for j in range(pop_num):
            if i != j:
                nest.Connect(neurons[i], neurons[j], syn_spec={"weight": w, "delay":delay_scale*dt})

    if LOG:
        return vm, sds, multimeter, neurons

if __name__ == '__main__':
    n = int(sys.argv[1])  # 每个population中神经元的数量
    pop_num = int(sys.argv[2])  # population数量
    LOG = int(sys.argv[3])
    dt = 0.1

    if LOG:
        vm, sds, multimeter, neurons = build_network(dt, n, pop_num)
    else:
        build_network(dt, n, pop_num)

    t1 = time.time()
    nest.Simulate(10000.0)
    t2 = time.time()

    print('Rank {0:d} total time: {1:.2f} seconds'.format(nest.Rank(), t2 - t1))

    if LOG:
        total_spike = 0
        with open('./tmp/rate_nest_{0}.log'.format(nest.Rank()), 'w') as f:
            for i in range(pop_num):
                f.write("{0}'s Number of spikes: {1}\n".format(i + 1, nest.GetStatus(sds[i], "n_events")[0]))
                total_spike += nest.GetStatus(sds[i], "n_events")[0]
                print(nest.GetStatus(sds[i], "events")[0]["senders"])
        # print("{0}'s Number of spikes: {1}".format(1, nest.GetStatus(sd, "n_events")[0]))
        dmm = nest.GetStatus(multimeter)[0]
        I_syn_ex = dmm["events"]["I_syn_ex"]
        weighted_spikes_ex = dmm["events"]["weighted_spikes_ex"]
        # I_syn_ex = dmm["events"]["I_syn_ex"]
        weighted_spikes_in = dmm["events"]["weighted_spikes_in"]
        V_m = dmm["events"]["V_m"]
        ts = dmm["events"]["times"]

        with open('./tmp/total_rate_nest_{0}.log'.format(nest.Rank()), 'w+') as f:
            # f.write('TOTAL SPIKE NUMBER: {0}\n'.format(total_spike))
            f.write(str(total_spike))
        
        with open('./tmp/spike_nest_{0}.log'.format(nest.Rank()), 'w') as f:
            for i in range(pop_num):
                print(nest.GetStatus(sds[i], "events")[0]["senders"])
                spikes = nest.GetStatus(sds[i], "events")[0]["senders"]
                for j in range(len(spikes)):
                    f.write(str(spikes[j]) + " ")
        
        if nest.Rank() == 0:
            with open('./tmp/neuron_gid.log', 'w') as f:
                for neuron in neurons:
                    # print("start: ", nest.GetStatus(neuron)[0])
                    # print("end: ", nest.GetStatus(neuron)[-1])
                    print(neuron)
                    for i in range(len(neuron)):
                        f.write(str(neuron[i]) + " ")
        