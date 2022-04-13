import nest
import sys
import matplotlib.pyplot as plt
import time


def build_network(dt, n, depth):
    # nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": dt,
        'total_num_virtual_procs': 448
    })

    neurons = []
    sds = []
    neurons.append('')

    if LOG:
        vm = nest.Create('voltmeter')
        # nest.SetStatus(vm, "withtime", True)

    for _ in range(depth):  # 只会创建3个神经元
        neuron = nest.Create('iaf_psc_exp', n)
        nest.SetStatus(neuron, "I_e", 376.0)

        if LOG:
            sd = nest.Create('spike_recorder')
            sds.append(sd)

            nest.Connect(vm, neuron)
            nest.Connect(neuron, sd)

        neurons.append(neuron)
    
    if LOG:
        multimeter = nest.Create("multimeter")
        nest.SetStatus(multimeter, {"record_from":["V_m", "I_syn_ex"]})
        nest.Connect(multimeter, neurons[2])

    scale = 1e3
    w1_2 = 2.4 * scale
    w2_3 = 2.9 * scale
    w3_1 = -1.0 * scale
    delay_scale = 3
    
    nest.Connect(neurons[1], neurons[2], syn_spec={"weight": w1_2 / n, "delay":delay_scale*dt})
    nest.Connect(neurons[2], neurons[3], syn_spec={"weight": w2_3 / n, "delay":delay_scale*dt})
    nest.Connect(neurons[3], neurons[1], syn_spec={"weight": w3_1 / n, "delay":delay_scale*dt})

    if LOG:
        return vm, sds, multimeter, neurons

if __name__ == '__main__':
    depth = 3  # 只有三个神经元连接成为的环状网络
    n = int(sys.argv[1])
    LOG = int(sys.argv[2])
    dt = 0.1

    if LOG:
        vm, sds, multimeter, neurons = build_network(dt, n, depth)
    else:
        build_network(dt, n, depth)

    t1 = time.time()
    nest.Simulate(10000.0)
    t2 = time.time()

    print('Rank {0:d} total time: {1:.2f} seconds'.format(nest.Rank(), t2 - t1))

    if LOG:
        total_spike = 0
        with open('./tmp/rate_nest_{0}.log'.format(nest.Rank()), 'w+') as f:
            for i in range(depth):
                f.write("{0}'s Number of spikes: {1}\n".format(i + 1, nest.GetStatus(sds[i], "n_events")[0]))
                total_spike += nest.GetStatus(sds[i], "n_events")[0]
        # print("{0}'s Number of spikes: {1}".format(1, nest.GetStatus(sd, "n_events")[0]))
        dmm = nest.GetStatus(multimeter)[0]
        I_syn_ex = dmm["events"]["I_syn_ex"]
        # I_syn_ex = dmm["events"]["I_syn_ex"]
        V_m = dmm["events"]["V_m"]
        ts = dmm["events"]["times"]

        with open('./tmp/total_rate_nest_{0}.log'.format(nest.Rank()), 'w+') as f:
            # f.write('TOTAL SPIKE NUMBER: {0}\n'.format(total_spike))
            f.write(str(total_spike))

        with open('./tmp/spike_nest_{0}.log'.format(nest.Rank()), 'w') as f:
            for i in range(depth):
                print(nest.GetStatus(sds[i], "events")[0]["senders"])
                spikes = nest.GetStatus(sds[i], "events")[0]["senders"]
                for j in range(len(spikes)):
                    f.write(str(spikes[j]) + " ")
        
        if nest.Rank() == 0:
            with open('./tmp/neuron_gid.log', 'w') as f:
                for neuron in neurons[1:]:
                    neuron_states = nest.GetStatus(neuron)
                    for neuron_state in neuron_states:
                        f.write(str(neuron_state['global_id']) + " ")
