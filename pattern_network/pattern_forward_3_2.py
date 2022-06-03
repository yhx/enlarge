import nest
import sys
import matplotlib.pyplot as plt
import time
import numpy as np


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

    for i in range(depth):
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
        # nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m", "weighted_spikes_ex", "I_syn_ex", "weighted_spikes_in"]})
        nest.SetStatus(multimeter, {"record_from":["V_m", "I_syn_ex"]})
        nest.Connect(multimeter, neurons[3])
        multimeters = []
        for i in range(depth):
            multimeters.append(nest.Create("multimeter"))
            # nest.SetStatus(multimeters[i], {"withtime":True, "record_from":["V_m", "weighted_spikes_ex", "I_syn_ex", "weighted_spikes_in"]})
            nest.SetStatus(multimeters[i], {"record_from":["V_m", "I_syn_ex"]})
            nest.Connect(multimeters[i], neurons[i+1])

    scale = 1e3
    w1_2 = 2.4 * scale
    w2_3 = 2.4 * scale
    w3_4 = 1.5 * scale
    w4_5 = -0.1 * scale
    w1_4 = 0.85 * scale
    delay_scale = 3
    
    for i in range(1, depth + 1, 4):
        if i + 1 <= depth:
            nest.Connect(neurons[i], neurons[i + 1], syn_spec={"weight": w1_2 / n, "delay":delay_scale*dt})
        
        if i + 2 <= depth:
            nest.Connect(neurons[i + 1], neurons[i + 2], syn_spec={"weight": w2_3 / n, "delay":delay_scale*dt})

        if i + 3 <= depth:
            nest.Connect(neurons[i + 2], neurons[i + 3], syn_spec={"weight": w3_4 / n, "delay":delay_scale*dt})
            nest.Connect(neurons[1], neurons[i + 3], syn_spec={"weight": w1_4 / n, "delay":delay_scale*dt})

        if i + 4 <= depth:
            nest.Connect(neurons[i + 3], neurons[i + 4], syn_spec={"weight": w4_5 / n, "delay":delay_scale*dt}) 

    if LOG:
        return vm, sds, multimeter, multimeters, neurons

if __name__ == '__main__':
    depth = int(sys.argv[1])
    n = int(sys.argv[2])
    LOG = int(sys.argv[3])
    dt = 0.1
    LOG_DETAIL = False

    if LOG:
        vm, sds, multimeter, multimeters, neurons = build_network(dt, n, depth)
    else:
        build_network(dt, n, depth)

    t1 = time.time()
    nest.Simulate(10000.0)
    t2 = time.time()

    print('Rank {0:d} total time: {1:.2f} seconds'.format(nest.Rank(), t2 - t1))

    if LOG:
        total_spike = 0 #np.zeors(())
        with open('./tmp/rate_nest_{0}.log'.format(nest.Rank()), "w") as f:
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
        
        if LOG_DETAIL:
            with open('./tmp/total_rate_nest_{0}.log'.format(nest.Rank()), 'w+') as f:
                # f.write('TOTAL SPIKE NUMBER: {0}\n'.format(total_spike))
                f.write(str(total_spike))

            np.savetxt('./tmp/volt_{0}.log'.format(nest.Rank()), V_m)

            V_ms = []
            for i in range(depth):
                dmm = nest.GetStatus(multimeters[i])[0]
                V_m = dmm["events"]["V_m"]
                V_ms.append(V_m)
            V_ms = np.array(V_ms).T
            print(V_ms.shape)
            np.savetxt('./tmp/volt.log', V_ms)

            I_syn_exs = []
            for i in range(depth):
                dmm = nest.GetStatus(multimeters[i])[0]
                I_syn_ex = dmm["events"]["I_syn_ex"]
                I_syn_exs.append(I_syn_ex)
            I_syn_exs = np.array(I_syn_exs).T
            print(I_syn_exs.shape)
            np.savetxt('./tmp/I_syn_ex.log', I_syn_exs, fmt='%.8f')

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
