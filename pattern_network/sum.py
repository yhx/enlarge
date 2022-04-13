import sys

if __name__ == '__main__':
    n = int(sys.argv[1])
    total_spikes = 0
    for i in range(n):
        with open('./tmp/total_rate_nest_{0}.log'.format(i), 'r') as f:
            # f.write('TOTAL SPIKE NUMBER: {0}\n'.format(total_spike))
            num = f.read()
            total_spikes += int(num)
    print('TOTAL SPIKE NUMBER: {0}'.format(total_spikes))
    