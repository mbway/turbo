#!/usr/bin/env python3
import sys
import time
import optimisation as op

def client(host, port):
    class MyEvaluator(op.Evaluator):
        def test_config(self, config):
            time.sleep(4)
            return config.a

    evaluator = MyEvaluator()
    evaluator.noisy = True
    evaluator.run_client(host, port)

def server(host, port):
    ranges = {
        'a' : [1,2,3], # linear
        'b' : [5], # constant
        'c' : [1,2,3] # linear
    }
    #optimiser = op.BayesianOptimisationOptimiser(ranges)
    optimiser = op.GridSearchOptimiser(ranges)
    optimiser.noisy = True # print log output
    optimiser.run_server(host, port, max_jobs=20)

def get_arg(i, default=None):
    if len(sys.argv) > i:
        return sys.argv[i]
    else:
        return default

if __name__ == '__main__':
    mode = get_arg(1)
    host = get_arg(2, '0.0.0.0.')
    port = get_arg(3, op.PORT)

    if mode == 'client':
        print('client connecting to {}:{}'.format(host, port))
        client(host, port)

    elif mode == 'server':
        print('server listening at {}:{}'.format(host, port))
        server(host, port)

    else:
        print('invalid mode: {}'.format(mode))
