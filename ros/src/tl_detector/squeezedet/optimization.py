import tensorflow as tf

import os
import hyperopt
from hyperopt import hp, tpe, fmin, Trials
import numpy as np
import math
import traceback
import tempfile

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

def optimize(max_evals=30):

    def objective(options):
        try:
            options['working_directory'] = tempfile.mkstemp()[1]
            return_value = {'loss': training(**options),
                            'status': hyperopt.STATUS_OK,
                            'options': options}
            print(return_value)
            tf.reset_default_graph()
            return return_value
        except Exception as err:
            traceback.print_exc()
            tf.reset_default_graph()
            return {'loss': np.inf,
                    'status': hyperopt.STATUS_FAIL,
                    'options': options}

    space = {'learning_rate': hp.loguniform('learning_rate', math.log(1e-6),
                                            math.log(2e-2)),
             'epochs': hp.choice('epochs', [10, 15, 20]),
             'top_N_detections': hp.quniform('top_N_detections', 40, 64, 1),
             'prob_thresh': hp.loguniform('prob_thresh', math.log(1e-6),
                                                         math.log(2e-2))}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials)

    return (best, trials)



def training(learning_rate, epochs, top_N_detections,
             prob_thresh, working_directory):

    """
    Caculate the loss
    """
    loss = 0
    return loss


if __name__ == "__main__":
    best, trials = optimize(max_evals=3)
    print("best is {}".format(best))
    print("trials are {}".format(trials))


