"""
This is a starting program implementing command-line features for training and testing models.
"""
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import fire
from utils.problem import Problem
from utils import registry
import tensorflow as tf
import hyperparams
import problems


tf.random.set_seed(666666)
np.random.seed(666666)

def list_registries(name='datasets'):
    acceptable_inputs = ['datasets', 'problems', 'hparams']
    if name not in acceptable_inputs:
        print('The acceptable input could be %s' % str(acceptable_inputs))

    prompt = None
    if name == 'datasets':
        prompt = ",".join(registry.datasets.keys())
    elif name == 'problems':
        prompt = ",".join(registry.problems.keys())
    elif name == 'hparams':
        prompt = ",".join(registry.hparams.keys())

    print(prompt)


def train(dataset: str = 'PDE',
          problem: str = 'GAN',
          hparam: str = 'basic_params1',
          gpu_id: int = 0
          ):
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        gpu = gpus[gpu_id]
        tf.config.experimental.set_visible_devices([gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpu, True)
    '''

    '''
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        devices = physical_devices[2:]
        tf.config.set_visible_devices(devices, 'GPU')
        tf.config.experimental.set_memory_growth(devices, True)
        logical_devices = tf.config.list_logical_devices('GPU')
        assert len(logical_devices) == len(physical_devices) - 1
    except:
        pass      
    '''
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.set_visible_devices(physical_devices[2:], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        assert len(logical_devices) == len(physical_devices) - 1
    except:
        pass



    hparam_fn = registry.get_hparam(dataset)
    problem_cls = registry.get_problem(problem)

    assert issubclass(problem_cls, Problem)

    this_hparam = hparam_fn(hparam)
    this_problem = problem_cls(this_hparam)



    this_problem.load_data()

    this_problem.train_model()



if __name__ == '__main__':
    fire.Fire({
        'list': list_registries,
        'train': train
    })
