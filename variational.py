#!/usr/bin/env python

import logging
import numpy as np
import theano
from argparse import ArgumentParser

from theano import tensor

from blocks.bricks import MLP, Tanh, Identity
from blocks.algorithms import GradientDescent, Adam
from blocks.initialization import IsotropicGaussian
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop

from blocks_contrib.bricks.filtering import VariationalSparseFilter
from blocks_contrib.extensions import DataStreamMonitoringAndSaving
from blocks_contrib.utils import _add_noise
floatX = theano.config.floatX


mnist = MNIST('train', sources=['features'])
data, _ = mnist._load_mnist()
means = data.mean(axis=0)


def _add_enumerator(n_steps, batch_size):
    def func(data):
        enum = np.zeros((n_steps, batch_size)).astype(floatX)
        return (enum,)
    return func


def _meanize(data):
    newfirst = data[0] - means[None, :]
    return (newfirst, data[1])


def main(save_to, num_epochs):
    rfunc = np.random.laplace
    dim = 500
    n_steps = 20
    batch_size = 100
    mlp = MLP([Identity()], [dim, 784], use_bias=False)
    filtering = VariationalSparseFilter(mlp=mlp, dim=dim, input_dim=784, batch_size=batch_size, n_steps=n_steps,
                                        weights_init=IsotropicGaussian(.01))
    filtering.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    e = tensor.tensor3('noise')

    z = filtering.apply(noise=e, inputs=x,
                        gamma=.1)[1][-1]
    prior = theano.gradient.disconnected_grad(z)
    cost, codes, recs = filtering.cost(inputs=x, noise=e, prior=prior)
    cost += 0*y.sum()

    cg = ComputationGraph([cost])
    cost.name = 'final_cost'

    mnist_train = MNIST("train")
    mnist_test = MNIST("test")
    trainstream = Mapping(DataStream(mnist_train,
                          iteration_scheme=SequentialScheme(mnist_train.num_examples, 100)),
                          _meanize)
    teststream = Mapping(DataStream(mnist_test,
                                    iteration_scheme=SequentialScheme(mnist_test.num_examples,
                                                                      100)),
                         _meanize)
    trainstream = Mapping(trainstream, _add_noise(n_steps, batch_size, dim, rfunc), add_sources=('noise', ))
    teststream = Mapping(teststream, _add_noise(n_steps, batch_size, dim, rfunc), add_sources=('noise', ))

    algorithm = GradientDescent(
        cost=cost, params=cg.parameters,
        step_rule=Adam())
    main_loop = MainLoop(
        algorithm,
        trainstream,
        model=Model(cost),
        extensions=[Timing(),
                    FinishAfter(after_n_epochs=num_epochs),
                    DataStreamMonitoring(
                        [cost],
                        teststream,
                        prefix="test"),
                    DataStreamMonitoringAndSaving(
                    [cost],
                    teststream,
                    [filtering],
                    'best_'+save_to+'.pkl',
                    cost_name=cost.name,
                    after_epoch=True,
                    prefix='valid'
                    ),
                    TrainingDataMonitoring(
                        [cost,
                         aggregation.mean(algorithm.total_gradient_norm)],
                        prefix="train",
                        after_epoch=True),
                    Plot(
                        save_to,
                        channels=[
                            ['test_final_cost',
                             'test_misclassificationrate_apply_error_rate'],
                            ['train_total_gradient_norm']]),
                    Printing()])
    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)
