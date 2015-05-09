#!/usr/bin/env python

import logging
import numpy as np
import theano

from argparse import ArgumentParser
from theano import tensor
from skimage.transform import rotate

from blocks.algorithms import GradientDescent, Adam
from blocks.initialization import IsotropicGaussian, Constant
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

from blocks_contrib.bricks.filtering import TemporalSparseFilter, SparseFilter
from blocks_contrib.extensions import DataStreamMonitoringAndSaving
floatX = theano.config.floatX


mnist = MNIST('train', sources=['features'])
data, _ = mnist._load_mnist()
means = data.mean(axis=0)


def _add_enumerator(n_steps, batch_size):
    def func(data):
        enum = np.zeros((n_steps, batch_size)).astype(floatX)
        return (enum,)
    return func


def allrotations(image, N):
    angles = np.linspace(0, 350, N)
    R = np.zeros((N, 784))
    for i in xrange(N):
        img = rotate(image, angles[i])
        R[i] = img.flatten()
    return R


def _meanize(n_steps):
    def func(data):
        newfirst = data[0] - means[None, :]
        Rval = np.zeros((n_steps, newfirst.shape[0], newfirst.shape[1]))
        for i, sample in enumerate(newfirst):
            Rval[:, i, :] = allrotations(sample.reshape((28, 28)), n_steps)
        # Rval = newfirst[np.newaxis].repeat(n_steps, axis=0)
        Rval = Rval.astype(floatX)
        return (Rval, data[1])
    return func


def main(save_to, num_epochs):
    dim = 500
    n_steps = 20
    proto = SparseFilter(dim=dim, input_dim=784, batch_size=100, n_steps=n_steps)
    filtering = TemporalSparseFilter(proto=proto, dim=dim, n_steps=n_steps, batch_size=100,
                                     weights_init=IsotropicGaussian(.01))
    filtering.initialize()
    x = tensor.tensor3('features')
    y = tensor.lmatrix('targets')

    cost, z, x_hat = filtering.cost(inputs=x, gamma=.1)
    cost += 0*y.sum()

    cg = ComputationGraph([cost])
    cost.name = 'final_cost'

    mnist_train = MNIST("train")
    mnist_test = MNIST("test")
    trainstream = Mapping(DataStream(mnist_train,
                          iteration_scheme=SequentialScheme(mnist_train.num_examples, 100)),
                          _meanize(n_steps))
    teststream = Mapping(DataStream(mnist_test,
                                    iteration_scheme=SequentialScheme(mnist_test.num_examples,
                                                                      100)),
                         _meanize(n_steps))

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
