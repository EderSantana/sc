#!/usr/bin/env python

import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import MLP, Identity, Sigmoid, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop

from blocks_contrib.bricks.filtering import SparseFilter, VarianceComponent
from blocks_contrib.extensions import DataStreamMonitoringAndSaving
from blocks_contrib.utils import batch_normalize


mnist = MNIST('train', sources=['features'])
data, _ = mnist._load_mnist()
means = data.mean(axis=0)


def _meanize(data):
    newfirst = data[0] - means[None, :]
    return (newfirst, data[1])


def main(save_to, num_epochs):
    dim = 1000
    filtering = SparseFilter(dim=dim, input_dim=784, activation=Identity(),
                             weights_init=IsotropicGaussian(.01), biases_init=Constant(0.))
    filtering.initialize()
    causes = VarianceComponent(dim=dim, input_dim=dim, activation=Identity(),
                               layer_below=filtering,
                               weights_init=Uniform(.02, .001),
                               use_bias=False)
    causes.initialize()
    clf = MLP([Sigmoid(), Softmax()], [dim, dim, 10],
              weights_init=IsotropicGaussian(0.01),
              use_bias=False, name="clf")
    clf.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    cost1 = filtering.cost(inputs=x, batch_size=100) + 0*y.sum()
    cost2, z = causes.cost(inputs=x, batch_size=100)
    cost = cost1 + cost2

    probs = clf.apply(z)
    nll = CategoricalCrossEntropy().apply(y.flatten(), probs)
    clf_error_rate = MisclassificationRate().apply(y.flatten(), probs)
    cost += nll
    cost.name = 'final_cost'
    cg = ComputationGraph([cost, clf_error_rate])
    # new_cg = ComputationGraph([cost])
    new_cg = batch_normalize(clf.linear_transformations, cg)

    mnist_train = MNIST("train")
    mnist_test = MNIST("test")
    trainstream = Mapping(DataStream(mnist_train,
                          iteration_scheme=SequentialScheme(
                              mnist_train.num_examples, 100)),
                          _meanize)
    teststream = Mapping(DataStream(mnist_test,
                         iteration_scheme=SequentialScheme(
                             mnist_test.num_examples, 100)),
                         _meanize)

    algorithm = GradientDescent(
        cost=new_cg.outputs[0], params=new_cg.parameters,
        step_rule=Adam())
    main_loop = MainLoop(
        algorithm,
        trainstream,
        model=Model(cost),
        extensions=[Timing(),
                    FinishAfter(after_n_epochs=num_epochs),
                    DataStreamMonitoring(
                        new_cg.outputs,
                        teststream,
                        prefix="test"),
                    DataStreamMonitoringAndSaving(
                    new_cg.outputs,
                    teststream,
                    [filtering, causes, clf],
                    'best_'+save_to+'.pkl',
                    cost_name='error_rate',
                    after_epoch=True,
                    prefix='valid'
                    ),
                    TrainingDataMonitoring(
                        new_cg.outputs,
                        prefix="train",
                        after_epoch=True),
                    Plot(
                        'Sparse-Coding',
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
