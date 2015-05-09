#!/usr/bin/env python

import logging
import theano
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
    dim = 400
    n_steps = 100
    batch_size = 100
    filtering = SparseFilter(dim=dim, input_dim=784, batch_size=batch_size, n_steps=n_steps,
                             weights_init=IsotropicGaussian(.01), biases_init=Constant(0.))
    filtering.initialize()
    causes = VarianceComponent(dim=9, input_dim=dim, n_steps=n_steps, batch_size=batch_size,
                               layer_below=filtering,
                               weights_init=IsotropicGaussian(.01),  # Uniform(.1, .001),
                               use_bias=False)
    causes.initialize()
    clf = MLP([Sigmoid(), Softmax()], [dim, dim, 10],
              weights_init=IsotropicGaussian(0.01),
              use_bias=False, name="clf")
    clf.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    cost1, code_1, rec_1 = filtering.cost(inputs=x, prior=0)
    cost2, code_2, rec_2 = causes.cost(prev_code=code_1, prior=0)
    cost1, code_1, rec_1 = filtering.cost(inputs=x, prior=0,
                                          gamma=theano.gradient.disconnected_grad(rec_2))
    cost2, code_2, rec_2 = causes.cost(prev_code=code_1, prior=0)
    cost = cost1 + cost2 + 0*y.sum()

    probs = clf.apply(code_1)
    nll = CategoricalCrossEntropy().apply(y.flatten(), probs)
    clf_error_rate = MisclassificationRate().apply(y.flatten(), probs)
    cost += nll
    cost.name = 'final_cost'
    cg = ComputationGraph([cost, clf_error_rate])
    new_cg = cg
    # new_cg = batch_normalize(clf.linear_transformations, cg)

    mnist_train = MNIST("train", stop=50000)
    mnist_valid = MNIST("train", start=50000, stop=60000)
    mnist_test = MNIST("test")

    trainstream = Mapping(DataStream(mnist_train,
                          iteration_scheme=SequentialScheme(
                              mnist_train.num_examples, 100)),
                          _meanize)
    teststream = Mapping(DataStream(mnist_test,
                         iteration_scheme=SequentialScheme(
                             mnist_test.num_examples, 100)),
                         _meanize)
    validstream = Mapping(DataStream(mnist_valid,
                          iteration_scheme=SequentialScheme(
                              mnist_test.num_examples, 100)),
                          _meanize)

    algorithm = GradientDescent(
        cost=new_cg.outputs[0], params=new_cg.parameters,
        step_rule=Adam())
    main_loop = MainLoop(
        algorithm,
        trainstream,
        extensions=[Timing(),
                    FinishAfter(after_n_epochs=num_epochs),
                    DataStreamMonitoring(
                        new_cg.outputs,
                        teststream,
                        prefix="test"),
                    DataStreamMonitoringAndSaving(
                    new_cg.outputs,
                    validstream,
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
                    # Plot(
                    #     save_to,
                    #     channels=[
                    #         ['test_final_cost',
                    #          'test_misclassificationrate_apply_error_rate'],
                    #         ['train_total_gradient_norm']]),
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
