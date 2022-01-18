import pytest
import itertools
import multiprocessing

import numpy as np
import torch

from distributed_optimization_library.algorithm import MarinaMasterAlgorithm, MarinaNodeAlgorithm, \
    GradientDescentNodeAlgorithm, GradientDescentMasterAlgorithm, MarinaPermutationMasterAlgorithm, \
    MarinaPermutationNodeAlgorithm, ZeroMarinaNodeAlgorithm, ZeroMarinaMasterAlgorithm, get_algorithm, \
    ZeroMarinaStochasticMasterAlgorithm, ZeroMarinaStochasticNodeAlgorithm, \
    StochasticGradientDescentMasterAlgorithm, StochasticGradientDescentNodeAlgorithm, ZeroMarinaPageNodeAlgorithm, \
    ZeroMarinaPageMasterAlgorithm, VRMarinaNodeAlgorithm, VRMarinaMasterAlgorithm, MarinaStochasticMasterAlgorithm, \
    MarinaStochasticNodeAlgorithm, DCGDStochasticMasterAlgorithm, DCGDStochasticNodeAlgorithm, \
    ZeroMarinaSyncStochasticMasterAlgorithm, ZeroMarinaSyncStochasticNodeAlgorithm
from distributed_optimization_library.function import QuadraticFunction, QuadraticTorchFunction, generate_random_vector, \
    QuadraticOptimizationProblemMeta, StochasticQuadraticFunction, MeanQuadraticFunction, OptimizationProblemMeta
from distributed_optimization_library.compressor import get_compressors
from distributed_optimization_library.transport import Transport
from distributed_optimization_library.signature import Signature


def mean(vectors):
    return sum(vectors) / float(len(vectors))


class OrthogonalityTestMarinaMasterAlgorithm(MarinaMasterAlgorithm):
    def _update_gradient_estimator(self):
        updates = self._transport.call_nodes_method(node_method="calculate_updates")
        updates = [update.decompress() for update in updates]
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                assert np.dot(updates[i], updates[j]) == 0
        self._gradient_estimator = self._gradient_estimator + self._average(updates)


@pytest.mark.parametrize("prob,number_of_iterations,gamma,compressor_name", [
                                                             (0.01, 10000, 0.01, 'permutation'), 
                                                             (0.9, 1000, 0.01, 'permutation'), 
                                                             (1.0, 100, 0.01, 'permutation'), 
                                                             (None, 1000, 0.01, 'permutation'),
                                                             (None, 1000, 0.01, 'rand_k'),
                                                             (None, 1000, None, 'rand_k')])
def test_marina_algorithm_with_quadratic_function(prob, number_of_iterations, gamma, compressor_name):
    dim = 100
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    functions = [QuadraticFunction.create_random(dim, generator) for _ in range(num_nodes)]
    if compressor_name == 'permutation':
        params = {'total_number_of_nodes': num_nodes, 'dim': dim}
        algorithms = [OrthogonalityTestMarinaMasterAlgorithm, MarinaMasterAlgorithm]
    elif compressor_name == 'rand_k':
        params = {'number_of_coordinates': 10, 'dim': dim}
        algorithms = [MarinaMasterAlgorithm]
    for master_cls in algorithms:
        compressors = get_compressors(compressor_name=compressor_name, 
                                      params=params, 
                                      total_number_of_nodes=num_nodes, 
                                      seed=generator)
        point = generate_random_vector(dim, generator)
        nodes = [Signature(MarinaNodeAlgorithm, functions[node_index], compressors[node_index])
                for node_index in range(num_nodes)]
        transort = Transport(nodes)
        optimizer = master_cls(transort, point, gamma, prob, seed=generator)
        solution = optimizer.optimize(number_of_iterations)
        
        analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                              mean([function._b for function in functions]))
        np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)
        
        points = optimizer._transport.call_nodes_method(node_method="get_point")
        for lhs, rhs in zip(points[:-1], points[1:]):
            np.testing.assert_array_equal(lhs, rhs)


@pytest.mark.parametrize("momentum,number_of_iterations,gamma,compressor_name_base", [
    (None, 1000, 0.01, 'rand_k'),
    (None, 1000, None, 'rand_k')])
def test_zero_marina_algorithm_with_quadratic_function(momentum, number_of_iterations, gamma, compressor_name_base):
    dim = 100
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    params = {'number_of_coordinates': 10, 'dim': dim}
    stats = [None, None]
    for is_torch in [False, True]:
        if not is_torch:
            functions = [QuadraticFunction.create_random(dim, generator) for _ in range(num_nodes)]
            compressor_name = compressor_name_base
        else:
            compressor_name = compressor_name_base + '_torch'
            functions = [QuadraticTorchFunction.create_random(dim, generator) for _ in range(num_nodes)]
        compressors = get_compressors(compressor_name=compressor_name,
                                      params=params, 
                                      total_number_of_nodes=num_nodes, 
                                      seed=generator)
        point = generate_random_vector(dim, generator)
        if is_torch:
            point = torch.tensor(point)
        nodes = [Signature(ZeroMarinaNodeAlgorithm, functions[node_index], compressors[node_index])
                for node_index in range(num_nodes)]
        transort = Transport(nodes)
        optimizer = ZeroMarinaMasterAlgorithm(transort, point, gamma, momentum)
        solution = optimizer.optimize(number_of_iterations)
        stat_from_nodes, stat_from_nodes = optimizer.get_stats()
        stats[is_torch] = [stat_from_nodes, stat_from_nodes]
        
        analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                            mean([function._b for function in functions]))
        np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)
    assert stats[True] == stats[False]


@pytest.mark.parametrize("algorithm_name", ['marina_stochastic', 'zero_marina_stochastic', 'zero_marina_sync_stochastic'])
def test_marina_stochastic_and_zero_marina_stochastic_algorithm_with_stochastic_quadratic_function(algorithm_name):
    gamma = 0.01
    compressor_name = 'rand_k'
    dim = 100
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    functions = [StochasticQuadraticFunction.create_random(dim, generator, noise=0.1) for _ in range(num_nodes)]
    params = {'number_of_coordinates': 10, 'dim': dim}
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params=params, 
                                  total_number_of_nodes=num_nodes, 
                                  seed=generator)
    point = generate_random_vector(dim, generator)
    if algorithm_name == 'zero_marina_stochastic':
        NodeAlgorithm = ZeroMarinaStochasticNodeAlgorithm
        MasterAlgorithm = ZeroMarinaStochasticMasterAlgorithm
        number_of_iterations = 1000
        params = {'noise_momentum': 0.01, 'initial_mega_batch_size': 100}
    elif algorithm_name == 'marina_stochastic':
        NodeAlgorithm = MarinaStochasticNodeAlgorithm
        MasterAlgorithm = MarinaStochasticMasterAlgorithm
        number_of_iterations = 10000
        params = {'mega_batch_size': 100, 'initial_mega_batch_size': 100}
    elif algorithm_name == 'zero_marina_sync_stochastic':
        NodeAlgorithm = ZeroMarinaSyncStochasticNodeAlgorithm
        MasterAlgorithm = ZeroMarinaSyncStochasticMasterAlgorithm
        number_of_iterations = 1000
        params = {'mega_batch_size': 100, 'prob_sync': 0.01, 'seed': 42, 'initial_mega_batch_size': 100}
    nodes = [Signature(NodeAlgorithm, functions[node_index], compressors[node_index])
             for node_index in range(num_nodes)]
    transort = Transport(nodes)
    optimizer = MasterAlgorithm(transort, point, gamma, **params)
    solution = optimizer.optimize(number_of_iterations)
    
    analytical_solution = np.linalg.solve(mean([function._quadratic_function._A for function in functions]),
                                          mean([function._quadratic_function._b for function in functions]))
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


@pytest.mark.parametrize("algorithm_name", ['vr_marina', 'zero_marina_page'])
def test_vr_and_zero_marina_page_algorithm_with_quadratic_function(algorithm_name):
    number_of_iterations = 1000
    if algorithm_name == 'vr_marina':
        gamma = None
    else:
        gamma = 0.01
    compressor_name = 'rand_k'
    dim = 100
    num_nodes = 10
    number_of_functions = 100
    batch_size = 4
    params = {'number_of_coordinates': 10, 'dim': dim}
    
    generator = np.random.default_rng(seed=42)
    functions = [MeanQuadraticFunction.create_random(dim, number_of_functions, generator) for _ in range(num_nodes)]
    compressors = get_compressors(compressor_name=compressor_name,
                                  params=params, 
                                  total_number_of_nodes=num_nodes, 
                                  seed=generator)
    point = generate_random_vector(dim, generator)
    if algorithm_name == 'zero_marina_page':
        NodeAlgorithm = ZeroMarinaPageNodeAlgorithm
        MasterAlgorithm = ZeroMarinaPageMasterAlgorithm
    elif algorithm_name == 'vr_marina':
        NodeAlgorithm = VRMarinaNodeAlgorithm
        MasterAlgorithm = VRMarinaMasterAlgorithm
        
    nodes = [Signature(NodeAlgorithm, functions[node_index], compressors[node_index])
             for node_index in range(num_nodes)]
    transort = Transport(nodes)
    optimizer = MasterAlgorithm(transort, point, gamma, batch_size=batch_size)
    solution = optimizer.optimize(number_of_iterations)
    A = mean([mean([qf._A for qf in function._quadratic_functions]) for function in functions])
    B = mean([mean([qf._b for qf in function._quadratic_functions]) for function in functions])
    analytical_solution = np.linalg.solve(A, B)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)


def test_marina_permutation_algorithm_with_quadratic_function():
    prob = None
    gamma = None
    number_of_iterations = 1000
    compressor_name = 'permutation'
    dim = 100
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    functions = [QuadraticFunction.create_random(dim, generator) for _ in range(num_nodes)]
    params = {'total_number_of_nodes': num_nodes, 'dim': dim}
    compressors = get_compressors(compressor_name=compressor_name, 
                                    params=params, 
                                    total_number_of_nodes=num_nodes, 
                                    seed=generator)
    point = generate_random_vector(dim, generator)
    nodes = [Signature(MarinaPermutationNodeAlgorithm, functions[node_index], compressors[node_index])
             for node_index in range(num_nodes)]
    transort = Transport(nodes)
    optimizer = MarinaPermutationMasterAlgorithm(transort, point, gamma, prob, seed=generator,
                                                 meta=OptimizationProblemMeta())
    solution = optimizer.optimize(number_of_iterations)
    
    analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                            mean([function._b for function in functions]))
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)


@pytest.mark.parametrize("gamma", [0.01, None])
def test_gradient_descent_algorithm_with_quadratic_function(gamma):
    dim = 100
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    number_of_iterations = 1000
    stats = [None, None]
    for is_torch in [False, True]:
        if not is_torch:
            functions = [QuadraticFunction.create_random(dim, generator) for _ in range(num_nodes)]
        else:
            functions = [QuadraticTorchFunction.create_random(dim, generator) for _ in range(num_nodes)]
        point = generate_random_vector(dim, generator)
        if is_torch:
            point = torch.tensor(point)
        
        nodes = [Signature(GradientDescentNodeAlgorithm, functions[node_index]) for node_index in range(num_nodes)]
        transort = Transport(nodes)
        optimizer = GradientDescentMasterAlgorithm(transort, point, gamma)
        solution = optimizer.optimize(number_of_iterations)
        stat_from_nodes, stat_from_nodes = optimizer.get_stats()
        stats[is_torch] = [stat_from_nodes, stat_from_nodes]
        
        analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                            mean([function._b for function in functions]))
        np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)
    assert stats[True] == stats[False]


def test_stochastic_gradient_descent_algorithm_with_stochastic_quadratic_function():
    number_of_iterations = 1000
    gamma = 0.01
    dim = 100
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    functions = [StochasticQuadraticFunction.create_random(dim, generator, noise=0.1) for _ in range(num_nodes)]
    point = generate_random_vector(dim, generator)
    nodes = [Signature(StochasticGradientDescentNodeAlgorithm, functions[node_index])
             for node_index in range(num_nodes)]
    transort = Transport(nodes)
    optimizer = StochasticGradientDescentMasterAlgorithm(transort, point, gamma)
    solution = optimizer.optimize(number_of_iterations)
    
    analytical_solution = np.linalg.solve(mean([function._quadratic_function._A for function in functions]),
                                          mean([function._quadratic_function._b for function in functions]))
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)


def test_dcgd_stochastic_gradient_descent_algorithm_with_stochastic_quadratic_function():
    number_of_iterations = 1000
    gamma = 0.01
    dim = 100
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    functions = [StochasticQuadraticFunction.create_random(dim, generator, noise=0.1) for _ in range(num_nodes)]
    point = generate_random_vector(dim, generator)
    compressors = get_compressors(compressor_name="rand_k",
                                  params={'number_of_coordinates': 90, 'dim': dim},
                                  total_number_of_nodes=num_nodes, 
                                  seed=generator)
    nodes = [Signature(DCGDStochasticNodeAlgorithm, functions[node_index], compressors[node_index]) 
             for node_index in range(num_nodes)]
    transort = Transport(nodes)
    optimizer = DCGDStochasticMasterAlgorithm(transort, point, gamma)
    solution = optimizer.optimize(number_of_iterations)
    
    analytical_solution = np.linalg.solve(mean([function._quadratic_function._A for function in functions]),
                                          mean([function._quadratic_function._b for function in functions]))
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


@pytest.mark.parametrize("gamma,init_with_gradients", itertools.product([0.0001, None], [True, False]))
def test_ef21_algorithm_with_quadratic_function(gamma, init_with_gradients):
    algorithm_name = "ef21"
    dim = 100
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    compressor_name = 'top_k'
    params = {'number_of_coordinates': 10, 'dim': dim}
    functions = [QuadraticFunction.create_random(dim, generator, reg=0.01) for _ in range(num_nodes)]
    point = generate_random_vector(dim, generator)
    
    algorithm_master_params = {'gamma': gamma,
                               'init_with_gradients': init_with_gradients}
    number_of_iterations = 5000
    optimizer = get_algorithm(functions, point, generator, algorithm_name, 
                              algorithm_master_params=algorithm_master_params,
                              compressor_name=compressor_name,
                              compressor_params=params,
                              meta=QuadraticOptimizationProblemMeta(functions))
    solution = optimizer.optimize(number_of_iterations)
    
    analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                          mean([function._b for function in functions]))
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=3)


multiprocessing.set_start_method("spawn")
torch.multiprocessing.set_sharing_strategy("file_system")

@pytest.mark.parametrize("algorithm_name", ["marina", "zero_marina"])
def test_parallel_algorithm_with_quadratic_function(algorithm_name):
    gamma = 0.01
    dim = 100
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    number_of_iterations = 200
    compressor_name = 'coordinate_sampling'
    params = {'total_number_of_nodes': num_nodes, 'dim': dim}
    
    functions = [QuadraticFunction.create_random(dim, generator) for _ in range(num_nodes)]
    point = generate_random_vector(dim, generator)
    optimizer = get_algorithm(functions, point, 
                              seed=42, algorithm_name=algorithm_name,
                              algorithm_master_params={'gamma': gamma},
                              compressor_name=compressor_name,
                              compressor_params=params)
    optimizer_parallel = get_algorithm(functions, point, 
                                       seed=42, algorithm_name=algorithm_name,
                                       algorithm_master_params={'gamma': gamma},
                                       compressor_name=compressor_name,
                                       compressor_params=params,
                                       parallel=True,
                                       shared_memory_size=dim,
                                       number_of_processes=3)
    
    for _ in range(number_of_iterations):
        optimizer.step()
        optimizer_parallel.step()
        point = optimizer.get_point()
        point_parallel = optimizer_parallel.get_point()
        np.testing.assert_array_almost_equal(point, point_parallel, decimal=3)
    
    analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                          mean([function._b for function in functions]))
    np.testing.assert_array_almost_equal(point_parallel, analytical_solution, decimal=3)
    stat_to_nodes, stat_from_nodes = optimizer.get_stats()
    stat_to_nodes_parallel, stat_from_nodes_parallel = optimizer_parallel.get_stats()
    max_stats_from_nodes = optimizer.get_max_stats()
    max_stats_from_nodes_parallel = optimizer_parallel.get_max_stats()
    assert stat_to_nodes == stat_to_nodes_parallel
    assert stat_from_nodes == stat_from_nodes_parallel
    assert max_stats_from_nodes == max_stats_from_nodes_parallel
    optimizer_parallel.stop()


def test_parallel_algorithm_with_torch_quadratic_function():
    cuda = False
    
    algorithm_name = "zero_marina"
    gamma = 0.01
    dim = 100
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    number_of_iterations = 200
    compressor_name = 'rand_k_torch'
    params = {'number_of_coordinates': 10, 'dim': dim}
    
    functions = [QuadraticTorchFunction.create_random(dim, generator) for _ in range(num_nodes)]
    point = generate_random_vector(dim, generator)
    point = torch.tensor(point)
    if cuda:
        params['is_cuda'] = True
        functions = [func.cuda() for func in functions]
        point = point.cuda()
    optimizer = get_algorithm(functions, point, 
                              seed=42, algorithm_name=algorithm_name,
                              algorithm_master_params={'gamma': gamma},
                              compressor_name=compressor_name,
                              compressor_params=params)
    optimizer_parallel = get_algorithm(functions, point, 
                                       seed=42, algorithm_name=algorithm_name,
                                       algorithm_master_params={'gamma': gamma},
                                       compressor_name=compressor_name,
                                       compressor_params=params,
                                       parallel=True,
                                       shared_memory_size=dim,
                                       number_of_processes=3)
    
    for _ in range(number_of_iterations):
        optimizer.step()
        optimizer_parallel.step()
        point = optimizer.get_point()
        point_parallel = optimizer_parallel.get_point()
        if cuda:
            np.testing.assert_array_almost_equal(point.cpu(), point_parallel.cpu(), decimal=3)
        else:
            np.testing.assert_array_almost_equal(point, point_parallel, decimal=3)
    
    if cuda:
        analytical_solution = np.linalg.solve(mean([function._A.cpu() for function in functions]),
                                              mean([function._b.cpu() for function in functions]))
        np.testing.assert_array_almost_equal(point_parallel.cpu(), analytical_solution, decimal=3)
    else:
        analytical_solution = np.linalg.solve(mean([function._A for function in functions]),
                                            mean([function._b for function in functions]))
        np.testing.assert_array_almost_equal(point_parallel, analytical_solution, decimal=3)
    stat_to_nodes, stat_from_nodes = optimizer.get_stats()
    stat_to_nodes_parallel, stat_from_nodes_parallel = optimizer_parallel.get_stats()
    max_stats_from_nodes = optimizer.get_max_stats()
    max_stats_from_nodes_parallel = optimizer_parallel.get_max_stats()
    assert stat_to_nodes == stat_to_nodes_parallel
    assert stat_from_nodes == stat_from_nodes_parallel
    assert max_stats_from_nodes == max_stats_from_nodes_parallel
    optimizer_parallel.stop()
