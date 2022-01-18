import numpy as np
import torch

from distributed_optimization_library.transport import Transport, find_total, find_max
from distributed_optimization_library.compressor import CompressedVector, CompressedTorchVector
from distributed_optimization_library.signature import Signature


class DummyNode():
    
    def do_nothing(self, foo_arr, baz_arr):
        return

    def return_array(self):
        output_arr = np.zeros((13,), dtype=np.float64)
        output_arr[:4] = 3.14
        return output_arr


def test_transport():
    transport = Transport([Signature(DummyNode)])
    foo_arr = np.zeros((54,), dtype=np.float32)
    foo_arr[1:7] = 3.14
    baz_arr = np.zeros((11,), dtype=np.float64)
    baz_arr[:2] = 3.14
    transport.call_node_method(node_index=0,
                               node_method="do_nothing",
                               foo_arr=foo_arr,
                               baz_arr=baz_arr) # 54 * 32 + 11 * 64
    compressed_foo_arr = CompressedVector(
        range(1, 7), np.array([3.14] * 6, dtype=np.float32), 54)
    assert compressed_foo_arr.decompress().tolist() == foo_arr.tolist()
    compresse_torch_foo_arr = CompressedTorchVector(
        torch.tensor(range(2, 7)), torch.tensor(np.array([3.14] * 5, dtype=np.float32)), 43)
    transport.call_node_method(node_index=0,
                               node_method="do_nothing",
                               foo_arr=compressed_foo_arr,
                               baz_arr=compresse_torch_foo_arr) # 6 * 32 + 5 * 32
    foo_arr_torch = torch.tensor(np.zeros((9,), dtype=np.float32))
    foo_arr[1:7] = 3.14
    baz_arr_torch = torch.tensor(np.zeros((7,), dtype=np.float64))
    baz_arr[:2] = 3.14
    transport.call_node_method(node_index=0,
                               node_method="do_nothing",
                               foo_arr=foo_arr_torch,
                               baz_arr=baz_arr_torch) # 9 * 32 + 7 * 64
    
    transport.call_node_method(node_index=0,
                               node_method="return_array") # 13 * 64
    transport.call_nodes_method(node_method="return_array") # 13 * 64
    
    with transport.ignore_statistics():
        transport.call_node_method(node_index=0,
                                   node_method="do_nothing",
                                   foo_arr=foo_arr_torch,
                                   baz_arr=baz_arr_torch)
        transport.call_node_method(node_index=0, node_method="return_array")
    
    stat_from_nodes = transport.get_stat_from_nodes()
    stat_to_nodes = transport.get_stat_to_nodes()
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    
    assert stat_to_nodes[0]["do_nothing"] == 54 * 32 + 11 * 64 + 6 * 32 + 5 * 32 + 9 * 32 + 7 * 64
    assert stat_from_nodes[0]["return_array"] == 13 * 64 + 13 * 64
    assert max_stat_from_nodes["return_array"] == stat_from_nodes[0]["return_array"]


def test_find_total():
    dct = [{"method_baz": 43}, {"method_foo": 21}]
    assert find_total(dct) == 43 + 21


def test_find_max():
    dct = [{"method_baz": 43}, {"method_foo": 21}]
    assert find_max(dct) == 43
