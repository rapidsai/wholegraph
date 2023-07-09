=======================
pylibwholegraph API doc
=======================

.. currentmodule:: pylibwholegraph

APIs
----
.. autosummary::
    :toctree: api/pylibwholegraph

    pylibwholegraph.torch.initialize.init_torch_env
    pylibwholegraph.torch.initialize.init_torch_env_and_create_wm_comm
    pylibwholegraph.torch.initialize.finalize
    pylibwholegraph.torch.comm.WholeMemoryCommunicator
    pylibwholegraph.torch.comm.set_world_info
    pylibwholegraph.torch.comm.create_group_communicator
    pylibwholegraph.torch.comm.destroy_communicator
    pylibwholegraph.torch.comm.get_global_communicator
    pylibwholegraph.torch.comm.get_local_node_communicator
    pylibwholegraph.torch.comm.get_local_device_communicator
    pylibwholegraph.torch.tensor.WholeMemoryTensor
    pylibwholegraph.torch.tensor.create_wholememory_tensor
    pylibwholegraph.torch.tensor.create_wholememory_tensor_from_filelist
    pylibwholegraph.torch.tensor.destroy_wholememory_tensor
    pylibwholegraph.torch.embedding.WholeMemoryOptimizer
    pylibwholegraph.torch.embedding.create_wholememory_optimizer
    pylibwholegraph.torch.embedding.destroy_wholememory_optimizer
    pylibwholegraph.torch.embedding.WholeMemoryCachePolicy
    pylibwholegraph.torch.embedding.create_wholememory_cache_policy
    pylibwholegraph.torch.embedding.create_builtin_cache_policy
    pylibwholegraph.torch.embedding.destroy_wholememory_cache_policy
    pylibwholegraph.torch.embedding.WholeMemoryEmbedding
    pylibwholegraph.torch.embedding.create_embedding
    pylibwholegraph.torch.embedding.create_embedding_from_filelist
    pylibwholegraph.torch.embedding.destroy_embedding
    pylibwholegraph.torch.embedding.WholeMemoryEmbeddingModule
    pylibwholegraph.torch.graph_structure.GraphStructure
