import numpy as np
import torch
from mpi4py import MPI

from .general_utils import (AttrDict, joinListDict, joinListDictList,
                            joinListList)


def update_mpi_config(cfg):
    rank = MPI.COMM_WORLD.Get_rank()
    cfg.mpi.rank = rank
    cfg.mpi.is_chef = rank == 0
    cfg.mpi.num_workers = MPI.COMM_WORLD.Get_size()

    # update conf
    cfg.seed = cfg.seed + rank


def mpi_sum(x):
    buf = np.zeros_like(np.array(x))
    MPI.COMM_WORLD.Allreduce(np.array(x), buf, op=MPI.SUM)
    return buf


def mpi_gather_experience_episode(experience_episode):
    buf = MPI.COMM_WORLD.allgather(experience_episode)
    return joinListDictList(buf)


def mpi_gather_experience_rollots(experience_rollouts):
    buf = MPI.COMM_WORLD.allgather(experience_rollouts)
    return joinListDict(buf)


def mpi_gather_experience_transitions(experience_transitions):
    buf = MPI.COMM_WORLD.allgather(experience_transitions)
    return joinListList(buf)


def mpi_gather_experience(experience_episode):
    """Gathers data across workers, can handle hierarchical and flat experience dicts."""
    return mpi_gather_experience_episode(experience_episode)


def mpi_gather_rollouts(rollouts):
    """Gathers data across workers, can handle hierarchical and flat experience dicts."""
    return mpi_gather_experience_rollots(rollouts)


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)


# get the flat params from the network
def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.state_dict().items():
        param_shape[key_name] = value.cpu().detach().numpy().shape
        if flat_params is None:
            flat_params = value.cpu().detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.cpu().detach().numpy().flatten())
    return flat_params, param_shape


# set the params from the network
def _set_flat_params(network, params_shape, params):
    pointer = 0
    device = torch.device("cuda:0")

    for key_name, values in network.state_dict().items():
        # get the length of the parameters
        len_param = int(np.prod(params_shape[key_name]))
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params).to(device)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param
