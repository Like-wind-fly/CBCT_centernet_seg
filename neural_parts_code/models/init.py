import torch
import sys

sys.path.append(".")
sys.path.append("..")

try:
    from radam import RAdam
except ImportError:
    pass

from .centerNet import CenterNetBuilder,\
    train_on_batch as train_on_batch_with_CN, \
    validate_on_batch as validate_on_batch_with_CN

from .centerNetMuti import CenterNetMutiBuilder,\
    train_on_batch as train_on_batch_with_CNM, \
    validate_on_batch as validate_on_batch_with_CNM

from .centerNetHead import CenterNetHeadBuilder,\
    train_on_batch as train_on_batch_with_CNH, \
    validate_on_batch as validate_on_batch_with_CNH

from .centerNetFcos import CenterNetFcosBuilder,\
    train_on_batch as train_on_batch_with_CNF, \
    validate_on_batch as validate_on_batch_with_CNF

from .vnet import VNetBuilder,\
    train_on_batch as train_on_batch_with_v, \
    validate_on_batch as validate_on_batch_with_v

from .sgaNet import SGAnetBuilder ,\
    train_on_batch as train_on_batch_with_SGA, \
    validate_on_batch as validate_on_batch_with_SGA


class OptimizerWrapper(object):
    def __init__(self, optimizer, aggregate=1):
        self.optimizer = optimizer
        self.aggregate = aggregate
        self._calls = 0

    def zero_grad(self):
        if self._calls == 0:
            self.optimizer.zero_grad()

    def step(self):
        self._calls += 1
        if self._calls == self.aggregate:
            self._calls = 0
            self.optimizer.step()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return OptimizerWrapper(
            torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                            weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "Adam":
        return OptimizerWrapper(
            torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "RAdam":
        return OptimizerWrapper(
            RAdam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    else:
        raise NotImplementedError()


def build_network(config, weight_file=None, device="cpu"):
    network, train_on_batch, validate_on_batch = get_network_with_type(config)
    network.to(device)
    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )

    return network, train_on_batch, validate_on_batch


def get_network_with_type(config):
    network_type = config["network"]["type"]

    if network_type == "centerNet":
        network = CenterNetBuilder(config).network
        train_on_batch = train_on_batch_with_CN
        validate_on_batch = validate_on_batch_with_CN

    elif network_type == "centerNetMuti":
        network = CenterNetMutiBuilder(config).network
        train_on_batch = train_on_batch_with_CNM
        validate_on_batch = validate_on_batch_with_CNM

    elif network_type == "centerNetHead":
        network = CenterNetHeadBuilder(config).network
        train_on_batch = train_on_batch_with_CNH
        validate_on_batch = validate_on_batch_with_CNH
    elif network_type == "centerNetFcos":
        network = CenterNetFcosBuilder(config).network
        train_on_batch = train_on_batch_with_CNF
        validate_on_batch = validate_on_batch_with_CNF

    elif network_type == "SGA":
        network = SGAnetBuilder(config).network
        train_on_batch = train_on_batch_with_SGA
        validate_on_batch = validate_on_batch_with_SGA
    elif network_type == "Vnet":
        network = VNetBuilder(config).network
        train_on_batch = train_on_batch_with_v
        validate_on_batch = validate_on_batch_with_v
    else:
        raise NotImplementedError()
    return network, train_on_batch, validate_on_batch
