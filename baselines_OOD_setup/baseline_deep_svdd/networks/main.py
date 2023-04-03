from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .kyoto import Kyoto_Net, Kyoto_Net_Autoencoder
from .kyoto2 import Kyoto_Net2, Kyoto_Net_Autoencoder2
from .kyoto_id import Kyoto_ID
from .kyoto_numeric import Kyoto_Numeric, Kyoto_Numeric_Autoencoder


def build_network(net_name, in_size=None):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'kyoto', 'kyoto2', 'kyoto_id', 'kyoto_numeric')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'kyoto':
        net = Kyoto_Net()
    
    if net_name == 'kyoto2':
        net = Kyoto_Net2()
    
    if net_name == 'kyoto_id':
        net = Kyoto_ID(in_size)

    if net_name == 'kyoto_numeric':
        net = Kyoto_Numeric()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'kyoto', 'kyoto2', 'kyoto_numeric')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'kyoto':
        ae_net = Kyoto_Net_Autoencoder()

    if net_name == 'kyoto2':
        ae_net = Kyoto_Net_Autoencoder2()

    if net_name == 'kyoto_numeric':
        ae_net = Kyoto_Numeric_Autoencoder()
    return ae_net
