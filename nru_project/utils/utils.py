from collections import OrderedDict
import _pickle as pickle
import inspect
import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import yaml
from copy import deepcopy


class MyContainer():

    def __init__(self):
        self.__dict__['tparams'] = OrderedDict()

    def __setattr__(self, name, array):
        tparams = self.__dict__['tparams']
        tparams[name] = array

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        tparams = self.__dict__['tparams']
        if name in tparams:
            return tparams[name]
        else:
            return None

    def remove(self, name):
        del self.__dict__['tparams'][name]

    def get(self):
        return self.__dict__['tparams']

    def values(self):
        tparams = self.__dict__['tparams']
        return list(tparams.values())

    def save(self, filename):
        tparams = self.__dict__['tparams']
        pickle.dump({p: tparams[p] for p in tparams}, open(filename, 'wb'), 2)

    def load(self, filename):
        tparams = self.__dict__['tparams']
        loaded = pickle.load(open(filename, 'rb'))
        for k in loaded:
            tparams[k] = loaded[k]

    def setvalues(self, values):
        tparams = self.__dict__['tparams']
        for p, v in zip(tparams, values):
            tparams[p] = v

    def add_from_dict(self, src_dict):
        for key in src_dict.keys():
            self.__setattr__(key, src_dict[key])

    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        self.__dict__['_env_locals'] = list(env_locals.keys())

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        prev_env_locals = self.__dict__['_env_locals']
        del self.__dict__['_env_locals']
        for k in list(env_locals.keys()):
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True

    def __deepcopy__(self, memo):
        new_container = MyContainer()
        new_container.add_from_dict(src_dict=deepcopy(self.get()))
        return new_container

def create_config(file_name):
    """Returns a config object.

    Args:
        file_name: yaml file name with absolute or relative path.
    """
    with open(file_name, 'r') as f:
        config_dict = yaml.load(f)

    config = MyContainer()

    assert("project_name" in config_dict.keys())
    assert("ex_name" in config_dict.keys())

    if "parent_config" in config_dict.keys():
        with open(config_dict["parent_config"], "r") as f:
            parent_config_dict = yaml.load(f)
            config.add_from_dict(parent_config_dict)

    config.add_from_dict(config_dict)

    key = "PROJ_SAVEDIR"
    assert(key in os.environ), "Environment variable named `{}` not set. Pls set it to the dir you wish to save the results".format(key)
    config.tflog_dir = os.path.join(os.environ[key], "logs", config.project_name, config.ex_name)
    config.save_dir = os.path.join(os.environ[key], "output", config.project_name, config.ex_name)
    create_folder(config.tflog_dir)
    create_folder(config.save_dir)

    # checking the device type
    device_type = config.device
    if device_type != "cpu":
        # check if cuda is available or not
        if not torch.cuda.is_available():
            device_type = "cpu"

    config.device = device_type

    return config


def act_name(activation):
    if activation == None:
        return 'linear'
    elif activation == torch.sigmoid:
        return 'sigmoid'
    elif activation == torch.nn.LogSoftmax:
        return 'sigmoid'
    elif activation == torch.tanh:
        return 'tanh'

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def my_variable(input, use_gpu=False, volatile=False, requires_grad=True):
    v = Variable(input, volatile=volatile, requires_grad=requires_grad)
    if use_gpu:
        v = v.cuda()
    return v

def modify_config_params(config, config_flag):
    config_flag = config_flag.split("__")
    if len(config_flag) % 2 != 0:
        print("format {config_param1}__{value1}__{config_param2}__{value2} ... ")
        assert(0)

    config_flag_values = [(flag, value) for flag, value in zip(config_flag[::2], config_flag[1::2])]
    for flag, value in config_flag_values:
        if flag in config.get():
            print(("Setting {} with value {}".format(flag, value)))
            if isinstance(config.get()[flag], str):
                config.get()[flag] = value
            else:
                config.get()[flag] = eval(value)
        else:
            assert("Flag {} not present in config !".format(flag))

def one_hot(x, n):
    if isinstance(x, list):
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def get_optimizer(params, config):
    if config.optim_name == "RMSprop":
        return optim.RMSprop(params, lr=config.lr, alpha=config.alpha, eps=config.eps, weight_decay=config.weight_decay, momentum=config.momentum, centered=config.centered)
    elif config.optim_name == "Adadelta":
        return optim.Adadelta(params, lr=config.lr, rho=config.rho, eps=config.eps, weight_decay=config.weight_decay)
    elif config.optim_name == "Adagrad":
        return optim.Adagrad(params, lr=config.lr, lr_decay=config.lr_decay, weight_decay=config.weight_decay)
    elif config.optim_name == "Adam":
        return optim.Adam(params, lr=config.lr, betas=(config.beta_0, config.beta_1), eps=config.eps, weight_decay=config.weight_decay,
                          amsgrad=config.amsgrad)
    elif config.optim_name == "SGD":
        return optim.SGD(params, lr=config.lr, momentum=config.momentum, dampening=config.dampening, weight_decay=config.weight_decay, nesterov=config.nesterov)
    else:
        assert("Unsupported optimizer : {}. Valid optimizers : Adadelta, Adagrad, Adam, RMSprop, SGD".format(config.optim_name))
