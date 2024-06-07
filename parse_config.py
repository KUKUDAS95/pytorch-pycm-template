import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json
import shutil

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, test_mode=False, processor_cores=None):
        """ Class to parse configuration json file.
            Handles hyperparameters for training, initializations of modules, checkpoint saving and logging module.
        
        Args:
            config: (str) Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
            resume: (str) String, path to the checkpoint being loaded.
            modification: (list) Dict keychain:value, specifying position values to be replaced from config dict.
            run_id: (str) Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
            test_mode: (bool) test mode or not.        
            processor_cores: (int) number of processor cores.
        """
        # load config file and apply modification
        self.processor_cores = processor_cores
        self._config = _update_config(config, modification)
        self.resume = resume
        resume_epoch = None if resume is None else Path(resume).name.split("-")[-1].split(".pth")[0] # best or int
        if test_mode: resume_epoch = 'trained'

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        exper_name += f"/{self.config['optimizer']['type']}-lr_{self.config['optimizer']['args']['lr']}"
        if 'lr_scheduler' in self.config.keys():
            exper_name += f"_{self.config['lr_scheduler']['type']}"
        exper_name += f"/{self.config['arch']['type']}"
        if run_id is None: # use timestamp as default run-id    
            acc_steps = '' if 'accumulation_steps' not in self.config['trainer'].keys() else f"X{self.config['trainer']['accumulation_steps']}"
            sampling = '' if 'data_sampling' not in self.config.keys() else f"-{self.config['data_sampling']['type']}_{self.config['data_sampling']['name']}"
            da = '' if 'data_augmentation' not in self.config.keys() else f"-{self.config['data_augmentation']['type']}"
            run_id =  f"{self.config['data_loader']['args']['batch_size']}batch{acc_steps}"
            run_id += f"-{self.config['trainer']['epochs']}epoch-{self.config['loss']}{da}{sampling}"
            run_id += f"-{datetime.now().strftime(r'%m%d_%H%M%S')}"
        self._checkpoint_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id
        self._output_dir = save_dir / 'output' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        if self.resume is None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.output_dir.mkdir(parents=True, exist_ok=exist_ok)
            # save updated config file to the checkpoint dir
            write_json(self.config, self._checkpoint_dir / 'config.json')
        else:
            copy_dir_path = self.log_dir.parent / f'{self.log_dir.name}-{resume_epoch}'
            if not copy_dir_path.is_dir(): shutil.copytree(self.log_dir, copy_dir_path)
            elif resume_epoch == 'trained':
                shutil.rmtree(self.log_dir)
                shutil.copytree(copy_dir_path, self.log_dir)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        run_id=None
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            try: args = args.parse_args()
            except: args = args.parse_args(args=[]) # for testing with jupyter lab

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            if resume.suffix not in ['.pth', '.pt']: raise ValueError('This is not a model path to resume.')
            cfg_fname = resume.parent / 'config.json'
            run_id = resume.parent.name
        else:
            if args.config is None: raise ValueError('Configuration file need to be specified. Add \'-c config.json\', for example.')
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        
        # 원본코드
        # return cls(config, resume, modification, run_id, args.test, args.core)

        # 수정코드
        return cls(config, resume, modification, run_id, args.test)

        
    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        if not all([k not in module_args for k in kwargs]): raise ValueError('Overwriting kwargs given in config file is not allowed')
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        if not all([k not in module_args for k in kwargs]): raise ValueError('Overwriting kwargs given in config file is not allowed')
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        if verbosity not in self.log_levels: ValueError(f'verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}.')
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def output_dir(self):
        return self._output_dir
        
# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
