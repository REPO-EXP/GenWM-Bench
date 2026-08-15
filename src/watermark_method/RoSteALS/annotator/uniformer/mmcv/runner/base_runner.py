
import copy
import logging
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod

import torch
from torch.optim import Optimizer

import annotator.uniformer.mmcv as mmcv
from ..parallel import is_module_wrapper
from .checkpoint import load_checkpoint
from .dist_utils import get_dist_info
from .hooks import HOOKS, Hook
from .log_buffer import LogBuffer
from .priority import Priority, get_priority
from .utils import get_time_str

class BaseRunner(metaclass=ABCMeta):
    
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                )
            warnings.warn('batch_processor is deprecated, please implement '
                          )
            
            if is_module_wrapper(model):
                _model = model.module
            else:
                _model = model
            if hasattr(_model, 'train_step') or hasattr(_model, 'val_step'):
                raise RuntimeError(
                    
                    )
        else:
            assert hasattr(model, 'train_step')

        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        
                        )
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                
                )

        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            )

        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                )

        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                )

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        
        self.log_buffer = LogBuffer()

    @property
    def model_name(self):
        
        return self._model_name

    @property
    def rank(self):
        
        return self._rank

    @property
    def world_size(self):
        
        return self._world_size

    @property
    def hooks(self):
        
        return self._hooks

    @property
    def epoch(self):
        
        return self._epoch

    @property
    def iter(self):
        
        return self._iter

    @property
    def inner_iter(self):
        
        return self._inner_iter

    @property
    def max_epochs(self):
        
        return self._max_epochs

    @property
    def max_iters(self):
        
        return self._max_iters

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass

    def current_lr(self):
        
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                )
        return lr

    def current_momentum(self):
        
        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                )
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg):
        
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name):
        
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def get_hook_info(self):
        
        stage_hook_map = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name
            except ValueError:
                priority = hook.priority
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,
            revise_keys=revise_keys)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault('hook_msgs', {})
        
        self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        if 'config' in checkpoint['meta']:
            config = mmcv.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
                    previous_gpu_ids) != self.world_size:
                self._iter = int(self._iter * len(previous_gpu_ids) /
                                 self.world_size)
                self.logger.info('the iteration number is changed due to '
                                 )

        self.meta = checkpoint['meta']

        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    
                    )

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def register_lr_hook(self, lr_config):
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook, priority='VERY_HIGH')

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = mmcv.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook, priority='HIGH')

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook, priority='NORMAL')

    def register_logger_hooks(self, log_config):
        if log_config is None:
            return
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_timer_hook(self, timer_config):
        if timer_config is None:
            return
        if isinstance(timer_config, dict):
            timer_config_ = copy.deepcopy(timer_config)
            hook = mmcv.build_from_cfg(timer_config_, HOOKS)
        else:
            hook = timer_config
        self.register_hook(hook, priority='LOW')

    def register_custom_hooks(self, custom_config):
        if custom_config is None:
            return

        if not isinstance(custom_config, list):
            custom_config = [custom_config]

        for item in custom_config:
            if isinstance(item, dict):
                self.register_hook_from_cfg(item)
            else:
                self.register_hook(item, priority='NORMAL')

    def register_profiler_hook(self, profiler_config):
        if profiler_config is None:
            return
        if isinstance(profiler_config, dict):
            profiler_config.setdefault('type', 'ProfilerHook')
            hook = mmcv.build_from_cfg(profiler_config, HOOKS)
        else:
            hook = profiler_config
        self.register_hook(hook)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                timer_config=dict(type='IterTimerHook'),
                                custom_hooks_config=None):
        
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_timer_hook(timer_config)
        self.register_logger_hooks(log_config)
        self.register_custom_hooks(custom_hooks_config)
