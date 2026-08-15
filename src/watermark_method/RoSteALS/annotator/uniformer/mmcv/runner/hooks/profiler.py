
import warnings
from typing import Callable, List, Optional, Union

import torch

from ..dist_utils import master_only
from .hook import HOOKS, Hook

@HOOKS.register_module()
class ProfilerHook(Hook):
    
    def __init__(self,
                 by_epoch: bool = True,
                 profile_iters: int = 1,
                 activities: List[str] = ['cpu', 'cuda'],
                 schedule: Optional[dict] = None,
                 on_trace_ready: Optional[Union[Callable, dict]] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:
        try:
            from torch import profiler  
        except ImportError:
            raise ImportError('profiler is the new feature of torch1.8.1, '
                              )

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean.'
        self.by_epoch = by_epoch

        if profile_iters < 1:
            raise ValueError('profile_iters should be greater than 0, but got '
                             )
        self.profile_iters = profile_iters

        if not isinstance(activities, list):
            raise ValueError(
                )
        self.activities = []
        for activity in activities:
            activity = activity.lower()
            if activity == 'cpu':
                self.activities.append(profiler.ProfilerActivity.CPU)
            elif activity == 'cuda':
                self.activities.append(profiler.ProfilerActivity.CUDA)
            else:
                raise ValueError(
                    )

        if schedule is not None:
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.json_trace_path = json_trace_path

    @master_only
    def before_run(self, runner):
        if self.by_epoch and runner.max_epochs < self.profile_iters:
            raise ValueError('self.profile_iters should not be greater than '
                             )

        if not self.by_epoch and runner.max_iters < self.profile_iters:
            raise ValueError('self.profile_iters should not be greater than '
                             )

        if callable(self.on_trace_ready):  
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):  
            trace_cfg = self.on_trace_ready.copy()
            trace_type = trace_cfg.pop('type')  
            if trace_type == 'log_trace':

                def _log_handler(prof):
                    print(prof.key_averages().table(**trace_cfg))

                _on_trace_ready = _log_handler
            elif trace_type == 'tb_trace':  
                try:
                    import torch_tb_profiler  
                except ImportError:
                    raise ImportError('please run "pip install '
                                      
                                      )
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    **trace_cfg)
            else:
                raise ValueError('trace_type should be "log_trace" or '
                                 )
        elif self.on_trace_ready is None:
            _on_trace_ready = None  
        else:
            raise ValueError('on_trace_ready should be handler, dict or None, '
                             )

        if runner.max_epochs > 1:
            warnings.warn(f'profiler will profile {runner.max_epochs} epochs '
                          
                          )

        self.profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=_on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops)

        self.profiler.__enter__()
        runner.logger.info('profiler is profiling...')

    @master_only
    def after_train_epoch(self, runner):
        if self.by_epoch and runner.epoch == self.profile_iters - 1:
            runner.logger.info('profiler may take a few minutes...')
            self.profiler.__exit__(None, None, None)
            if self.json_trace_path is not None:
                self.profiler.export_chrome_trace(self.json_trace_path)

    @master_only
    def after_train_iter(self, runner):
        self.profiler.step()
        if not self.by_epoch and runner.iter == self.profile_iters - 1:
            runner.logger.info('profiler may take a few minutes...')
            self.profiler.__exit__(None, None, None)
            if self.json_trace_path is not None:
                self.profiler.export_chrome_trace(self.json_trace_path)
