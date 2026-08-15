import os.path as osp

from annotator.uniformer.mmcv.runner import DistEvalHook as _DistEvalHook
from annotator.uniformer.mmcv.runner import EvalHook as _EvalHook

class EvalHook(_EvalHook):
    
    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def after_train_iter(self, runner):
        
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from annotator.uniformer.mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from annotator.uniformer.mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

class DistEvalHook(_DistEvalHook):
    
    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def after_train_iter(self, runner):
        
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from annotator.uniformer.mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from annotator.uniformer.mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
