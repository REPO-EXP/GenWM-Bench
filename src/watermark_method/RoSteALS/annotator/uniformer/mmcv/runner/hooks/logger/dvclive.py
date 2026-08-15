
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook

@HOOKS.register_module()
class DvcliveLoggerHook(LoggerHook):
    
    def __init__(self,
                 path,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):

        super(DvcliveLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.path = path
        self.import_dvclive()

    def import_dvclive(self):
        try:
            import dvclive
        except ImportError:
            raise ImportError(
                )
        self.dvclive = dvclive

    @master_only
    def before_run(self, runner):
        self.dvclive.init(self.path)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            for k, v in tags.items():
                self.dvclive.log(k, v, step=self.get_iter(runner))
