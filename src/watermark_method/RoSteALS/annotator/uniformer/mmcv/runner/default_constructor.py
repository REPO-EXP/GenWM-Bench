from .builder import RUNNER_BUILDERS, RUNNERS

@RUNNER_BUILDERS.register_module()
class DefaultRunnerConstructor:
    
    def __init__(self, runner_cfg, default_args=None):
        if not isinstance(runner_cfg, dict):
            raise TypeError('runner_cfg should be a dict',
                            )
        self.runner_cfg = runner_cfg
        self.default_args = default_args

    def __call__(self):
        return RUNNERS.build(self.runner_cfg, default_args=self.default_args)
