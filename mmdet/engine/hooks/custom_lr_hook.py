from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class CustomLrSchedulerHook(Hook):
    def __init__(self, epochs, values):
        assert len(epochs) == len(values), "Epochs and values must have the same length"
        self.epochs = epochs
        self.values = values

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch
        if current_epoch in self.epochs:
            index = self.epochs.index(current_epoch)
            new_lr = self.values[index]
            if hasattr(runner, 'optim_wrapper'):
                for param_group in runner.optim_wrapper.optimizer.param_groups:
                    param_group['lr'] = new_lr
                runner.logger.info(f'Setting learning rate to {new_lr} at epoch {current_epoch}')
            else:
                runner.logger.warning('Optimizer attribute not found in runner')