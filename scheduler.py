
from torch.optim import lr_scheduler

class PolynomialLR(lr_scheduler._LRScheduler):
  def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
    self.max_epoch = max_epoch
    self.power = power
    super(PolynomialLR, self).__init__(optimizer, last_epoch)

  def _decay_rate(self):
    return (1 - self.last_epoch / self.max_epoch) ** self.power

  def get_lr(self):
    return [group['lr'] * self._decay_rate()
        for group in self.optimizer.param_groups]

class LRRangeTest(lr_scheduler._LRScheduler):
  def __init__(self, optimizer, min_lr, growth_cons, last_epoch=-1):
    super(PolynomialLR, self).__init__(optimizer, last_epoch)
    self.min_lr = min_lr
    self.growth_cons = growth_cons

  def get_lr(self):
    return [group['lr'] * self._decay_rate()
        for group in self.optimizer.param_groups]

