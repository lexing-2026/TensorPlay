import numpy as np


class Scheduler:
    """学习率调度器基类"""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [optimizer.lr]  # 保存初始学习率

    def step(self):
        """更新学习率"""
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        """计算当前学习率，子类必须重写此方法"""
        raise NotImplementedError


class StepLR(Scheduler):
    """固定步长学习率调度器"""

    def __init__(self, optimizer, step_size=30, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 每过step_size个epoch，学习率乘以gamma
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))


class MultiStepLR(Scheduler):
    """多步学习率调度器"""

    def __init__(self, optimizer, milestones=None, gamma=0.1, last_epoch=-1):
        if milestones is None:
            milestones = [30, 60, 90]
        self.milestones = set(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 在指定milestones中的epoch，学习率乘以gamma
        if self.last_epoch in self.milestones:
            return self.optimizer.lr * self.gamma
        return self.optimizer.lr


class ExponentialLR(Scheduler):
    """指数衰减学习率调度器"""

    def __init__(self, optimizer, gamma=0.99, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 学习率按指数规律衰减: lr = lr * gamma^epoch
        return self.base_lrs[0] * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(Scheduler):
    """余弦退火学习率调度器"""

    def __init__(self, optimizer, t_max=10, eta_min=0, last_epoch=-1):
        self.T_max = t_max  # 周期长度
        self.eta_min = eta_min  # 最小学习率
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 学习率按余弦曲线周期性变化
        return self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * \
            (1 + np.cos(np.pi * self.last_epoch / self.T_max))


class ReduceLROnPlateau:
    """自适应调度器，当指标停止改善时降低学习率"""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', min_lr=0):
        self.optimizer = optimizer
        self.mode = mode  # min表示指标越小越好，max表示指标越大越好
        self.factor = factor  # 学习率衰减因子
        self.patience = patience  # 容忍多少个epoch没有改善
        self.threshold = threshold  # 改善的阈值
        self.threshold_mode = threshold_mode  # rel表示相对变化，abs表示绝对变化
        self.min_lr = min_lr  # 最小学习率
        self.best = None
        self.num_bad_epochs = 0  # 记录连续没有改善的epoch数

    def step(self, metric):
        """
        根据监测指标更新学习率
        metric: 需要监测的指标值
        """
        if self.best is None:
            self.best = metric
            return

        if self.threshold_mode == 'rel':
            if self.mode == 'min':
                # 对于最小值指标，新值需要小于 best * (1 - threshold)才算改善
                improvement_threshold = self.best * (1 - self.threshold)
            else:
                # 对于最大值指标，新值需要大于 best * (1 + threshold)才算改善
                improvement_threshold = self.best * (1 + self.threshold)
        else:
            if self.mode == 'min':
                # 对于最小值指标，新值需要小于 best - threshold才算改善
                improvement_threshold = self.best - self.threshold
            else:
                # 对于最大值指标，新值需要大于 best + threshold才算改善
                improvement_threshold = self.best + self.threshold

        if (self.mode == 'min' and metric < improvement_threshold) or \
                (self.mode == 'max' and metric > improvement_threshold):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            # 如果超过容忍次数，则降低学习率
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0


class EarlyStopping:
    """早停机制，用于监测验证集指标，连续多轮无改善则触发早停"""

    def __init__(self, patience=10, delta=1e-4, mode='min', verbose=False, save=True,
                 path='best_model.tp'):
        self.patience = patience  # 容忍连续无改善的轮次
        self.delta = delta  # 指标改善的最小阈值（避免微小波动被判定为改善）
        self.mode = mode  # min：指标越小越好（如损失）；max：指标越大越好（如准确率）
        self.verbose = verbose  # 是否打印早停相关日志

        self.save = save  # 是否保存性能最优的模型
        self.path = path  # 最优模型保存路径

        self.best_score = None  # 记录历史最优指标值
        self.num_bad_epochs = 0  # 连续无改善的轮次计数
        self.early_stop = False  # 是否触发早停的标志

    def __call__(self, val_metric, model=None):
        """
        每轮验证后调用，判断是否触发早停
        val_metric: 当前轮次的验证集指标（如val_loss、val_acc）
        model: 当前训练的模型实例（需支持state_dict()方法，仅当save_best_model=True时需传入）
        """
        # 1. 计算当前指标对应的得分（统一转为最小化逻辑，方便比较）
        current_score = -val_metric if self.mode == 'min' else val_metric

        # 2. 初始化历史最优得分（第一轮调用时）
        if self.best_score is None:
            self.best_score = current_score
            return

        # 3. 判断当前指标是否有效改善：当前得分 > 历史最优得分 + delta（delta避免微小波动）
        if current_score > self.best_score + self.delta:
            self.best_score = current_score  # 更新历史最优得分
            self._save_best_model(val_metric, model)  # 保存新的最优模型
            self.num_bad_epochs = 0  # 重置连续无改善计数
        else:
            self.num_bad_epochs += 1  # 累加连续无改善计数
            if self.verbose:
                print(
                    f"EarlyStopping: consecutive {self.num_bad_epochs} epoches has no improvement(current: {val_metric:.6f})")

            if self.num_bad_epochs >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarlyStopping: consecutive {self.patience} epoches has no improvement, early stop!")
                    score = (-self.best_score) if self.mode == 'min' else self.best_score
                    print(f"EarlyStopping: best model's metric on validation set: {score:.6f}")

    def _save_best_model(self, val_metric, model):
        """保存最优模型参数（仅当开启保存功能且传入模型时）"""
        if self.save and model is not None:
            model.save(self.path)
            if self.verbose:
                print(f"EarlyStopping: find a better model(metric: {val_metric:.6f}), save to {self.path}")
