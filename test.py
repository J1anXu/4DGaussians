class RunningMean:
    def __init__(self):
        self.mean = 0.0  # 当前均值
        self.count = 0   # 采样次数

    def update(self, value):
        self.count += 1
        self.mean += (value - self.mean) / self.count  # 更新均值

    def get_mean(self):
        return self.mean

# 示例使用
running_mean = RunningMean()
samples = [10, 20, 30, 40, 50]  # 模拟采样数据

for sample in samples:
    running_mean.update(sample)
    print(f"当前均值: {running_mean.get_mean()}")
