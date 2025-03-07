_base_ = "./default.py"
from constants import *
OptimizationParams = dict(
    batch_size=2,
    densify_from_iter=0,  # 动态调整开始
    densify_until_iter=0,  # 动态调整结束

    simp_iteration1=14001,  # 先删点 14001
    admm_start_iter1=14200,  # 再开始admm  14200
    admm_stop_iter1=14200,  # 停止admm 18200
    simp_iteration2=14200,  # 再删点 18201
    iterations=14200,  # 整体迭代 20200
    
    admm_interval=25,  #
    rho_lr=0.0005,

    opacity_admm_threshold1=0.50,
    opacity_admm_threshold2=0.50,
    
    coarse_iterations=0,


    admm=True,
)
