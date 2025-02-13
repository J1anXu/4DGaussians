_base_ = './default.py'
OptimizationParams = dict(
    {
        "densify_from_iter": 0,  # 动态调整开始
        "densify_until_iter": 0,  # 动态调整结束
        "admm_start_iter1": 14001,  # admm迭代开始
        "admm_stop_iter1": 18000,  # admm迭代结束
        "simp_iteration2": 18000,
        "iterations": 20000,
        "admm_interval": 50,  # 50
        "admm": True,
        "rho_lr": 0.0005,
        "opacity_admm_threshold1": 0.50,
        "opacity_admm_threshold2": 0.50,
        "coarse_iterations": 0,
    }
)