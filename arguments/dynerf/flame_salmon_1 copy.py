_base_ = './default.py'
OptimizationParams = dict(
    {
        "densify_from_iter" : 100,
        "densify_until_iter" : 200,
        "admm_start_iter1":200,
        "admm_stop_iter1":500,
        "simp_iteration2":500,
        "iterations":600,
        "admm_interval" : 1, # 50
        "admm":True,
        "rho_lr": 0.0005,
        "opacity_admm_threshold1":0.50,
        "opacity_admm_threshold2": 0.8,
        "coarse_iterations" : 10

    }
)