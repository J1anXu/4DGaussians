_base_ = "./default.py"
OptimizationParams = dict(
    batch_size=2,
    densify_from_iter=0,  # 动态调整开始
    densify_until_iter=0,  # 动态调整结束
    simp_iteration1=14001,  # 删点执行 18000
    admm_start_iter1=16000,  # admm迭代开始  test
    admm_stop_iter1=20000,  # admm迭代结束 18000
    simp_iteration2=20000,  # 删点执行 18000
    iterations=25000,  # 整体迭代 20000
    admm_interval=50,  #
    rho_lr=0.0005,
    opacity_admm_threshold1=0.80,
    opacity_admm_threshold2=0.50,
    coarse_iterations=0,
    important_score_type="opacity", # opacity init_blending_weight
    important_score_3_outdoor=False,
    prune_points = True,# 是否进行删点
    admm=True,

)
