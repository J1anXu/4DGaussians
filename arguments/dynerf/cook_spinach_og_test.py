_base_ = "./default.py"
OptimizationParams = dict(
    batch_size=2,
    densify_from_iter=0,  # 动态调整开始
    densify_until_iter=0,  # 动态调整结束
    simp_iteration1=14000,  # 删点执行
    admm_start_iter1=16000,  # admm迭代开始
    admm_stop_iter1=25000,  # admm迭代结束 19400
    simp_iteration2=25000,  # 删点执行 19401
    iterations=14002,  # 整体迭代
    admm_interval=50,
    rho_lr=0.0005,
    opacity_admm_threshold1=0.5,  # 0.8
    opacity_admm_threshold2=0.5,
    coarse_iterations=0,
    outdoor=False,
    prune_points=False,  # 是否进行删点
    admm=True,
    related_gs_num=1,
    simp_iteration1_score_type = 0,
    score_function = 36,
    )

