_base_ = "./default.py"
OptimizationParams = dict(
    batch_size=2,

    coarse_iterations=0,
    densify_from_iter=0,  # 动态调整开始
    densify_until_iter=0,  # 动态调整结束

    simp_iteration1=14001,  # admm之前的硬阈值删点执行
    admm_start_iter1=16000,  # admm迭代开始
    admm_stop_iter1=25000,  # admm迭代结束
    simp_iteration2=25000,  # admm软阈值删点
    quant_start_iter = 25000, # quant迭代开始
    quant_stop_iter = 27000, # quant迭代结束
    iterations=27000,  # 整体迭代次数

    admm=True, #是否admm迭代
    admm_interval=50,
    rho_lr=0.0005,

    prune_points=True,  # 是否进行删点
    opacity_admm_threshold1=0.8,  # 第一次删点比例
    opacity_admm_threshold2=0.5, # 第二次删点比例

    outdoor=False,
    related_gs_num=1,
    simp_iteration1_score_type = 0,
    score_function = 36,
    add_extra_scores = False,

    quant = True,
    kmeans_ncls = 4096,
    kmeans_ncls_sh = 4096,
    kmeans_ncls_dc = 4096,
    kmeans_iters = 10,
    kmeans_freq = 100,
    grad_thresh = 0.002,
    quant_params = ['sh', 'dc', 'scale', 'rot']
    )