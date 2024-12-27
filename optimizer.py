from torch.optim import AdamW, Adam


def separate_weight_decayable_params(params):
    """
    将模型参数分为需要权重衰减的参数和不需要权重衰减的参数。

    参数:
        params (List[torch.nn.Parameter]): 所有的模型参数列表。

    返回:
        Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]: 
            - wd_params (List[torch.nn.Parameter]): 需要权重衰减的参数列表。
            - no_wd_params (List[torch.nn.Parameter]): 不需要权重衰减的参数列表。
    """
    # 初始化两个空列表
    wd_params, no_wd_params = [], []

    for param in params:
        # 如果参数的维度小于2（例如偏置），则添加到不需要权重衰减的参数列表中
        # 否则，添加到需要权重衰减的参数列表中
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    # 返回分类后的参数列表
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True
):
    """
    获取优化器实例，支持根据参数是否需要权重衰减进行分组。

    参数:
        params (List[torch.nn.Parameter]): 模型参数列表。
        lr (float, 可选): 学习率，默认为1e-4。
        wd (float, 可选): 权重衰减系数，默认为1e-2。
        betas (Tuple[float, float], 可选): Adam 优化器的 beta 参数，默认为 (0.9, 0.99)。
        eps (float, 可选): Adam 优化器的 epsilon 参数，默认为1e-8。
        filter_by_requires_grad (bool, 可选): 是否仅过滤需要梯度的参数，默认为 False。
        group_wd_params (bool, 可选): 是否将参数分组以应用不同的权重衰减，默认为 True。

    返回:
        Union[Adam, AdamW]: 返回 Adam 或 AdamW 优化器实例。
    """
    # 检查是否需要权重衰减
    has_wd = wd > 0

    if filter_by_requires_grad:
        # 仅保留需要梯度的参数
        params = list(filter(lambda t: t.requires_grad, params))

    if group_wd_params and has_wd:
        # 分离需要和不需要权重衰减的参数
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        # 准备参数组
        params = [
            {'params': wd_params},  # 需要权重衰减的参数组
            {'params': no_wd_params, 'weight_decay': 0},  # 不需要权重衰减的参数组，weight_decay 设置为0
        ]

    if not has_wd:
        # 如果不需要权重衰减，则使用 Adam 优化器
        return Adam(params, lr = lr, betas = betas, eps = eps)
    
    # 如果需要权重衰减，则使用 AdamW 优化器
    return AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)
