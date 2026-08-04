import logging  # 导入 logging 模块,用于记录日志信息

import click  # 导入 click 库,用于构建命令行接口(CLI)

from chemsmart.cli.gaussian.gaussian import gaussian  # 导入 gaussian 命令组,作为 opt 子命令的父级组

# Import and register qmmm subcommand
# 导入并注册 qmmm(量子力学/分子力学)子命令的创建函数
from chemsmart.cli.gaussian.qmmm import create_qmmm_subcommand
from chemsmart.cli.job import click_job_options  # 导入通用的作业命令行选项装饰器
from chemsmart.utils.cli import MyGroup  # 导入自定义的 click Group 类,用于支持特殊命令组行为
from chemsmart.utils.utils import check_charge_and_multiplicity  # 导入校验电荷与自旋多重度合法性的工具函数

logger = logging.getLogger(__name__)  # 创建本模块的 logger 实例,日志名以当前模块名命名


@gaussian.group("opt", cls=MyGroup, invoke_without_command=True)  # 在 gaussian 命令组下注册名为 opt 的子命令组;使用自定义 MyGroup 类,并允许不带子命令直接调用
@click_job_options  # 应用通用的作业选项(如电荷、多重度、输入文件等)
@click.option(  # 添加一个 click 命令行选项
    "-f",  # 短选项名
    "--freeze-atoms",  # 长选项名
    type=str,  # 选项值为字符串类型
    help="Indices of atoms to freeze for constrained optimization. 1-indexed.",  # 帮助文本:冻结原子索引(从 1 开始),用于约束优化
)
@click.pass_context  # 将 click 的上下文对象 ctx 传入函数,以便访问全局共享数据
def opt(ctx, freeze_atoms, skip_completed, **kwargs):  # 定义 opt 子命令的处理函数,参数包括上下文、冻结原子、跳过已完成作业标志及其他关键字参数
    """CLI subcommand for running Gaussian optimization calculation."""  # 函数文档字符串:用于运行 Gaussian 结构优化计算的 CLI 子命令

    # get jobrunner for optimization
    jobrunner = ctx.obj["jobrunner"]  # 从上下文中取出作业执行器 jobrunner,负责实际提交和运行作业

    # get settings from project
    project_settings = ctx.obj["project_settings"]  # 从上下文中取出项目级配置对象
    opt_settings = project_settings.opt_settings()  # 从项目配置中获取优化作业专用的设置对象

    # job setting from filename or default, with updates from user in cli
    # specified in keywords
    # e.g., `sub.py gaussian -c <user_charge> -m <user_multiplicity>`
    job_settings = ctx.obj["job_settings"]  # 从上下文中取出基于文件名或默认值生成的作业设置
    keywords = ctx.obj["keywords"]  # 从上下文中取出用户在 CLI 中输入的关键字(如计算方法、基组等)

    # merge project opt settings with job settings from cli keywords from
    # cli.gaussian.py subcommands
    opt_settings = opt_settings.merge(job_settings, keywords=keywords)  # 将项目优化设置与 CLI 传入的作业设置、关键字合并,得到最终生效的设置

    # get molecules
    molecules = ctx.obj["molecules"]  # 从上下文中取出待处理的分子对象列表

    # get label for the job
    label = ctx.obj["label"]  # 从上下文中取出本次作业的标签(用于命名输出文件等)

    # Set atoms to freeze

    from chemsmart.utils.utils import (  # 延迟导入工具函数,避免循环依赖或不必要的启动开销
        convert_list_to_gaussian_frozen_list,  # 将原子索引列表转换为 Gaussian 冻结原子格式
        get_list_from_string_range,  # 将字符串形式的范围(如 "1-3,5")解析为整数列表
    )

    logger.info(f"Opt job settings from project: {opt_settings.__dict__}")  # 记录日志:打印最终合并后的优化作业设置内容

    # Store parent context for potential qmmm subcommand
    ctx.obj["parent_skip_completed"] = skip_completed  # 将 skip_completed 存入上下文,供后续 qmmm 子命令使用
    ctx.obj["parent_freeze_atoms"] = freeze_atoms  # 将冻结原子设置存入上下文,供后续 qmmm 子命令使用
    ctx.obj["parent_kwargs"] = kwargs  # 将其他关键字参数存入上下文,供后续 qmmm 子命令使用
    ctx.obj["parent_settings"] = opt_settings  # 将合并后的优化设置存入上下文,供后续 qmmm 子命令使用
    ctx.obj["parent_jobtype"] = "opt"  # 标记父命令的作业类型为 opt,供后续 qmmm 子命令识别

    from chemsmart.jobs.gaussian.opt import GaussianOptJob  # 延迟导入 Gaussian 优化作业类,用于构建具体作业对象

    # Get the original molecule indices from context
    molecule_indices = ctx.obj["molecule_indices"]  # 从上下文中取出分子的原始索引列表(用于多分子场景下区分每个分子)

    if ctx.invoked_subcommand is None:  # 若未调用任何子命令(即直接执行 opt 而非 opt qmmm),则执行以下逻辑
        check_charge_and_multiplicity(opt_settings)  # 校验电荷与自旋多重度是否合法(例如电子数与多重度是否匹配)

        # Handle multiple molecules: create one job per molecule
        if len(molecules) > 1 and molecule_indices is not None:  # 当存在多个分子且提供了分子索引时,为每个分子创建一个独立作业
            logger.info(f"Creating {len(molecules)} optimization jobs")  # 记录日志:即将创建多个优化作业
            jobs = []  # 初始化作业列表,用于收集创建好的作业对象
            for molecule, idx in zip(molecules, molecule_indices):  # 遍历每个分子及其对应的索引
                # Create a copy to avoid side effects from mutation
                molecule = molecule.copy()  # 复制分子对象,避免后续修改影响原始数据
                molecule_label = f"{label}_idx{idx}"  # 为当前分子生成专属标签(在原标签后附加索引)
                logger.info(  # 记录日志:打印当前正在优化的分子信息
                    f"Optimizing molecule {idx}: {molecule} with label {molecule_label}"
                )

                # Apply frozen atoms if specified
                if freeze_atoms is not None:  # 如果用户指定了要冻结的原子
                    frozen_atoms_list = get_list_from_string_range(  # 将用户输入的字符串范围解析为整数列表
                        freeze_atoms
                    )
                    logger.debug(f"Freezing atoms: {frozen_atoms_list}")  # 调试日志:打印即将冻结的原子索引列表
                    molecule.frozen_atoms = (  # 将转换后的 Gaussian 冻结原子格式赋值给分子的 frozen_atoms 属性
                        convert_list_to_gaussian_frozen_list(
                            frozen_atoms_list, molecule
                        )
                    )
                else:  # 若未指定冻结原子
                    logger.debug("No atoms will be frozen during optimization")  # 调试日志:说明优化过程中不冻结任何原子

                job = GaussianOptJob(  # 创建一个 Gaussian 优化作业对象
                    molecule=molecule,  # 指定待优化的分子
                    settings=opt_settings,  # 指定优化作业设置
                    label=molecule_label,  # 指定作业标签
                    jobrunner=jobrunner,  # 指定作业执行器
                    skip_completed=skip_completed,  # 指定是否跳过已完成的作业
                    **kwargs,  # 透传其他关键字参数
                )
                jobs.append(job)  # 将创建好的作业加入作业列表
            return jobs  # 返回所有创建的作业列表
        else:  # 单分子场景
            # Single molecule case
            molecule = molecules[-1]  # 取分子列表中的最后一个分子作为待优化对象
            molecule = molecule.copy()  # 复制分子对象,避免修改原始数据
            logger.info(f"Optimizing molecule: {molecule}.")  # 记录日志:打印正在优化的分子信息

            if freeze_atoms is not None:  # 如果用户指定了要冻结的原子
                frozen_atoms_list = get_list_from_string_range(freeze_atoms)  # 将字符串范围解析为整数列表
                logger.debug(f"Freezing atoms: {frozen_atoms_list}")  # 调试日志:打印即将冻结的原子索引列表
                molecule.frozen_atoms = convert_list_to_gaussian_frozen_list(  # 将冻结列表转换为 Gaussian 格式并赋值给分子
                    frozen_atoms_list, molecule
                )
            else:  # 若未指定冻结原子
                logger.debug("No atoms will be frozen during optimization")  # 调试日志:说明优化过程中不冻结任何原子

            return GaussianOptJob(  # 创建并返回单个 Gaussian 优化作业对象
                molecule=molecule,  # 指定待优化的分子
                settings=opt_settings,  # 指定优化作业设置
                label=label,  # 指定作业标签
                jobrunner=jobrunner,  # 指定作业执行器
                skip_completed=skip_completed,  # 指定是否跳过已完成的作业
                **kwargs,  # 透传其他关键字参数
            )


create_qmmm_subcommand(opt)  # 在 opt 命令组上注册 qmmm 子命令,使 opt 命令组支持 opt qmmm 形式的调用
