import os.path
import sys
sys.path.append(".")

from RLILT import *
from setting import *
from utility import *
from lithobench.dataset import *
import gc


def main(folder, logger=None, profiler=None):
    Benchmark = "MetalSet"
    ImageSize = FULL_SIZE
    BatchSize = 1
    NJobs = 1
    train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    ilt = RLILT(profiler=profiler, size=FULL_SIZE)
    # ilt.load(r"/storage/d01/daify/rlilt/trivial/103/best_model.pth")

    if MULTIGPU and torch.cuda.device_count() > 1:
        ilt.model_para()

    if PRE_TRAIN is True:
        epoches_reward, epoches_average_min_loss = ilt.pretrain(train_loader, val_loader, epochs=EPOCH, logger=logger, folder=folder)
        ilt.save(os.path.join(folder, 'train.pth'))
        if logger:
            logger.info(f"epoches_reward:{epoches_reward}, epoches_average_min_loss:{epoches_average_min_loss}")
        save_list_json(epoches_reward, os.path.join(folder, 'epoches_reward.json'))
        if logger:
            logger.info(f"epoches_reward is saved at {os.path.join(folder, 'epoches_reward.json')}")
        save_list_json(epoches_average_min_loss, os.path.join(folder, 'epoches_average_min_loss.json'))
        if logger:
            logger.info(f"epoches_average_min_loss is saved at {os.path.join(folder, 'epoches_average_min_loss.json')}")

    ilt.evaluate(targets, finetune=False, folder=folder, shot=False)


if __name__ == "__main__":
    my_folder = CreateFolder()
    global_logger = setup_logger('my_logger', my_folder.get_path())
    global_profiler = CodeProfiler(my_folder.get_path())

    global_logger.info('start')
    global_logger.info(
        f'BATCH_SIZE: {BATCH_SIZE}, UPDATE_TARGET_ITER: {UPDATE_TARGET_ITER}, MAX_TIMESTEP: {MAX_TIMESTEP}, '
        f'EPOCH:{EPOCH}, EPISODE:{EPISODE}, FLIP_SIZE:{FLIP_SIZE}, MEMORY_LENGTH:{MEMORY_LENGTH}, '
        f'EPS_START:{EPS_START}, EPS_END:{EPS_END}, EPS_DECAY:{EPS_DECAY}, FINE_TUNE_EPOCH:{FINE_TUNE_EPOCH}, '
        f'LEARNING_RATE:{LEARNING_RATE}, FINE_TUNE_EPS_START:{FINE_TUNE_EPS_START}, '
        f'FINE_TUNE_EPS_END:{FINE_TUNE_EPS_END}, FINE_TUNE_EPS_DECAY:{FINE_TUNE_EPS_DECAY}, '
        f'L2_LOSS_REWARD_RATIO:{L2_LOSS_REWARD_RATIO}, PVBAND_REWARD_RATIO:{PVBAND_REWARD_RATIO}, '
        f'INFERENCE_EPOCH:{INFERENCE_EPOCH}, OPTIMIZE_MODEL_ITER:{OPTIMIZE_MODEL_ITER}'
    )
    global_logger.info(
        f'KERNEL_SIZE: {KERNEL_SIZE}, FLIP_CONSISTENCE:{FLIP_CONSISTENCE}, ACTION_MASK:{ACTION_MASK}, '
        f'PRE_TRAIN:{PRE_TRAIN}, FINE_TUNE:{FINE_TUNE}, BINARY_IMG:{BINARY_IMG}, '
        f'FINE_TUNE_AUGMENTATION:{FINE_TUNE_AUGMENTATION}, MASK_SAVE_BY:{MASK_SAVE_BY}, '
        f'DOUBLE_DQN:{DOUBLE_DQN}, MULTIGPU:{MULTIGPU}'
    )

    gc.enable()

    if gc.isenabled():
        global_logger.info("Garbage collection is enabled")
    else:
        global_logger.info("Garbage collection is disabled")

    main(my_folder.get_path(), global_logger, global_profiler)

    global_profiler.report()