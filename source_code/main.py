import os
from src.utils import get_cfg
from src.oven import Oven
from loguru import logger
import torch.distributed as dist
from src.utils import set_random_seed, setup_distributed, setup_default_logging, setup_default_logging_wt_dir
import pprint

def main():
    cfg, metadata = get_cfg()
    set_random_seed(cfg["seed"] if cfg["seed"] > 0 else 1, deterministic=False)

    if cfg['distributed']:
        rank, word_size = setup_distributed()
        
        if not os.path.exists(cfg["log_path"]) and rank == 0:
            os.makedirs(cfg["log_path"])

        if rank == 0:
            # curr_timestr = setup_default_logging(cfg["log_path"], False)
            curr_timestr = setup_default_logging_wt_dir(cfg["log_path"])
            cfg["log_path"] = os.path.join(cfg["log_path"], curr_timestr)
            os.makedirs(cfg["log_path"], exist_ok=True)
            csv_path = os.path.join(cfg["log_path"], "out_stat.csv")

            from shutil import copyfile
            output_yaml = os.path.join(cfg["log_path"], "config.yaml")
            copyfile(cfg['config'], output_yaml) 

        else:
            csv_path = None

        if rank == 0:
            logger.info("\n{}".format(pprint.pformat(cfg)))
        
        # make sure all folder are correctly created at rank == 0
        dist.barrier()
    else:
        if not os.path.exists(cfg["log_path"]):
            os.makedirs(cfg["log_path"])
        # curr_timestr = setup_default_logging(cfg["log_path"], False)
        curr_timestr = setup_default_logging_wt_dir(cfg["log_path"])
        cfg["log_path"] = os.path.join(cfg["log_path"], curr_timestr)
        os.makedirs(cfg["log_path"], exist_ok=True)
        csv_path = os.path.join(cfg["log_path"], "info_{}_stat.csv".format(curr_timestr))

        from shutil import copyfile
        output_yaml = os.path.join(cfg["log_path"], "config.yaml")
        copyfile(cfg['config'], output_yaml)

        logger.info("\n{}".format(pprint.pformat(cfg)))

    oven = Oven(cfg, metadata)
    oven.train()
    return


if __name__ == "__main__":
    main()