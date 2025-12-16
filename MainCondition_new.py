from DiffusionFreeGuidence.TrainCondition import train, eval
import os

def main(model_config=None):
    modelConfig = {
        "state": "eval", # or eval or train
        "epoch": 1000,
        "batch_size": 12,
        "T": 1000,
        "channel": 32,
        "channel_mult": [1, 2, 2, 4],
        "num_res_blocks": 1,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 1.5,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda",
        "w": 1.8,
        "save_dir": "D:/Pycharm_Professional/MCDDPM-main/check-xiaorong",
        "training_load_weight": None,
        "test_load_weight": "final_model_after_1000_epochs.pt",
        "nrow":12,
        "porosity":0.254,
        "pore_mean":3,
        "por_std":2,
        "throat_mean":7,
        "throat_std":2,
        "coord_mean":5,
        "coord_std":2,
        "path":"D:/Pycharm_Professional/MCDDPM-main/data/xiaorong"
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
    # os.system("/usr/bin/shutdown")
