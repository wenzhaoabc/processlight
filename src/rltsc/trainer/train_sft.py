from __future__ import annotations


import hydra


from verl.trainer.sft_trainer import run_sft


from verl.utils.device import auto_set_device


@hydra.main(config_path="configs", config_name="sft", version_base=None)
def main(config) -> None:
    auto_set_device(config)

    run_sft(config)


if __name__ == "__main__":
    main()
