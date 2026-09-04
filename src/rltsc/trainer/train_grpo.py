from __future__ import annotations


import os


from importlib import import_module


import hydra


import ray


from verl.experimental.reward_loop import migrate_legacy_reward_impl


from verl.trainer.main_ppo import TaskRunner, run_ppo


from verl.utils.device import auto_set_device


class ProcessLightTaskRunner(TaskRunner):
    def run(self, config):
        import_module("rltsc.trainer.advantage")

        return super().run(config)


@hydra.main(config_path="configs", config_name="grpo_processlight", version_base=None)
def main(config) -> None:
    modules = os.environ["VERL_USE_EXTERNAL_MODULES"] if "VERL_USE_EXTERNAL_MODULES" in os.environ else ""

    module_list = [item for item in modules.split(",") if item]

    if "rltsc.trainer.advantage" not in module_list:
        module_list.append("rltsc.trainer.advantage")

    os.environ["VERL_USE_EXTERNAL_MODULES"] = ",".join(module_list)

    import_module("rltsc.trainer.advantage")

    auto_set_device(config)

    config = migrate_legacy_reward_impl(config)

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ProcessLightTaskRunner))


if __name__ == "__main__":
    main()
