"""Entry point for the OPSD (on-policy self-distillation) math recipe.

The student samples GSM8K solutions without help; its own frozen base model,
shown the dataset's worked solution as privileged context, scores every sampled
token. No second model is needed. The loop lives in
:mod:`training_sessions_cookbook.distill.train`::

    python -m training_sessions_cookbook.recipes.opsd_math.train --help

Programmatic use::

    from training_sessions_cookbook.distill.train import build_parser, parse_with_config, run_async

    args = parse_with_config(
        build_parser("opsd", envs=("gsm8k",)),
        argv=["--base_model", "Qwen/Qwen3.6-27B",
              "--dataset_path", "data/gsm8k_train.jsonl"],
    )
    await run_async(args)      # use run(args) in scripts
"""

from training_sessions_cookbook.distill.train import main as distill_main


def main() -> None:
    distill_main("opsd", envs=("gsm8k",))


if __name__ == "__main__":
    main()
