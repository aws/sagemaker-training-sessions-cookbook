"""Entry point for the KTO preference-alignment recipe.

The training loop lives in :mod:`training_sessions_cookbook.preference.train`.
This module exists so every recipe is launched the same way::

    python -m training_sessions_cookbook.recipes.preference_kto.train --help

To drive the same loop from your own code, build the args, override what you
need, and call the loop directly. ``train_batches`` bypasses ``--dataset_path``
entirely; it accepts a ready list of ``KtoExample`` batches or a callable
``(tokenizer) -> batches``, because the service-resolved tokenizer only exists
once the session opens::

    from training_sessions_cookbook.preference.train import build_parser, run_async

    args = build_parser().parse_args([
        "--base_model_arn", "arn:aws:sagemaker:...",
        "--model_package_group_arn", "arn:aws:sagemaker:...",
        "--role_arn", "arn:aws:iam::...",
        "--output_s3_path", "s3://my-bucket/output/",
    ])
    args.max_steps = 10
    await run_async(args)   # use run(...) in scripts
"""

from training_sessions_cookbook.preference.train import parse_args, run


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
