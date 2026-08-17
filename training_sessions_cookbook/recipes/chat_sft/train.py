"""Entry point for the chat SFT recipe.

The training loop lives in :mod:`training_sessions_cookbook.sft.train`. This
module exists so every recipe is launched the same way::

    python -m training_sessions_cookbook.recipes.chat_sft.train --help

To drive the same loop from your own code, build the args, override what you
need, and call the loop directly. ``train_batches`` and ``eval_batches``
bypass ``--dataset_path`` entirely; each accepts a ready list of datum
batches or a callable ``(tokenizer) -> batches``, because the
service-resolved tokenizer only exists once the session opens::

    from training_sessions_cookbook.sft.train import build_parser, run_async

    args = build_parser().parse_args([
        "--base_model_arn", "arn:aws:sagemaker:...",
        "--model_package_group_arn", "arn:aws:sagemaker:...",
        "--role_arn", "arn:aws:iam::...",
        "--output_s3_path", "s3://my-bucket/output/",
    ])
    args.max_steps = 3
    await run_async(args, train_batches=my_batches)   # use run(...) in scripts
"""

from training_sessions_cookbook.sft.train import build_parser, run


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
