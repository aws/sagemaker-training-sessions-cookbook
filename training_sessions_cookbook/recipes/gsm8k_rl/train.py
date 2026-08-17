"""Entry point for the GSM8K math-RL recipe.

The training loop lives in :mod:`training_sessions_cookbook.rl.train`. This
entry builds a parser restricted to the gsm8k environment, so ``--help``
shows exactly this recipe's flags::

    python -m training_sessions_cookbook.recipes.gsm8k_rl.train --help

To drive the same loop from your own code (a script or a notebook), build the
args, override what you need, and call the loop directly. The example below
supplies the one required flag; everything else keeps its default::

    from training_sessions_cookbook.rl.train import build_parser, parse_with_config, run_async

    args = parse_with_config(
        build_parser(envs=("gsm8k",)),
        argv=["--base_model", "openai/gpt-oss-20b"],
    )
    args.base_model_arn = "arn:aws:sagemaker:..."   # your account values
    args.role_arn = "arn:aws:iam::..."
    args.max_steps = 10
    await run_async(args)      # notebooks / async code; use run(args) in scripts

See the recipes README for registering a custom environment.
"""

from training_sessions_cookbook.rl.train import build_parser, parse_with_config, run


def main() -> None:
    # Parser restricted to this recipe's environment; with a single env it
    # is also the --env_type default, and --help lists only its flags.
    parser = build_parser(envs=("gsm8k",))
    args = parse_with_config(parser)
    run(args)


if __name__ == "__main__":
    main()
