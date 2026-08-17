"""Entry point for the code-RL recipe.

The training loop lives in :mod:`training_sessions_cookbook.rl.train`. This
entry builds a parser restricted to the code_rl environment, so ``--help``
shows exactly this recipe's flags::

    python -m training_sessions_cookbook.recipes.code_rl.train --help

Code-RL grades generated programs by executing them against unit tests in a
local sandbox; start one with ``bash tools/sandbox_up.sh`` first. See this
recipe's README for commands and the recipes README for programmatic use.
"""

from training_sessions_cookbook.rl.train import build_parser, parse_with_config, run


def main() -> None:
    # Parser restricted to this recipe's environment; with a single env it
    # is also the --env_type default, and --help lists only its flags.
    parser = build_parser(envs=("code_rl",))
    args = parse_with_config(parser)
    run(args)


if __name__ == "__main__":
    main()
