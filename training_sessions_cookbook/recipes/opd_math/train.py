"""Entry point for the OPD (on-policy distillation) math recipe.

The student samples GSM8K solutions; a teacher scores every sampled token, and
the per-token log-prob gap is the advantage. The preset uses the student's own
frozen base, sharpened with ``--teacher_temperature``; ``--teacher checkpoint``
or ``--teacher session`` swap in a stronger teacher. The loop lives
in :mod:`training_sessions_cookbook.distill.train`::

    python -m training_sessions_cookbook.recipes.opd_math.train --help

Programmatic use::

    from training_sessions_cookbook.distill.train import build_parser, parse_with_config, run_async

    args = parse_with_config(
        build_parser("opd", envs=("gsm8k",)),
        argv=["--base_model", "Qwen/Qwen3.6-27B",
              "--teacher", "base", "--teacher_temperature", "0.5"],
    )
    args.max_steps = 10
    await run_async(args)      # use run(args) in scripts
"""

from training_sessions_cookbook.distill.train import main as distill_main


def main() -> None:
    distill_main("opd", envs=("gsm8k",))


if __name__ == "__main__":
    main()
