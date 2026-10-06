"""Entry point for the SDPO (self-distillation policy optimization) code recipe.

The student writes solutions to DeepCoder problems, graded in the sandbox; the
student's frozen base, shown the sandbox's feedback on the attempt and/or a
sibling rollout that passed, scores every sampled token. The loop
lives in :mod:`training_sessions_cookbook.distill.train`::

    python -m training_sessions_cookbook.recipes.sdpo_code.train --help

Needs the ``code-rl`` extras and a SandboxFusion endpoint (see the code_rl
recipe README).
"""

from training_sessions_cookbook.distill.train import main as distill_main


def main() -> None:
    distill_main("sdpo", envs=("code_rl",))


if __name__ == "__main__":
    main()
