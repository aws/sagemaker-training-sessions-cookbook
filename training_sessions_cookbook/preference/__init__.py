"""KTO preference-alignment training loop over SageMaker Training Sessions.

Peer of :mod:`training_sessions_cookbook.sft` and
:mod:`training_sessions_cookbook.rl`. KTO (Kahneman-Tversky Optimization)
aligns a model from a **binary** desirable/undesirable signal - no paired
preferences required - and is trained here through the stock ``forward`` /
``forward_backward`` service API via the surrogate-gradient trick, with no
custom server-side loss kernel.

* :mod:`.kto` - the pure KTO arithmetic (loss, per-sequence gradient, KL point).
* :mod:`.data` - preference-dataset loading, rendering, and datum shaping.
* :mod:`.train` - the training loop (``python -m
  training_sessions_cookbook.preference.train``).
"""
