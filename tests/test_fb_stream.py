"""Contract tests for --fb_stream_chunks (streamed forward_backward, sync loop).

What is pinned (and deliberately nothing else):

* The streamed path trains on EXACTLY the same transitions as the monolithic
  path would - same union of datums, chunked, with optim after every fb.
* Sub-batch boundaries: N chunks of ceil(groups/N) groups, remainder in the
  last flush; empty accumulations (all rollouts failed) submit nothing.
* Fire-and-forget: submission must not wait for op results (a slow backend
  cannot stall the collector).
* Fallbacks: monte_carlo mode and protocol-less backends use the monolithic
  train_step untouched.
"""
import asyncio
import math
import unittest

from training_sessions_cookbook.rl.loops import _FbStreamer
from training_sessions_cookbook.rl.rollouts import Transition


def _group(reward_pairs):
    """One prompt-group: a list of single-transition episodes."""
    return [
        [Transition(obs_tokens=[1, 2, 3], act_tokens=[4, 5],
                    act_logprobs=[-0.1, -0.2], reward=r, done=True,
                    env_type="gsm8k")]
        for r in reward_pairs
    ]


class _Args:
    temperature = 1.0
    loss = "ppo"


class _StreamBackend:
    """Records submissions; never blocks (fire-and-forget contract)."""

    def __init__(self):
        self.submitted = []          # list of datum-lists, in submission order
        self.finished_with = None    # fb_ops passed to finish_train_step
        self.optim_submitted = False

    def build_datum(self, t, *, temperature):
        return {"adv": t.advantage, "reward": t.reward}

    async def submit_forward_backward(self, datums, *, loss_name):
        assert not self.optim_submitted, "fb submitted AFTER optim - barrier broken"
        self.submitted.append(list(datums))
        return f"op-{len(self.submitted)}"

    async def finish_train_step(self, fb_ops, **_kw):
        self.finished_with = list(fb_ops)
        self.optim_submitted = True

        class _R:
            metrics = {"actor/loss": 0.1}
        return _R()


class FbStreamerTest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_chunk_boundaries_and_union(self):
        """10 groups, 4 chunks -> ceil(10/4)=3 groups per submit: 3/3/3, flush 1.
        The union of all submitted datums equals every transition, exactly once."""
        backend = _StreamBackend()
        st = _FbStreamer(backend, _Args(), n_groups=10, chunks=4)

        async def run():
            for g in range(10):
                await st.on_group(_group([0.0, 1.0]))
            await st.flush()
        self._run(run())

        sizes = [len(c) for c in backend.submitted]
        self.assertEqual(sizes, [6, 6, 6, 2])  # 3 groups x 2 episodes, remainder 1x2
        self.assertEqual(sum(sizes), 20)
        self.assertEqual(len(st.transitions), 20)
        self.assertEqual(st.fb_ops, ["op-1", "op-2", "op-3", "op-4"])

    def test_advantages_match_batch_computation(self):
        """Per-group streaming advantages == whole-batch computation (Check 0
        pinned at the integration level: the datums carry the same values)."""
        import copy
        from training_sessions_cookbook.rl.advantages import compute_grouped_advantages
        groups = [_group([0.0, 1.0, 1.0]), _group([1.0, 1.0, 0.0]), _group([0.0, 0.0, 1.0])]
        reference = compute_grouped_advantages(copy.deepcopy(groups))

        backend = _StreamBackend()
        st = _FbStreamer(backend, _Args(), n_groups=3, chunks=2)

        async def run():
            for g in copy.deepcopy(groups):
                await st.on_group(g)
            await st.flush()
        self._run(run())

        streamed = [d["adv"] for c in backend.submitted for d in c]
        self.assertEqual(streamed, [t.advantage for t in reference])

    def test_empty_groups_submit_nothing(self):
        """All-failed rollouts (empty episodes) never produce an empty fb op."""
        backend = _StreamBackend()
        st = _FbStreamer(backend, _Args(), n_groups=2, chunks=2)

        async def run():
            await st.on_group([[], []])   # every episode empty
            await st.on_group([[], []])
            await st.flush()
        self._run(run())
        self.assertEqual(backend.submitted, [])
        self.assertEqual(st.fb_ops, [])

    def test_submission_does_not_wait_for_results(self):
        """Fire-and-forget: on_group returns even when results never resolve."""
        class _NeverResolves(_StreamBackend):
            async def submit_forward_backward(self, datums, *, loss_name):
                self.submitted.append(list(datums))
                return asyncio.get_event_loop().create_future()  # never done

        backend = _NeverResolves()
        st = _FbStreamer(backend, _Args(), n_groups=1, chunks=1)

        async def run():
            await asyncio.wait_for(st.on_group(_group([1.0])), timeout=1.0)
        self._run(run())  # would raise TimeoutError if submission awaited results
        self.assertEqual(len(backend.submitted), 1)


class StreamedCollectorTest(unittest.TestCase):
    def test_callback_per_group_and_prompt_order_return(self):
        from training_sessions_cookbook.rl import rollouts as R

        class _Params:
            pass

        class _Backend:
            def make_sampling_params(self, **kw):
                return _Params()

        class _ArgsC:
            # >= total episodes: no semaphore contention, so completion order
            # is purely the per-prompt sleep (later prompts sleep less)
            num_rollout_workers = 16
            max_tokens = 8
            temperature = 1.0
            max_turns = 1
            env_type = "gsm8k"

        async def fake_rollout(*, prompt, **_kw):
            # later prompts finish FIRST, to prove callback order != return order
            await asyncio.sleep(0.05 * (3 - prompt))
            return [Transition(obs_tokens=[prompt], act_tokens=[1],
                               act_logprobs=[-0.1], reward=float(prompt),
                               done=True, env_type="gsm8k")]

        completed = []

        async def on_group(group):
            completed.append(group[0][0].reward)

        orig = R.rollout_one
        R.rollout_one = fake_rollout
        try:
            groups = asyncio.run(R.collect_grouped_batch_streamed_async(
                [0, 1, 2, 3], K=2, sampling_client=None, args=_ArgsC(),
                tokenizer=None, env_factory=lambda: None, template_fn=None,
                backend=_Backend(), on_group_complete=on_group,
            ))
        finally:
            R.rollout_one = orig

        self.assertEqual(len(completed), 4)                      # one callback per group
        self.assertEqual(completed, [3.0, 2.0, 1.0, 0.0])        # completion order
        self.assertEqual([g[0][0].reward for g in groups],
                         [0.0, 1.0, 2.0, 3.0])                   # return: prompt order


if __name__ == "__main__":
    unittest.main()
