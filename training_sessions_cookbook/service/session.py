"""Opening a SageMaker Training Session and resolving its tokenizer.

Both training paths open a session the same way: create a ``ServiceClient``
carrying the account-level configuration (role, model package group, output
S3 path), create a LoRA training client (which blocks until the session is
running), and fetch the service-resolved tokenizer. Doing this in one place
keeps the two paths from diverging on session setup or tokenizer resolution.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def open_session(
    *,
    session_name: str,
    base_model_arn: str,
    model_package_group_arn: str,
    role_arn: str,
    output_s3_path: str,
    lora_rank: int,
    lora_alpha: int,
    region: str = "us-west-2",
    user_metadata: dict[str, str] | None = None,
    load_state_arn: str | None = None,
):
    """Open a LoRA training session and return ``(svc, training_client, tokenizer)``.

    ``svc`` is the control-plane ``ServiceClient`` (used later to stop/delete
    the session and manage model packages); ``training_client`` drives
    forward/backward, optimizer steps, sampling, and checkpointing;
    ``tokenizer`` is the one the service resolved for the base model, so
    client-side and service-side tokenization cannot diverge. The SDK
    auto-resolves its endpoints (override via
    ``SAGEMAKER_TRAINING_SESSION_ENDPOINT`` / ``_RUNTIME_ENDPOINT``).

    ``user_metadata`` (optional string map) is attached to the session for
    provenance, e.g. the run name and cookbook version.

    ``load_state_arn`` resumes from a saved checkpoint: the fresh session
    loads the given model package (weights + optimizer state) before this
    function returns.
    """
    # Lazy import so merely importing the ``service`` package (e.g. for the
    # read_jsonl / metrics helpers, or during test collection) never requires
    # the ``sagemaker-train`` SDK wheel to be installed. The SDK is only
    # needed when a session is actually opened.
    from sagemaker.train.training_session import ServiceClient

    logger.info(
        "Opening training session: region=%s session=%s base_model_arn=%s",
        region, session_name, base_model_arn,
    )
    svc = ServiceClient(
        region=region,
        role_arn=role_arn,
        model_package_group_arn=model_package_group_arn,
        s3_output_path=output_s3_path,
    )
    create_kwargs = dict(
        rank=lora_rank,
        alpha=lora_alpha,
        training_session_name=session_name,
        user_metadata=user_metadata,
    )
    if load_state_arn:
        logger.info("Resuming session from checkpoint: %s", load_state_arn)
        training_client = svc.create_training_client_from_state_with_optimizer(
            model_package_arn=load_state_arn, base_model=base_model_arn, **create_kwargs
        )
    else:
        training_client = svc.create_lora_training_client(base_model_arn, **create_kwargs)
    logger.info("Session running: %s", training_client.training_session_name)

    tokenizer = training_client.get_tokenizer()
    return svc, training_client, tokenizer
