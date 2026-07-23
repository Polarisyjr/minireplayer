from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# native-agent-replay post-sampling commitment"
NEEDLE = "        sampled = sampled.long()\n\n        if num_logprobs is None:\n"
REPLACEMENT = (
    "        sampled = sampled.long()\n\n"
    "        # native-agent-replay post-sampling commitment\n"
    "        # Sampling above is intentionally unchanged. A replay processor may only\n"
    "        # replace the token committed to the next decode step after sampling.\n"
    "        for processor in sampling_metadata.logitsprocs.all:\n"
    '            override = getattr(processor, "native_replay_override", None)\n'
    "            if override is not None:\n"
    "                sampled = override(sampled)\n\n"
    "        if num_logprobs is None:\n"
)

RUNNER_MARKER = "# native-agent-replay valid-sample mask"
RUNNER_NEEDLE = (
    "        self.input_batch.update_async_output_token_ids()\n"
    "        if spec_decode_metadata is None:\n"
)
RUNNER_REPLACEMENT = (
    "        self.input_batch.update_async_output_token_ids()\n"
    "\n"
    "        # native-agent-replay valid-sample mask\n"
    "        # The sampler tensor contains every persistent-batch row, including\n"
    "        # rows that vLLM will discard for this step. Publish the authoritative\n"
    "        # mask before sampling so replay only records or overrides real output.\n"
    "        native_replay_discarded = self.discard_request_mask.np[\n"
    "            : self.input_batch.num_reqs\n"
    "        ]\n"
    "        for processor in sampling_metadata.logitsprocs.all:\n"
    "            set_mask = getattr(\n"
    "                processor, \"native_replay_set_discard_mask\", None\n"
    "            )\n"
    "            if set_mask is not None:\n"
    "                set_mask(native_replay_discarded)\n"
    "\n"
    "        if spec_decode_metadata is None:\n"
)


def patch(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        return False
    if source.count(NEEDLE) != 1:
        raise RuntimeError("unsupported vLLM sampler source; patch anchor is not unique")
    path.write_text(source.replace(NEEDLE, REPLACEMENT), encoding="utf-8")
    return True


def patch_model_runner(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if RUNNER_MARKER in source:
        return False
    if source.count(RUNNER_NEEDLE) != 1:
        raise RuntimeError("unsupported vLLM model runner source; patch anchor is not unique")
    path.write_text(
        source.replace(RUNNER_NEEDLE, RUNNER_REPLACEMENT),
        encoding="utf-8",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sampler", type=Path)
    parser.add_argument("model_runner", type=Path)
    args = parser.parse_args()
    patch(args.sampler)
    patch_model_runner(args.model_runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
