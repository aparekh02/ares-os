#!/usr/bin/env python3
"""MGS Smoke Test — runs the full pipeline with mock data + optional MongoDB.

Usage:
    python scripts/mgs_smoke_test.py              # local-only (no DB needed)
    python scripts/mgs_smoke_test.py --mongodb     # includes MongoDB Atlas test

Local tests verify:
  1. Adapter processes 384-dim inputs and produces correct output shapes
  2. Gradient flow through all parameters
  3. GuidanceCompiler produces formatted directives
  4. Outcome reward computation works
  5. Checkpoint save/load (local filesystem)

MongoDB tests (--mongodb) verify:
  6. TraceBuffer CRUD via MongoDB collection
  7. Trainer runs 5 async online steps
  8. Adapter checkpoint save/load via GridFS
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import torch

sys.path.insert(0, ".")


def test_local() -> None:
    """Run local tests that don't require MongoDB."""
    from mem_alpha.steering.adapter import SteeringAdapter
    from mem_alpha.steering.config import MGSConfig
    from mem_alpha.steering.guidance import GuidanceCompiler
    from mem_alpha.steering.outcome import Outcome
    from mem_alpha.steering.schemas import MemorySlot

    import tempfile
    from pathlib import Path

    config = MGSConfig()

    print("=" * 60)
    print("MGS SMOKE TEST — LOCAL")
    print("=" * 60)

    # ── 1. Adapter forward pass ──────────────────────────────
    print("\n[1] SteeringAdapter forward pass")
    adapter = SteeringAdapter(config)
    print(f"    Parameter count: {adapter.param_count():,}")

    mock_slots = []
    for i in range(8):
        slot = MemorySlot(
            id=f"mem-{i}",
            tier=["core", "episodic", "semantic"][i % 3],
            content=f"Mock memory content #{i}: user prefers approach {chr(65 + i)}",
            score=0.9 - i * 0.05,
            confidence=0.8 - i * 0.05,
            embedding=[float(x) for x in torch.randn(384).tolist()],
        )
        mock_slots.append(slot)

    query_embedding = torch.randn(384).tolist()

    t0 = time.perf_counter()
    inputs = adapter.prepare_input(query_embedding, mock_slots)
    steering_vec, guidance_logits, attn_weights = adapter(**inputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"    Steering vector shape: {steering_vec.shape}")
    print(f"    Guidance logits shape: {guidance_logits.shape}")
    print(f"    Attention weights shape: {attn_weights.shape}")
    print(f"    Forward latency: {latency_ms:.1f}ms")

    assert steering_vec.shape == (1, config.steering_dim)
    assert guidance_logits.shape == (1, config.num_guidance_slots, config.num_categories)
    print("    ✓ Shapes correct")

    # ── 2. Gradient flow ─────────────────────────────────────
    print("\n[2] Gradient flow check")
    adapter.train()
    inputs_grad = adapter.prepare_input(query_embedding, mock_slots)
    sv, gl, aw = adapter(**inputs_grad)
    loss = sv.sum() + gl.sum()
    loss.backward()

    has_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in adapter.parameters() if p.requires_grad
    )
    print(f"    All parameters have gradients: {has_grads}")
    assert has_grads
    print("    ✓ Gradient flow OK")
    adapter.zero_grad()
    adapter.eval()

    # ── 3. GuidanceCompiler ──────────────────────────────────
    print("\n[3] GuidanceCompiler")
    compiler = GuidanceCompiler(config)

    with torch.no_grad():
        sv2, gl2, aw2 = adapter(**inputs)

    directives = compiler.compile(gl2, aw2, mock_slots)
    print(f"    Produced {len(directives)} directives")
    for d in directives:
        print(f"      [{d.category}] {d.instruction[:60]}... (conf={d.confidence:.2f})")

    prompt_block = GuidanceCompiler.format_for_prompt(directives)
    if prompt_block:
        print(f"    Prompt block:\n{prompt_block}")
    print("    ✓ Compiler OK")

    # ── 4. Outcome ───────────────────────────────────────────
    print("\n[4] Outcome reward computation")
    outcome = Outcome(
        deltas={"accuracy": -0.1, "latency": 0.05},
        weights={"accuracy": 5.0, "latency": 2.0},
        human_approved=True,
    )
    reward = outcome.compute_reward()
    print(f"    Computed reward: {reward:.2f}")
    assert reward > 0
    print("    ✓ Outcome OK")

    # ── 5. Local checkpoint save/load ────────────────────────
    print("\n[5] Checkpoint save/load (local)")
    tmpdir = Path(tempfile.mkdtemp())
    ckpt_path = tmpdir / "smoke_test.pt"
    adapter.save_local(ckpt_path)

    adapter2 = SteeringAdapter(config)
    adapter2.load_local(ckpt_path)

    with torch.no_grad():
        sv_orig = adapter(**inputs)[0]
        sv_loaded = adapter2(**inputs)[0]
    diff = (sv_orig - sv_loaded).abs().max().item()
    print(f"    Max diff after load: {diff:.8f}")
    assert diff < 1e-5
    print("    ✓ Checkpoint OK")

    print("\n" + "=" * 60)
    print("LOCAL SMOKE TESTS PASSED")
    print(f"  Adapter params:  {adapter.param_count():,}")
    print(f"  Forward latency: {latency_ms:.1f}ms")
    print("=" * 60)


async def test_mongodb() -> None:
    """Run MongoDB integration tests (requires MONGODB_URI env var)."""
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

    from mem_alpha.steering.adapter import SteeringAdapter
    from mem_alpha.steering.config import MGSConfig
    from mem_alpha.steering.outcome import Outcome
    from mem_alpha.steering.schemas import MGSTrace, MemorySlot, SteeringResult
    from mem_alpha.steering.trace_buffer import MGSTraceBuffer
    from mem_alpha.steering.trainer import AdapterTrainer

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("\n⚠ MONGODB_URI not set — skipping MongoDB tests")
        return

    db_name = "mgs_smoke_test"
    client = AsyncIOMotorClient(uri)
    db = client[db_name]

    config = MGSConfig(
        traces_collection="smoke_traces",
        checkpoints_bucket="smoke_checkpoints",
        online_update_every_n=5,
    )

    print("\n" + "=" * 60)
    print("MGS SMOKE TEST — MONGODB")
    print("=" * 60)

    try:
        # ── 6. TraceBuffer CRUD ──────────────────────────────
        print("\n[6] TraceBuffer (MongoDB)")
        traces_coll = db[config.traces_collection]
        trace_buffer = MGSTraceBuffer(traces_coll)
        await trace_buffer.ensure_indexes()

        adapter = SteeringAdapter(config)
        mock_slots = [
            MemorySlot(
                id=f"mem-{i}",
                tier=["core", "episodic", "semantic"][i % 3],
                content=f"Mock memory #{i}",
                score=0.9 - i * 0.05,
                confidence=0.8 - i * 0.05,
                embedding=torch.randn(384).tolist(),
            )
            for i in range(8)
        ]
        query_embedding = torch.randn(384).tolist()

        with torch.no_grad():
            inputs = adapter.prepare_input(query_embedding, mock_slots)
            sv, gl, aw = adapter(**inputs)

        steering_result = SteeringResult(
            directives=[],
            steering_vector=sv[0].tolist(),
            num_memories_retrieved=len(mock_slots),
        )

        trace_ids = []
        for i in range(10):
            trace = MGSTrace(
                trace_id=f"smoke-trace-{i}",
                timestamp=time.time(),
                user_id="smoke-test-user",
                query=f"Test query {i}",
                query_embedding=query_embedding,
                memory_slots=mock_slots,
                steering_result=steering_result,
                response=f"Test response {i}",
            )
            await trace_buffer.record(trace)
            trace_ids.append(trace.trace_id)

        count = await trace_buffer.size()
        print(f"    Recorded {count} traces")
        assert count == 10

        # Backfill outcomes
        for i, tid in enumerate(trace_ids):
            o = Outcome(
                deltas={"quality": -0.1 * (i - 5)},
                weights={"quality": 2.0},
            )
            await trace_buffer.attach_outcome(tid, o)

        # Sample
        sampled = await trace_buffer.sample_traces(3, only_with_outcome=True)
        print(f"    Sampled {len(sampled)} traces with outcomes")
        assert len(sampled) == 3

        # Get single trace
        single = await trace_buffer.get_trace(trace_ids[0])
        assert single is not None
        assert single["outcome"] is not None
        print(f"    get_trace OK  reward={single['reward']:.2f}")

        # Contrastive pairs
        pairs = await trace_buffer.sample_contrastive_pairs(3)
        print(f"    Sampled {len(pairs)} contrastive pairs")
        print("    ✓ TraceBuffer MongoDB OK")

        # ── 7. Trainer (async online update) ─────────────────
        print("\n[7] Training — 5 async online steps")
        trainer = AdapterTrainer(adapter, trace_buffer, config)

        losses = []
        for step in range(5):
            avg_loss = await trainer.online_update()
            losses.append(avg_loss)
            print(f"    Step {step + 1}: loss = {avg_loss:.4f}")

        print(f"    Loss trajectory: {[f'{l:.4f}' for l in losses]}")
        print("    ✓ Async training OK")

        # ── 8. GridFS checkpoint ─────────────────────────────
        print("\n[8] Adapter checkpoint (GridFS)")
        gridfs = AsyncIOMotorGridFSBucket(db, bucket_name=config.checkpoints_bucket)

        await adapter.save_gridfs(gridfs, "smoke_test.pt", update_count=1)

        adapter2 = SteeringAdapter(config)
        await adapter2.load_gridfs(gridfs, "smoke_test.pt")

        with torch.no_grad():
            sv_orig = adapter(**inputs)[0]
            sv_loaded = adapter2(**inputs)[0]
        diff = (sv_orig - sv_loaded).abs().max().item()
        print(f"    Max diff after GridFS load: {diff:.8f}")
        assert diff < 1e-5
        print("    ✓ GridFS checkpoint OK")

        print("\n" + "=" * 60)
        print("MONGODB SMOKE TESTS PASSED")
        print("=" * 60)

    finally:
        # Cleanup test database
        await client.drop_database(db_name)
        client.close()
        print(f"\n    Cleaned up test database: {db_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MGS Smoke Test")
    parser.add_argument(
        "--mongodb", action="store_true",
        help="Include MongoDB Atlas integration tests (requires MONGODB_URI env var)",
    )
    args = parser.parse_args()

    # Always run local tests
    test_local()

    # Optionally run MongoDB tests
    if args.mongodb:
        asyncio.run(test_mongodb())


if __name__ == "__main__":
    main()
