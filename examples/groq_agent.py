#!/usr/bin/env python3
"""
Groq Agent with mem-alpha Memory-Guided Steering

Demonstrates the full RL-replacement loop:
  inject guidance → call Groq → store response → evaluate → feedback

Run 15 cycles of SQL optimization questions. Watch the guidance directives
appear and refine as the system learns what makes a good response.

Setup:
    1. pip install groq
    2. Set GROQ_API_KEY and MONGODB_URI in your .env
    3. python examples/groq_agent.py

You'll see the agent's responses improve as the memory system accumulates
feedback and the adapter learns which patterns produce better outcomes.
"""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from groq import AsyncGroq

from mem_alpha import MemAlpha, Outcome

# ── Domain: SQL Performance Optimization ─────────────────────

SYSTEM_PROMPT = """You are a senior database engineer specializing in SQL performance optimization.
When answering questions:
- Always be specific and actionable
- Include concrete SQL examples when relevant
- Mention tools like EXPLAIN ANALYZE when appropriate
- Consider indexing strategies
- Keep answers concise but thorough (3-5 paragraphs max)"""

QUERIES = [
    "How do I optimize a slow SQL JOIN between a large orders table and a customers table?",
    "My GROUP BY query on a 50 million row table takes 30 seconds. How do I speed it up?",
    "What's the best indexing strategy for a table with heavy read and moderate write workloads?",
    "How do I optimize a query that uses multiple subqueries in the WHERE clause?",
    "My LIKE '%search_term%' query is very slow. What are my options?",
    "How should I optimize pagination for large datasets — OFFSET vs cursor-based?",
    "What's the best way to handle a slow COUNT(*) on a large partitioned table?",
    "How do I optimize a query that joins 5+ tables with complex WHERE conditions?",
    "My INSERT performance degrades as the table grows. What causes this and how do I fix it?",
    "How do I identify and fix N+1 query problems in a web application?",
    "What are covering indexes and when should I use them?",
    "How do I optimize a UNION ALL query that combines results from multiple large tables?",
    "My database has lock contention issues during peak hours. How do I diagnose and resolve this?",
    "What's the difference between a hash join and a merge join, and when is each faster?",
    "How do I optimize a recursive CTE that traverses a deep hierarchy?",
]

EVAL_PROMPT = """Rate the following SQL optimization response on a scale of 1-10.

Criteria:
- Specificity: Does it give concrete, actionable advice (not generic platitudes)?
- Examples: Does it include SQL code examples or tool commands?
- Depth: Does it explain WHY, not just WHAT?
- Practicality: Could a developer act on this immediately?

Question: {query}

Response: {response}

Reply with ONLY a JSON object like: {{"score": 7, "reason": "Good specificity but missing EXPLAIN example"}}"""


# ── Groq client ──────────────────────────────────────────────

async def call_groq(
    client: AsyncGroq,
    system: str,
    user_msg: str,
    model: str = "llama-3.1-8b-instant",
) -> str:
    """Call Groq and return the response text."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return resp.choices[0].message.content


async def evaluate_response(
    client: AsyncGroq,
    query: str,
    response: str,
) -> tuple[int, str]:
    """Use Groq to evaluate the response quality. Returns (score, reason)."""
    prompt = EVAL_PROMPT.format(query=query, response=response)
    raw = await call_groq(client, "You are a strict evaluator. Reply only with JSON.", prompt)

    # Parse score from response
    try:
        # Try to find JSON in the response
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 5))
            reason = data.get("reason", "")
            return min(max(score, 1), 10), reason
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to find a bare number
    nums = re.findall(r'\b(\d+)\b', raw)
    if nums:
        return min(max(int(nums[0]), 1), 10), raw[:100]
    return 5, "Could not parse evaluation"


# ── Display helpers ──────────────────────────────────────────

def print_cycle(cycle: int, total: int, query: str, guidance: str,
                response: str, score: int, reason: str, reward: float,
                latency_ms: float, n_memories: int):
    """Print a formatted cycle summary."""
    width = 60
    print(f"\n{'='*width}")
    print(f"  CYCLE {cycle}/{total}  |  memories: {n_memories}  |  steering: {latency_ms:.0f}ms")
    print(f"{'='*width}")

    print(f"\n  Query: {query[:80]}")

    if guidance.strip():
        print(f"\n  Guidance injected:")
        for line in guidance.strip().split("\n"):
            print(f"    {line}")
    else:
        print(f"\n  Guidance injected: (none yet — building memory)")

    # Truncate response for display
    resp_lines = response.strip().split("\n")
    preview = "\n    ".join(resp_lines[:6])
    if len(resp_lines) > 6:
        preview += "\n    ..."
    print(f"\n  Response:")
    print(f"    {preview}")

    score_bar = "#" * score + "." * (10 - score)
    color_start = "\033[32m" if score >= 7 else "\033[33m" if score >= 5 else "\033[31m"
    color_end = "\033[0m"
    print(f"\n  Evaluation: {color_start}{score}/10{color_end} [{score_bar}]")
    print(f"  Reason: {reason[:80]}")
    print(f"  Reward: {reward:+.1f}")
    print(f"{'='*width}")


def print_summary(scores: list[int], rewards: list[float]):
    """Print final learning summary."""
    width = 60
    print(f"\n{'#'*width}")
    print(f"  LEARNING SUMMARY")
    print(f"{'#'*width}")

    # Split into thirds
    n = len(scores)
    third = max(1, n // 3)
    early = scores[:third]
    mid = scores[third:third*2]
    late = scores[third*2:]

    early_avg = sum(early) / len(early) if early else 0
    mid_avg = sum(mid) / len(mid) if mid else 0
    late_avg = sum(late) / len(late) if late else 0

    print(f"\n  Avg score (first {third} cycles):  {early_avg:.1f}/10")
    print(f"  Avg score (middle {len(mid)} cycles): {mid_avg:.1f}/10")
    print(f"  Avg score (last {len(late)} cycles):   {late_avg:.1f}/10")

    delta = late_avg - early_avg
    if delta > 0.5:
        print(f"\n  Improvement: +{delta:.1f} points")
    elif delta < -0.5:
        print(f"\n  Decline: {delta:.1f} points")
    else:
        print(f"\n  Stable: {delta:+.1f} points (more cycles needed for visible learning)")

    print(f"\n  Score progression: {' '.join(str(s) for s in scores)}")

    # Reward progression
    total_reward = sum(rewards)
    print(f"  Total reward: {total_reward:+.1f}")
    print(f"{'#'*width}\n")


# ── Main loop ────────────────────────────────────────────────

async def main():
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("Error: GROQ_API_KEY not set in .env")
        sys.exit(1)

    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("Error: MONGODB_URI not set in .env")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  mem-alpha + Groq: SQL Optimization Agent")
    print("  Watch the agent improve as it learns from feedback")
    print("=" * 60)

    # Initialize
    client = AsyncGroq(api_key=groq_key)
    ma = MemAlpha()

    print("\nConnecting to MongoDB...")
    await ma.connect()
    print("Connected. Loading embedding model (first run downloads ~80MB)...\n")

    user_id = "groq_demo_user"
    scores = []
    rewards = []
    total_cycles = len(QUERIES)

    # Store the agent's specialization as core memory
    await ma.set_core(user_id, "role", "Senior SQL performance optimization expert")
    await ma.set_core(user_id, "quality_bar", "Responses must include specific examples, EXPLAIN output, and indexing strategies")

    try:
        for i, query in enumerate(QUERIES):
            cycle = i + 1

            # 1. INJECT — get guidance from memory
            result = await ma.inject(user_id, query)
            guidance = ma.get_prompt_block(result)

            # 2. CALL — build the system prompt with guidance and call Groq
            full_system = SYSTEM_PROMPT
            if guidance.strip():
                full_system += f"\n\n{guidance}"

            response = await call_groq(client, full_system, query)

            # 3. STORE — record the trace
            trace_id = await ma.store(user_id, query, response)

            # 4. EVALUATE — use Groq to rate the response
            score, reason = await evaluate_response(client, query, response)

            # 5. FEEDBACK — convert evaluation to outcome
            # Score 1-10 → delta centered at 0 (5 = neutral, 10 = very good, 1 = bad)
            quality_delta = -(score - 5) / 5.0  # negative delta = improvement
            reward_estimate = (score - 5) * 2.0  # maps 1-10 to -8..+10

            outcome = Outcome(
                deltas={"quality": quality_delta},
                weights={"quality": 10.0},
                human_approved=score >= 7,
            )
            await ma.feedback(trace_id, outcome)
            outcome.compute_reward()

            scores.append(score)
            rewards.append(outcome.reward)

            # Display
            print_cycle(
                cycle, total_cycles, query, guidance, response,
                score, reason, outcome.reward,
                result.total_latency_ms, result.num_memories_retrieved,
            )

            # Small delay to avoid rate limits
            if i < total_cycles - 1:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Summary
        if scores:
            print_summary(scores, rewards)

        print("Shutting down...")
        await ma.shutdown()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
