from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg

from mem_alpha.manager import MemoryManager


def create_memory_tools(
    manager: MemoryManager, default_user_id: Optional[str] = None
) -> list:
    """Create LangChain tools bound to a MemoryManager instance.

    Args:
        manager: An initialized MemoryManager (must have called connect()).
        default_user_id: Default user_id injected at runtime (hidden from LLM).

    Returns:
        List of LangChain Tool objects ready for an agent.
    """

    # ── Core Memory Tools ──────────────────────────────────────────────

    @tool
    async def store_core_memory(
        key: str,
        content: str,
        session_id: Optional[str] = None,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Store or update a core working memory entry. Use this for key facts about
        the current task, user preferences, or any information that needs to persist
        for the current session. Memories are stored by key -- if a key already exists,
        it will be updated.

        Common keys: 'user_name', 'current_task', 'user_preferences', 'important_context'.
        """
        memory_id = await manager.core.add(
            user_id=user_id, key=key, content=content, session_id=session_id
        )
        return f"Stored core memory '{key}': {content[:80]}... (id: {memory_id})"

    @tool
    async def get_core_memory(
        key: str,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Retrieve a specific core memory by its key name."""
        doc = await manager.core.get_by_key(user_id, key)
        if doc:
            return f"[{doc['key']}]: {doc['content']}"
        return f"No core memory found with key '{key}'."

    @tool
    async def list_core_memory_keys(
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """List all available core memory keys to see what working memory is stored."""
        keys = await manager.core.get_all_keys(user_id)
        if keys:
            return "Core memory keys: " + ", ".join(keys)
        return "No core memories stored yet."

    @tool
    async def delete_core_memory(
        key: str,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Delete a core memory by its key. Use when information is no longer relevant."""
        deleted = await manager.core.delete_by_key(user_id, key)
        if deleted:
            return f"Deleted core memory '{key}'."
        return f"No core memory found with key '{key}'."

    # ── Episodic Memory Tools ──────────────────────────────────────────

    @tool
    async def store_episode(
        title: str,
        content: str,
        outcome: str = "success",
        outcome_detail: str = "",
        tags: Optional[list[str]] = None,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Record a complete task episode with its outcome. Use this after completing
        a task to log what happened, what actions were taken, and whether it succeeded.
        This helps learn from past experiences.

        Args:
            title: Brief summary (e.g., 'Debugged authentication error')
            content: Full description of what happened
            outcome: One of 'success', 'failure', 'partial', 'abandoned'
            outcome_detail: Why this outcome occurred
            tags: Categories like 'coding', 'research', 'debugging'
        """
        memory_id = await manager.episodic.add(
            user_id=user_id,
            title=title,
            content=content,
            outcome=outcome,
            outcome_detail=outcome_detail,
            tags=tags,
        )
        return f"Recorded episode '{title}' [{outcome}] (id: {memory_id})"

    @tool
    async def search_episodes(
        query: str,
        outcome: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 5,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Search past episodes by semantic similarity. Useful for finding
        past experiences relevant to the current task. Can filter by outcome
        ('success', 'failure', 'partial', 'abandoned') and/or tags."""
        results = await manager.episodic.search(
            user_id, query, limit=limit, outcome=outcome, tags=tags
        )
        if not results:
            return "No relevant episodes found."
        lines = []
        for r in results:
            lines.append(
                f"- [{r.get('outcome', '?')}] {r.get('title', 'Untitled')} "
                f"(score: {r.get('score', 0):.2f}): {r.get('content', '')[:120]}..."
            )
        return "Past episodes:\n" + "\n".join(lines)

    # ── Semantic Memory Tools ──────────────────────────────────────────

    @tool
    async def store_semantic_knowledge(
        content: str,
        category: str = "fact",
        confidence: float = 0.5,
        source_episode_ids: Optional[list[str]] = None,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Store a piece of distilled knowledge or insight. Use this for general
        facts, rules, patterns, user preferences, or skills learned from experience.

        Categories: 'preference', 'fact', 'rule', 'pattern', 'skill'.
        Confidence: 0.0 to 1.0 (how sure you are about this knowledge).
        """
        memory_id = await manager.semantic.add(
            user_id=user_id,
            content=content,
            category=category,
            confidence=confidence,
            source_episode_ids=source_episode_ids,
        )
        return (
            f"Stored semantic memory [{category}] (confidence: {confidence}): "
            f"{content[:80]}... (id: {memory_id})"
        )

    @tool
    async def search_knowledge(
        query: str,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 5,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Search long-term knowledge by semantic similarity. Use this to recall
        known facts, user preferences, learned rules, or established patterns.
        Can filter by category and minimum confidence level."""
        results = await manager.semantic.search(
            user_id,
            query,
            limit=limit,
            category=category,
            min_confidence=min_confidence,
        )
        if not results:
            return "No relevant knowledge found."
        lines = []
        for r in results:
            lines.append(
                f"- [{r.get('category', '?')}] (conf: {r.get('confidence', 0):.2f}) "
                f"{r.get('content', '')[:150]}"
            )
        return "Known knowledge:\n" + "\n".join(lines)

    @tool
    async def reinforce_knowledge(
        memory_id: str,
        source_episode_id: Optional[str] = None,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Reinforce an existing piece of semantic knowledge, increasing its
        confidence score. Use this when a known fact is confirmed by new evidence."""
        success = await manager.semantic.reinforce(memory_id, source_episode_id)
        if success:
            return f"Reinforced knowledge {memory_id}."
        return f"Memory {memory_id} not found."

    # ── Cross-Tier Tools ───────────────────────────────────────────────

    @tool
    async def get_relevant_context(
        query: str,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Get relevant context from ALL memory tiers for the current task.
        This searches core working memory, past episodes, and long-term knowledge
        simultaneously. Use this at the start of a task to load relevant context."""
        context = await manager.get_context(user_id, query)
        sections = []

        if context["core"]["relevant"]:
            core_lines = [
                f"  - [{r.get('key', '?')}]: {r.get('content', '')[:100]}"
                for r in context["core"]["relevant"]
            ]
            sections.append("Working Memory:\n" + "\n".join(core_lines))

        if context["episodic"]:
            ep_lines = [
                f"  - [{r.get('outcome', '?')}] {r.get('title', '')}: "
                f"{r.get('content', '')[:80]}..."
                for r in context["episodic"]
            ]
            sections.append("Relevant Past Episodes:\n" + "\n".join(ep_lines))

        if context["semantic"]:
            sem_lines = [
                f"  - [{r.get('category', '?')}] {r.get('content', '')[:120]}"
                for r in context["semantic"]
            ]
            sections.append("Known Knowledge:\n" + "\n".join(sem_lines))

        if not sections:
            return "No relevant context found in any memory tier."
        return "\n\n".join(sections)

    @tool
    async def promote_to_knowledge(
        episode_id: str,
        knowledge: str,
        category: str = "pattern",
        confidence: float = 0.6,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Distill an episode into long-term semantic knowledge. Use this when
        a task episode reveals a generalizable insight, rule, or pattern worth
        remembering. If similar knowledge already exists, it will be reinforced
        instead of duplicated.

        Args:
            episode_id: The episode_id to distill from
            knowledge: The distilled knowledge statement
            category: 'preference', 'fact', 'rule', 'pattern', or 'skill'
            confidence: Initial confidence (0.0-1.0)
        """
        memory_id = await manager.promote_episode_to_semantic(
            user_id, episode_id, knowledge, category, confidence
        )
        return (
            f"Distilled episode {episode_id} into semantic knowledge "
            f"(id: {memory_id}): {knowledge[:80]}..."
        )

    @tool
    async def search_all_memories(
        query: str,
        limit_per_tier: int = 3,
        user_id: Annotated[str, InjectedToolArg] = default_user_id,
    ) -> str:
        """Search across all memory tiers simultaneously. Returns results grouped
        by tier (core, episodic, semantic). Use for broad recall across all memory."""
        results = await manager.search_all_tiers(user_id, query, limit_per_tier)
        sections = []
        for tier_name, tier_results in results.items():
            if tier_results:
                lines = [
                    f"  - {r.get('content', '')[:120]}... "
                    f"(score: {r.get('score', 0):.2f})"
                    for r in tier_results
                ]
                sections.append(f"{tier_name.title()} Memory:\n" + "\n".join(lines))

        if sections:
            return "\n\n".join(sections)
        return "No memories found across any tier."

    return [
        store_core_memory,
        get_core_memory,
        list_core_memory_keys,
        delete_core_memory,
        store_episode,
        search_episodes,
        store_semantic_knowledge,
        search_knowledge,
        reinforce_knowledge,
        get_relevant_context,
        promote_to_knowledge,
        search_all_memories,
    ]
