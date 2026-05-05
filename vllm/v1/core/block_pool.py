# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)

# CacheSage protection class constants (avoids hard dependency).
_CACHESAGE_P0 = 0  # ProtectionClass.P0.value


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        cachesage_coordinator: Any | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        # CacheSage: optional coordinator for blast-radius-aware eviction.
        # When set, eviction uses BRAE scores instead of LRU, and P0-protected
        # blocks in the prefix cache are immune to eviction.
        self._cachesage = cachesage_coordinator

        # CacheSage: per-hash hit tracking for score computation.
        # Always initialized — _cachesage may be set after __init__ via
        # EngineCore patching (block_pool._cachesage = coordinator).
        self._hash_hit_count: dict[Any, int] = {}
        self._hash_last_touch: dict[Any, int] = {}
        self._cachesage_step: int = 0
        self._cachesage_decay: float = float(
            os.environ.get("CACHESAGE_DECAY", "0.1")
        )
        # Cached P0 threshold — recomputed every N steps, not per call.
        self._p0_threshold: int = 2
        self._p0_threshold_step: int = 0
        # Policy: "brae" (hit-count + decay) or "predictive" (learned
        # transition-based survival). Default brae to preserve backward compat.
        self._cachesage_policy: str = os.environ.get(
            "CACHESAGE_POLICY", "brae"
        )
        # Predictive: map block_hash_with_group_id -> agent_id (a stable
        # fingerprint of the request's leading blocks past any shared
        # chat-template prefix). Enables agent-level survival prediction,
        # which concentrates probability mass vs per-block Markov
        # (|agents| << |blocks|).
        self._block_to_agent: dict[Any, str] = {}
        # Agent fingerprint window: hash block_hashes[SKIP : SKIP + N]
        # (stripped of kv_cache_group_id) to identify the request's
        # agent. Must skip past the shared chat-template prefix (e.g.
        # gpt-oss harmony fills ~28 blocks with ChatGPT/cutoff/date/
        # channel-config content identical across agents) and cover
        # enough agent-unique blocks to collide rarely. Stripping
        # group_id makes the fingerprint equal across KV cache groups.
        self._cachesage_agent_prefix_skip: int = int(
            os.environ.get("CACHESAGE_AGENT_PREFIX_SKIP", "28")
        )
        self._cachesage_agent_prefix_blocks: int = int(
            os.environ.get("CACHESAGE_AGENT_PREFIX_BLOCKS", "8")
        )
        # Dedupe agent observations within contiguous cache_full_blocks
        # calls for the same request (e.g. multiple KV cache groups fire
        # one call each, and chunked prefill may fire several). We
        # observe only when the fingerprint changes.
        self._cachesage_last_agent_id: str | None = None
        # Weight on agent-level survival probability when computing the
        # eviction score. score = W * p_survive + recency. Typical
        # p_survive in [0, 0.5], recency in [0, 1], so W ≥ 2 makes
        # predictions dominate stale recency for cold-but-predicted
        # agents, while W ≤ 8 keeps just-touched agents from being
        # evicted accidentally.
        self._cachesage_predict_weight: float = float(
            os.environ.get("CACHESAGE_PREDICT_WEIGHT", "4.0")
        )
        # Predictive admission: agents with one-step-survival probability
        # below this threshold are NOT inserted into the GPU prefix
        # cache at `cache_full_blocks` time. Set to 0 to disable (all
        # requests admit as usual — identity behavior). Warm-up
        # requests (max_tokens <= 1) are always admitted so client-side
        # prefetching is unaffected.
        self._cachesage_admit_p_min: float = float(
            os.environ.get("CACHESAGE_ADMIT_P_MIN", "0.0")
        )
        # Debug / audit counters for admission decisions.
        self._cachesage_admit_stats = {"admit": 0, "reject": 0}

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        # CacheSage: observe agent fingerprint on EVERY request, even when
        # the prefix is a perfect cache hit (num_cached_blocks ==
        # num_full_blocks). The early return below skips block-cache
        # bookkeeping when there's nothing new to insert, but agent
        # transitions are an event stream — they fire once per request
        # regardless of cache state. Without this, on workloads with
        # high prefix-hit rates the coordinator never learns transitions.
        if (
            self._cachesage is not None
            and self._cachesage_policy in ("predictive", "kvflow")
            and num_cached_blocks >= num_full_blocks
            and len(request.block_hashes) > self._cachesage_agent_prefix_skip
        ):
            skip = self._cachesage_agent_prefix_skip
            take = self._cachesage_agent_prefix_blocks
            end = min(len(request.block_hashes), skip + take)
            prefix = tuple(request.block_hashes[i] for i in range(skip, end))
            agent_id_hit = str(hash(prefix))
            is_warmup_hit = getattr(request, "max_tokens", 2) <= 1
            if (not is_warmup_hit
                    and agent_id_hit != self._cachesage_last_agent_id):
                try:
                    self._cachesage.observe_agent_touch(agent_id_hit)
                except AttributeError:
                    pass
                self._cachesage_last_agent_id = agent_id_hit
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )

        # CacheSage predictive/kvflow: derive a per-agent fingerprint
        # from a slice of the request's block_hashes. We skip past the
        # shared chat-template prefix (e.g. gpt-oss harmony fills ~28
        # blocks identically across agents) and take the next N blocks
        # as the signature. The raw BlockHash is used (not the group-id
        # variant) so the fingerprint matches across KV cache groups.
        # KVFlow uses the same block→agent mapping as predictive — its
        # only difference is reading the transition graph from a frozen
        # JSON file rather than learning online.
        agent_id: str | None = None
        if (
            self._cachesage is not None
            and self._cachesage_policy in ("predictive", "kvflow")
            and len(block_hashes) > self._cachesage_agent_prefix_skip
        ):
            skip = self._cachesage_agent_prefix_skip
            take = self._cachesage_agent_prefix_blocks
            end = min(len(block_hashes), skip + take)
            prefix = tuple(block_hashes[i] for i in range(skip, end))
            agent_id = str(hash(prefix))
            # Heuristic: requests with max_tokens <= 1 are treated as
            # warm-ups from a client-side prefetcher and are NOT fed
            # into the coordinator's transition graph. This avoids
            # polluting the Markov model with spurious
            # real_event → predicted_next_event pairs generated by the
            # prefetcher itself. Block→agent mapping is still recorded
            # so predictive scoring can recognize these blocks when
            # the real request eventually fires.
            is_warmup = getattr(request, "max_tokens", 2) <= 1
            # Observe whenever the agent fingerprint changes. Dedupes
            # repeat calls for the same request (multi-group or chunked
            # prefill) but does NOT skip cache-hit requests (which have
            # num_cached_blocks > 0 but still need to register as an
            # agent fire).
            if not is_warmup and agent_id != self._cachesage_last_agent_id:
                try:
                    self._cachesage.observe_agent_touch(agent_id)
                except AttributeError:
                    pass
                self._cachesage_last_agent_id = agent_id
                if os.environ.get("CACHESAGE_DEBUG_BP"):
                    import sys as _sys
                    print(
                        f"[BlockPool.cache_full] num_full={num_full_blocks} "
                        f"num_cached={num_cached_blocks} "
                        f"fp={agent_id[:10]} group={kv_cache_group_id}",
                        file=_sys.stderr, flush=True,
                    )

        # Predictive admission: if the current request's agent has
        # predicted one-step survival below the threshold, skip the
        # GPU prefix-cache insert. The blocks will still be allocated
        # and used for this request's prefill, but they are not
        # registered as cached — future requests will not find them
        # via prefix lookup. This is the "don't pollute GPU with
        # predicted-cold content" admission policy.
        admit_to_gpu = True
        if (
            self._cachesage_admit_p_min > 0.0
            and self._cachesage is not None
            and self._cachesage_policy == "predictive"
            and agent_id is not None
            and getattr(request, "max_tokens", 2) > 1
        ):
            try:
                p = self._cachesage.predict_agent_survival(agent_id)
            except AttributeError:
                p = 1.0
            if p < self._cachesage_admit_p_min:
                admit_to_gpu = False
                self._cachesage_admit_stats["reject"] += 1
                if os.environ.get("CACHESAGE_DEBUG_ADMIT"):
                    import sys as _sys
                    print(
                        f"[BlockPool.admit] REJECT fp={agent_id[:10]} "
                        f"p={p:.3f} threshold={self._cachesage_admit_p_min}",
                        file=_sys.stderr, flush=True,
                    )
            else:
                self._cachesage_admit_stats["admit"] += 1

        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            # Always assign the block's hash so the CPU-offload store
            # path can recognise it and copy its KV data to CPU even
            # when admission rejects it from the GPU prefix cache.
            # Without this, the store scanner hits `block_hash is None`
            # and breaks out without offloading.
            blk.block_hash = block_hash_with_group_id
            if admit_to_gpu:
                self.cached_block_hash_to_block.insert(
                    block_hash_with_group_id, blk)

                # CacheSage: initialize hit count for newly cached
                # blocks. Only tracked for admitted blocks since
                # rejected ones won't be scored on the GPU tier.
                if self._cachesage is not None:
                    self._cachesage_step += 1
                    h = block_hash_with_group_id
                    self._hash_hit_count.setdefault(h, 0)
                    self._hash_last_touch[h] = self._cachesage_step
                    # Predictive/KVFlow: attribute this block to the
                    # request's agent. Both policies need block→agent for
                    # the agent-survival lookup in _update_brae_scores;
                    # they differ only in whether transitions are learned
                    # online (predictive) or read from a frozen graph
                    # (kvflow). Without this mapping the kvflow score
                    # collapses to recency = LRU.
                    if (self._cachesage_policy in ("predictive", "kvflow")
                            and agent_id):
                        self._block_to_agent[h] = agent_id

                if new_hashes is not None:
                    new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events and admit_to_gpu:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.
        When CacheSage is enabled, P0-protected cached blocks are skipped
        during eviction — they are put back and the next candidate is tried.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        if self._cachesage is not None and self.enable_caching:
            # For tiers whose free_blocks() is called once per request
            # (and thus can only score a single agent's blocks at a
            # time — the CPU offload tier), we must re-score the whole
            # free queue here so cross-request ordering reflects the
            # current predictions. Skip this refresh when scores are
            # still fresh since the last observation.
            if (
                self._cachesage_policy == "predictive"
                and getattr(self, "_cachesage_tier", "gpu") == "cpu"
            ):
                try:
                    coord_step = self._cachesage._step
                except AttributeError:
                    coord_step = 0
                last = getattr(self, "_cachesage_refresh_step", -1)
                if coord_step != last:
                    if os.environ.get("CACHESAGE_DEBUG_ORDER"):
                        import sys as _sys
                        n_free = self.get_num_free_blocks()
                        print(
                            f"[BlockPool.refresh tier=cpu] "
                            f"n_free={n_free} coord_step={coord_step}",
                            file=_sys.stderr, flush=True,
                        )
                    self.refresh_brae_scores()
                    self._cachesage_refresh_step = coord_step
            return self._get_new_blocks_brae(num_blocks)

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _get_new_blocks_brae(self, num_blocks: int) -> list[KVCacheBlock]:
        """CacheSage: allocate blocks with P0 prefix cache protection.

        When a popped block is cached and P0-protected, skip it by putting
        it back at the tail and try the next candidate.
        """
        ret: list[KVCacheBlock] = []
        skipped: list[KVCacheBlock] = []
        max_attempts = self.get_num_free_blocks() + len(skipped)
        attempts = 0

        while len(ret) < num_blocks and attempts < max_attempts:
            block = self.free_block_queue.popleft()
            attempts += 1

            if block.block_hash is not None and self._is_p0_protected(block):
                # P0-protected cached block — skip it.
                skipped.append(block)
                continue

            self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
            ret.append(block)

        # Put skipped P0 blocks back into the free queue.
        if skipped:
            self.free_block_queue.append_n(skipped)

        if len(ret) < num_blocks:
            raise ValueError(
                f"Cannot allocate {num_blocks} blocks: only {len(ret)} "
                f"non-P0 free blocks available"
            )
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block. This can
            # happen when CacheSage predictive admission rejected the
            # block from the GPU prefix cache but still stamped its
            # hash so the CPU offload scheduler could store it. Reset
            # the hash anyway so a later reuse passes the
            # `blk.block_hash is None` precondition in
            # `cache_full_blocks`.
            block.reset_hash()
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def _is_p0_protected(self, block: KVCacheBlock) -> bool:
        """CacheSage: do NOT hard-protect any block.

        Eviction ordering is already handled by BRAE scoring in free_blocks.
        Hard P0 protection risks engine deadlock when too many blocks are
        marked unreachable, which is worse than sub-optimal eviction.
        The score-ordering is sufficient — highest-hit blocks are simply
        last in the eviction queue, so they survive unless cache is
        truly saturated.
        """
        return False

    def _refresh_p0_threshold(self) -> None:
        """No-op kept for API compatibility."""
        pass

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # Predictive agent observation is driven from cache_full_blocks
        # (which has access to the full request.block_hashes) rather
        # than touch(), so touch() is not a reliable source for the
        # fingerprint window (it may fire on a prefix shorter than the
        # shared chat-template skip).
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

            # CacheSage: track cumulative hits per block hash.
            if self._cachesage is not None and block.block_hash is not None:
                self._cachesage_step += 1
                h = block.block_hash
                self._hash_hit_count[h] = self._hash_hit_count.get(h, 0) + 1
                self._hash_last_touch[h] = self._cachesage_step
                self._refresh_p0_threshold()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        When CacheSage is enabled, blocks are re-ordered by BRAE score
        (ascending) so that lowest-value blocks are evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        free_blocks = [
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ]

        if self._cachesage is not None and free_blocks:
            # CacheSage: re-order by BRAE score ascending (lowest score
            # = best eviction candidate = appended first = evicted first).
            pre_order = [b.block_id for b in free_blocks]
            self._update_brae_scores(free_blocks)
            free_blocks.sort(key=lambda b: b.brae_score)
            if os.environ.get("CACHESAGE_DEBUG_ORDER"):
                post_order = [b.block_id for b in free_blocks]
                reordered = sum(1 for i, bid in enumerate(pre_order)
                                if post_order[i] != bid)
                import sys as _sys
                tier = getattr(self, "_cachesage_tier", "?")
                print(
                    f"[BlockPool.free tier={tier}] n={len(free_blocks)} "
                    f"reordered={reordered} "
                    f"score_range=[{free_blocks[0].brae_score:.3f},"
                    f"{free_blocks[-1].brae_score:.3f}]",
                    file=_sys.stderr, flush=True,
                )

        self.free_block_queue.append_n(free_blocks)

    def _update_brae_scores(self, blocks: list[KVCacheBlock]) -> None:
        """CacheSage: compute eviction scores.

        Three policies, selected by CACHESAGE_POLICY env var:

        - "brae" (default): score = hit_count * exp(-decay * age)
          Protects frequently-touched blocks like shared system prompts.
          Essentially LFU-with-decay on block hashes.

        - "predictive": score = predict_survival(hash, K) + recency
          Uses learned block-to-block Markov transitions to predict which
          cached blocks will be touched soon. Conditions on the current
          active state (recent touches), not just past statistics.

        - "kvflow": same scoring path as "predictive" but the agent
          transition table is loaded from a static JSON file at startup
          and frozen — no online learning. Mirrors KVFlow's "declared
          Agent Step Graph" assumption.
        """
        step = self._cachesage_step
        decay = self._cachesage_decay
        # Both predictive and kvflow take the survival-prediction code
        # path; they differ only in whether transitions are learned.
        predictive = self._cachesage_policy in ("predictive", "kvflow")

        _debug_score = bool(os.environ.get("CACHESAGE_DEBUG_SCORE"))
        if _debug_score:
            _debug_score_stats = {
                "count": 0, "has_agent": 0, "p_sum": 0.0, "p_max": 0.0,
            }
        for block in blocks:
            if block.block_hash is None:
                block.brae_score = 0.0
                continue
            h = block.block_hash
            if predictive:
                # Agent-level survival: look up the agent_id this block
                # belongs to (first block hash of its originating request)
                # and query P(agent fires within K steps). Unlike block
                # Markov (probability diluted across ~160 blocks/agent),
                # agent Markov concentrates mass over ~16 agents, so
                # predictions actually dominate recency when they should.
                agent_id = self._block_to_agent.get(h)
                p_survive = 0.0
                if agent_id is not None:
                    try:
                        p_survive = self._cachesage.predict_agent_survival(
                            agent_id)
                    except AttributeError:
                        p_survive = 0.0
                last = self._hash_last_touch.get(h, 0)
                age = max(step - last, 0)
                recency = math.exp(-decay * age)
                # Agent-level p_survive can be meaningful (e.g., 0.3 on a
                # clean chain), so we weight it higher than block-level.
                block.brae_score = (
                    self._cachesage_predict_weight * p_survive + recency
                )
                if _debug_score:
                    _debug_score_stats["count"] += 1
                    _debug_score_stats["has_agent"] += (
                        1 if agent_id is not None else 0
                    )
                    _debug_score_stats["p_sum"] += p_survive
                    _debug_score_stats["p_max"] = max(
                        _debug_score_stats["p_max"], p_survive
                    )
            else:
                _debug_score = False  # only predictive path emits stats
                # Default: hit-count × temporal decay.
                hits = self._hash_hit_count.get(h, 0)
                last = self._hash_last_touch.get(h, 0)
                age = max(step - last, 0)
                block.brae_score = hits * math.exp(-decay * age)

        if _debug_score and _debug_score_stats["count"] > 0:
            import sys as _sys
            s = _debug_score_stats
            pavg = s["p_sum"] / max(s["count"], 1)
            print(
                f"[BlockPool.scores] n={s['count']} "
                f"with_agent={s['has_agent']} "
                f"p_avg={pavg:.4f} p_max={s['p_max']:.4f}",
                file=_sys.stderr, flush=True,
            )

    def refresh_brae_scores(self) -> None:
        """CacheSage: re-score all free blocks and re-order the free queue.

        Call this on phase transitions when protection classes change
        (e.g., BRANCH_PRUNE demotes P0→P2).
        """
        if self._cachesage is None:
            return
        # Drain the free queue, re-score, re-sort, and re-insert.
        free_blocks = self.free_block_queue.get_all_free_blocks()
        if not free_blocks:
            return
        # Remove all from queue.
        for block in free_blocks:
            self.free_block_queue.remove(block)
        # Re-score and re-sort.
        self._update_brae_scores(free_blocks)
        free_blocks.sort(key=lambda b: b.brae_score)
        # Re-insert in new order.
        self.free_block_queue.append_n(free_blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
