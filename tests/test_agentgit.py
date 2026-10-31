from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentgit import (
    AgentGitRepo,
    CodingAgentSandboxTools,
    TokenizerBridge,
    run_end_to_end_debugging_simulation,
    run_quantitative_benchmarks,
)


@pytest.fixture()
def temp_workspace():
    tmp = Path(tempfile.mkdtemp(prefix="agentgit_pytest_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end_simulation_and_bisect(temp_workspace: Path):
    sim = run_end_to_end_debugging_simulation(work_dir=temp_workspace)
    repo: AgentGitRepo = sim["repo"]
    try:
        main_commits = sim["main_commits"]
        fork_commits = sim["fork_commits"]
        bisect_rep = sim["bisect_report"]

        assert len(main_commits) == 10
        assert len(fork_commits) == 3
        assert bisect_rep["culprit_step_idx"] == 5
        assert bisect_rep["culprit_commit_id"] == main_commits[4].commit_id
        assert bisect_rep["bisect_probes_executed"] <= 4

        # Verify CoW KV-cache sharing across fork
        f5 = fork_commits[0]
        c4 = main_commits[3]
        shared_blocks = len(c4.kv_snapshot.block_hashes)
        assert f5.kv_snapshot.block_hashes[:shared_blocks] == c4.kv_snapshot.block_hashes
        assert f5.kv_snapshot.prefill_tokens_reused >= shared_blocks * repo.kv_manager.block_size

        # Verify divergent status
        assert main_commits[-1].tool_status == "TEST_FAIL"
        assert fork_commits[-1].tool_status == "TEST_PASS"

        # Verify checkout prunes untracked future files
        chk_c2 = repo.checkout(main_commits[1].commit_id)
        assert not (repo.workspace_dir / "config.py").exists()
        assert chk_c2["restored_kv_shape"][0] == 2
    finally:
        repo.close()


def test_automatic_lru_kv_cache_eviction_and_hydration(temp_workspace: Path):
    repo = AgentGitRepo(
        temp_workspace,
        block_size=16,
        num_layers=4,
        num_kv_heads=2,
        head_dim=32,
        max_ram_blocks=6,
    )
    try:
        tools = CodingAgentSandboxTools(repo)
        c1 = tools.execute_and_commit(
            "Step 1", "write_file", {"path": "mod_a.py", "content": "a = 1\n" * 100}
        )
        repo.fork(c1.commit_id, "exp/lru-branch")
        tools.execute_and_commit(
            "Step 2", "write_file", {"path": "mod_b.py", "content": "b = 2\n" * 120}
        )

        telem = repo.kv_manager.memory_telemetry()
        assert telem["auto_evictions"] > 0
        assert telem["in_ram_blocks"] <= 6

        # Checking out c1 must transparently hydrate offloaded blocks from disk
        chk = repo.checkout(c1.commit_id)
        assert chk["restored_tokens"] == c1.kv_snapshot.total_tokens
        assert repo.kv_manager.total_auto_hydrations > 0
    finally:
        repo.close()


def test_huggingface_past_key_values_bridge(temp_workspace: Path):
    repo = AgentGitRepo(
        temp_workspace, block_size=16, num_layers=4, num_kv_heads=2, head_dim=16
    )
    try:
        hf_layers = [
            (
                np.ones((1, 2, 32, 16), dtype=np.float16) * (i + 1),
                np.ones((1, 2, 32, 16), dtype=np.float16) * -(i + 1),
            )
            for i in range(4)
        ]
        extracted = repo.kv_manager.extract_from_hf_past_key_values(hf_layers)
        assert extracted.shape == (2, 4, 2, 32, 16)

        snap, _ = repo.kv_manager.append_sequence_and_snapshot(
            list(range(32)), external_kv_tensors=extracted
        )
        restored, _ = repo.kv_manager.export_to_hf_past_key_values(snap, as_torch=False)
        assert len(restored) == 4
        assert np.allclose(restored[0][0], hf_layers[0][0])
        assert np.allclose(restored[3][1], hf_layers[3][1])
    finally:
        repo.close()


def test_sandbox_path_traversal_and_metadata_guard(temp_workspace: Path):
    repo = AgentGitRepo(temp_workspace)
    try:
        tools = CodingAgentSandboxTools(repo)
        c_escape = tools.execute_and_commit(
            "Escape attempt", "write_file", {"path": "../../outside.py", "content": "x=1"}
        )
        assert c_escape.tool_status == "ERROR"
        assert "PermissionError" in c_escape.tool_output

        c_meta = tools.execute_and_commit(
            "Meta overwrite attempt",
            "write_file",
            {"path": ".agentgit/dag.sqlite3", "content": "corrupt"},
        )
        assert c_meta.tool_status == "ERROR"
        assert "PermissionError" in c_meta.tool_output
    finally:
        repo.close()


def test_binary_blob_diff_and_tokenizer_bridge(temp_workspace: Path):
    repo = AgentGitRepo(temp_workspace, tokenizer_backend="auto")
    try:
        (temp_workspace / "model.bin").write_bytes(b"\x00\x01\x02\x03weights_v1")
        m1, _ = repo.workspace_store.snapshot_workspace(temp_workspace)
        (temp_workspace / "model.bin").write_bytes(b"\x00\x01\x02\x03weights_v2_new")
        m2, _ = repo.workspace_store.snapshot_workspace(temp_workspace)

        diff = repo.workspace_store.diff_trees(m1.tree_hash, m2.tree_hash)
        assert "model.bin" in diff["modified"]
        assert "Binary files" in diff["unified_diffs"]["model.bin"]

        bridge = TokenizerBridge(backend="deterministic")
        tokens = bridge.encode("def settle_batch(fills): return 0")
        assert len(tokens) > 0
    finally:
        repo.close()


def test_quantitative_benchmark_suite():
    res = run_quantitative_benchmarks()
    assert res["total_commits_recorded"] == 32
    assert res["branches_evaluated"] == 5
    assert res["mean_fork_kv_prefix_hit_rate_pct"] > 70.0
    assert res["workspace_merkle_dedup_ratio_x"] > 1.0
