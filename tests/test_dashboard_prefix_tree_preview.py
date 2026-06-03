from dashboard.backend.prefix_tree_preview import build_prefix_tree_preview


def _step(step_key, prompt_uid, trajectory_uid, step_index, tokens):
    split = max(1, len(tokens) // 2)
    return {
        "step_key": step_key,
        "prompt_uid": prompt_uid,
        "trajectory_uid": trajectory_uid,
        "step_index": step_index,
        "prompt_ids": tokens[:split],
        "response_ids": tokens[split:],
    }


def test_prefix_tree_preview_shared_prefix():
    preview = build_prefix_tree_preview(
        [
            _step("a:0", "p", "a", 0, [1, 2, 3, 4, 5]),
            _step("b:0", "p", "b", 0, [1, 2, 3, 6, 7]),
            _step("c:0", "p", "c", 0, [1, 2, 3, 6, 7, 8, 9]),
        ]
    )

    assert preview["original_tokens"] == 17
    assert preview["packed_tokens"] == 9
    assert preview["token_ratio"] == 9 / 17
    assert preview["nodes"][0]["tokens"] == [1, 2, 3]
    assert preview["sequence_paths"][2]["node_ranges"] == [[0, 2], [5, 6], [7, 8]]


def test_prefix_tree_preview_no_shared_prefix():
    preview = build_prefix_tree_preview(
        [
            _step("a:0", "p", "a", 0, [1, 2, 3]),
            _step("b:0", "p", "b", 0, [4, 5, 6]),
            _step("c:0", "p", "c", 0, [7, 8, 9]),
        ]
    )

    assert preview["packed_tokens"] == 9
    assert preview["token_ratio"] == 1.0


def test_prefix_tree_preview_identical_sequences():
    preview = build_prefix_tree_preview(
        [
            _step("a:0", "p", "a", 0, [1, 2, 3, 4]),
            _step("b:0", "p", "b", 0, [1, 2, 3, 4]),
        ]
    )

    assert preview["packed_tokens"] == 4
    assert preview["token_ratio"] == 0.5
    assert preview["nodes"][0]["sequence_ids"] == [0, 1]
