"""Score API request body construction."""

from __future__ import annotations

from typing import Any

from profiler.config import WorkloadConfig


# Use a single repeated word for text-mode payloads.
_FILLER_WORD = "hello "


def _generate_text(target_tokens: int) -> str:
    """Generate filler text of approximately `target_tokens` tokens."""
    return _FILLER_WORD * target_tokens


def _normalize_buckets(raw: list[int], fallback: int) -> list[int]:
    if not raw:
        return [int(fallback)]
    out = sorted({int(v) for v in raw if int(v) > 0})
    if not out:
        return [int(fallback)]
    return out


def _pick_bucket(
    *,
    logical: int,
    buckets: list[int],
    strict: bool,
    dim_name: str,
    workload_name: str,
) -> tuple[int, bool]:
    logical = int(logical)
    for bucket in buckets:
        if logical <= int(bucket):
            return int(bucket), False
    if strict:
        raise ValueError(
            f"workload={workload_name} {dim_name}={logical} exceeds max bucket {buckets[-1]}"
        )
    # Non-strict mode permits a dynamic fallback shape that is outside approved buckets.
    return logical, True


def _build_padded_token_sequence(
    *,
    logical_len: int,
    bucket_len: int,
    fill_token_id: int,
    pad_token_id: int,
) -> list[int]:
    if bucket_len < logical_len:
        raise ValueError(f"bucket_len({bucket_len}) < logical_len({logical_len})")
    return ([int(fill_token_id)] * int(logical_len)) + (
        [int(pad_token_id)] * int(bucket_len - logical_len)
    )


def build_score_request_with_shape_contract(
    workload: WorkloadConfig,
    model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a /v1/score request plus shape-contract diagnostics."""
    logical_query = int(workload.query_tokens)
    logical_items = int(workload.num_items)
    logical_item_tokens = int(workload.item_tokens)

    query_buckets = _normalize_buckets(workload.query_token_buckets, logical_query)
    item_buckets = _normalize_buckets(workload.item_token_buckets, logical_item_tokens)
    num_items_buckets = _normalize_buckets(workload.num_items_buckets, logical_items)

    strict = bool(workload.enforce_shape_contract)
    query_bucket, query_dynamic_fallback = _pick_bucket(
        logical=logical_query,
        buckets=query_buckets,
        strict=strict,
        dim_name="query_tokens",
        workload_name=workload.name,
    )
    item_bucket, item_dynamic_fallback = _pick_bucket(
        logical=logical_item_tokens,
        buckets=item_buckets,
        strict=strict,
        dim_name="item_tokens",
        workload_name=workload.name,
    )
    num_items_bucket, num_items_dynamic_fallback = _pick_bucket(
        logical=logical_items,
        buckets=num_items_buckets,
        strict=strict,
        dim_name="num_items",
        workload_name=workload.name,
    )

    payload_mode = "token_ids" if bool(workload.use_token_ids) else "text"
    if strict and payload_mode != "token_ids":
        raise ValueError(
            f"workload={workload.name} enforce_shape_contract=true requires use_token_ids=true"
        )

    if payload_mode == "token_ids":
        query_payload: list[int] = _build_padded_token_sequence(
            logical_len=logical_query,
            bucket_len=query_bucket,
            fill_token_id=workload.query_fill_token_id,
            pad_token_id=workload.pad_token_id,
        )
        item_real = _build_padded_token_sequence(
            logical_len=logical_item_tokens,
            bucket_len=item_bucket,
            fill_token_id=workload.item_fill_token_id,
            pad_token_id=workload.pad_token_id,
        )
        item_pad = [int(workload.pad_token_id)] * int(item_bucket)
        items_payload: list[list[int]] = [list(item_real) for _ in range(logical_items)]
        if num_items_bucket > logical_items:
            items_payload.extend(
                [list(item_pad) for _ in range(int(num_items_bucket - logical_items))]
            )
    else:
        query_payload = _generate_text(logical_query)
        items_payload = [_generate_text(logical_item_tokens) for _ in range(logical_items)]

    request_body: dict[str, Any] = {
        "model": model,
        "query": query_payload,
        "items": items_payload,
        "label_token_ids": workload.label_token_ids,
        "apply_softmax": workload.apply_softmax,
    }

    request_num_items = len(items_payload)
    item_lengths = [len(item) if isinstance(item, list) else len(str(item)) for item in items_payload]
    item_len_min = min(item_lengths) if item_lengths else 0
    item_len_max = max(item_lengths) if item_lengths else 0
    request_query_len = len(query_payload) if isinstance(query_payload, list) else len(str(query_payload))

    request_matches_bucket = (
        payload_mode == "token_ids"
        and request_query_len == query_bucket
        and request_num_items == num_items_bucket
        and item_len_min == item_bucket
        and item_len_max == item_bucket
    )
    bucket_shape_is_approved = (
        query_bucket in query_buckets
        and num_items_bucket in num_items_buckets
        and item_bucket in item_buckets
    )
    dynamic_bucket_fallback = {
        "query_tokens": bool(query_dynamic_fallback),
        "num_items": bool(num_items_dynamic_fallback),
        "item_tokens": bool(item_dynamic_fallback),
    }

    violations: list[str] = []
    if strict and payload_mode != "token_ids":
        violations.append("strict_contract_requires_token_ids")
    if strict and not request_matches_bucket:
        violations.append("request_shape_outside_bucket")
    if not bucket_shape_is_approved:
        violations.append("dynamic_bucket_fallback")

    shape_contract = {
        "enabled": strict,
        "use_token_ids": payload_mode == "token_ids",
        "logical_shape": {
            "query_tokens": logical_query,
            "num_items": logical_items,
            "item_tokens": logical_item_tokens,
        },
        "bucket_shape": {
            "query_tokens": query_bucket,
            "num_items": num_items_bucket,
            "item_tokens": item_bucket,
        },
        "approved_buckets": {
            "query_tokens": query_buckets,
            "num_items": num_items_buckets,
            "item_tokens": item_buckets,
        },
        "padding_added": {
            "query_tokens": max(0, query_bucket - logical_query),
            "num_items": max(0, num_items_bucket - logical_items),
            "item_tokens": max(0, item_bucket - logical_item_tokens),
        },
        "request_shape": {
            "query_len": request_query_len,
            "num_items": request_num_items,
            "item_len_min": item_len_min,
            "item_len_max": item_len_max,
        },
        "bucket_shape_is_approved": bucket_shape_is_approved,
        "dynamic_bucket_fallback": dynamic_bucket_fallback,
        "request_matches_bucket": request_matches_bucket,
        "violations": violations,
    }
    return request_body, shape_contract


def build_score_request(
    workload: WorkloadConfig,
    model: str,
) -> dict[str, Any]:
    """Build only the /v1/score request body from workload config."""
    request_body, _ = build_score_request_with_shape_contract(workload, model)
    return request_body
