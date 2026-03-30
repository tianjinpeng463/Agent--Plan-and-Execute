"""Execution loop helper utilities.

Low-level helpers used by run_exec_loop() and run_react_loop():
  - Tool invocation and step status updates
  - Tool result trimming to prevent context overflow
  - Sliding window message history management
  - Watchdog detection of repeatedly failing tools
  - Replan wrapper with timeout handling
  - apply_fixers: single entry point for all tool-call corrections
"""

import asyncio

from langchain_core.tools import ToolException

from agent.base.fixers import _fix_args, _fix_content, _fix_tool_name
from config import (
    FEATURES,
    MESSAGE_WINDOW_HEAD,
    MESSAGE_WINDOW_SIZE,
    TOOL_RESULT_DEFAULT_MAX_CHARS,
    TOOL_RESULT_MAX_CHARS,
)
from core.models import Step


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------

async def _invoke_tool(tc: dict, tool_map: dict) -> tuple[str, bool]:
    """调用工具并返回 (result_str, is_error)。

    Unknown tool names return a descriptive error instead of raising KeyError.
    """
    if tc["name"] not in tool_map:
        available = ", ".join(sorted(tool_map.keys()))
        return (
            f"Error: Unknown tool '{tc['name']}'. "
            f"Available tools: {available}",
            True,
        )
    try:
        result = await tool_map[tc["name"]].ainvoke(tc["args"])
        result_str = str(result)
        is_error = "Error:" in result_str or "SQL error:" in result_str
    except ToolException as e:
        result_str = f"Tool error: {e}"
        is_error = True
    except Exception as e:
        # Catches McpError (Connection closed), asyncio.CancelledError leakage, etc.
        result_str = f"Tool error: {type(e).__name__}: {e}"
        is_error = True
    return result_str, is_error


# ---------------------------------------------------------------------------
# Step status
# ---------------------------------------------------------------------------

def _update_step(steps: list[Step], idx: int, is_error: bool, result_str: str) -> int:
    """更新当前步骤状态并返回下一个索引。"""
    if idx < len(steps):
        if is_error:
            steps[idx].status = "failed"
            steps[idx].note = result_str[:120]
        else:
            steps[idx].status = "done"
            steps[idx].note = result_str[:80]
            idx += 1
    return idx


# ---------------------------------------------------------------------------
# Fixer pipeline
# ---------------------------------------------------------------------------

def apply_fixers(
    tc: dict,
    tool_map: dict,
    logger,
) -> tuple[dict, str | None, list[str]]:
    """Apply tool_name_fixer, arg_fixer, and content_fixer to a tool call.

    Consolidates all three correction passes so loops only depend on this
    module rather than importing agent.base.fixers directly.

    Returns:
        (fixed_tc, tool_name_fix, arg_fixes)
        tool_name_fix — fix description string, or None if unchanged
        arg_fixes     — list of arg fix descriptions (empty if none)
    """
    tool_name_fix: str | None = None
    arg_fixes: list[str] = []

    if FEATURES.get("tool_name_fixer", True):
        tc, tool_name_fix = _fix_tool_name(tc, tool_map)
        if tool_name_fix:
            logger.warning(f"[tool_fix] {tool_name_fix}")

    if FEATURES.get("arg_fixer", True):
        tc, arg_fixes = _fix_args(tc, tool_map)
        if arg_fixes:
            logger.warning(f"[arg_fix] {tc['name']}: {', '.join(arg_fixes)}")

    if FEATURES.get("content_fixer", True):
        tc, content_fix = _fix_content(tc)
        if content_fix:
            logger.warning(f"[content_fix] {tc['name']}: {content_fix}")

    return tc, tool_name_fix, arg_fixes


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def _trim_tool_result(tool_name: str, result: str) -> tuple[str, int]:
    """按上下文上限裁剪工具结果。

    可在 TOOL_RESULT_MAX_CHARS 中为不同工具设置上限。
    未配置的工具使用 TOOL_RESULT_DEFAULT_MAX_CHARS。
    上限为 0 时不进行裁剪。

    Returns:
        (trimmed_result, original_length)
    """
    original_len = len(result)
    limit = TOOL_RESULT_MAX_CHARS.get(tool_name, TOOL_RESULT_DEFAULT_MAX_CHARS)
    if not limit or original_len <= limit:
        return result, original_len
    trimmed = result[:limit]
    trimmed += f"\n... [truncated: {original_len - limit} chars omitted]"
    return trimmed, original_len


def _apply_window(messages: list) -> tuple[list, bool]:
    """仅保留最近 MESSAGE_WINDOW_SIZE 条消息并返回消息列表。

    前 MESSAGE_WINDOW_HEAD 条（System + Task）始终保留。
    若应用了窗口裁剪则返回 True（用于日志）。
    """
    head = messages[:MESSAGE_WINDOW_HEAD]
    tail = messages[MESSAGE_WINDOW_HEAD:]
    if len(tail) <= MESSAGE_WINDOW_SIZE:
        return messages, False
    return head + tail[-MESSAGE_WINDOW_SIZE:], True


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

def _build_watchdog_hint(tool_failure_counts: dict) -> str:
    repeated = {k: v for k, v in tool_failure_counts.items() if v >= 2}
    if not repeated:
        return ""
    return (
        f"[WATCHDOG] The following tools have failed {repeated} times total. "
        "Do NOT call them with the same arguments again. "
        "Use a completely different tool or a different approach."
    )


# ---------------------------------------------------------------------------
# Replan wrapper
# ---------------------------------------------------------------------------

async def _do_replan(
    prompt, steps, execution_history, tools, replan_model, logger,
    tool_failure_counts, remaining_fn,
    tool_map: dict | None = None,
):
    """执行重规划并返回 (new_steps, new_step_idx)，超时时返回 None。

    remaining_fn() — 返回剩余秒数的 callable（用于超时计算）。
    tool_map       — 用于 plan_tool_name_fixer。
    """
    from agent.components.planner import _apply_replan  # avoid circular import at module level

    watchdog_hint = ""
    if FEATURES.get("watchdog", True):
        watchdog_hint = _build_watchdog_hint(tool_failure_counts)
        if watchdog_hint:
            logger.warning(f"[watchdog] {watchdog_hint}")

    try:
        return await asyncio.wait_for(
            _apply_replan(
                prompt, steps, execution_history, tools, replan_model, logger,
                watchdog_hint=watchdog_hint,
                tool_map=tool_map,
            ),
            timeout=remaining_fn(),
        )
    except asyncio.TimeoutError:
        return None
