"""
Tool verifier — three-phase gatekeeper for generated tools.

Phase 1: Static AST inspection (block dangerous imports/calls + dunder escapes)
Phase 2: Runtime sandbox via isolated subprocess (sandbox_runner.py)
          — tool code NEVER executes in the main process.
          — communication is strictly JSON over stdout.
          — hard timeout, POSIX resource limits, network disabled.
Phase 3: Determinism check (identical outputs across multiple sandbox runs)
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import TOOL_VERIFY_DETERMINISM_RUNS, TOOL_VERIFY_TIMEOUT_SEC
from prediction_agent.evolution.schemas import ToolSpec, VerificationResult
from prediction_agent.evolution.sandbox_runner import SandboxResult, run_tool_in_sandbox
from schemas import EventInput

logger = logging.getLogger(__name__)

# ── Imports never allowed in generated tools ───────────────────────────────────
_BLOCKED_IMPORTS = frozenset({
    "os",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "http",
    "requests",
    "urllib",
    "random",
    "multiprocessing",
    "threading",
    "ctypes",
    "cffi",
    "pickle",
    "shelve",
    "tempfile",
    "io",
    "pathlib",
    "glob",
    "fnmatch",
})

# ── Function/builtins calls never allowed ──────────────────────────────────────
_BLOCKED_CALLS = frozenset({
    "exec",
    "eval",
    "__import__",
    "compile",
    "globals",
    "locals",
    "vars",
    "setattr",
    "delattr",
    "getattr",
    "open",
    "print",
    "input",
    "breakpoint",
})

# ── Dangerous module.method attribute call pairs ───────────────────────────────
_BLOCKED_ATTR_CALLS = frozenset({
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execve"),
    ("os", "fork"),
    ("os", "kill"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_output"),
    ("socket", "socket"),
    ("socket", "create_connection"),
    ("requests", "get"),
    ("requests", "post"),
    ("urllib", "request"),
})

# ── Dunder attributes that enable sandbox escapes ─────────────────────────────
_BLOCKED_DUNDERS = frozenset({
    "__class__", "__bases__", "__subclasses__",
    "__globals__", "__builtins__", "__import__",
    "__code__", "__closure__", "__reduce__",
})


_SYNTHETIC_EVENT = EventInput(
    event_id="verify-evt-001",
    market_id="VERIFY-MKT-001",
    market_title="Verification Test Market",
    current_price=0.50,
    timestamp=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
)


def verify_tool(tool_path: Path, spec: ToolSpec) -> VerificationResult:
    """
    Run all verification phases on a generated tool file.

    Args:
        tool_path: Path to the generated .py file.
        spec: The ToolSpec that produced this tool.

    Returns:
        VerificationResult with pass/fail status and check details.
    """
    checks: Dict[str, bool] = {}

    # ── Phase 1: AST inspection ────────────────────────────────────────────────
    ast_ok, ast_reason = _check_ast(tool_path)
    checks["ast_inspection"] = ast_ok
    if not ast_ok:
        logger.warning("Tool '%s' failed AST inspection: %s", spec.tool_name, ast_reason)
        return VerificationResult(
            tool_name=spec.tool_name,
            passed=False,
            checks=checks,
            rejection_reason=f"AST inspection failed: {ast_reason}",
        )
    logger.info("Tool '%s' passed Phase 1 (AST).", spec.tool_name)

    # ── Phase 2: Subprocess sandbox runtime ───────────────────────────────────
    sb_ok, sb_reason = _check_sandbox(tool_path, spec)
    checks["runtime_sandbox"] = sb_ok
    if not sb_ok:
        logger.warning("Tool '%s' failed subprocess sandbox: %s", spec.tool_name, sb_reason)
        return VerificationResult(
            tool_name=spec.tool_name,
            passed=False,
            checks=checks,
            rejection_reason=f"Runtime sandbox failed: {sb_reason}",
        )
    logger.info("Tool '%s' passed Phase 2 (subprocess sandbox).", spec.tool_name)

    # ── Phase 3: Determinism check ─────────────────────────────────────────────
    det_ok, det_reason = _check_determinism(tool_path, TOOL_VERIFY_DETERMINISM_RUNS)
    checks["determinism"] = det_ok
    if not det_ok:
        logger.warning("Tool '%s' failed determinism check: %s", spec.tool_name, det_reason)
        return VerificationResult(
            tool_name=spec.tool_name,
            passed=False,
            checks=checks,
            rejection_reason=f"Determinism check failed: {det_reason}",
        )
    logger.info("Tool '%s' passed all 3 verification phases.", spec.tool_name)

    # ── Manual Approval Gate ───────────────────────────────────────────────────
    from config import EVOLUTION_REQUIRE_MANUAL_APPROVAL, GENERATED_TOOLS_DIR

    if EVOLUTION_REQUIRE_MANUAL_APPROVAL:
        _write_pending(tool_path, spec, checks, GENERATED_TOOLS_DIR)
    else:
        logger.info("Tool '%s' auto-approved (manual approval disabled)", spec.tool_name)

    return VerificationResult(
        tool_name=spec.tool_name,
        passed=True,
        checks=checks,
        rejection_reason=None,
    )


# ── Phase 1: AST Inspection ────────────────────────────────────────────────────

def _check_ast(tool_path: Path) -> Tuple[bool, str]:
    """Parse the file and check for disallowed constructs. Returns (passed, reason)."""
    try:
        source = tool_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(tool_path))
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return False, f"Blocked import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return False, f"Blocked import-from: {node.module}"

        elif isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in _BLOCKED_CALLS:
                return False, f"Blocked call: {func_name}"

            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    pair = (node.func.value.id, node.func.attr)
                    if pair in _BLOCKED_ATTR_CALLS:
                        return False, f"Blocked attribute call: {node.func.value.id}.{node.func.attr}"
                    if node.func.value.id in _BLOCKED_IMPORTS:
                        return False, f"Blocked call on prohibited module: {node.func.value.id}.{node.func.attr}"

        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_DUNDERS:
                return False, f"Blocked dunder access: {node.attr}"

    return True, ""


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


# ── Phase 2: Subprocess Sandbox ────────────────────────────────────────────────

def _check_sandbox(tool_path: Path, spec: ToolSpec) -> Tuple[bool, str]:
    """
    Run the tool inside an isolated subprocess.
    Tool code NEVER executes in the main process.
    Returns (passed, reason).
    """
    event_dict = _SYNTHETIC_EVENT.model_dump(mode="json")
    result: SandboxResult = run_tool_in_sandbox(
        tool_path=tool_path,
        event_input_dict=event_dict,
        timeout_sec=TOOL_VERIFY_TIMEOUT_SEC,
    )
    if result.stderr_text:
        logger.debug("Sandbox stderr for '%s': %s", spec.tool_name, result.stderr_text[:500])
    if not result.success:
        return False, result.error or "Unknown sandbox failure"
    return True, ""


# ── Phase 3: Determinism Check ─────────────────────────────────────────────────

def _check_determinism(tool_path: Path, num_runs: int) -> Tuple[bool, str]:
    """
    Run the tool num_runs times via subprocess sandbox and verify identical output.
    """
    event_dict = _SYNTHETIC_EVENT.model_dump(mode="json")
    outputs: List[List[float]] = []

    for i in range(num_runs):
        result = run_tool_in_sandbox(
            tool_path=tool_path,
            event_input_dict=event_dict,
            timeout_sec=TOOL_VERIFY_TIMEOUT_SEC,
        )
        if not result.success:
            return False, f"Run {i + 1} failed: {result.error}"
        outputs.append(result.output_vector)

    reference = outputs[0]
    for i, output in enumerate(outputs[1:], start=2):
        if output != reference:
            return False, (
                f"Non-deterministic: run 1={reference}, run {i}={output}"
            )
    return True, ""


# ── Manual Approval Gate ───────────────────────────────────────────────────────

def _write_pending(
    tool_path: Path,
    spec: ToolSpec,
    checks: Dict[str, bool],
    generated_tools_dir: Path,
) -> None:
    """Copy verified tool to pending/ and write a human review document."""
    pending_dir = generated_tools_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    pending_path = pending_dir / f"{spec.tool_name}.py"
    shutil.copy(tool_path, pending_path)

    review_path = pending_dir / f"{spec.tool_name}_REVIEW.md"
    first_50 = "\n".join(tool_path.read_text(encoding="utf-8").splitlines()[:50]) if tool_path.exists() else "N/A"
    review_content = f"""# Manual Review Required: {spec.tool_name}

## Tool Specification
- **Description**: {spec.description}
- **Risk Level**: {spec.risk_level.value}
- **Deterministic**: {spec.deterministic}
- **Expected Accuracy Gain**: {spec.expected_accuracy_gain:.2%}
- **Expected Token Reduction**: {spec.expected_token_reduction}
- **Data Sources**: {', '.join(spec.data_sources) if spec.data_sources else 'None specified'}

## Verification Results (all 3 phases via subprocess sandbox)
```json
{json.dumps(checks, indent=2)}
```

## Sandbox Security Properties
- Tool executed in isolated subprocess — never in main process
- Network access disabled (socket.socket replaced with stub)
- Resource limits applied (POSIX: {256}MB RSS, {8}s CPU)
- Communication strictly via JSON stdout
- Hard timeout: {TOOL_VERIFY_TIMEOUT_SEC}s

## Tool Location
- **Pending File**: `{pending_path}`
- **Verification Date**: {datetime.now(timezone.utc).isoformat()}

## To Approve
```bash
mv "{pending_path}" "{generated_tools_dir / f'{spec.tool_name}.py'}"
# then add to approved.json: {{"tool_name": "{spec.tool_name}", "version": "0.1.0"}}
```

## To Reject
```bash
rm "{pending_path}" "{review_path}"
```

## Generated Code (first 50 lines)
```python
{first_50}
```
"""
    review_path.write_text(review_content, encoding="utf-8")
    logger.info("Tool '%s' written to pending/ — manual approval required.", spec.tool_name)
    print(f"\n{'='*60}")
    print("NEW TOOL REQUIRES MANUAL REVIEW")
    print(f"{'='*60}")
    print(f"Tool:       {spec.tool_name}")
    print(f"Location:   {pending_path}")
    print(f"Review doc: {review_path}")
    print(f"{'='*60}\n")
