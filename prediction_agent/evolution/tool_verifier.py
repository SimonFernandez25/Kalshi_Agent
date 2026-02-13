"""
Tool verifier -- three-phase gatekeeper for generated tools.

Phase 1: Static AST inspection (block dangerous imports/calls)
Phase 2: Runtime sandbox execution (timeout, output validation)
Phase 3: Determinism check (identical outputs across multiple runs)
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from config import TOOL_VERIFY_DETERMINISM_RUNS, TOOL_VERIFY_TIMEOUT_SEC
from prediction_agent.evolution.schemas import ToolSpec, VerificationResult
from schemas import EventInput, ToolOutput
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Imports that are never allowed in generated tools
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
})

# Function calls that are never allowed
_BLOCKED_CALLS = frozenset({
    "exec",
    "eval",
    "__import__",
    "compile",
    "globals",
    "locals",
})


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

    # Phase 1: AST inspection
    ast_ok, ast_reason = _check_ast(tool_path)
    checks["ast_inspection"] = ast_ok
    if not ast_ok:
        return VerificationResult(
            tool_name=spec.tool_name,
            passed=False,
            checks=checks,
            rejection_reason=f"AST inspection failed: {ast_reason}",
        )

    # Phase 2: Runtime sandbox
    runtime_ok, runtime_reason, tool_class = _check_runtime(tool_path, spec)
    checks["runtime_sandbox"] = runtime_ok
    if not runtime_ok:
        return VerificationResult(
            tool_name=spec.tool_name,
            passed=False,
            checks=checks,
            rejection_reason=f"Runtime sandbox failed: {runtime_reason}",
        )

    # Phase 3: Determinism
    det_ok, det_reason = _check_determinism(tool_class, TOOL_VERIFY_DETERMINISM_RUNS)
    checks["determinism"] = det_ok
    if not det_ok:
        return VerificationResult(
            tool_name=spec.tool_name,
            passed=False,
            checks=checks,
            rejection_reason=f"Determinism check failed: {det_reason}",
        )

    logger.info("Tool '%s' passed all verification checks.", spec.tool_name)
    
    # Manual Approval Gate (if enabled)
    from config import EVOLUTION_REQUIRE_MANUAL_APPROVAL, GENERATED_TOOLS_DIR
    import shutil
    import json
    
    if EVOLUTION_REQUIRE_MANUAL_APPROVAL:
        # Write to pending directory for manual review
        pending_dir = GENERATED_TOOLS_DIR / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        
        pending_path = pending_dir / f"{spec.tool_name}.py"
        shutil.copy(tool_path, pending_path)
        
        # Generate review document
        review_path = pending_dir / f"{spec.tool_name}_REVIEW.md"
        review_content = f"""# Manual Review Required: {spec.tool_name}

## Tool Specification
- **Description**: {spec.description}
- **Risk Level**: {spec.risk_level.value}
- **Deterministic**: {spec.deterministic}
- **Expected Accuracy Gain**: {spec.expected_accuracy_gain:.2%}
- **Expected Token Reduction**: {spec.expected_token_reduction}
- **Data Sources**: {', '.join(spec.data_sources) if spec.data_sources else 'None specified'}

## Verification Results
All checks passed:
```json
{json.dumps(checks, indent=2)}
```

## Tool Location
- **Pending File**: `{pending_path}`
- **Verification Date**: {datetime.now(timezone.utc).isoformat()}

## Review Instructions

### To Approve This Tool:
1. Review the generated code at `{pending_path}`
2. Verify the implementation matches the specification
3. Move the file to the parent directory:
   ```bash
   mv "{pending_path}" "{GENERATED_TOOLS_DIR / f'{spec.tool_name}.py'}"
   ```
4. Add entry to `approved.json`:
   ```json
   {{
     "tool_name": "{spec.tool_name}",
     "version": "0.1.0"
   }}
   ```

### To Reject This Tool:
```bash
rm "{pending_path}"
rm "{review_path}"
```

## Generated Code Preview
First 50 lines of `{spec.tool_name}.py`:
```python
{tool_path.read_text(encoding='utf-8').splitlines()[:50] if tool_path.exists() else 'N/A'}
```
"""
        review_path.write_text(review_content, encoding="utf-8")
        
        logger.info(
            "Tool '%s' written to pending/ directory - manual approval required", 
            spec.tool_name
        )
        print(f"\n{'='*60}")
        print(f"NEW TOOL REQUIRES MANUAL REVIEW")
        print(f"{'='*60}")
        print(f"Tool: {spec.tool_name}")
        print(f"Location: {pending_path}")
        print(f"Review doc: {review_path}")
        print(f"{'='*60}\n")
    else:
        logger.info("Tool '%s' auto-approved (manual approval disabled)", spec.tool_name)
    
    return VerificationResult(
        tool_name=spec.tool_name,
        passed=True,
        checks=checks,
        rejection_reason=None,
    )


# ------------------------------------------------------------------
# Phase 1: AST Inspection
# ------------------------------------------------------------------

def _check_ast(tool_path: Path) -> tuple[bool, str]:
    """
    Parse the file with the ast module and check for disallowed constructs.

    Returns:
        (passed, reason) tuple.
    """
    try:
        source = tool_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(tool_path))
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in _BLOCKED_IMPORTS:
                    return False, f"Blocked import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module in _BLOCKED_IMPORTS:
                    return False, f"Blocked import from: {node.module}"

        # Check function calls
        elif isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in _BLOCKED_CALLS:
                return False, f"Blocked function call: {func_name}"

            # Block os.system, subprocess.run, etc.
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in _BLOCKED_IMPORTS:
                        return False, f"Blocked call: {node.func.value.id}.{node.func.attr}"

            # Block open() in write mode
            if func_name == "open" and len(node.args) >= 2:
                mode_arg = node.args[1]
                if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                    if "w" in mode_arg.value or "a" in mode_arg.value:
                        return False, "Blocked: open() in write/append mode"

            # Also check keyword mode argument
            if func_name == "open":
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        if isinstance(kw.value.value, str) and ("w" in kw.value.value or "a" in kw.value.value):
                            return False, "Blocked: open() in write/append mode"

    return True, ""


def _get_call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


# ------------------------------------------------------------------
# Phase 2: Runtime Sandbox
# ------------------------------------------------------------------

def _check_runtime(
    tool_path: Path,
    spec: ToolSpec,
) -> tuple[bool, str, Optional[Type[BaseTool]]]:
    """
    Dynamically import the tool, instantiate it, and run with a synthetic event.

    Returns:
        (passed, reason, tool_class) tuple.
    """
    # Dynamic import
    try:
        module_spec = importlib.util.spec_from_file_location(
            f"generated_{spec.tool_name}",
            str(tool_path),
        )
        if module_spec is None or module_spec.loader is None:
            return False, "Could not create module spec", None

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    except Exception as exc:
        return False, f"Import failed: {exc}", None

    # Find the BaseTool subclass
    tool_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseTool)
            and attr is not BaseTool
        ):
            tool_class = attr
            break

    if tool_class is None:
        return False, "No BaseTool subclass found in module", None

    # Instantiate and run with timeout
    synthetic_event = EventInput(
        event_id="verify-evt-001",
        market_id="VERIFY-MKT-001",
        market_title="Verification Test Market",
        current_price=0.50,
        timestamp=datetime.now(timezone.utc),
    )

    result_holder: Dict[str, Any] = {"output": None, "error": None}

    def _run_tool():
        try:
            tool_instance = tool_class()
            output = tool_instance.run(synthetic_event)
            result_holder["output"] = output
        except Exception as exc:
            result_holder["error"] = str(exc)

    thread = threading.Thread(target=_run_tool, daemon=True)
    thread.start()
    thread.join(timeout=TOOL_VERIFY_TIMEOUT_SEC)

    if thread.is_alive():
        return False, f"Tool execution timed out after {TOOL_VERIFY_TIMEOUT_SEC}s", None

    if result_holder["error"] is not None:
        return False, f"Runtime error: {result_holder['error']}", None

    output = result_holder["output"]

    # Validate output type
    if not isinstance(output, ToolOutput):
        return False, f"Output is not ToolOutput, got {type(output).__name__}", None

    # Validate output_vector is numeric
    if not output.output_vector:
        return False, "output_vector is empty", None

    for i, val in enumerate(output.output_vector):
        if not isinstance(val, (int, float)):
            return False, f"output_vector[{i}] is not numeric: {type(val).__name__}", None

    return True, "", tool_class


# ------------------------------------------------------------------
# Phase 3: Determinism Check
# ------------------------------------------------------------------

def _check_determinism(
    tool_class: Type[BaseTool],
    num_runs: int,
) -> tuple[bool, str]:
    """
    Run the tool multiple times with identical input and verify identical output.
    """
    synthetic_event = EventInput(
        event_id="verify-evt-001",
        market_id="VERIFY-MKT-001",
        market_title="Verification Test Market",
        current_price=0.50,
        timestamp=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )

    outputs: List[List[float]] = []

    for i in range(num_runs):
        try:
            tool_instance = tool_class()
            result = tool_instance.run(synthetic_event)
            outputs.append(result.output_vector)
        except Exception as exc:
            return False, f"Run {i + 1} crashed: {exc}"

    # Compare all outputs to the first
    reference = outputs[0]
    for i, output in enumerate(outputs[1:], start=2):
        if output != reference:
            return False, (
                f"Non-deterministic: run 1 produced {reference}, "
                f"run {i} produced {output}"
            )

    return True, ""
