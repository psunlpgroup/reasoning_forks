import subprocess
import tempfile
import os
import sys
import json
import textwrap
import re

class SandboxExecutor:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def _build_script(self, user_code: str):
        import textwrap

        lines = user_code.strip().split("\n")
        *body, last = lines

        body_code = "\n".join(body)
        last_line = last.strip()

        return textwrap.dedent(f"""
    import sys
    import builtins as _builtins

    # 🔥 Save original import BEFORE overriding
    _real_import = _builtins.__import__

    SAFE_MODULES = {{
        "math",
        "sympy",
        "mpmath",
        "numpy",
    }}

    BLOCKED_MODULES = {{
        "os",
        "subprocess",
        "socket",
    }}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        
        # Check who is calling the import
        caller = globals.get("__name__", "") if globals else ""

        # ❌ block dangerous modules ONLY if the user script is calling them
        if root in BLOCKED_MODULES and caller == "__main__":
            raise ImportError(f"Import of '{{name}}' is blocked")

        # ✅ allow safe modules + internal library imports
        return _real_import(name, globals, locals, fromlist, level)

    SAFE_BUILTINS = dict(_builtins.__dict__)

    # ❌ Remove dangerous functions
    # NOTE: If you run into `NameError: name 'eval' is not defined`, 
    # you may need to restore `eval` and `compile`, as sympy uses them heavily internally.
    for k in [
        "input",
        "__import__",  
    ]:
        SAFE_BUILTINS.pop(k, None)

    # ✅ Add controlled import back
    SAFE_BUILTINS["__import__"] = safe_import

    # Override builtins safely
    _builtins.__dict__.clear()
    _builtins.__dict__.update(SAFE_BUILTINS)

    # REMOVED: sys.modules["os"] = None (This breaks library internals)

    try:
{textwrap.indent(body_code, " " * 8)}

        _result = {last_line}
        print(_result)

    except Exception as e:
        print("ERROR:", e)
    """)

    # def execute(self, code: str):
    #     with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    #         script = self._build_script(code)
    #         f.write(script)
    #         script_path = f.name

    #     try:
    #         result = subprocess.run(
    #             [sys.executable, script_path],
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             timeout=self.timeout,
    #             text=True
    #         )

    #         output = result.stdout.strip()
    #         error = result.stderr.strip()

    #         if result.returncode != 0:
    #             return "", error or "Runtime Error"

    #         return output, "Done"

    #     except subprocess.TimeoutExpired:
    #         return "", "Timeout"

    #     finally:
    #         os.remove(script_path)

    def execute(self, code: str):
        script = self._build_script(code)

        try:
            # Pass the script directly via 'input' instead of a file
            result = subprocess.run(
                [sys.executable], 
                input=script,            # <-- Send code directly to stdin
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                text=True
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode != 0:
                return "", error or "Runtime Error"

            return output, "Done"

        except subprocess.TimeoutExpired:
            return "", "Timeout"

def extract_llm_code(text: str):
    pattern = r"<llm-code>(.*?)</llm-code>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]

def run_llm_code(text, executor):
    codes = extract_llm_code(text)
    outputs = []
    if len(codes) == 0:
        return outputs

    # for code in codes:

    code = codes[-1]
    result, status = executor.execute(code)
    outputs.append({
        "code": code,
        "result": result,
        "status": status
    })

    return outputs

# ---------------- Example ----------------
def _test():
    text = """Let's solve this problem using Python's sympy library.
<llm-code>
import sympy as sp

# Let's denote the number of boxes of pizza as x
x = sp.symbols('x')

# total cost of pizza is 50
total_cost_of_pizza = x * 8.50

# total cost of all the items is 50
eq = sp.Eq(total_cost_of_pizza + 12 + 5 * 3 + 4 * 1.5, 50)

# solving for x we get the answer
sp.solve(eq, x)[0]
</llm-code>
<llm-code-output>
5
</llm-code-output>
Thus Marie ordered \boxed{5} boxes of pizza.
"""

    executor = SandboxExecutor()
    results = run_llm_code(text, executor)

    print(results)


if __name__ == "__main__":
    _test()