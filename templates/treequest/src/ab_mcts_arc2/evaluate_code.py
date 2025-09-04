import multiprocessing
import os
import re
import sys
import tempfile
import time
import types
import unittest
from ast import literal_eval
from multiprocessing import Manager, Value
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np

from ab_mcts_arc2.data_types import Action, Grid, GridType
from ab_mcts_arc2.utils import (
    TIMEOUT_LIMIT,
    create_tempdir,
    reliability_guard,
    safe_environment,
    swallow_io,
    time_limit,
)


TEST_TRANSFORM = (
    Path(__file__).parent / "unittest_templates" / "test_transform.py"
).resolve()

TIMEOUT_LIMIT = 60.0


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def unsafe_execute(
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,  # Value
    details,  # Array
):
    with safe_environment(), create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import builtins
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        # Set necessary attributes for the module
        new_module.__dict__.update(
            {
                "__builtins__": builtins,
                "__file__": f"{module_name}.py",
                "__package__": None,
                "__doc__": None,
                "sys": sys,
                "os": os,
                "environ": os.environ,
            }
        )

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(
                    compile(full_code, f"{module_name}.py", "exec"), new_module.__dict__
                )
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, "TestCases")
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace
            stat.value = _SUCCESS
        except BaseException as e:
            details["ALL"] = str(e)
            stat.value = _FAILED
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    code: str,
    test_code: str,
    max_as_limit: float = 30 * 1024,
    max_data_limit: float = 30 * 1024,
    max_stack_limit: float = 10,
    min_time_limit: float = 10,
    gt_time_limit: float = 60,
) -> Tuple[str, np.ndarray]:
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = (
        max(
            int(os.getenv("TIMEOUT_PER_TASK", TIMEOUT_LIMIT)),
            min_time_limit,
        )
        + 1
    )
    # shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    # convert details to a dict
    details = dict(details)

    if not stat:
        stat = TIMEOUT
    if stat == PASS:
        if details:
            stat = FAIL

    return stat, details


def parse_funcname_from_answer(answer_code: str) -> Optional[str]:
    funcname_pat = re.compile(r"^def\s+([a-zA-Z_][a-zA-Z_0-9]*)", re.MULTILINE)

    funcnames = funcname_pat.findall(answer_code)
    if len(funcnames) == 0:
        return None
    else:
        if "transform" in funcnames:
            return "transform"
        return funcnames[-1]


def prepare_test_code(
    test_code: str,
    funcname: str,
    input: Grid,
    output: Grid,
    output_fpath: Optional[Path],
) -> str:
    """
    Populate template test code with grid input, output and task function name
    """
    input_grid_pat = re.compile(r"^__PROBLEM_INPUT__\s*=.*$", re.MULTILINE)
    output_grid_pat = re.compile(r"^__PROBLEM_OUTPUT__\s*=.*$", re.MULTILINE)
    task_func_pat = re.compile(r"^__TASK_FUNC__\s*=.*$", re.MULTILINE)
    output_file_pat = re.compile(r"^__OUTPUT_FILE__\s*=.*$", re.MULTILINE)

    input_str = f"__PROBLEM_INPUT__ = {repr(input)}"
    output_str = f"__PROBLEM_OUTPUT__ = {repr(output)}"
    task_func_str = f"__TASK_FUNC__ = {str(funcname)}"
    output_file_str = (
        f"__OUTPUT_FILE__ = '{str(output_fpath.resolve()) if output_fpath else ''}'"
    )

    for repl, pat in zip(
        (input_str, output_str, task_func_str, output_file_str),
        (input_grid_pat, output_grid_pat, task_func_pat, output_file_pat),
    ):
        test_code = pat.sub(repl, test_code)

    return test_code


def generate_code_and_test(
    example: Dict[GridType, Grid],
    answer_code: str,
    kind: Action,
    output_fpath: Optional[Path] = None,
) -> Optional[Tuple[str, str]]:
    if kind == "transform":
        test_code = TEST_TRANSFORM.read_text()
    else:
        raise NotImplementedError()

    funcname = parse_funcname_from_answer(answer_code)
    if funcname is None:
        return None

    test_code = prepare_test_code(
        test_code, funcname, example["input"], example["output"], output_fpath
    )
    return answer_code, test_code


def eval_if_test_pass(
    example: Dict[GridType, Grid],
    answer_code: str,
    kind: Action,
) -> Tuple[bool, Optional[List[List[int]]]]:
    """
    For transform test, we can let test code output the transformed grid to output_fpath.
    We can parse its content by repr(output_fpath.read_text())
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_fpath = Path(temp_dir) / "output.txt"
        parsed = generate_code_and_test(example, answer_code, kind, output_fpath)
        if parsed is None:
            return False, None

        code, test_code = parsed
        stat, details = untrusted_check(code, test_code)

        output = None
        if output_fpath.exists():
            try:
                output = literal_eval(output_fpath.read_text())
            except:
                pass

        return stat == "pass", output
