import os
import subprocess
from set_env import set_env
import sys

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test", "infiniop")
)
os.chdir(PROJECT_DIR)


def run_tests(args):
    failed = []
    for test in [
        "add.py",
        "attention.py",
        "causal_softmax.py",
        "clip.py",
        "gemm.py",
        "mul.py",
        "random_sample.py",
        "rearrange.py",
        "rms_norm.py",
        "rope.py",
        "sub.py",
        "swiglu.py",
        "silu.py",
        "div.py",
        "logical_and.py",
        "logical_or.py",
        "equal.py",
        "all_equal.py",
        "relu_backward.py",
        "gelu.py",
        "gelu_backward.py",
        "cross_entropy_loss_backward.py"
    ]:
        result = subprocess.run(
            f"python {test} {args} --debug", text=True, encoding="utf-8", shell=True
        )
        if result.returncode != 0:
            failed.append(test)

    return failed


if __name__ == "__main__":
    set_env()
    failed = run_tests(" ".join(sys.argv[1:]))
    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for test in failed:
            print(f"\033[91m - {test}\033[0m")
    exit(len(failed))
