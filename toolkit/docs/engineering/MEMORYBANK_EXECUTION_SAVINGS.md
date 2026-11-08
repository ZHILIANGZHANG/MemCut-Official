# MemoryBank early stop on a valid wrong answer

`python run.py memorybank-efficient prepare --system memorybank ...` freezes `two-trial-valid-failure-short-circuit-v1` for new experiments. When trial 1 is valid and wrong, 2/2 can no longer hold, and trial 2 is explicitly recorded as not executed; when trial 1 is correct, trial 2 is still executed. An invalid answer or a failed request must not trigger the short circuit as if it were a wrong answer.

This policy does not change the option order, the 2/2 recovery condition or the deletion semantics, but it lacks the diagnostics of two complete answers. Historical runs keep their own execution policy; this optimization is not applied by rewriting a PLAN. The four-system `efficient` entry is a separate frozen extension and does not automatically replace all base entries.
