# threaded_runner_utils.py
import sys
import traceback
from enum import Enum
from typing import Callable, Optional, Any, List, Iterator, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pandas as pd # Import pandas here
import attrs
import tqdm
from tqdm import tqdm

class TaskRunStatus(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    TIMEOUT = 2

@attrs.define(eq=False, repr=False)
class TaskResult:
    status: TaskRunStatus
    # task_id: Any # <<<--- Make sure this line exists!
    result: Optional[Any] = None
    exception_tb: Optional[str] = None
    original_task_index: int = -1 # Store original position
    # original_task_args: Optional[tuple] = None # For debugging

    def is_success(self) -> bool:
        return self.status == TaskRunStatus.SUCCESS

    def is_timeout(self) -> bool:
        return self.status == TaskRunStatus.TIMEOUT

    def is_exception(self) -> bool:
        return self.status == TaskRunStatus.EXCEPTION

def run_tasks_in_parallel_threaded(
    func: Callable,
    tasks: List[Tuple], # Expect tuples as tasks
    num_workers: int,
    timeout_per_task: Optional[int] = None,
    use_progress_bar: bool = False,
    progress_bar_desc: Optional[str] = None,
) -> Iterator[TaskResult]:
    """
    A threading-based equivalent of your run_tasks_in_parallel utility.
    
    Args:
        func: The static worker function to run. Must accept a single 'task' argument.
        tasks: A list of tasks (the arguments for 'func').
        num_workers: Maximum number of parallel threads.
        timeout_per_task: The timeout, in seconds, to use per task.
    Returns:
        A list of TaskResult objects, one per task.
    """
    # Initialize results list with placeholders to maintain order
    task_results: List[Optional[TaskResult]] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Map Future object back to its original index in the tasks list
        future_to_index: Dict[Future, int] = {}

        for i, task in enumerate(tasks):
            future = pool.submit(func, task)
            future_to_index[future] = i

        iterator = as_completed(future_to_index)
        if use_progress_bar:
            pbar = tqdm(
                iterator,
                desc=progress_bar_desc,
                total=len(tasks),
                dynamic_ncols=True,
                file=sys.stdout,
            )
        else:
            pbar = iterator

        succ = timeouts = exceptions = 0

        for future in pbar:
            original_index = future_to_index[future]
            try:
                result = future.result(timeout=timeout_per_task)
                # task_results[original_index] = TaskResult(
                #     status=TaskRunStatus.SUCCESS,
                #     result=result,
                #     original_task_index=original_index
                # )
                yield TaskResult(
                    status=TaskRunStatus.SUCCESS,
                    result=result,
                    original_task_index=original_index
                )
                succ += 1
            except TimeoutError:
                # task_results[original_index] = TaskResult(
                #     status=TaskRunStatus.TIMEOUT,
                #     original_task_index=original_index
                # )
                yield TaskResult(
                    status=TaskRunStatus.TIMEOUT,
                    original_task_index=original_index
                )
                timeouts += 1
            except Exception:
                exception_tb = traceback.format_exc()
                # task_results[original_index] = TaskResult(
                #     status=TaskRunStatus.EXCEPTION,
                #     exception_tb=exception_tb,
                #     original_task_index=original_index
                # )
                yield TaskResult(
                    status=TaskRunStatus.EXCEPTION,
                    exception_tb=exception_tb,
                    original_task_index=original_index
                )
                exceptions += 1

            if use_progress_bar:
                pbar.set_postfix(
                    succ=succ, timeouts=timeouts, exc=exceptions
                )

    # # Replace any None placeholders if tasks somehow didn't complete (shouldn't happen)
    # final_results = []
    # for i, res in enumerate(task_results):
    #      if res is None:
    #           logger.error(f"Result for task index {i} is missing!")
    #           final_results.append(TaskResult(
    #                status=TaskRunStatus.EXCEPTION, # Or a custom MISSING status
    #                exception_tb="Result placeholder was never filled.",
    #                original_task_index=i
    #           ))
    #      else:
    #           final_results.append(res)

    # if len(final_results) != len(tasks):
    #      # This check should ideally never fail now, but keep it as a safeguard
    #      print(f"CRITICAL WARNING: Final length mismatch! Input: {len(tasks)}, Output: {len(final_results)}")

    # return final_results