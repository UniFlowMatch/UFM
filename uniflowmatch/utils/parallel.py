# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for multiprocessing
# --------------------------------------------------------

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm


class IndependentParallelProcessing:
    def __init__(
        self,
        node_id: int = 0,
        all_nodes: int = 1,
        num_processes: int = 16,
        num_threads: int = 32,
        debug: bool = False,
        **kwargs
    ):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.debug = debug

    @classmethod
    def get_default_parser(cls) -> argparse.ArgumentParser:
        # return an argparse.ArgumentParser object

        parser = argparse.ArgumentParser(description="Parallel Processing")

        parser.add_argument("--node_id", type=int, default=0, help="Process ID")
        parser.add_argument("--all_nodes", type=int, default=1, help="Number of total processes")
        parser.add_argument("--num_processes", type=int, default=16, help="Number of processes")
        parser.add_argument("--num_threads", type=int, default=32, help="Number of threads")

        parser.add_argument("--debug", action="store_true", help="Debug mode")

        return parser

    def get_total_job_args(self) -> List[Any]:
        # return a list of arguments to be passed to the final thread executor
        raise NotImplementedError

    def on_before_process_processing(self, node_job_args: List[Any], manager):
        # cache any data that needs to be shared across processes
        pass

    def on_before_thread_processing(self, job_args: List[Any]):
        # cache any data that needs to be shared across threads
        pass

    def thread_worker(self, job_arg: Any):
        # process a single job
        raise NotImplementedError

    def process_worker(self, list_job_args: List[Any]):
        self.on_before_thread_processing(list_job_args)

        if not self.debug:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                executor.map(self.thread_worker, list_job_args)
        else:
            for job_arg in list_job_args:
                self.thread_worker(job_arg)

    def node_worker(self, list_job_args: List[Any]):
        with Manager() as manager:
            self.on_before_process_processing(list_job_args, manager)

            # Process level work distribution
            local_batch_index = np.arange(len(list_job_args))
            local_batch_index_for_process = np.array_split(local_batch_index, self.num_processes)

            process_batched_splits = [
                [list_job_args[i] for i in local_batch_index_for_process[j]] for j in range(self.num_processes)
            ]

            if not self.debug:
                with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                    executor.map(self.process_worker, process_batched_splits)
            else:
                for i in range(self.num_processes):
                    self.process_worker(process_batched_splits[i])

    def run(self):
        job_args = self.get_total_job_args()

        # Node level work distribution
        work_ids = np.arange(len(job_args))
        work_ids_for_node = np.array_split(work_ids, self.all_nodes)[self.node_id]
        job_args_node = [job_args[i] for i in work_ids_for_node]

        self.node_worker(job_args_node)


class ParallelProcessing(IndependentParallelProcessing):
    def __init__(
        self,
        node_id: int = 0,
        all_nodes: int = 1,
        num_processes: int = 16,
        num_threads: int = 32,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(node_id, all_nodes, num_processes, num_threads, debug, **kwargs)

    def on_after_thread_processing(self, job_args_results_list: List[Any]):
        # post processing after thread processing
        return job_args_results_list

    def on_after_process_processing(self, job_args_results_list: List[Any]):
        # post processing after process processing
        pass

    def thread_worker(self, job_arg: Any) -> Any:
        # process a single job
        raise NotImplementedError

    def warped_thread_worker(self, job_arg: Any) -> Tuple[Any, Any]:
        return job_arg, self.thread_worker(job_arg)

    def process_worker(self, list_job_args: List[Any]) -> List[Any]:
        results = []
        if not self.debug:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(executor.map(self.warped_thread_worker, list_job_args))
        else:
            for job_arg in list_job_args:
                results.append(self.warped_thread_worker(job_arg))

        return results

    def node_worker(self, list_job_args: List[Any]):
        with Manager() as manager:
            self.on_before_process_processing(list_job_args, manager)

            # Process level work distribution
            local_batch_index = np.arange(len(list_job_args))
            local_batch_index_for_process = np.array_split(local_batch_index, self.num_processes)

            process_batched_splits = [
                [list_job_args[i] for i in local_batch_index_for_process[j]] for j in range(self.num_processes)
            ]

            process_results = []
            if not self.debug:
                with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                    process_results = list(executor.map(self.process_worker, process_batched_splits))
            else:
                for i in range(self.num_processes):
                    process_results.append(self.process_worker(process_batched_splits[i]))

            self.on_after_process_processing(process_results)


def parallel_threads(
    function, args, workers=0, star_args=False, kw_args=False, front_num=1, Pool=ThreadPool, **tqdm_kw
):
    """tqdm but with parallel execution.

    Will essentially return
      res = [ function(arg) # default
              function(*arg) # if star_args is True
              function(**arg) # if kw_args is True
              for arg in args]

    Note:
        the <front_num> first elements of args will not be parallelized.
        This can be useful for debugging.
    """
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float("inf")

    # convert into an iterable
    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    # sequential execution first
    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # end of the iterable
        front.append(function(*a) if star_args else function(**a) if kw_args else function(a))

    # then parallel execution
    out = []
    with Pool(workers) as pool:
        # Pass the elements of args into function
        if star_args:
            futures = pool.imap(starcall, [(function, a) for a in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, a) for a in args])
        else:
            futures = pool.imap(function, args)
        # Print out the progress as tasks complete
        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out


def parallel_processes(*args, **kwargs):
    """Same as parallel_threads, with processes"""
    import multiprocessing as mp

    kwargs["Pool"] = mp.Pool
    return parallel_threads(*args, **kwargs)


def starcall(args):
    """convenient wrapper for Process.Pool"""
    function, args = args
    return function(*args)


def starstarcall(args):
    """convenient wrapper for Process.Pool"""
    function, args = args
    return function(**args)
