import threading
import queue

def run_tasks_in_parallel(*tasks):
    """
    并行执行多个任务，并返回所有任务的结果。
    
    参数:
        tasks: 一个或多个 (函数, *参数) 形式的任务。
    
    返回:
        任务的执行结果列表，顺序与传入任务一致。
    """
    result_queues = [queue.Queue() for _ in tasks]
    threads = []

    # 定义通用的线程包装函数
    def thread_wrapper(func, args, result_queue):
        result = func(*args)  # 运行任务
        result_queue.put(result)  # 结果放入队列

    # 创建并启动线程
    for i, (func, *args) in enumerate(tasks):
        t = threading.Thread(target=thread_wrapper, args=(func, args, result_queues[i]))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 收集所有任务的执行结果
    return [q.get() for q in result_queues]