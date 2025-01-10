from multiprocessing import Process, Queue
import os
import time

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f1(name, f, queue):
    """Worker function to perform task and send a value back using queue."""
    info('function f1')
    print('hello', name)
    i = 0
    f = 50  # Modifying the value of f
    while i < 5:
        print(f'f1 - i: {i}')
        time.sleep(1)
        i += 1
    queue.put(f)  # Send the final value of f back to the main process

def tp(f):
    f = 20  # Changing the value of f
    return f

if __name__ == '__main__':
    info('main line')
    f = 0
    f = tp(f)  # Updating f
    print(f"Updated f in main: {f}")

    queue = Queue()  # Create a queue for communication
    p = Process(target=f1, args=('bob', f, queue))  # Pass queue to the process
    p.start()

    print(f"Process is alive: {p.is_alive()}")

    p.join() 

    j = 0
    while j < 5:
        print(f'main - j: {j}')
        time.sleep(1)
        j += 1

    p.join()  # Wait for the process to complete
    
    # Retrieve the value sent from the worker process
    if not queue.empty():
        f = queue.get()
    
    print(f"Final value of f received from process: {f}")