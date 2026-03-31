import threading
import time

counter = 0
lock = threading.Lock()

def increment_counter(num_increments):
    global counter
    for _ in range(num_increments):
        with lock:              # acquire lock
            counter += 1        # critical section
        # lock released automatically here

def run_test(num_threads=4, num_increments=100_000):
    global counter
    counter = 0

    threads = []
    start = time.perf_counter()

    for _ in range(num_threads):
        t = threading.Thread(target=increment_counter, args=(num_increments,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end = time.perf_counter()

    expected = num_threads * num_increments
    print(f"threads   : {num_threads}")
    print(f"expected  : {expected}")
    print(f"counter   : {counter}")
    print(f"time      : {end - start:.4f} seconds")
    print("-" * 30)

if __name__ == "__main__":
    for n in [1, 2, 4, 8]:
        run_test(num_threads=n)