# gunicorn_config.py

bind = "127.0.0.1:8000"  # Address and port to bind to
workers = 4  # Number of worker processes
threads = 2  # Number of threads per worker
timeout = 120  # Timeout for worker processes
