import logging
from celery import Celery
from kombu import Queue

from app.core.config import settings

logger = logging.getLogger(__name__)

# ===================================================================
#   Celery Application Instantiation
# ===================================================================

# Instantiate the Celery application.
# The `main` argument is the conventional name for the app module.
# The `broker` and `backend` are configured to use the Redis instance
# defined in our central settings.
celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.worker.tasks"]  # Auto-discover tasks from this module
)

# ===================================================================
#   Task Queue and Routing Configuration
# ===================================================================

# Define the task queues. This is the core of our resource management strategy.
# We create two distinct queues: one for GPU-intensive tasks and one for
# general-purpose CPU tasks.
celery_app.conf.task_queues = (
    Queue("gpu_tasks", routing_key="task.gpu.#"),
    Queue("cpu_tasks", routing_key="task.cpu.#"),
)

# Set the default queue for any task that doesn't have a specific queue assigned.
# We default to CPU tasks to be safe.
celery_app.conf.task_default_queue = "cpu_tasks"
celery_app.conf.task_default_exchange = "tasks"
celery_app.conf.task_default_routing_key = "task.cpu.default"

# Define a routing rule. This is a powerful feature that allows us to send
# tasks to specific queues based on their names.
celery_app.conf.task_routes = {
    # Any task whose name includes 'transcribe' or 'diarize' will be
    # automatically routed to the 'gpu_tasks' queue.
    "app.worker.tasks.run_transcription_task": {
        "queue": "gpu_tasks",
        "routing_key": "task.gpu.transcription",
    },
    "app.worker.tasks.run_diarization_task": {
        "queue": "gpu_tasks",
        "routing_key": "task.gpu.diarization",
    },
    # You could add routes for CPU-bound tasks here if needed, but they
    # will go to the default 'cpu_tasks' queue anyway.
    # "app.worker.tasks.some_cpu_task": {
    #     "queue": "cpu_tasks",
    #     "routing_key": "task.cpu.general",
    # }
}


# ===================================================================
#   Celery Configuration Settings
# ===================================================================

celery_app.conf.update(
    # Acknowledge tasks only after they have been successfully executed.
    # If a worker crashes mid-task, the task will be re-queued.
    task_acks_late=True,

    # Set the worker concurrency to 1 by default for GPU tasks. This is
    # crucial as a single GPU can typically only handle one heavy model
    # at a time. The actual concurrency is set in the docker-compose command.
    worker_prefetch_multiplier=1,

    # Use JSON as the serializer for task messages and results. It's safe
    # and universally compatible.
    result_serializer='json',
    task_serializer='json',
    accept_content=['json'],

    # Timezone settings to ensure timestamps are consistent
    timezone='Asia/Ho_Chi_Minh',
    enable_utc=True,
)

logger.info(f"Celery app configured with broker at {settings.REDIS_URL}")