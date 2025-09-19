from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker_url = settings.REDIS_URL,
    backend = settings.REDIS_URL,
    include = [
        "app.worker.tasks"
    ]  
)

celery_app.conf.broker_transport_options = {
    'visibility_timeout': 3600,
    'broker_heartbeat': 0
}

celery_app.conf.task_default_queue = 'gpu_tasks'

celery_app.conf.update(

    task_acks_late=True,

    worker_prefetch_multiplier=1,
    result_serializer='json',
    task_serializer='json',
    accept_content=['json'],

    timezone='UTC',
    enable_utc=True

)
