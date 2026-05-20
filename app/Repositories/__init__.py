"""
Media worker repositories — transport boundaries (no DB).

  KafkaConsumer    consume media_tasks
  KafkaProducer    publish media_results
  RedisRepository  dequeue live tasks + fetch frame bytes
  S3Client         fetch batch source bytes from MinIO
  StorageClient    POST crops to storage service for persistence
"""
from app.Repositories.KafkaConsumer import KafkaConsumer
from app.Repositories.KafkaProducer import KafkaProducer
from app.Repositories.RedisRepository import RedisRepository
from app.Repositories.S3Client import S3Client
from app.Repositories.StorageClient import StorageClient

__all__ = ["KafkaConsumer", "KafkaProducer", "RedisRepository", "S3Client", "StorageClient"]