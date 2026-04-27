from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import yaml
import os

project_root = Path(__file__).parent.parent.parent.parent  # Remonte de 4 niveaux
env_path = project_root / ".env"

load_dotenv(dotenv_path=env_path)

config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

client = QdrantClient(
    host=config['qdrant']['host'],
    port=config['qdrant']['port'],
    timeout=config['qdrant']['timeout'],
    check_compatibility=config['qdrant']['check_compatibility'],
    api_key=os.getenv(config['qdrant']['api_key']),
    https=False
)
model = SentenceTransformer(config['model']['name'])
COLLECTION = config['collection']['name']
BATCH_SIZE = config['indexing']['batch_size']
