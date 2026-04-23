from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import yaml

config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

client = QdrantClient(
    host=config['qdrant']['host'], 
    port=config['qdrant']['port'],
    timeout=config['qdrant']['timeout'],
    check_compatibility=config['qdrant']['check_compatibility']
)
model = SentenceTransformer(config['model']['name'])
COLLECTION = config['collection']['name']
BATCH_SIZE = config['indexing']['batch_size']