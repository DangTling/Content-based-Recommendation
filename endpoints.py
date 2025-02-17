from flask import Flask, request, jsonify
import numpy as np
import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
import ipaddress
import jwt
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

qdrant = QdrantClient(url='http://qdrant:6333')  
model = SentenceTransformer('all-MiniLM-L6-v2')

PG_HOST = os.getenv("DB_HOST")
PG_PORT = os.getenv("DB_PORT")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
PG_DATABASE = os.getenv("DB_NAME")
PG_SCHEMA = os.getenv("DB_SCHEMA")
PG_TABLE = "song"

def connect_postgres():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE
    )

def fetch_songs_from_db():
    conn = connect_postgres()
    cursor = conn.cursor()

    query = f"SELECT s.id AS song_id,s.title AS song_title, s.description as song_description,a.name AS artist_name,STRING_AGG(DISTINCT c.name, ', ') AS categories,STRING_AGG(DISTINCT t.name, ', ') AS tags, s.updated_at FROM cms.song s LEFT JOIN cms.artist a ON s.artist_id = a.id LEFT JOIN cms.song_category sc ON s.draft_id = sc.draft_song_id  LEFT JOIN cms.category c ON sc.category_id  = c.id LEFT JOIN cms.tag_song ts ON s.draft_id  = ts.draft_song_id  LEFT JOIN cms.tag t ON ts.tag_id  = t.id where s.type = 'SONG'  GROUP BY s.id, s.title, a.name"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    existing_songs = qdrant.scroll(collection_name='songs_collection_database', scroll_filter=None)[0]
    existing_ids = {int(song.payload['id']): song.payload['updated_at'] for song in existing_songs}

    updated_songs = []
    new_songs = []

    for row in rows:
        song_id, title, description, artist, categories, tags, updated_at = row

        if song_id in existing_ids:

            if str(updated_at) != existing_ids[song_id]: 
                print(f"Song ID {song_id} has been updated.")
                updated_songs.append((song_id, title, description, artist, categories, tags, updated_at))
        else:
            print(f"New song detected: Title={title}, Artist={artist}")
            new_songs.append((song_id, title, description, artist, categories, tags, updated_at))
    
    for song in updated_songs:
        song_id, title, description, artist, categories, tags, updated_at = song
        qdrant.delete(collection_name="songs_collection_database", points_selector=[song_id])

        title_vector = model.encode(title)
        description_vector = model.encode(description)
        artist_vector = model.encode(artist)
        category_vector = model.encode(categories)
        tags_vector = model.encode(tags)

        query_vector = np.hstack([title_vector, category_vector, description_vector, artist_vector, tags_vector])

        qdrant.upsert(
            collection_name="songs_collection_database",
            points=[
                {
                    'id': song_id, 
                    'vector': query_vector.tolist(),
                    'payload': {
                        'title': title,
                        'artist_name': artist,
                        'category_name': categories,
                        'description': description,
                        'tags': tags,
                        'updated_at': str(updated_at),
                        'id': str(song_id)
                    }
                }
            ],
        )

    for song in new_songs:
        song_id, title, description, artist, categories, tags, updated_at = song

        title_vector = model.encode(title)
        description_vector = model.encode(description)
        artist_vector = model.encode(artist)
        category_vector = model.encode(categories)
        tags_vector = model.encode(tags)

        query_vector = np.hstack([title_vector, category_vector, description_vector, artist_vector, tags_vector])

        qdrant.upsert(
            collection_name="songs_collection_database",
            points=[
                {
                    'id': song_id, 
                    'vector': query_vector.tolist(),
                    'payload': {
                        'title': title,
                        'artist_name': artist,
                        'category_name': categories,
                        'description': description,
                        'tags': tags,
                        'updated_at': str(updated_at),
                        'id': str(song_id)
                    }
                }
            ],
        )

    print("✅ Update completed!")

def initialize_qdrant():
    collections = qdrant.get_collections()

    if 'songs_collection_database' not in [col.name for col in collections.collections]:
        print("Creating new Qdrant collection...")
        qdrant.recreate_collection(
            collection_name="songs_collection_database",
            vectors_config=models.VectorParams(
                size=384 * 5,
                distance=models.Distance.COSINE  
            )
        )
    else:
        print("Qdrant collection already exists.")

def fetch_and_store_songs():
    conn = connect_postgres()
    cursor = conn.cursor()

    query = f"SELECT s.id AS song_id,s.title AS song_title, s.description as song_description,a.name AS artist_name,STRING_AGG(DISTINCT c.name, ', ') AS categories,STRING_AGG(DISTINCT t.name, ', ') AS tags, s.updated_at FROM cms.song s LEFT JOIN cms.artist a ON s.artist_id = a.id LEFT JOIN cms.song_category sc ON s.draft_id = sc.draft_song_id  LEFT JOIN cms.category c ON sc.category_id  = c.id LEFT JOIN cms.tag_song ts ON s.draft_id  = ts.draft_song_id  LEFT JOIN cms.tag t ON ts.tag_id  = t.id where s.type = 'SONG'  GROUP BY s.id, s.title, a.name;"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No songs found in the database.")
        return

    vectors = []
    for row in rows:
        song_id, title, description, artist, category, tags, updated_at = row

        title_vector = model.encode(title)
        description_vector = model.encode(description)
        artist_vector = model.encode(artist)
        category_vector = model.encode(category)
        tags_vector = model.encode(tags)

        query_vector = np.hstack([title_vector, category_vector, description_vector, artist_vector, tags_vector])

        vectors.append({
            'id': song_id,
            'vector': query_vector.tolist(),
            'payload': {
                'title': title,
                'artist_name': artist,
                'category_name': category,
                'description': description,
                'tags': tags,
                'updated_at': str(updated_at),
                'id': str(song_id)
            }
        })

    if vectors:
        qdrant.upsert(collection_name='songs_collection_database', points=vectors)
        print(f"Inserted {len(vectors)} songs into Qdrant.")
    

allowed_range = os.getenv('ALLOWED_RANGE')
secret_key = os.getenv('JWT_SECRET_KEY')

def is_ip_in_range(ip, ip_range):
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(ip_range)
    except ValueError:
        return False
    
def is_token_valid(token):
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False
    
@app.route('/generate-token', methods=['POST'])
def generate_token():
    clientIp = request.remote_addr

    if not is_ip_in_range(clientIp, allowed_range):
        return jsonify({"error": "IP not allowed"}), 403
    
    token_payload = request.json.get('payload')
    expire_time = request.json.get('expire_time')
    
    if expire_time: 
        expire_time = datetime.datetime.utcnow()+datetime.timedelta(minutes=expire_time)
        token_payload["exp"] = expire_time

    return jwt.encode(token_payload, secret_key, algorithm='HS256')

@app.route('/search', methods=['POST'])
def search_songs():
    clientIp = request.remote_addr

    if not is_ip_in_range(clientIp, allowed_range):
        return jsonify({"error": "IP not allowed"}), 403
    
    # token = request.headers.get('Authorization')

    # if not token or not is_token_valid(token):
    #     return jsonify({"error": "Invalid or expired token"}), 401
    
    top_k=request.json.get('top_k', 2)

    song_title = request.json.get('title')
    song_artist = request.json.get('artist')
    song_category = request.json.get('category')
    song_description = request.json.get('description')
    song_tags = request.json.get('tags')

    if not all([song_title, song_category, song_description, song_artist, song_tags]):
        return jsonify({"error": "Missing required fields"}), 400
    
    title_artist_vector = model.encode(song_title)
    author_vector = model.encode(song_artist)
    category_vector = model.encode(song_category)
    description_vector = model.encode(song_description)
    tags_vector = model.encode(song_tags)

    query_vector = np.hstack([title_artist_vector, category_vector, description_vector, author_vector, tags_vector])

    results = qdrant.search(collection_name='songs_collection_database', query_vector=query_vector.tolist(), limit=top_k)

    recommendations = [res.payload for res in results]
 
    return jsonify(recommendations)



@app.route('/vectors', methods=['GET'])
def get_all_vectors():
    clientIp = request.remote_addr

    if not is_ip_in_range(clientIp, allowed_range):
        return jsonify({"error": "IP not allowed"}), 403

    vectors = qdrant.scroll(
        collection_name='songs_collection_database', 
        scroll_filter=None,  
        limit=100 
    )
    
    return jsonify([vector.payload for vector in vectors[0]]), 200

initialize_qdrant()
fetch_and_store_songs()

scheduler = BackgroundScheduler()
scheduler.add_job(fetch_songs_from_db, 'interval', minutes=int(os.getenv('TIME_DURATION')))
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
