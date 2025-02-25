import jwt
from flask import Flask, request, jsonify
import numpy as np
import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler
import psycopg2.pool
from sentence_transformers import SentenceTransformer
import ipaddress
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

PG_HOST = os.getenv("DB_HOST")
PG_PORT = os.getenv("DB_PORT")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
PG_DATABASE = os.getenv("DB_NAME")
PG_SCHEMA = os.getenv("DB_SCHEMA")
PG_TABLE = "song"

LIKE_UPDATE_TIMEFRAME = os.getenv("LIKE_UPDATE_TIMEFRAME", 2)

def connect_postgres():
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,  
        maxconn=10,  
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE
    )
    return pg_pool.getconn()

def get_user_liked_songs():
    conn = connect_postgres()
    cursor = conn.cursor()
    query = """
        SELECT created_by, song_id
        FROM cms.reactions
        WHERE reaction_type = 'like' AND is_deleted = FALSE LIMIT 5000;
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    user_songs = {}
    for user_id, song_id in rows:
        if user_id not in user_songs:
            user_songs[user_id] = []
        user_songs[user_id].append(str(song_id))

    return user_songs

def generate_user_embeddings():
    user_songs = get_user_liked_songs()
    user_embeddings = {}

    for user_id, songs in user_songs.items():
        song_vectors = [model.encode(song) for song in songs]  
        user_embedding = np.mean(song_vectors, axis=0) 
        user_embeddings[user_id] = user_embedding
    # print(user_embeddings)
    return user_embeddings

def save_user_embeddings():
    user_embeddings = generate_user_embeddings()
    conn = connect_postgres()
    cursor = conn.cursor()

    for user_id, embedding in user_embeddings.items():
        cursor.execute(
            "INSERT INTO recommendation.user_embeddings (user_id, embedding) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding;",
            (user_id, embedding.tolist())
        )

    conn.commit()
    conn.close()

def find_similar_users(user_id, top_k=5):
    conn = connect_postgres()
    cursor = conn.cursor()

    cursor.execute('SET search_path TO recommendation;')
    query = """
        SELECT user_id FROM recommendation.user_embeddings
        WHERE user_id != %s
        ORDER BY embedding <=> (SELECT embedding FROM recommendation.user_embeddings WHERE user_id = %s)
        LIMIT %s;
    """
    
    cursor.execute(query, (user_id, user_id, top_k))
    similar_users = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return similar_users

def recommend_songs_from_similar_users(user_id, top_k=5):
    similar_users = find_similar_users(user_id, top_k)

    conn = connect_postgres()
    cursor = conn.cursor()

    cursor.execute("SELECT song_id FROM cms.reactions WHERE created_by = %s", (user_id,))
    user_liked_songs = {row[0] for row in cursor.fetchall()}

    recommended_songs = set()

    for sim_user in similar_users:
        cursor.execute("SELECT song_id FROM cms.reactions WHERE created_by = %s", (sim_user,))
        sim_user_liked_songs = {row[0] for row in cursor.fetchall()}
        
        new_songs = sim_user_liked_songs - user_liked_songs
        recommended_songs.update(new_songs)
    
    conn.close()
    
    return list(recommended_songs)[:top_k]

def fetch_songs_from_db():
    conn = connect_postgres()
    cursor = conn.cursor()

    query = f"SELECT s.id AS song_id,s.title AS song_title, s.description as song_description,a.name AS artist_name,STRING_AGG(DISTINCT c.name, ', ') AS categories,STRING_AGG(DISTINCT t.name, ', ') AS tags, s.updated_at FROM cms.song s LEFT JOIN cms.artist a ON s.artist_id = a.id LEFT JOIN cms.song_category sc ON s.draft_id = sc.draft_song_id  LEFT JOIN cms.category c ON sc.category_id  = c.id LEFT JOIN cms.tag_song ts ON s.draft_id  = ts.draft_song_id  LEFT JOIN cms.tag t ON ts.tag_id  = t.id where s.type = 'SONG'  GROUP BY s.id, s.title, a.name"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    conn = connect_postgres()
    cursor = conn.cursor()
    cursor.execute("SELECT song_id, updated_at FROM recommendation.song_embeddings")
    existing_songs = cursor.fetchall()
    existing_ids = {song_id: updated_at for song_id, updated_at in existing_songs}
    conn.close()

    updated_songs = []
    new_songs = []

    for row in rows:
        song_id, title, description, artist, categories, tags, updated_at = row

        if song_id in existing_ids:
            
            if str(updated_at) != str(existing_ids[song_id]): 
                print(f"Song ID {song_id} has been updated.")
                updated_songs.append((song_id, title, description, artist, categories, tags, updated_at))
        else:
            print(f"New song detected: Title={title}, Artist={artist}")
            new_songs.append((song_id, title, description, artist, categories, tags, updated_at))
    
    for song in updated_songs:
        song_id, title, description, artist, categories, tags, updated_at = song

        title_vector = model.encode(title)
        description_vector = model.encode(description)
        artist_vector = model.encode(artist)
        category_vector = model.encode(categories)
        tags_vector = model.encode(tags)

        query_vector = np.hstack([title_vector, category_vector, description_vector, artist_vector, tags_vector])
        conn = connect_postgres()
        cursor = conn.cursor()
        query = """
            UPDATE recommendation.song_embeddings
            SET embedding = %s, updated_at = %s
            WHERE song_id = %s
        """
        cursor.execute(query, (query_vector.tolist(), str(updated_at), song_id))
        conn.commit()
        conn.close()

    for song in new_songs:
        song_id, title, description, artist, categories, tags, updated_at = song

        title_vector = model.encode(title)
        description_vector = model.encode(description)
        artist_vector = model.encode(artist)
        category_vector = model.encode(categories)
        tags_vector = model.encode(tags)

        query_vector = np.hstack([title_vector, category_vector, description_vector, artist_vector, tags_vector])

        conn = connect_postgres()
        cursor = conn.cursor()
        query = """
            INSERT INTO recommendation.song_embeddings (song_id, title, artist, category, description, tags, embedding, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (song_id, title, artist, categories, description, tags, query_vector.tolist(), str(updated_at)))
        conn.commit()
        conn.close()

    print("✅ Update completed!")

def initialize_postgres():
    conn = connect_postgres()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS recommendation.song_embeddings (
        song_id SERIAL PRIMARY KEY,
        title VARCHAR(255),
        artist VARCHAR(255),
        category VARCHAR(255),
        description TEXT,
        tags TEXT,
        embedding VECTOR(1920),
        updated_at TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def fetch_and_store_songs():
    conn = connect_postgres()
    cursor = conn.cursor()

    query = f"""
    SELECT s.id AS song_id, s.title AS song_title, s.description as song_description,
           a.name AS artist_name,
           STRING_AGG(DISTINCT c.name, ', ') AS categories,
           STRING_AGG(DISTINCT t.name, ', ') AS tags, s.updated_at
    FROM cms.song s
    LEFT JOIN cms.artist a ON s.artist_id = a.id
    LEFT JOIN cms.song_category sc ON s.draft_id = sc.draft_song_id  
    LEFT JOIN cms.category c ON sc.category_id = c.id
    LEFT JOIN cms.tag_song ts ON s.draft_id = ts.draft_song_id  
    LEFT JOIN cms.tag t ON ts.tag_id = t.id
    WHERE s.type = 'SONG'  
    GROUP BY s.id, s.title, a.name;
    """
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
        
        category_vector = model.encode(category if category else '')
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
        for vector in vectors:
            song_id = vector['id']
            query_vector = vector['vector']
            payload = vector['payload']

            conn2 = connect_postgres()
            cursor2 = conn2.cursor()
            cursor2.execute("""
                INSERT INTO recommendation.song_embeddings (song_id, title, artist, category, description, tags, embedding, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (song_id) 
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at;
            """, (
                song_id,
                payload['title'],
                payload['artist_name'],
                payload['category_name'],
                payload['description'],
                payload['tags'],
                query_vector,  # Assuming the vector is stored in the correct format
                payload['updated_at']
            ))
            conn2.commit()
            conn2.close()
        print(f"Inserted {len(vectors)} songs into PG-Vector.")
    

def caculate_user_embeddings():
    conn = connect_postgres()
    cursor = conn.cursor()

    query = """
        SELECT created_by, COUNT(song_id) AS total_likes
        FROM cms.reactions
        WHERE reaction_type = 'like'
        GROUP BY created_by
        LIMIT 5000;
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    user_embeddings = {}

    for user_id, total_count in rows:
        cursor.execute("""
            SELECT s.embedding
            FROM cms.reactions r
            JOIN recommendation.song_embeddings s ON r.song_id = s.song_id
            WHERE r.created_by = %s AND r.reaction_type = 'like';
        """, (user_id,))
        
        song_embeddings = cursor.fetchall()

        try:
            song_embeddings = [np.fromstring(embedding[0][1:-1], sep=',', dtype=np.float32) if isinstance(embedding[0], str) else np.array(embedding[0], dtype=np.float32) for embedding in song_embeddings]
        except Exception as e:
            print(f"Error converting embeddings for user {user_id}: {e}")
            continue

        if not song_embeddings:
            continue

        embedding_dim = song_embeddings[0].shape[0]
        if embedding_dim != 384:
            print(f"Warning: User {user_id} has embeddings with unexpected size {embedding_dim}")

            if embedding_dim % 384 == 0:
                num_parts = embedding_dim // 384
                song_embeddings = [embedding.reshape(num_parts, 384).mean(axis=0) for embedding in song_embeddings]
            else:
                continue  

        user_embedding = song_embeddings[0]
        for i in range(1, len(song_embeddings)):
            user_embedding = (user_embedding * i + song_embeddings[i]) / (i + 1)

        user_embeddings[user_id] = user_embedding

    conn.close()
    return user_embeddings

def get_users_with_reaction_changes():
    conn = connect_postgres()
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT created_by
        FROM cms.reactions
        WHERE reaction_type = 'like' 
        AND updated_at >= NOW() - INTERVAL %s
        LIMIT 5000;
    """
    cursor.execute(query, (f"{LIKE_UPDATE_TIMEFRAME} minutes",))

    users = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return users

def update_user_embeddings():
    users = get_users_with_reaction_changes()
    if not users:
        print("Không có thay đổi nào trong bảng reactions.")
        return
    
    conn = connect_postgres()
    cursor = conn.cursor()

    for user_id in users:
        cursor.execute("""
            SELECT song_id FROM cms.reactions
            WHERE created_by = %s AND reaction_type = 'like'
            LIMIT 5000;
        """, (user_id,))
        liked_songs = [str(row[0]) for row in cursor.fetchall()]

        if liked_songs:
            song_vectors = [model.encode(song) for song in liked_songs]
            user_embedding = np.mean(song_vectors, axis=0)

            cursor.execute("""
                INSERT INTO recommendation.user_embeddings (user_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """, (user_id, user_embedding.tolist()))
       
    
    conn.commit()
    conn.close()
    print("✅ User embeddings updated successfully!")




allowed_range = os.getenv('ALLOWED_RANGE')
secret_key = os.getenv('JWT_SECRET_KEY')

def is_ip_in_range(ip, ip_range):
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(ip_range)
    except ValueError:
        return False
    
    
@app.route('/recommend/generate-token', methods=['POST'])
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

@app.route('/recommend/search', methods=['POST'])
def search_songs():
    clientIp = request.remote_addr

    if not is_ip_in_range(clientIp, allowed_range):
        return jsonify({"error": "IP not allowed"}), 403
    
    top_k = request.json.get('top_k', 10)

    exclude_ids = request.json.get('exclude_ids', [])

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

    exclude_clause = ""
    if exclude_ids:
        exclude_clause = "AND song_id NOT IN %s"

    conn = connect_postgres()
    cursor = conn.cursor()

    cursor.execute('SET search_path TO recommendation;')
    if exclude_ids:
        query = """
            SELECT song_id, title, artist, category, description, tags
            FROM recommendation.song_embeddings
            WHERE song_id NOT IN %s
            ORDER BY embedding <=> CAST(%s AS vector(1920))
            LIMIT %s;
        """
        cursor.execute(query, (tuple(exclude_ids), query_vector.tolist(), top_k))
    else:
        query = """
            SELECT song_id, title, artist, category, description, tags
            FROM recommendation.song_embeddings
            ORDER BY embedding <=> CAST(%s AS vector(1920))
            LIMIT %s;
        """
        cursor.execute(query, (query_vector.tolist(), top_k))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify({"error": "No songs found"}), 404

    recommendations = []
    for row in rows:
        song_id, title, artist, category, description, tags = row
        recommendations.append(song_id)
    
    return jsonify(recommendations)

@app.route('/recommend/by-user', methods=['POST'])
def recommend_by_user():
    clientIp = request.remote_addr

    if not is_ip_in_range(clientIp, allowed_range):
        return jsonify({"error": "IP not allowed"}), 403
    
    user_id = request.json.get('user_id')
    top_k = request.json.get('top_k', 10)
    return jsonify(recommend_songs_from_similar_users(user_id, top_k))


# initialize_postgres()
fetch_and_store_songs()
save_user_embeddings()

scheduler = BackgroundScheduler()
scheduler.add_job(fetch_songs_from_db, 'interval', minutes=int(os.getenv('TIME_DURATION')), max_instances=2)
# scheduler.add_job(update_user_embeddings, 'interval', minutes=2, max_instances=2)
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
