from flask import Flask, jsonify, render_template, request
from sqlalchemy import create_engine
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

DB_CONFIG = {
    'dbname': 'new',
    'user': 'postgres',
    'password': 'Pusdatin@2023!',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    return create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

def get_popular_books(limit=10):
    engine = get_db_connection()
    query = """
    SELECT cat.ID, cat.TITLE, cat.AUTHOR, cat.PUBLISHER, COUNT(*) as borrow_count
    FROM COLLECTIONLOANITEMS cli
    JOIN COLLECTIONS c ON cli.COLLECTION_ID = c.ID
    JOIN CATALOGS cat ON c.CATALOG_ID = cat.ID
    GROUP BY cat.ID, cat.TITLE, cat.AUTHOR, cat.PUBLISHER
    ORDER BY borrow_count DESC
    LIMIT %s
    """
    return pd.read_sql(query, engine, params=[limit])

@app.route('/catalogs', methods=['GET'])
def get_catalogs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agama")
        catalogs = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(catalogs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_subject_based_recommendations(member_no, limit=10):
    engine = get_db_connection()
    query = """
    WITH UserSubjects AS (
        SELECT DISTINCT cat.SUBJECT
        FROM COLLECTIONLOANITEMS cli
        JOIN COLLECTIONS c ON cli.COLLECTION_ID = c.ID
        JOIN CATALOGS cat ON c.CATALOG_ID = cat.ID
        JOIN MEMBERS m ON cli.MEMBER_ID = m.ID
        WHERE m.MEMBERNO = %s
    )
    SELECT DISTINCT cat.ID, cat.TITLE, cat.AUTHOR, cat.PUBLISHER
    FROM CATALOGS cat
    WHERE cat.SUBJECT IN (SELECT SUBJECT FROM UserSubjects)
    AND cat.ID NOT IN (
        SELECT DISTINCT c2.CATALOG_ID
        FROM COLLECTIONLOANITEMS cli2
        JOIN COLLECTIONS c2 ON cli2.COLLECTION_ID = c2.ID
        JOIN MEMBERS m ON cli2.MEMBER_ID = m.ID
        WHERE m.MEMBERNO = %s
    )
    LIMIT %s
    """
    return pd.read_sql(query, engine, params=[member_no, member_no, limit])

def get_user_history_count(member_no):
    engine = get_db_connection()
    query = """
    SELECT COUNT(DISTINCT c.CATALOG_ID) as borrow_count
    FROM COLLECTIONLOANITEMS cli
    JOIN COLLECTIONS c ON cli.COLLECTION_ID = c.ID
    JOIN MEMBERS m ON cli.MEMBER_ID = m.ID
    WHERE m.MEMBERNO = %s
    """
    result = pd.read_sql(query, engine, params=[member_no])
    return result.iloc[0]['borrow_count']

def get_recommendations(member_no):
    history_count = get_user_history_count(member_no)
    
    if history_count == 0:
        # Cold start: Return popular books
        recommendations = get_popular_books()
        recommendations['recommendation_type'] = 'Popular Books (New User)'
    elif history_count < 5:
        # Limited history: Use subject-based recommendations
        recommendations = get_subject_based_recommendations(member_no)
        recommendations['recommendation_type'] = 'Based on Your Interests'
    else:
        # Sufficient history: Use collaborative filtering
        recommendations = get_collaborative_recommendations(member_no)
        recommendations['recommendation_type'] = 'Based on Similar Users'
    
    return recommendations.to_dict('records')

def get_collaborative_recommendations(member_no):
    engine = get_db_connection()
    similar_users = get_similar_users(member_no)
    
    query = """
    SELECT DISTINCT cat.ID, cat.TITLE, cat.AUTHOR, cat.PUBLISHER
    FROM COLLECTIONLOANITEMS cli
    JOIN COLLECTIONS c ON cli.COLLECTION_ID = c.ID
    JOIN CATALOGS cat ON c.CATALOG_ID = cat.ID
    WHERE cli.MEMBER_ID = ANY(%s)
    AND cat.ID NOT IN (
        SELECT DISTINCT c2.CATALOG_ID
        FROM COLLECTIONLOANITEMS cli2
        JOIN COLLECTIONS c2 ON cli2.COLLECTION_ID = c2.ID
        JOIN MEMBERS m ON cli2.MEMBER_ID = m.ID
        WHERE m.MEMBERNO = %s
    )
    LIMIT 10
    """
    return pd.read_sql(query, engine, params=[similar_users, member_no])

@app.route('/')
def index():
    return render_template('index_pg.html')

@app.route('/recommendations/<member_no>')
def get_book_recommendations(member_no):
    try:
        recommendations = get_recommendations(member_no)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'status': 'success',
                'recommendations': recommendations
            })
        return render_template('recommendations.html', recommendations=recommendations)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)