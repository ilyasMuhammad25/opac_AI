from flask import Flask, jsonify, request, render_template
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import google.generativeai as genai
import os
from flask import request


app = Flask(__name__)
# Configure Gemini AI
GOOGLE_API_KEY = 'AIzaSyA1ZHwqVwzj1z9A8lWNhlXZfX5-sby__8A'  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Database configuration
DB_CONFIG = {
    'dbname': 'perpusnas_inlis',
    'user': 'postgres',
    'password': 'Pusdatin@2023!',
    'host': 'localhost',
    'port': '5432'
}

# Helper function to connect to the database
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# API route to fetch all catalogs
@app.route('/catalogs', methods=['GET'])
def get_catalogs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT "ID", "MEMBERNO", "FULLNAME" FROM "MEMBERS" WHERE "MEMBERNO" = \'19021900162\'')
        catalogs = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(catalogs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_user_item_matrix(member_no):
    """Create user-item interaction matrix"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all loan records
            cur.execute("""
                WITH member_loans AS (
                    SELECT DISTINCT 
                        m."MEMBERNO",
                        c."ID" as collection_id,
                        c."TITLE",
                        COUNT(*) as borrow_count
                    FROM "COLLECTIONLOANITEMS" cli
                    JOIN "MEMBERS" m ON cli."MEMBER_ID" = m."ID"
                    JOIN "COLLECTIONS" c ON cli."COLLECTION_ID" = c."ID"
                    GROUP BY m."MEMBERNO", c."ID", c."TITLE"
                )
                SELECT * FROM member_loans
                WHERE "MEMBERNO" = %s 
                OR "MEMBERNO" IN (
                    SELECT DISTINCT m2."MEMBERNO"
                    FROM "COLLECTIONLOANITEMS" cli2
                    JOIN "MEMBERS" m2 ON cli2."MEMBER_ID" = m2."ID"
                    JOIN "COLLECTIONS" c2 ON cli2."COLLECTION_ID" = c2."ID"
                    WHERE c2."ID" IN (
                        SELECT collection_id 
                        FROM member_loans 
                        WHERE "MEMBERNO" = %s
                    )
                    AND m2."MEMBERNO" != %s
                    LIMIT 10
                )
            """, (member_no, member_no, member_no))
            
            loans = cur.fetchall()
            
            if not loans:
                return None
            
            # Create DataFrames for analysis
            df = pd.DataFrame(loans)
            
            # Create interaction matrix
            matrix = pd.pivot_table(
                df,
                values='borrow_count',
                index='MEMBERNO',
                columns=['collection_id', 'TITLE'],
                fill_value=0
            )
            
            # Calculate user similarity
            user_similarity = cosine_similarity(matrix.values)
            
            # Get user and item mappings
            users = matrix.index.tolist()
            items = [(str(col[0]), col[1]) for col in matrix.columns]
            
            return {
                'matrix': matrix.values,
                'users': users,
                'items': items,
                'user_similarity': user_similarity,
                'target_user_idx': users.index(member_no)
            }

def get_user_loan_history(member_no):
    """Get user's loan history"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT cli."COLLECTION_ID", c."TITLE", c."AUTHOR", cli."LOANDATE", cat."COVERURL"
                FROM "COLLECTIONLOANITEMS" cli
                JOIN "MEMBERS" m ON cli."MEMBER_ID" = m."ID"
                JOIN "COLLECTIONS" c ON cli."COLLECTION_ID" = c."ID"
                LEFT JOIN "CATALOGS" cat ON c."CATALOG_ID" = cat."ID"
                WHERE m."MEMBERNO" = %s
                ORDER BY cli."LOANDATE" DESC
            """, (member_no,))
            return cur.fetchall()

def get_recommendations(member_no):
    """Generate book recommendations using matrix factorization"""
    matrix_data = create_user_item_matrix(member_no)
    
    if matrix_data is None:
        return []
    
    target_idx = matrix_data['target_user_idx']
    user_similarity = matrix_data['user_similarity']
    matrix = matrix_data['matrix']
    
    # Get similar users (excluding self)
    similar_users = [(i, user_similarity[target_idx][i]) 
                    for i in range(len(matrix_data['users'])) 
                    if i != target_idx]
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Get recommendations based on similar users
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            placeholders = ','.join(['%s'] * len(similar_users))
            similar_member_nos = [matrix_data['users'][idx] for idx, _ in similar_users]
            
            cur.execute(f"""
                SELECT DISTINCT 
                    c."ID",
                    c."TITLE",
                    c."AUTHOR",
                    c."PUBLISHER",
                    cat."COVERURL",
                    COUNT(DISTINCT cli."MEMBER_ID") as borrow_count
                FROM "COLLECTIONS" c
                LEFT JOIN "CATALOGS" cat ON c."CATALOG_ID" = cat."ID"
                JOIN "COLLECTIONLOANITEMS" cli ON c."ID" = cli."COLLECTION_ID"
                JOIN "MEMBERS" m ON cli."MEMBER_ID" = m."ID"
                WHERE m."MEMBERNO" IN ({placeholders})
                AND c."ID" NOT IN (
                    SELECT DISTINCT cli2."COLLECTION_ID"
                    FROM "COLLECTIONLOANITEMS" cli2
                    JOIN "MEMBERS" m2 ON cli2."MEMBER_ID" = m2."ID"
                    WHERE m2."MEMBERNO" = %s
                )
                GROUP BY 
                    c."ID", 
                    c."TITLE", 
                    c."AUTHOR", 
                    c."PUBLISHER",
                    cat."COVERURL"
                ORDER BY borrow_count DESC
                LIMIT 10
            """, (*similar_member_nos, member_no))
            
            recommendations = cur.fetchall()
            
            # Format recommendations
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append({
                    'id': rec['ID'],
                    'title': rec['TITLE'],
                    'author': rec['AUTHOR'],
                    'publisher': rec['PUBLISHER'],
                    'borrow_count': rec['borrow_count'],
                    'cover_url': rec['COVERURL']
                })
            
            return formatted_recommendations

import requests

@app.route('/cover-proxy/<path:cover_url>')
def cover_proxy(cover_url):
    """Proxy for book cover images"""
    try:
        response = requests.get(f'https://opac.perpusnas.go.id/uploaded_files/sampul_koleksi/original/Monograf/{cover_url}')
        return Response(
            response.content, 
            content_type=response.headers['content-type']
        )
    except Exception as e:
        return '', 404




def generate_sql_query(prompt):
    """Convert natural language to SQL PostgreSQL using Gemini AI"""
    system_prompt = """
    You are a SQL expert. Convert the following natural language query into a PostgreSQL query.
    Use these table structures and relationships:
    
    Tables:
    - COLLECTIONS: Contains book information
      * ID (numeric)
      * NOMORBARCODE (varchar)
      * STATUS (varchar)
      * CATALOG_ID (numeric) - references CATALOGS.ID
    
    - COLLECTIONLOANITEMS: Contains loan records
      * COLLECTION_ID (numeric) - references COLLECTIONS.ID
      * MEMBER_ID (numeric) - references MEMBERS.ID
      * LOANDATE (timestamp)
    
    - CATALOGS: Contains book metadata
      * ID (numeric)
      * TITLE (varchar)
      * AUTHOR (varchar)
      * PUBLISHER (varchar)
      * SUBJECT (varchar)
      * COVERURL (varchar)
    
    - MEMBERS: Contains member information
      * ID (numeric)
      * MEMBERNO (varchar)
      * FULLNAME (varchar)
    
    Rules:
    1. Always use double quotes for table and column names
    2. Include error handling
    3. Optimize for performance
    4. Return only the SQL query
    5. For book covers and titles, always join COLLECTIONS with CATALOGS using CATALOG_ID
    6. When counting borrows, join COLLECTIONS with COLLECTIONLOANITEMS using COLLECTIONS.ID
    7. in every result alwasy show AUTHOR, TITLE, PUBLISHER, COVERURL
    
    Example for most borrowed books:
    SELECT 
        cat."TITLE",
        cat."AUTHOR",
        cat."PUBLISHER",
        cat."COVERURL",
        COUNT(cli."ID") as borrow_count
    FROM "COLLECTIONS" c
    LEFT JOIN "CATALOGS" cat ON c."CATALOG_ID" = cat."ID"
    LEFT JOIN "COLLECTIONLOANITEMS" cli ON c."ID" = cli."COLLECTION_ID"
    GROUP BY cat."ID", cat."TITLE", cat."AUTHOR", cat."PUBLISHER", cat."COVERURL"
    ORDER BY borrow_count DESC
    LIMIT 10;
    """
    
    try:
        response = model.generate_content(f"{system_prompt}\n\nQuery: {prompt}")
        sql_query = response.text.strip()
        # Remove markdown code blocks if present
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        return sql_query
    except Exception as e:
        print(f"Error generating SQL: {str(e)}")
        return None

@app.route('/api/search', methods=['POST'])
def natural_language_search():
    """API endpoint for natural language search"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
            
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
            
        # For common queries, use predefined SQL
        if query.lower() in ['show top 10 most borrowed books', 'show me top 10 most borrowed books']:
            sql_query = """
                SELECT 
                    cat."TITLE",
                    cat."AUTHOR",
                    cat."PUBLISHER",
                    cat."COVERURL",
                    COUNT(cli."ID") as borrow_count
                FROM "COLLECTIONS" c
                LEFT JOIN "CATALOGS" cat ON c."CATALOG_ID" = cat."ID"
                LEFT JOIN "COLLECTIONLOANITEMS" cli ON c."ID" = cli."COLLECTION_ID"
                GROUP BY cat."ID", cat."TITLE", cat."AUTHOR", cat."PUBLISHER", cat."COVERURL"
                ORDER BY borrow_count DESC
                LIMIT 10
            """
        else:
            # Generate SQL from natural language
            sql_query = generate_sql_query(query)
            
        if not sql_query:
            return jsonify({
                'success': False,
                'error': 'Failed to generate SQL query'
            }), 500
        
        # Execute the generated query
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    cur.execute(sql_query)
                    results = cur.fetchall()
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'id': result.get('ID'),
                            'title': result.get('TITLE'),
                            'author': result.get('AUTHOR', 'Unknown'),
                            'cover_url': result.get('COVERURL'),
                            'publisher': result.get('PUBLISHER', 'Unknown'),
                            'borrow_count': result.get('borrow_count', 0)
                        })
                    
                    return jsonify({
                        'success': True,
                        'data': formatted_results,
                        'sql_query': sql_query  # For debugging
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Error executing query: {str(e)}',
                        'sql_query': sql_query  # For debugging
                    }), 500
                    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.route('/')
def index():
    return render_template('index_pg.html')

@app.route('/api/matrix/<member_no>')
def get_matrix_data(member_no):
    """API endpoint to get matrix calculations"""
    try:
        matrix_data = create_user_item_matrix(member_no)
        
        if matrix_data is None:
            return jsonify({
                'success': False,
                'error': 'No loan data found for this member'
            }), 404
        
        # Get similar users data
        target_idx = matrix_data['target_user_idx']
        similar_users = []
        
        for i in range(len(matrix_data['users'])):
            if i != target_idx:
                similar_users.append({
                    'user': matrix_data['users'][i],
                    'similarity_score': float(matrix_data['user_similarity'][target_idx][i])
                })
        
        # Sort by similarity score
        similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        response_data = {
            'target_user': member_no,
            'interaction_matrix': {
                'users': matrix_data['users'],
                'items': matrix_data['items'],
                'matrix': matrix_data['matrix'].tolist()
            },
            'similar_users': similar_users[:5]  # Top 5 similar users
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommendations/<member_no>', methods=['GET'])
def get_book_recommendations(member_no):
    """API endpoint to get book recommendations"""
    try:
        recommendations = get_recommendations(member_no)
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history/<member_no>', methods=['GET'])
def get_user_history(member_no):
    """API endpoint to get user's loan history"""
    try:
        history = get_user_loan_history(member_no)
        formatted_history = []
        for item in history:
            formatted_history.append({
                'collection_id': item['COLLECTION_ID'],
                'title': item['TITLE'],
                'author': item['AUTHOR'],
                'loandate': item['LOANDATE'],
                'cover_url': item['COVERURL']
            })
        return jsonify({
            'success': True,
            'history': formatted_history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    
if __name__ == '__main__':
    app.run(debug=True)
