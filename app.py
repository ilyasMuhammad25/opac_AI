from flask import Flask, request, render_template, jsonify
import mysql.connector
from mysql.connector import pooling
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict
from datetime import datetime, timedelta
import google.generativeai as genai
import asyncio
from typing import Dict, Any
import pandas as pd
import numpy as np

app = Flask(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'inlislite_pangkalpinang'
}

# Initialize the connection pool
db_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,  # Number of connections in the pool
    **db_config
)

def get_db_connection():
    """
    Fetch a database connection from the connection pool.
    """
    try:
        # Get a connection from the pool
        conn = db_pool.get_connection()
        if conn.is_connected():
            return conn
        else:
            raise Exception("Failed to establish a database connection.")
    except Exception as e:
        app.logger.error(f"Database connection error: {e}")
        raise

# start kodingan klasifikasi
class LibraryClassifier:
    def __init__(self, db_config):
        """
        Initialize the classifier with database configuration
        
        Parameters:
        db_config (dict): Database connection configuration
        """
        self.db_config = db_config
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.subject_vectors = None
        self.book_metadata = None
        
    def _fetch_book_data(self):
        """Fetch book data from database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT 
                c.ID,
                c.Title,
                c.Author,
                c.Publisher,
                c.Subject,
                c.PublishYear,
                COUNT(cli.ID) as borrow_count
            FROM catalogs c
            LEFT JOIN collectionloanitems cli ON c.ID = cli.Collection_id
            WHERE c.Subject IS NOT NULL
            GROUP BY c.ID
            """
            
            cursor.execute(query)
            return pd.DataFrame(cursor.fetchall())
            
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()
                
    def train_classifier(self):
        """Train the classifier on the current library catalog"""
        # Fetch book data
        df = self._fetch_book_data()
        self.book_metadata = df
        
        # Create subject vectors
        subjects = df['Subject'].fillna('')
        self.subject_vectors = self.vectorizer.fit_transform(subjects)
        
        # Create subject categories
        self.subject_categories = self._create_subject_categories(df)
        
    def _create_subject_categories(self, df):
        """Create hierarchical subject categories"""
        categories = defaultdict(list)
        
        for _, row in df.iterrows():
            subject = row['Subject']
            if pd.isna(subject) or not subject:
                continue
                
            # Split subject into hierarchical levels
            levels = [s.strip() for s in subject.split('-')]
            
            # Add book to each level of hierarchy
            current = ''
            for level in levels:
                current = (current + ' - ' + level).strip()
                categories[current].append(row['ID'])
                
        return categories
        
    def get_similar_books(self, book_id, n=5):
        """
        Get similar books based on subject similarity
        
        Parameters:
        book_id (int): ID of the target book
        n (int): Number of similar books to return
        
        Returns:
        list: Similar books with similarity scores
        """
        if self.subject_vectors is None:
            self.train_classifier()
            
        # Get book index
        book_idx = self.book_metadata[self.book_metadata['ID'] == book_id].index
        if len(book_idx) == 0:
            return []
            
        book_idx = book_idx[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            self.subject_vectors[book_idx:book_idx+1], 
            self.subject_vectors
        ).flatten()
        
        # Get top similar books
        similar_idxs = similarities.argsort()[::-1][1:n+1]
        
        similar_books = []
        for idx in similar_idxs:
            book = self.book_metadata.iloc[idx]
            similar_books.append({
                'id': book['ID'],
                'title': book['Title'],
                'author': book['Author'],
                'subject': book['Subject'],
                'similarity': similarities[idx]
            })
            
        return similar_books

def get_similar_books(title, author, subject):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT 
            Title, 
            Author, 
            Subject,
            COUNT(cli.ID) AS borrow_count
        FROM 
            catalogs c
        LEFT JOIN 
            collectionloanitems cli ON c.ID = cli.Collection_id
        WHERE 
            (c.Title LIKE %s OR c.Author LIKE %s OR c.Subject LIKE %s)
            AND NOT (c.Title = %s AND c.Author = %s AND c.Subject = %s)
        GROUP BY 
            c.ID
        ORDER BY 
            borrow_count DESC
        LIMIT 5;
        """
        cursor.execute(query, (f"%{title}%", f"%{author}%", f"%{subject}%", title, author, subject))
        return cursor.fetchall()
    except Exception as e:
        app.logger.error(f"Error in get_similar_books: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()


def get_suggested_categories(title, author, subject):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT 
            Subject AS category, 
            COUNT(*) AS similarity
        FROM 
            catalogs
        WHERE 
            Title LIKE %s OR
            Author LIKE %s OR
            Subject LIKE %s
        GROUP BY 
            Subject
        ORDER BY 
            similarity DESC
        LIMIT 5;
        """
        cursor.execute(query, (f"%{title}%", f"%{author}%", f"%{subject}%"))
        return cursor.fetchall()
    except Exception as e:
        app.logger.error(f"Error in get_suggested_categories: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        title = data.get('title', '').strip()
        author = data.get('author', '').strip()
        subject = data.get('subject', '').strip()

        # Validate inputs
        if not title or not author or not subject:
            raise ValueError("All fields (title, author, subject) are required.")

        # Fetch data from the database
        suggested_categories = get_suggested_categories(title, author, subject)
        similar_books = get_similar_books(title, author, subject)

        # Return the classification results
        return jsonify({
            "success": True,
            "data": {
                "suggested_categories": suggested_categories,
                "similar_books": similar_books,
            }
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error in classify route: {str(e)}")
        return jsonify({"success": False, "error": "Internal Server Error"}), 500

#------ end kodingan klasifikasi---------


# ------startn kodingan input search to sql query------
class GeminiSQLConverter:
    def __init__(self, api_key):
        # Configure Gemini AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # System prompt that defines the table structure
        self.system_prompt = """
        You are a SQL expert. Convert the following natural language query into a MySQL query.
        Use these table structures and relationships:

        Tables:
        - catalogs: Contains book information
          * ID (double)
          * Title (varchar)
          * Author (varchar)
          * Publisher (varchar)
          * PublishYear (varchar)
          * Subject (varchar)
          * CoverURL (varchar)
          * CreateDate (datetime)

        - collectionloanitems: Contains loan records
          * ID (double)
          * Collection_id (double)
          * member_id (double)
          * LoanDate (datetime)
          * DueDate (datetime)
          * ActualReturn (datetime)

        - members: Contains member information
          * ID (double)
          * MemberNo (varchar)
          * Fullname (varchar)

        Rules:
        1. Always include Title, Author, Publisher, Subject, CoverURL in SELECT
        2. For popularity queries, count loans using collectionloanitems
        3. When filtering by date, use proper date formats
        4. Always include proper JOINs between tables
        5. Always use proper MySQL syntax
        6. Consider case-insensitive searches using LIKE
        7. Add appropriate GROUP BY when using COUNT
        8. Include ORDER BY for proper sorting
        9. Add LIMIT for reasonable result sets
        10. Handle both Indonesian and English queries
        
        Examples:
        Query: "Show popular books"
        SQL: 
        SELECT 
            c.Title,
            c.Author,
            c.Publisher,
            c.Subject,
            c.CoverURL,
            COUNT(cli.ID) as borrow_count
        FROM catalogs c
        LEFT JOIN collectionloanitems cli ON cli.Collection_id = c.ID
        GROUP BY c.ID, c.Title, c.Author, c.Publisher, c.Subject, c.CoverURL
        HAVING COUNT(cli.ID) > 0
        ORDER BY borrow_count DESC
        LIMIT 10;

        Return only the SQL query without any explanation.
        """

    def generate_sql(self, query: str) -> Dict[str, Any]:
        """Generate SQL query using Gemini AI"""
        try:
            # Combine system prompt with user query
            full_prompt = f"{self.system_prompt}\n\nQuery: {query}\nSQL:"
            
            # Generate response from Gemini
            response = self.model.generate_content(full_prompt)
            sql_query = response.text.strip()
            
            # Clean up the SQL query - remove markdown and code formatting
            sql_query = self._clean_sql_response(sql_query)
            
            # Basic validation
            if not self._validate_sql(sql_query):
                raise ValueError("Generated SQL query is invalid")
            
            return {
                'query': sql_query,
                'success': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _clean_sql_response(self, sql: str) -> str:
        """Clean up the SQL response from Gemini AI"""
        # Remove markdown code blocks if present
        if sql.startswith('```sql'):
            sql = sql[6:]
        if sql.startswith('```'):
            sql = sql[3:]
        if sql.endswith('```'):
            sql = sql[:-3]
            
        # Remove any extra whitespace and newlines
        sql = sql.strip()
        
        # Remove any other markdown formatting if present
        sql = sql.replace('`', '')
        
        return sql

    def _validate_sql(self, sql: str) -> bool:
        """Basic validation of generated SQL query"""
        required_elements = [
            'SELECT',
            'FROM',
            'Title',
            'Author',
            'Publisher'
        ]
        
        sql_upper = sql.upper()
        return all(element.upper() in sql_upper for element in required_elements)

# Flask route for handling search
@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        nl_query = data.get('query', '')
        
        if not nl_query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            })

        # Initialize converter with your Gemini API key
        api_key = setup_gemini()
        if not api_key:
            raise Exception("Gemini API key not found")
            
        converter = GeminiSQLConverter(api_key)
        
        # Generate SQL using Gemini AI
        sql_data = converter.generate_sql(nl_query)
        
        if not sql_data['success']:
            raise Exception(sql_data['error'])

        # Execute generated query
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_data['query'])
        results = cursor.fetchall()
        
        return jsonify({
            'success': True,
            'data': results,
            'sql_query': sql_data['query']  # For debugging
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# Load Gemini API key from environment or config
def setup_gemini():
    try:
        with open('.env') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY'):
                    api_key = line.split('=')[1].strip()
                    return api_key
    except:
        return None
    
# ----end kodingan input search to sql-----


# -----start kodingan rekomendasi-----
def get_user_loan_history(member_no):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    loan_query = """
    SELECT cl.member_id, cl.Collection_id, c.Catalog_id, cat.Title, cat.CoverURL
    FROM collectionloanitems cl
    JOIN collections c ON cl.Collection_id = c.ID
    JOIN catalogs cat ON c.Catalog_id = cat.ID
    WHERE cl.member_id = %s
    """
    try:
        cursor.execute(loan_query, (member_no,))
        result = cursor.fetchall()
        return pd.DataFrame(result)
    except Exception as e:
        return {'error': str(e)}
    finally:
        cursor.close()
        connection.close()


def get_recommendations(member_no):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Query untuk mendapatkan data peminjaman, termasuk Author dan Subject
    loan_query = """
    SELECT cl.member_id, cl.Collection_id, c.Catalog_id, cat.Title, cat.Author, cat.Subject, cat.CoverURL
    FROM collectionloanitems cl
    JOIN collections c ON cl.Collection_id = c.ID
    JOIN catalogs cat ON c.Catalog_id = cat.ID
    """
    try:
        cursor.execute(loan_query)
        result = cursor.fetchall()
        df_loans = pd.DataFrame(result)

        # Cek apakah MemberNo ada dalam data
        member_query = """
        SELECT ID FROM members WHERE MemberNo = %s
        """
        cursor.execute(member_query, (member_no,))
        member_id_result = cursor.fetchone()

        if not member_id_result:
            return [], 0.0, 0.0, 0.0, 0.0  # Kembalikan rekomendasi kosong dan nilai metrik nol

        member_id = member_id_result['ID']

        # Pivot table untuk collaborative filtering
        pivot_table = df_loans.pivot_table(index='member_id', columns='Catalog_id', aggfunc='size', fill_value=0)

        # Menghitung cosine similarity
        cosine_sim = cosine_similarity(pivot_table)
        similarity_df = pd.DataFrame(cosine_sim, index=pivot_table.index, columns=pivot_table.index)

        # Mendapatkan anggota yang paling mirip
        similar_members = similarity_df[member_id].sort_values(ascending=False).iloc[1:11].index.tolist()

        # Mendapatkan rekomendasi buku
        recommended_books = df_loans[df_loans['member_id'].isin(similar_members)]
        recommended_books = recommended_books.groupby('Catalog_id').size().reset_index(name='count')
        recommended_books = recommended_books.sort_values(by='count', ascending=False).head(10)

        # Mengambil judul buku beserta Author dan Subject
        book_ids = recommended_books['Catalog_id'].tolist()
        book_query = """
        SELECT ID, Title, Author, Subject, CoverURL FROM catalogs WHERE ID IN (%s)
        """ % ','.join(['%s'] * len(book_ids))
        
        cursor.execute(book_query, book_ids)
        books = cursor.fetchall()
        
        recommendations = [dict(book) for book in books]

        # Perhitungan Metrik Evaluasi

        # Ground truth: Buku yang telah dipinjam oleh pengguna (buku relevan)
        user_books = set(df_loans[df_loans['member_id'] == member_id]['Catalog_id'].tolist())

        # Buku yang direkomendasikan: ID Katalog dari rekomendasi teratas
        recommended_books_set = set(recommended_books['Catalog_id'].tolist())

        # Precision = jumlah buku relevan yang direkomendasikan / jumlah total buku yang direkomendasikan
        relevant_recommended_books = recommended_books_set.intersection(user_books)
        precision = len(relevant_recommended_books) / len(recommended_books_set) if len(recommended_books_set) > 0 else 0.0

        # Recall = jumlah buku relevan yang direkomendasikan / jumlah total buku relevan
        recall = len(relevant_recommended_books) / len(user_books) if len(user_books) > 0 else 0.0

        # Akurasi = jumlah buku relevan yang direkomendasikan / jumlah buku relevan dalam sistem
        accuracy = len(relevant_recommended_books) / len(user_books) if len(user_books) > 0 else 0.0

        # Perhitungan NDCG (menggunakan peringkat dari buku yang direkomendasikan)
        # Membuat skor relevansi (1 jika relevan, 0 jika tidak) untuk buku yang direkomendasikan
        relevance_scores = [1 if book_id in user_books else 0 for book_id in recommended_books['Catalog_id']]
        
        # Menghitung DCG (Discounted Cumulative Gain)
        dcg = np.sum([relevance_scores[i] / np.log2(i + 2) for i in range(len(relevance_scores))])
        
        # Menghitung Ideal DCG (IDCG) berdasarkan skor relevansi yang diurutkan
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        idcg = np.sum([ideal_relevance_scores[i] / np.log2(i + 2) for i in range(len(ideal_relevance_scores))])
        
        # NDCG = DCG / IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return recommendations, precision, recall, ndcg, accuracy

    except Exception as e:
        return [], 0.0, 0.0, 0.0, 0.0  # Kembalikan rekomendasi kosong dan nilai metrik nol
    finally:
        cursor.close()
        connection.close()




# ------end kodingan rekomendasi--------        



@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    member_no = ''
    is_new_user = False
    precision = recall = ndcg = accuracy = None  # Tambahkan variabel accuracy
    
    if request.method == 'POST':
        member_no = request.form.get('member_no')
        if member_no:
            # Dapatkan riwayat peminjaman pengguna
            user_history = get_user_loan_history(member_no)
            # Mendapatkan rekomendasi dan metrik
            recommendations, precision, recall, ndcg, accuracy = get_recommendations(member_no)  # Ambil akurasi juga
            is_new_user = user_history.empty  # Gunakan .empty untuk memeriksa apakah DataFrame kosong
    
    return render_template('index.html', 
                           recommendations=recommendations,
                           member_no=member_no,
                           is_new_user=is_new_user,
                           precision=precision,
                           recall=recall,
                           ndcg=ndcg,
                           accuracy=accuracy)  # Kirimkan akurasi ke template




if __name__ == '__main__':
    app.run(debug=True)