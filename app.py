from flask import Flask, request, render_template, jsonify
import mysql.connector
from mysql.connector import pooling
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    'database': 'inlislite_pangkalpinang21122024'
}

# Initialize the connection pool
db_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,  # Number of connections in the pool
    **db_config
)

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

def get_popular_in_category(self, category, n=5):
        """
        Get popular books in a specific category
        
        Parameters:
        category (str): Target category
        n (int): Number of books to return
        
        Returns:
        list: Popular books in category
        """
        if category not in self.subject_categories:
            return []
            
        book_ids = self.subject_categories[category]
        category_books = self.book_metadata[
            self.book_metadata['ID'].isin(book_ids)
        ]
        
        return category_books.nlargest(n, 'borrow_count').to_dict('records')

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
    
def get_user_loan_history(member_no):
    """Get loan history for a specific member"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT DISTINCT c.Title, c.Author, c.Subject
        FROM members m
        JOIN collectionloans cl ON m.ID = cl.Member_id
        JOIN collectionloanitems cli ON cl.ID = cli.CollectionLoan_id
        JOIN catalogs c ON cli.Collection_id = c.ID
        WHERE m.MemberNo = %s
        """
        
        cursor.execute(query, (member_no,))
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_popular_books(limit=5):
    """Get most popular books from the last 3 months"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        three_months_ago = datetime.now() - timedelta(days=90)
        
        query = """
        SELECT 
            c.Title,
            c.Author,
            c.Subject,
            COUNT(cli.ID) as borrow_count
        FROM catalogs c
        JOIN collectionloanitems cli ON c.ID = cli.Collection_id
        WHERE cli.LoanDate >= %s
        GROUP BY c.ID
        ORDER BY borrow_count DESC
        LIMIT %s
        """
        
        cursor.execute(query, (three_months_ago, limit))
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_subject_recommendations(subject, limit=5):
    """Get books based on a specific subject"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            c.Title,
            c.Author,
            c.Subject,
            COUNT(cli.ID) as borrow_count
        FROM catalogs c
        LEFT JOIN collectionloanitems cli ON c.ID = cli.Collection_id
        WHERE c.Subject LIKE %s
        GROUP BY c.ID
        ORDER BY borrow_count DESC
        LIMIT %s
        """
        
        cursor.execute(query, (f"%{subject}%", limit))
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_new_books(limit=5):
    """Get recently added books"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            Title,
            Author,
            Subject,
            CreateDate
        FROM catalogs
        WHERE CreateDate IS NOT NULL
        ORDER BY CreateDate DESC
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_similar_users(member_no):
    """Get similar users based on borrowing patterns"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get books borrowed by the target user
        query = """
        SELECT DISTINCT c.Subject
        FROM members m
        JOIN collectionloans cl ON m.ID = cl.Member_id
        JOIN collectionloanitems cli ON cl.ID = cli.CollectionLoan_id
        JOIN catalogs c ON cli.Collection_id = c.ID
        WHERE m.MemberNo = %s
        """
        cursor.execute(query, (member_no,))
        user_subjects = {row['Subject'] for row in cursor.fetchall()}
        
        if not user_subjects:
            return []
            
        # Find users who borrowed books with similar subjects
        query = """
        SELECT DISTINCT m.ID, m.MemberNo
        FROM members m
        JOIN collectionloans cl ON m.ID = cl.Member_id
        JOIN collectionloanitems cli ON cl.ID = cli.CollectionLoan_id
        JOIN catalogs c ON cli.Collection_id = c.ID
        WHERE c.Subject IN %s
        AND m.MemberNo != %s
        LIMIT 10
        """
        cursor.execute(query, (tuple(user_subjects), member_no))
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_user_interests(member_no):
    """Get user interests based on profile or initial quiz"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get user's education and job information
        query = """
        SELECT 
            m.EducationLevel_id,
            m.Job_id,
            m.InstitutionName,
            m.JenjangPendidikan_id,
            m.Fakultas_id,
            m.Jurusan_id
        FROM members m
        WHERE m.MemberNo = %s
        """
        
        cursor.execute(query, (member_no,))
        return cursor.fetchone()
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_recommendations(member_no):
    """Get book recommendations with cold start handling"""
    user_history = get_user_loan_history(member_no)
    
    # If user has no history, use cold start strategy
    if not user_history:
        recommendations = []
        
        # 1. Get user interests from profile
        user_interests = get_user_interests(member_no)
        if user_interests:
            # Get recommendations based on education/profession
            if user_interests['Fakultas_id']:
                faculty_books = get_subject_recommendations(str(user_interests['Fakultas_id']), 2)
                recommendations.extend(faculty_books)
            
            if user_interests['Jurusan_id']:
                department_books = get_subject_recommendations(str(user_interests['Jurusan_id']), 2)
                recommendations.extend(department_books)
        
        # 2. Add some popular books
        popular_books = get_popular_books(3)
        recommendations.extend(popular_books)
        
        # 3. Add some new books
        new_books = get_new_books(2)
        recommendations.extend(new_books)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for book in recommendations:
            if book['Title'] not in seen:
                seen.add(book['Title'])
                unique_recommendations.append(book)
        
        return unique_recommendations[:5]  # Return top 5 recommendations
    
    # If user has history, use collaborative filtering
    similar_users = get_similar_users(member_no)
    if similar_users:
        return get_collaborative_recommendations(member_no, similar_users)
    
    # Fallback to popular books if collaborative filtering fails
    return get_popular_books()

def get_collaborative_recommendations(member_no, similar_users):
    """Get recommendations based on collaborative filtering"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        similar_member_ids = [user['ID'] for user in similar_users]
        
        query = """
        SELECT DISTINCT 
            c.Title, 
            c.Author, 
            c.Subject, 
            COUNT(*) as borrow_count
        FROM collectionloans cl
        JOIN collectionloanitems cli ON cl.ID = cli.CollectionLoan_id
        JOIN catalogs c ON cli.Collection_id = c.ID
        WHERE cl.Member_id IN %s
        AND c.ID NOT IN (
            SELECT DISTINCT cli2.Collection_id
            FROM members m
            JOIN collectionloans cl2 ON m.ID = cl2.Member_id
            JOIN collectionloanitems cli2 ON cl2.ID = cli2.CollectionLoan_id
            WHERE m.MemberNo = %s
        )
        GROUP BY c.ID
        ORDER BY borrow_count DESC
        LIMIT 5
        """
        
        cursor.execute(query, (tuple(similar_member_ids), member_no))
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    member_no = ''
    user_history = []
    is_new_user = False
    
    if request.method == 'POST':
        member_no = request.form.get('member_no')
        if member_no:
            user_history = get_user_loan_history(member_no)
            recommendations = get_recommendations(member_no)
            is_new_user = not bool(user_history)
    
    return render_template('index.html', 
                         recommendations=recommendations,
                         member_no=member_no,
                         user_history=user_history,
                         is_new_user=is_new_user)

if __name__ == '__main__':
    app.run(debug=True)