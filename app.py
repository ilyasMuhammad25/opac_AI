from flask import Flask, request, render_template, jsonify
import mysql.connector
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import google.generativeai as genai
import asyncio
from typing import Dict, Any

app = Flask(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'inlislite_pangkalpinang'
}

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