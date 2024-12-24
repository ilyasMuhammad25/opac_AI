from flask import Flask, request, render_template, jsonify
import mysql.connector
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

app = Flask(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'inlis_pangkalpinang'
}

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