import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

class DBClient:
    """Handles database connections and queries for DropTruck."""
    
    def __init__(self):
        self.config = {
            'user': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT')),
            'database': os.getenv('DB_DATABASE'),
            'raise_on_warnings': True
        }
        self.connection = None

    def connect(self):
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            return True
        except mysql.connector.Error as err:
            print(f"❌ Database connection failed: {err}")
            return False

    def close(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def get_truck_types(self):
        """Fetch all active truck types."""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT id, name FROM truck_types WHERE deleted_at IS NULL"
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except mysql.connector.Error as err:
            print(f"❌ Failed to fetch truck types: {err}")
            return []
        finally:
            self.close()

    def get_body_types(self):
        """Fetch all active body types."""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT id, name FROM body_types WHERE deleted_at IS NULL"
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except mysql.connector.Error as err:
            print(f"❌ Failed to fetch body types: {err}")
            return []
        finally:
            self.close()

    def get_truck_type_id(self, name: str):
        """Get truck type ID by name."""
        if not self.connect():
            return None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT id FROM truck_types WHERE name = %s AND deleted_at IS NULL"
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            cursor.close()
            return result['id'] if result else None
        except mysql.connector.Error as err:
            print(f"❌ Failed to fetch truck type ID: {err}")
            return None
        finally:
            self.close()

    def get_body_type_id(self, name: str):
        """Get body type ID by name."""
        if not self.connect():
            return None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT id FROM body_types WHERE name = %s AND deleted_at IS NULL"
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            cursor.close()
            return result['id'] if result else None
        except mysql.connector.Error as err:
            print(f"❌ Failed to fetch body type ID: {err}")
            return None
        finally:
            self.close()
