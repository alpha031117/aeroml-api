"""
Script to reset the database by dropping all tables and recreating them.
WARNING: This will delete all data in the database!
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.db.database import engine, Base
from app.db import models  # Import models to register them


def reset_database():
    """Drop all tables and recreate them."""
    print("⚠️  WARNING: This will delete all data in the database!")
    print("Dropping all tables...")
    
    # Drop all tables
    with engine.connect() as conn:
        # Disable foreign key checks temporarily (PostgreSQL)
        conn.execute(text("SET session_replication_role = 'replica';"))
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        
        # Re-enable foreign key checks
        conn.execute(text("SET session_replication_role = 'origin';"))
        
        conn.commit()
    
    print("✅ All tables dropped.")
    print("Creating tables...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("✅ Database reset complete!")
    print("All tables have been recreated with the latest schema.")


if __name__ == "__main__":
    confirm = input("Are you sure you want to reset the database? This will delete ALL data. (yes/no): ")
    if confirm.lower() == "yes":
        reset_database()
    else:
        print("Database reset cancelled.")

