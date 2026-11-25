import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate('services.json')  # Replace with your Firebase credentials file
firebase_admin.initialize_app(cred)
db = firestore.client()
coordinates_collection = db.collection('Coordinates')

# Function to clear documents with names starting with "palarivattom"
def clear_palarivattom_documents():
    print("Clearing documents with names starting with 'palarivattom'...")
    
    # List all documents in the collection
    docs = coordinates_collection.list_documents()
    
    deleted_count = 0
    for doc in docs:
        # Check if the document ID starts with "palarivattom_"
        if doc.id.startswith("sarath_"):
            print(f"Deleting document: {doc.id}")
            doc.delete()
            deleted_count += 1
    
    print(f"Deleted {deleted_count} documents.")

# Call the function to clear the documents
clear_palarivattom_documents()