import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
def initialize_firestore():
    current_dir = os.path.dirname(__file__)
    service_account_path = os.path.join(current_dir, '..', 'config', 'credentials', 'credentials.json')
    print(f'Path to service account file: {service_account_path}')
    cred = credentials.Certificate(service_account_path)

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db

def create_firestore_collection(db):
    doc_ref = db.collection('parameters').document('model_params')
    doc_ref.set({
        'n_estimators': 100,
        'criterion': 'gini'
    })

if __name__ == '__main__':
    try:
        db = initialize_firestore()
        create_firestore_collection(db)
    except Exception as e:
        print(f"An error occurred during initialization: {e}")



