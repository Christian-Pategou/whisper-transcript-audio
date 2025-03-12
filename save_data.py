from pymongo import MongoClient
import pandas as pd
from datetime import datetime

# Connexion à MongoDB
client = MongoClient('mongodb://SoluChange:SoluChange@31.207.35.98:27017/')

# consultations data
medical_maia_db = client['medical-maia-database']
consultations_collection = medical_maia_db["consultation"]

# patient informations
user_maia_db = client['user-maia-database']
patients_collection = user_maia_db['patient']

# Fonction pour récupérer les données de MongoDB et les sauvegarder dans des DataFrames
def save_mongo_to_dataframe():
    # Récupérer les données des patients
    patients_data = list(patients_collection.find())
    patients_df = pd.DataFrame(patients_data)

    # Convertir les champs de date
    try:
        patients_df['birthday'] = pd.to_datetime(patients_df['birthday'])
        patients_df['createdAt'] = pd.to_datetime(patients_df['createdAt'])
        patients_df['updatedAt'] = pd.to_datetime(patients_df['updatedAt'])
    except:
        pass

    # Récupérer les données des consultations
    consultations_data = list(consultations_collection.find())
    consultations_df = pd.DataFrame(consultations_data)
    consultations_df['consultedAt']=pd.to_datetime(consultations_df['consultedAt'])

    
    # Sauvegarder les DataFrames en fichiers CSV si nécessaire
    patients_df.to_csv('patients.csv', index=False)
    consultations_df.to_csv('consultations.csv', index=False)

if __name__ == "__main__":
    # Exécution de la fonction
    save_mongo_to_dataframe()
