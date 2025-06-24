import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pickle
import os

# Créer le dossier nécessaire
os.makedirs('model', exist_ok=True)

def train_cameroon_pricing_model():
    """
    Entraîne un modèle de prédiction de prix basé sur modele_ml_reseau
    """
    try:
        # Charger le dataset
        csv_files = [
            "dataset_projet_reseau_cameroun_enrichi.csv",
            "dataset_projet_reseau_100.csv"
        ]

        df = None
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"✅ Dataset chargé avec succès: {csv_file}")
                print(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")
                break
            except FileNotFoundError:
                print(f"Fichier {csv_file} non trouvé, essai du suivant...")
                continue

        if df is None:
            print("❌ Aucun dataset trouvé. Veuillez d'abord générer le dataset.")
            return None, None

        # Afficher des informations sur le dataset
        print("\n INFORMATIONS SUR LE DATASET")
        print("=" * 50)
        print(f"Nombre total d'entrées: {len(df)}")
        print(f"\n Répartition par domaine:")
        domain_counts = df['Domaine'].value_counts()
        for domain, count in domain_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {domain}: {count} entrées ({percentage:.1f}%)")

        print("\n Statistiques descriptives:")
        print(df.describe().round(4))

        # Vérifier les valeurs manquantes
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n Valeurs manquantes détectées:")
            print(missing_values[missing_values > 0])
            df = df.dropna()
            print(f"Dataset nettoyé: {len(df)} entrées restantes")

        # Encoder la variable catégorielle 'Domaine'
        le = LabelEncoder()
        df['Domaine_encoded'] = le.fit_transform(df['Domaine'])

        print(f"\n DOMAINES ENCODÉS:")
        domain_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        for domain, encoded in domain_mapping.items():
            count = domain_counts[domain]
            print(f"  {domain}: {encoded} ({count} entrées)")

        # Préparer les données pour l'entraînement
        X = df[['Domaine_encoded', 'Prix_concurrent', 'Cout_production', 'Marge_voulue']]
        y = df['Prix_marchandise']

        print(f"\n PRÉPARATION DE L'ENTRAÎNEMENT:")
        print(f"Caractéristiques utilisées: {list(X.columns)}")
        print(f"Variable cible: Prix_marchandise")
        print(f"Nombre total d'échantillons: {X.shape[0]}")

        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        print(f"Ensemble d'entraînement: {X_train.shape[0]} échantillons")
        print(f"Ensemble de test: {X_test.shape[0]} échantillons")

        # Entraîner le modèle
        model = LinearRegression()
        print(f"\n ENTRAÎNEMENT DU MODÈLE: Linear Regression")
        model.fit(X_train, y_train)

        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Évaluer le modèle
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

        print(f"\n RÉSULTATS DU MODÈLE")
        print("=" * 40)
        print(f"Score R² (train): {train_score:.4f}")
        print(f"Score R² (test): {test_score:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # Afficher les coefficients
        print(f"\n COEFFICIENTS DU MODÈLE")
        for feature, coef in zip(X.columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")
        print(f"Ordonnée à l'origine: {model.intercept_:.4f}")

        # Sauvegarder le modèle et les composants
        model_files = {
            'model/pricing_model.pkl': model,
            'model/label_encoder.pkl': le,
            'model/feature_names.pkl': list(X.columns),
            'model/domain_mapping.pkl': domain_mapping
        }

        print(f"\n SAUVEGARDE DES FICHIERS:")
        for filepath, obj in model_files.items():
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            print(f"✅ {filepath}")

        print(f"\n ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"Modèle utilisé: Linear Regression")
        print(f"Domaines supportés: {len(le.classes_)} ({', '.join(le.classes_)})")
        print(f"Précision (R²): {test_score:.1%}")

        return model, le

    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print(" DÉMARRAGE DE L'ENTRAÎNEMENT - MODÈLE PRIX CAMEROUN")
    print("=" * 60)
    train_cameroon_pricing_model()