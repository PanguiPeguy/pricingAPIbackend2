from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import traceback
from datetime import datetime

app = Flask(__name__)

# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Chemins vers les fichiers du modèle
model_path = os.path.join(os.path.dirname(__file__), 'model', 'pricing_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')
features_path = os.path.join(os.path.dirname(__file__), 'model', 'feature_names.pkl')
domain_mapping_path = os.path.join(os.path.dirname(__file__), 'model', 'domain_mapping.pkl')

# Variables globales pour le modèle
model = None
label_encoder = None
feature_names = None
domain_mapping = None
model_info = {}

def load_model_components():
    global model, label_encoder, feature_names, domain_mapping, model_info
    try:
        logging.info("🔄 Chargement des composants du modèle...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        try:
            with open(domain_mapping_path, 'rb') as f:
                domain_mapping = pickle.load(f)
        except FileNotFoundError:
            domain_mapping = {domain: int(label_encoder.transform([domain])[0])
                              for domain in label_encoder.classes_}
        model_type = type(model).__name__
        model_info = {
            'type': model_type,
            'features': feature_names,
            'domains': list(label_encoder.classes_),
            'n_domains': len(label_encoder.classes_),
            'loaded_at': datetime.now().isoformat()
        }
        if hasattr(model, 'coef_'):
            model_info['coefficients'] = {
                feature_names[i]: float(model.coef_[i])
                for i in range(len(feature_names))
            }
            model_info['intercept'] = float(model.intercept_)
        elif hasattr(model, 'feature_importances_'):
            model_info['feature_importances'] = {
                feature_names[i]: float(model.feature_importances_[i])
                for i in range(len(feature_names))
            }
        logging.info("✅ Modèle chargé avec succès!")
        logging.info(f"   Type: {model_type}")
        logging.info(f"   Caractéristiques: {feature_names}")
        logging.info(f"   Domaines disponibles ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
        return True
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return False

model_loaded = load_model_components()


@app.route('/domains', methods=['GET'])
def get_domains():
    if not model_loaded:
        return jsonify({'error': 'Modèle non chargé'}), 500
    domains_info = {}
    for domain in label_encoder.classes_:
        encoded_value = int(label_encoder.transform([domain])[0])
        domains_info[domain] = {
            'code': encoded_value,
            'description': get_domain_description(domain)
        }
    return jsonify({
        'domaines_disponibles': domains_info,
        'total': len(label_encoder.classes_),
        'format_accepte': {
            'nom': 'Utilisez le nom du domaine (ex: "Électronique")',
            'code': 'Ou utilisez le code numérique (ex: 0)'
        }
    })

def get_domain_description(domain):
    descriptions = {
        'Électronique': 'Appareils électroniques, composants, gadgets',
        'Mode': 'Vêtements, accessoires, chaussures',
        'Maison': 'Articles ménagers, décoration, mobilier',
        'Sport': 'Équipements sportifs, vêtements de sport',
        'Automobile': 'Pièces auto, accessoires véhicules',
        'Livres': 'Ouvrages, manuels, littérature',
        'Beauté': 'Cosmétiques, soins, parfums',
        'Alimentation': 'Produits alimentaires, boissons'
    }
    return descriptions.get(domain, 'Secteur économique spécialisé')

def validate_numeric_ranges(prix_concurrent, cout_production, marge_voulue):
    errors = []
    if prix_concurrent < 0:
        errors.append('prix_concurrent doit être positif')
    if cout_production < 0:
        errors.append('cout_production doit être positif')
    if marge_voulue < 0:
        errors.append('marge_voulue doit être positive')
    if cout_production > prix_concurrent * 0.9:
        errors.append('cout_production semble trop élevé par rapport au prix_concurrent')
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Modèle non chargé',
            'message': 'Veuillez entraîner le modèle avec le script d\'entraînement'
        }), 500
    try:
        data = request.get_json()
        logging.debug(f"Requête reçue: {data}")
        if not data:
            return jsonify({
                'error': 'Aucune donnée JSON fournie',
                'format_attendu': {
                    'domaine': 'string ou int',
                    'prix_concurrent': 'float',
                    'cout_production': 'float',
                    'marge_voulue': 'float'
                }
            }), 400
        domaine = data.get('domaine')
        prix_concurrent = data.get('prix_concurrent')
        cout_production = data.get('cout_production')
        marge_voulue = data.get('marge_voulue')
        missing_fields = []
        if domaine is None: missing_fields.append('domaine')
        if prix_concurrent is None: missing_fields.append('prix_concurrent')
        if cout_production is None: missing_fields.append('cout_production')
        if marge_voulue is None: missing_fields.append('marge_voulue')
        if missing_fields:
            logging.error(f"Champs manquants: {missing_fields}")
            return jsonify({
                'error': 'Caractéristiques manquantes',
                'manquantes': missing_fields,
                'domaines_disponibles': list(label_encoder.classes_),
                'exemple_complet': {
                    'domaine': 'Électronique',
                    'prix_concurrent': 750.0,
                    'cout_production': 500.0,
                    'marge_voulue': 0.6
                }
            }), 400
        try:
            if isinstance(domaine, str):
                if domaine not in label_encoder.classes_:
                    domaine_matches = [d for d in label_encoder.classes_ if
                                       domaine.lower() in d.lower() or d.lower() in domaine.lower()]
                    if domaine_matches:
                        domaine = domaine_matches[0]
                    else:
                        logging.error(f"Domaine non reconnu: {domaine}")
                        return jsonify({
                            'error': f'Domaine "{domaine}" non reconnu',
                            'domaines_disponibles': list(label_encoder.classes_),
                            'suggestion': 'Vérifiez l\'orthographe ou utilisez un code numérique'
                        }), 400
                domaine_encode = label_encoder.transform([domaine])[0]
                domaine_nom = domaine
            else:
                domaine_encode = int(domaine)
                if domaine_encode < 0 or domaine_encode >= len(label_encoder.classes_):
                    logging.error(f"Code domaine invalide: {domaine}")
                    return jsonify({
                        'error': f'Code domaine {domaine} invalide',
                        'codes_valides': f'0 à {len(label_encoder.classes_) - 1}',
                        'domaines_disponibles': domain_mapping
                    }), 400
                domaine_nom = label_encoder.classes_[domaine_encode]
        except (ValueError, IndexError) as e:
            logging.error(f"Erreur d'encodage du domaine: {str(e)}")
            return jsonify({
                'error': f'Erreur d\'encodage du domaine: {str(e)}',
                'domaines_disponibles': list(label_encoder.classes_)
            }), 400
        try:
            prix_concurrent = float(prix_concurrent)
            cout_production = float(cout_production)
            marge_voulue = float(marge_voulue)
        except (ValueError, TypeError) as e:
            logging.error(f"Erreur de conversion numérique: {str(e)}")
            return jsonify({
                'error': f'Erreur de conversion numérique: {str(e)}',
                'message': 'Toutes les valeurs numériques doivent être des nombres valides'
            }), 400
        validation_errors = validate_numeric_ranges(prix_concurrent, cout_production, marge_voulue)
        if validation_errors:
            logging.error(f"Erreurs de validation: {validation_errors}")
            return jsonify({
                'error': 'Erreurs de validation',
                'details': validation_errors,
                'valeurs_recues': {
                    'prix_concurrent': prix_concurrent,
                    'cout_production': cout_production,
                    'marge_voulue': marge_voulue
                }
            }), 400
        features_df = pd.DataFrame({
            feature_names[0]: [domaine_encode],
            feature_names[1]: [prix_concurrent],
            feature_names[2]: [cout_production],
            feature_names[3]: [marge_voulue]
        })
        predicted_price = model.predict(features_df)[0]
        marge_calculee = (predicted_price - cout_production) / cout_production if cout_production > 0 else 0
        benefice_unitaire = predicted_price - cout_production
        ratio_concurrent = predicted_price / prix_concurrent if prix_concurrent > 0 else None
        strategie = "Équilibrée"
        if ratio_concurrent and ratio_concurrent < 0.9:
            strategie = "Agressive (prix bas)"
        elif ratio_concurrent and ratio_concurrent > 1.1:
            strategie = "Premium (prix élevé)"
        response = {
            'prix_predit': round(float(predicted_price), 2),
            'caracteristiques_utilisees': {
                'domaine': domaine_nom,
                'domaine_encode': int(domaine_encode),
                'prix_concurrent': prix_concurrent,
                'cout_production': cout_production,
                'marge_voulue': marge_voulue
            },
            'analyse_economique': {
                'marge_realisee': round(float(marge_calculee), 4),
                'benefice_unitaire': round(float(benefice_unitaire), 2),
                'ratio_prix_concurrent': round(float(ratio_concurrent), 3) if ratio_concurrent else None,
                'strategie_prix': strategie
            },
            'recommandations': {
                'competitivite': "Compétitif" if ratio_concurrent and 0.95 <= ratio_concurrent <= 1.05 else "À ajuster",
                'rentabilite': "Bonne" if marge_calculee >= marge_voulue * 0.8 else "Faible"
            },
            'statut': 'success',
            'timestamp': datetime.now().isoformat()
        }
        logging.info(f"Prédiction réussie: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'error': 'Erreur interne lors de la prédiction',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not model_loaded:
        return jsonify({
            'error': 'Modèle non chargé',
            'message': 'Veuillez entraîner le modèle'
        }), 500
    try:
        data = request.get_json()
        logging.debug(f"Requête batch reçue: {data}")
        if not data or 'predictions' not in data:
            return jsonify({
                'error': 'Format incorrect. Utilisez: {"predictions": [{"domaine": ..., "prix_concurrent": ..., etc.}]}',
                'exemple': {
                    'predictions': [
                        {
                            'domaine': 'Électronique',
                            'prix_concurrent': 750.0,
                            'cout_production': 500.0,
                            'marge_voulue': 0.6
                        }
                    ]
                }
            }), 400
        predictions_data = data['predictions']
        results = []
        for i, item in enumerate(predictions_data):
            try:
                domaine = item.get('domaine')
                prix_concurrent = item.get('prix_concurrent')
                cout_production = item.get('cout_production')
                marge_voulue = item.get('marge_voulue')
                missing_fields = []
                if domaine is None: missing_fields.append('domaine')
                if prix_concurrent is None: missing_fields.append('prix_concurrent')
                if cout_production is None: missing_fields.append('cout_production')
                if marge_voulue is None: missing_fields.append('marge_voulue')
                if missing_fields:
                    results.append({
                        'index': i,
                        'error': 'Caractéristiques manquantes',
                        'manquantes': missing_fields,
                        'caracteristiques': item,
                        'statut': 'error'
                    })
                    continue
                try:
                    if isinstance(domaine, str):
                        if domaine not in label_encoder.classes_:
                            domaine_matches = [d for d in label_encoder.classes_ if domaine.lower() in d.lower()]
                            if domaine_matches:
                                domaine = domaine_matches[0]
                            else:
                                results.append({
                                    'index': i,
                                    'error': f'Domaine "{domaine}" non reconnu',
                                    'caracteristiques': item,
                                    'statut': 'error'
                                })
                                continue
                        domaine_encode = label_encoder.transform([domaine])[0]
                        domaine_nom = domaine
                    else:
                        domaine_encode = int(domaine)
                        if domaine_encode < 0 or domaine_encode >= len(label_encoder.classes_):
                            results.append({
                                'index': i,
                                'error': f'Code domaine {domaine} invalide',
                                'caracteristiques': item,
                                'statut': 'error'
                            })
                            continue
                        domaine_nom = label_encoder.classes_[domaine_encode]
                except (ValueError, IndexError) as e:
                    results.append({
                        'index': i,
                        'error': f'Erreur d\'encodage du domaine: {str(e)}',
                        'caracteristiques': item,
                        'statut': 'error'
                    })
                    continue
                try:
                    prix_concurrent = float(prix_concurrent)
                    cout_production = float(cout_production)
                    marge_voulue = float(marge_voulue)
                except (ValueError, TypeError) as e:
                    results.append({
                        'index': i,
                        'error': f'Erreur de conversion numérique: {str(e)}',
                        'caracteristiques': item,
                        'statut': 'error'
                    })
                    continue
                validation_errors = validate_numeric_ranges(prix_concurrent, cout_production, marge_voulue)
                if validation_errors:
                    results.append({
                        'index': i,
                        'error': 'Erreurs de validation',
                        'details': validation_errors,
                        'caracteristiques': item,
                        'statut': 'error'
                    })
                    continue
                features_df = pd.DataFrame({
                    feature_names[0]: [domaine_encode],
                    feature_names[1]: [prix_concurrent],
                    feature_names[2]: [cout_production],
                    feature_names[3]: [marge_voulue]
                })
                predicted_price = model.predict(features_df)[0]
                marge_calculee = (predicted_price - cout_production) / cout_production if cout_production > 0 else 0
                ratio_concurrent = predicted_price / prix_concurrent if prix_concurrent > 0 else None
                results.append({
                    'index': i,
                    'prix_predit': round(float(predicted_price), 2),
                    'domaine': domaine_nom,
                    'marge_realisee': round(float(marge_calculee), 4),
                    'ratio_concurrent': round(float(ratio_concurrent), 3) if ratio_concurrent else None,
                    'caracteristiques': item,
                    'statut': 'success'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'caracteristiques': item,
                    'statut': 'error'
                })
        successful_predictions = [r for r in results if r['statut'] == 'success']
        if successful_predictions:
            avg_price = sum(r['prix_predit'] for r in successful_predictions) / len(successful_predictions)
            price_range = {
                'min': min(r['prix_predit'] for r in successful_predictions),
                'max': max(r['prix_predit'] for r in successful_predictions),
                'moyenne': round(avg_price, 2)
            }
        else:
            price_range = None
        return jsonify({
            'predictions': results,
            'statistiques': {
                'total': len(results),
                'succes': len(successful_predictions),
                'erreurs': len([r for r in results if r['statut'] == 'error']),
                'taux_succes': round(len(successful_predictions) / len(results) * 100, 1) if results else 0,
                'prix_range': price_range
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Erreur lors du traitement en lot: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'error': 'Erreur lors du traitement en lot',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info_endpoint():
    if not model_loaded:
        return jsonify({
            'error': 'Modèle non chargé'
        }), 500
    try:
        info = {
            'modele': {
                'type': model_info.get('type', 'N/A'),
                'caracteristiques': feature_names,
                'domaines_supportes': list(label_encoder.classes_),
                'nombre_domaines': len(label_encoder.classes_),
                'charge_le': model_info.get('loaded_at', 'N/A')
            },
            'encodage_domaines': {
                domain: int(label_encoder.transform([domain])[0])
                for domain in label_encoder.classes_
            }
        }
        if 'coefficients' in model_info:
            info['coefficients'] = model_info['coefficients']
            info['ordonnee_origine'] = model_info['intercept']
        info['conseils_utilisation'] = {
            'format_donnees': 'Utilisez des valeurs réelles (ex: 750.0 pour 750€)',
            'domaines': 'Nom complet ou code numérique acceptés',
            'validation': 'L\'API valide automatiquement la cohérence économique',
            'precision': 'Les prédictions sont arrondies à 2 décimales'
        }
        return jsonify(info)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des informations: {str(e)}")
        return jsonify({
            'error': 'Erreur lors de la récupération des informations du modèle',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logging.info("Démarrage de l'API de prédiction de prix - Version Multi-Domaines...")
    logging.info("Endpoints disponibles:")
    logging.info("  GET  / - Informations générales")
    logging.info("  GET  /domains - Liste des domaines supportés")
    logging.info("  POST /predict - Prédiction individuelle")
    logging.info("  POST /predict_batch - Prédictions en lot")
    logging.info("  GET  /model_info - Informations détaillées sur le modèle")
    if model_loaded:
        logging.info(f"✅ Modèle chargé: {len(label_encoder.classes_)} domaines disponibles")
        logging.info(f"   Domaines: {', '.join(list(label_encoder.classes_))}")
    else:
        logging.info("❌ Modèle non chargé - Entraînez d'abord le modèle!")
    logging.info("API accessible sur: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)