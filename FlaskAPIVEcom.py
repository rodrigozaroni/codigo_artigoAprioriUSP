import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS 
# Carregar o arquivo CSV
file_path = 'association_rulesEcom.csv'
rules_df = pd.read_csv(file_path)

def recommend_products(product):
    # Encontrar regras onde o produto é um antecedente
    relevant_rules = rules_df[rules_df['Rule'].str.contains(product)]
    # Extrair produtos recomendados
    recommendations = set()
    for _, row in relevant_rules.iterrows():
        items = row['Rule'].split(' -> ')
        if items[0] == product:
            recommendations.add(items[1])
    return list(recommendations)

def list_skus():
    # Extrair todos os SKUs antes do "->"
    skus = set()
    for _, row in rules_df.iterrows():
        items = row['Rule'].split(' -> ')
        skus.add(items[0])
    return list(skus)

app = Flask(__name__)
CORS(app) 

@app.route('/recommendacao', methods=['GET'])
def get_recommendations():
    product = request.args.get('sku')
    if product:
        recommendations = recommend_products(product)
        return jsonify(recommendations)
    else:
        return jsonify({"error": "Produto não especificado"})

@app.route('/listaskus', methods=['GET'])
def get_skus_list():
    skus = list_skus()
    return jsonify({"SKUS para Vendas Adicionais": skus})

@app.route('/swagger', methods=['GET'])
def swagger():
    swagger_data = {
        "swagger": "2.0",
        "info": {
            "version": "1.0",
            "title": "API de Recomendações",
            "description": "API para obter recomendações de produtos"
        },
        "paths": {
            "/recommendacao": {
                "get": {
                    "summary": "Obter recomendações de produtos",
                    "responses": {
                        "200": {
                            "description": "Lista de recomendações",
                            "schema": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            },
            "/listaskus": {
                "get": {
                    "summary": "Obter lista de SKUs",
                    "responses": {
                        "200": {
                            "description": "Lista de SKUs",
                            "schema": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return jsonify(swagger_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=443)
