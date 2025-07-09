# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from typing import List, Dict, Any
import numpy as np # Nueva importación
import sklearn # Nueva importación

# ==============================================================================
# IMPORTANTE: AJUSTA ESTA LISTA
# Define explícitamente las características que tu modelo espera,
# EN EL ORDEN CORRECTO en que fueron entrenadas.
# ==============================================================================
EXPECTED_FEATURES = [
    'bedrooms',
    'bathrooms',
    'sq_footage'
    # Asegúrate de que esta lista coincida exactamente con tu FEATURES en trainer/train.py
]

# Inicializa la aplicación Flask
app = Flask(__name__)

# ==============================================================================
# Función para cargar el modelo
# ==============================================================================
def load_model_pipeline():
    """
    Carga el pipeline de scikit-learn entrenado desde la ruta local.
    El modelo se copia directamente en la imagen en /app/model/.
    """
    # La ruta del modelo es ahora fija dentro del contenedor, ya que se incrusta en la imagen.
    model_path = "/app/model/"
    
    # Construye la ruta completa al archivo del modelo
    full_model_path = os.path.join(model_path, 'model.joblib')
    
    print(f"Intentando cargar el modelo desde: {full_model_path}")
    try:
        # Carga el pipeline completo (preprocesador + modelo)
        model = joblib.load(full_model_path)
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo desde {full_model_path}: {e}")
        import traceback
        traceback.print_exc()
        # Si el modelo no se carga, el worker no puede arrancar.
        raise RuntimeError(f"Failed to load model: {e}")

# ==============================================================================
# Cargar el modelo al inicio del módulo
# Esto asegura que el modelo se cargue una vez por cada worker de Gunicorn
# cuando el módulo app.py es importado.
# ==============================================================================
MODEL_PIPELINE = load_model_pipeline()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para realizar predicciones.
    Espera un cuerpo JSON con una lista de instancias.
    """
    if MODEL_PIPELINE is None:
        print("ERROR: MODEL_PIPELINE no está cargado en el endpoint /predict.")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        instances: List[Dict[str, Any]] = request.get_json(force=True)
        input_df = pd.DataFrame(instances)
        for feature in EXPECTED_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = pd.NA 
        input_df = input_df[EXPECTED_FEATURES]
        predictions = MODEL_PIPELINE.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud para verificar que el servidor está activo y el modelo cargado."""
    if MODEL_PIPELINE is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False, "reason": "Model not loaded"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
