# app.py (Versión de Diagnóstico)


import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from typing import List, Dict, Any

print(f"[INFO] numpy version: {np.__version__}")
print(f"[INFO] pandas version: {pd.__version__}")
try:
    import sklearn
    print(f"[INFO] scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("[INFO] scikit-learn not installed")
print(f"[INFO] joblib version: {joblib.__version__}")

print(f"[INFO] numpy version: {np.__version__}")

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
# Función para cargar el modelo (con logging de diagnóstico)
# ==============================================================================
def load_model_pipeline_diagnostic():
    """
    Intenta cargar el pipeline de scikit-learn entrenado y registra
    información de diagnóstico sobre el directorio del modelo.
    """
    # La ruta estándar de montaje del modelo en Vertex AI
    model_path = "/mnt/models/" 
    
    # Construye la ruta completa al archivo del modelo
    full_model_path = os.path.join(model_path, 'model.joblib')
    
    print(f"DEBUG: Intentando cargar el modelo desde: {full_model_path}")
    
    # ==========================================================================
    # LOGGING DE DIAGNÓSTICO: Inspeccionar el directorio /mnt/models/
    # ==========================================================================
    print(f"DEBUG: Verificando el directorio del modelo: {model_path}")
    if os.path.exists(model_path):
        print(f"DEBUG: Directorio {model_path} EXISTE.")
        if os.path.isdir(model_path):
            print(f"DEBUG: {model_path} es un directorio.")
            # Listar contenido del directorio
            print(f"DEBUG: Contenido de {model_path}:")
            try:
                contents = os.listdir(model_path)
                if not contents:
                    print(f"  El directorio {model_path} está VACÍO.")
                else:
                    for item in contents:
                        item_path = os.path.join(model_path, item)
                        if os.path.isfile(item_path):
                            print(f"  - Archivo: {item} (Tamaño: {os.path.getsize(item_path)} bytes)")
                        elif os.path.isdir(item_path):
                            print(f"  - Directorio: {item}")
                        else:
                            print(f"  - Otro: {item}")
            except Exception as e:
                print(f"DEBUG: Error al listar el contenido de {model_path}: {e}")
        else:
            print(f"DEBUG: {model_path} EXISTE, pero NO es un directorio.")
    else:
        print(f"DEBUG: Directorio {model_path} NO EXISTE.")
    # ==========================================================================

    try:
        # Carga el pipeline completo (preprocesador + modelo)
        model = joblib.load(full_model_path)
        print("Modelo cargado exitosamente.")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file NOT FOUND at {full_model_path}. Este es el problema principal.")
        raise RuntimeError(f"Model file not found: {full_model_path}")
    except Exception as e:
        print(f"ERROR: Fallo al cargar el modelo desde {full_model_path}: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load model: {e}")

# ==============================================================================
# Cargar el modelo al inicio del módulo
# ==============================================================================
MODEL_PIPELINE = load_model_pipeline_diagnostic()


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

