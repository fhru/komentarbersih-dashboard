import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Dict, Union
import time
import numpy as np

# Global variable untuk caching model
_model = None
_tokenizer = None
_classifier = None
_model_loaded = False

def load_model():
    """
    Memuat model IndoBERT untuk klasifikasi komentar judi.
    Model hanya dimuat sekali dan disimpan dalam cache.
    """
    global _model, _tokenizer, _classifier, _model_loaded
    
    if _model_loaded:
        print("Model sudah dimuat sebelumnya")
        return True
    
    try:
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained("fhru/indobert-komentarbersih")
        # Load model
        _model = TFAutoModelForSequenceClassification.from_pretrained("fhru/indobert-komentarbersih")
        _model_loaded = True
        print("Model dan tokenizer berhasil dimuat!")
        return True
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return False

def predict_text(text: str) -> Dict[str, Union[str, float, int]]:
    """
    Melakukan prediksi untuk satu teks secara manual (tanpa pipeline).
    """
    if not _model_loaded:
        if not load_model():
            return {"label": "Error", "confidence": 0.0, "prediction": -1}
    if not text or not text.strip():
        return {"label": "Komentar Normal", "confidence": 0.0, "prediction": 0}
    try:
        # Tokenisasi dan padding
        inputs = _tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        outputs = _model(inputs)
        logits = outputs.logits.numpy()[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = "Komentar Judi" if pred_id == 1 else "Komentar Normal"
        return {"label": label, "confidence": confidence, "prediction": pred_id}
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return {"label": "Komentar Normal", "confidence": 0.0, "prediction": 0}

def predict_batch(texts: List[str]) -> List[Dict[str, Union[str, float, int]]]:
    """
    Melakukan prediksi batch manual (tanpa pipeline).
    """
    if not _model_loaded:
        if not load_model():
            return [{"label": "Error", "confidence": 0.0, "prediction": -1}] * len(texts)
    if not texts:
        return []
    try:
        # Tokenisasi batch
        inputs = _tokenizer(texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
        outputs = _model(inputs)
        logits = outputs.logits.numpy()
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        pred_ids = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append({"label": "Komentar Normal", "confidence": 0.0, "prediction": 0})
            else:
                label = "Komentar Judi" if pred_ids[i] == 1 else "Komentar Normal"
                results.append({
                    "label": label,
                    "confidence": float(confidences[i]),
                    "prediction": int(pred_ids[i])
                })
        return results
    except Exception as e:
        print(f"Error saat prediksi batch: {e}")
        return [{"label": "Komentar Normal", "confidence": 0.0, "prediction": 0}] * len(texts)

def get_model_info() -> Dict[str, str]:
    """
    Mendapatkan informasi model.
    
    Returns:
        Dict[str, str]: Informasi model
    """
    return {
        "model_name": "fhru/indobert-komentarbersih",
        "labels": {0: "Komentar Normal", 1: "Komentar Judi"},
        "device": "GPU" if tf.config.list_physical_devices('GPU') else "CPU",
        "model_loaded": _model_loaded
    }
    
# Contoh penggunaan dan testing
if __name__ == "__main__":
    # Test single prediction
    print("=== Test Single Prediction ===")
    test_text = "Promo judi online terbaik, deposit 10rb dapat bonus 100rb!"
    result = predict_text(test_text)
    print(f"Input: {test_text}")
    print(f"Hasil: {result}")
    
    # Test batch prediction
    print("\n=== Test Batch Prediction ===")
    test_texts = [
        "Promo judi online terbaik!",
        "Hari ini cuaca bagus sekali",
        "Slot gacor deposit pulsa",
        "Makan siang enak sekali"
    ]
    
    results = predict_batch(test_texts)
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"Text {i+1}: {text}")
        print(f"Hasil: {result}")
        print()