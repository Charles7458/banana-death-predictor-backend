import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. Pydantic Response Contract
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    ripeness: str
    num_days_to_death: float
    comment: str


# Global model cache dictionary
ml_models = {}

MOBILENET_PATH = "banana_ripeness_mobilenetv2.keras"
REGRESSOR_PATH = "banana_days_regressor.pkl"

# Keras image_dataset_from_directory assigns alphabetical order by default
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

# Formatting map for UI presentation
RIPENESS_DISPLAY = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Over ripe",
    "rotten": "Rotten",
}


# ---------------------------------------------------------------------------
# 2. Server Lifespan (Preloads both models once at startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load MobileNetV2 Classifier
    if Path(MOBILENET_PATH).is_file():
        ml_models["classifier"] = tf.keras.models.load_model(MOBILENET_PATH)
        print(f"[OK] Loaded MobileNetV2 classifier from {MOBILENET_PATH}")
    else:
        print(f"[WARN] {MOBILENET_PATH} not found. Ripeness will fall back to heuristic.")
        ml_models["classifier"] = None

    # 2. Load Random Forest Regressor
    if Path(REGRESSOR_PATH).is_file():
        ml_models["regressor"] = joblib.load(REGRESSOR_PATH)
        print(f"[OK] Loaded shelf-life regressor from {REGRESSOR_PATH}")
    else:
        print(f"[WARN] {REGRESSOR_PATH} not found. Shelf-life will fall back to heuristic.")
        ml_models["regressor"] = None

    yield
    ml_models.clear()


app = FastAPI(title="Banana Death Predictor API", lifespan=lifespan)

# Enable CORS for React frontend (localhost:5173, localhost:3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 3. Preprocessing & Feature Extraction
# ---------------------------------------------------------------------------
def preprocess_for_mobilenet(image_bytes: bytes) -> np.ndarray:
    """Prepares 224x224 RGB image batch for MobileNetV2."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def extract_features_from_bytes(image_bytes: bytes) -> Optional[dict]:
    """Extracts peel HSV metrics for Random Forest regressor."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.resize(img, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Otsu threshold on Saturation channel
    _, peel_mask = cv2.threshold(
        s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    peel_mask = cv2.morphologyEx(peel_mask, cv2.MORPH_CLOSE, kernel)
    peel_mask = cv2.morphologyEx(peel_mask, cv2.MORPH_OPEN, kernel)

    total_peel_pixels = cv2.countNonZero(peel_mask)
    if total_peel_pixels == 0:
        return None

    # Necrotic spots mask
    lower_brown = np.array([0, 20, 0])
    upper_brown = np.array([30, 255, 95])
    spot_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    spot_mask = cv2.bitwise_and(spot_mask, peel_mask)
    brown_spot_pixels = cv2.countNonZero(spot_mask)

    spot_percentage = (brown_spot_pixels / total_peel_pixels) * 100.0
    healthy_mask = cv2.bitwise_and(peel_mask, cv2.bitwise_not(spot_mask))

    if cv2.countNonZero(healthy_mask) > 0:
        mean_peel_hue = float(np.mean(h_channel[healthy_mask > 0]))
        mean_peel_sat = float(np.mean(s_channel[healthy_mask > 0]))
        mean_peel_val = float(np.mean(v_channel[healthy_mask > 0]))
    else:
        mean_peel_hue, mean_peel_sat, mean_peel_val = 0.0, 0.0, 0.0

    overall_mean_val = float(np.mean(v_channel[peel_mask > 0]))

    return {
        "spot_percentage": spot_percentage,
        "mean_peel_hue": mean_peel_hue,
        "mean_peel_sat": mean_peel_sat,
        "mean_peel_val": mean_peel_val,
        "overall_mean_val": overall_mean_val,
    }


def generate_context_comment(ripeness_key: str, days_to_rot: float, confidence: float) -> str:
    """Produces biological and culinary advice matching model findings."""
    if ripeness_key == "unripe":
        return (
            f"High chlorophyll and resistant starch levels ({confidence * 100:.1f}% confidence). "
            "Pulp is still starchy; store at room temperature away from direct sunlight."
        )
    elif ripeness_key == "ripe":
        return (
            f"Peak physiological ripeness reached ({confidence * 100:.1f}% confidence). "
            "Optimal sugar-to-acid ratio; ideal for direct consumption over the next few days."
        )
    elif ripeness_key == "overripe":
        return (
            f"Enzymatic browning active ({confidence * 100:.1f}% confidence). "
            "Elevated sucrose and softening tissue; perfect for smoothies, banana bread, or freezing."
        )
    else:  # rotten
        return (
            f"Advanced microbial breakdown and fermentation detected ({confidence * 100:.1f}% confidence). "
            "Tissue senescence is complete; discard or compost."
        )


# ---------------------------------------------------------------------------
# 4. Main Inference Route
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_banana(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await image.read()

        # -------------------------------------------------------------------
        # A. Classification via MobileNetV2
        # -------------------------------------------------------------------
        classifier = ml_models.get("classifier")
        if classifier is not None:
            mobilenet_input = preprocess_for_mobilenet(contents)
            class_probs = classifier.predict(mobilenet_input, verbose=0)[0]
            top_idx = int(np.argmax(class_probs))
            raw_ripeness = CLASS_NAMES[top_idx]
            classification_conf = float(class_probs[top_idx])
        else:
            # Fallback if MobileNetV2 model file is missing
            raw_ripeness = "ripe"
            classification_conf = 0.50

        display_ripeness = RIPENESS_DISPLAY.get(raw_ripeness, raw_ripeness.capitalize())

        # -------------------------------------------------------------------
        # B. Shelf-Life Estimation via Random Forest
        # -------------------------------------------------------------------
        regressor = ml_models.get("regressor")
        feats = extract_features_from_bytes(contents)

        if feats is not None and regressor is not None:
            feature_vector = np.array([[
                feats["spot_percentage"],
                feats["mean_peel_hue"],
                feats["mean_peel_sat"],
                feats["mean_peel_val"],
                feats["overall_mean_val"],
            ]])
            days_since_ripe = float(regressor.predict(feature_vector)[0])
            days_since_ripe = max(0.0, days_since_ripe)
            # Assuming full decomposition occurs ~5.5 days after initial ripening
            days_to_rot = max(0.0, 6.5 - days_since_ripe)
            if raw_ripeness == "unripe":
                days_to_rot = np.random.uniform(7.0,6.0)
            elif raw_ripeness == "rotten":
                days_to_rot = np.random.uniform(0.0,1.0)
        else:
            # Class-guided fallback if peel segmentation fails or regressor missing
            fallback_days = {
                "unripe": 6.0,
                "ripe": 4.0,
                "overripe": 1.5,
                "rotten": 0.0,
            }
            days_to_rot = fallback_days.get(raw_ripeness, 2.0)

        # -------------------------------------------------------------------
        # C. Generate Response
        # -------------------------------------------------------------------
        comment = generate_context_comment(raw_ripeness, days_to_rot, classification_conf)

        return PredictionResponse(
            ripeness=display_ripeness,
            num_days_to_death=round(days_to_rot, 1),
            comment=comment,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "mobilenet_loaded": ml_models.get("classifier") is not None,
        "regressor_loaded": ml_models.get("regressor") is not None,
    }