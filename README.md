🍌 Banana Death Predictor
An end-to-end computer vision and machine learning platform designed to classify banana varieties, determine their ripeness stage (Unripe, Ripe, Overripe, Rotten), and predict their expiration horizon in days using spectral bio-features.

📂 Project Architecture
Plaintext
banana_death_predictor/
│
├── backend/                        # FastAPI Inference Server
│   ├── main.py                     # Dual-model pipeline (MobileNetV2 + Random Forest)
│   ├── banana_ripeness_mobilenetv2.keras # 4-class Keras CNN weights
│   ├── banana_days_regressor.pkl   # Scikit-Learn Random Forest Regressor
│   └── requirements.txt            # Python dependencies
│
└── frontend/                       # React Cyberpunk Web UI
    ├── src/
    │   ├── App.jsx                 # Landing, Loading, and Results views
    │   └── App.css                 # HUD grid animations & styling
    ├── package.json
    └── index.html
⚙️ Backend Setup & Execution (FastAPI)
1. Prerequisites
Python 3.10+ installed on your system.

2. Install Dependencies
Navigate into your backend folder and install the required packages:

Bash
cd backend
pip install -r requirements.txt
(If you don't have a requirements file yet, install manually: pip install fastapi uvicorn python-multipart pillow numpy joblib scikit-learn tensorflow)

3. Place Model Weights
Ensure your pre-trained model files are placed in the root of the backend/ directory:

banana_ripeness_mobilenetv2.keras

banana_days_regressor.pkl

4. Run the FastAPI Server
Start the local server using Uvicorn with auto-reload enabled:

Bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
API Documentation: Accessible at http://localhost:8000/docs (Swagger UI).

Health Check: http://localhost:8000/health

💻 Frontend Setup & Execution (React)
1. Prerequisites
Node.js (v18+ recommended) installed.

2. Install Dependencies
Navigate into your frontend folder and install dependencies (including Lucide icons):

Bash
cd frontend
npm install
npm install lucide-react
3. Run the Development Server
Bash
npm run dev
The web app will launch locally (typically at http://localhost:5173 or http://localhost:3000).

🔌 API Contract Reference
The backend communicates with the frontend via the following JSON contract:

Endpoint: POST /predict

Payload: multipart/form-data with key image (single image file).

Response Structure:

JSON
{
  "ripeness": "Over ripe",
  "num_days_to_death": 2.4,
  "comment": "Enzymatic browning active. High sucrose concentration; ideal for smoothies or banana bread."
}