import hashlib
import io
import logging
import os
import uuid

import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import shap
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash

from src.models import db, User, Farm, PredictionHistory

# ─── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

# ─── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# Fix #2: Fail loudly if secret key is missing — never silently rotate sessions
_secret_key = os.environ.get('FLASK_SECRET_KEY')
if not _secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.config['SECRET_KEY'] = _secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///farm.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB upload limit
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # CSRF tokens expire after 1 hour

# ─── Structured Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Extensions ──────────────────────────────────────────────────────────────
db.init_app(app)
csrf = CSRFProtect(app)

# Fix #9: Security headers via flask-talisman
# CSP is set to permissive to allow CDN fonts/scripts used in templates
Talisman(
    app,
    force_https=False,  # Set True behind a real HTTPS proxy in production
    strict_transport_security=False,
    content_security_policy={
        'default-src': "'self'",
        'script-src': [
            "'self'", "'unsafe-inline'", "'unsafe-eval'",
            'cdn.jsdelivr.net', 'cdnjs.cloudflare.com',
            'translate.google.com', 'translate.googleapis.com',
            'translate-pa.googleapis.com',
            'www.gstatic.com',
        ],
        'style-src': [
            "'self'", "'unsafe-inline'",
            'fonts.googleapis.com', 'cdnjs.cloudflare.com',
            'translate.googleapis.com', 'www.gstatic.com',
        ],
        'font-src': [
            "'self'", 'fonts.gstatic.com', 'cdnjs.cloudflare.com',
        ],
        'img-src': ["'self'", 'data:', 'translate.googleapis.com', 'www.google.com', 'www.gstatic.com', '*'],
        'connect-src': ["'self'", 'api.open-meteo.com', 'translate.googleapis.com', 'translate-pa.googleapis.com'],
        'frame-src': ["'self'", 'translate.googleapis.com'],
        'frame-ancestors': "'none'",
    },
    x_content_type_options=True,
    referrer_policy='strict-origin-when-cross-origin',
)

# Rate limiting — 200/day, 50/hour globally; stricter on auth/prediction routes
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# ─── Paths & Constants ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FERTILIZER_MODEL_PATH = os.path.join(BASE_DIR, "npk_model.joblib")
SEED_MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "disease_model.h5")
SEED_CLASSES_PATH = os.path.join(BASE_DIR, "seed_class_names.json")
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'img')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Load seed class names (saved by train_seed.py). Maps class index → folder name.
# Quality mapping: 'pure' → High, 'discolored'/'silkcut' → Average, 'broken' → Poor
import json as _json
SEED_CLASS_NAMES = []
if os.path.exists(SEED_CLASSES_PATH):
    with open(SEED_CLASSES_PATH) as _f:
        SEED_CLASS_NAMES = _json.load(_f)
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.info(f"Seed class names loaded: {SEED_CLASS_NAMES}")
else:
    # Fallback: alphabetical order from training (broken=0, discolored=1, pure=2, silkcut=3)
    SEED_CLASS_NAMES = ['broken', 'discolored', 'pure', 'silkcut']

DEFAULT_SEED_CLASSES = ['broken', 'discolored', 'pure', 'silkcut']

def _interpret_seed_class(class_name: str):
    """Map a seed class name to a human-friendly quality label and advice.
    Handles both merged quality labels (High/Average/Poor) and raw folder names
    (pure/discolored/silkcut/broken) for backwards compatibility.
    """
    name = class_name.lower()
    if name in ('high', 'pure'):
        return (
            'High Quality (Pure & Highly Suitable)',
            'The neural network detected uniform seed morphology, healthy coloring, and no structural defects.',
            'These seeds possess extremely high germination viability. Ready for planting without any treatment.'
        )
    elif name in ('average', 'discolored', 'silkcut'):
        return (
            'Average Quality (Minor Defects Detected)',
            'The seeds show minor morphological inconsistencies — slight discoloration or surface cuts — but core structure is intact.',
            'These seeds are viable for planting. Consider seed priming or fungicide treatment before sowing to improve germination rates.'
        )
    else:  # 'poor', 'broken'
        return (
            'Poor Quality (Broken / Defective)',
            'Visual analysis indicates severe structural damage — cracked or broken seed coats — which compromises germination potential.',
            'Do NOT plant these seeds. Discard them immediately and replace with a fresh, certified seed lot.'
        )

# Fix #14: Named constants replacing magic numbers in the fertilizer optimizer
# Fertilizer prices in INR/kg: [Urea, DAP, MOP]
FERT_COST_PER_KG = [6, 24, 20]
# Nutrient content coefficients (negative for linprog minimization):
# Urea supplies 46% N; DAP supplies 18% N and 46% P; MOP supplies 60% K
NUTRIENT_COEFFICIENTS = [
    [-0.46, -0.18, 0.00],  # Nitrogen row
    [ 0.00, -0.46, 0.00],  # Phosphorus row
    [ 0.00,  0.00, -0.60], # Potassium row
]

# ─── Load Models ─────────────────────────────────────────────────────────────
fertilizer_model = None
seed_model = None
disease_model = None

try:
    fertilizer_model = joblib.load(FERTILIZER_MODEL_PATH)
    logger.info("NPK Regressor model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading NPK Regressor model: {e}")

try:
    seed_model = load_model(SEED_MODEL_PATH)
    logger.info("Seed model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading seed model: {e}")

try:
    if os.path.exists(DISEASE_MODEL_PATH):
        disease_model = load_model(DISEASE_MODEL_PATH)
        logger.info("PlantVillage Disease model loaded successfully.")
    else:
        logger.info("Disease model not found. Building MobileNetV2 placeholder for demonstration.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        disease_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(38, activation='softmax')
        ])
        disease_model.predict(np.zeros((1, 224, 224, 3)))
        logger.info("Dynamic placeholder disease model ready.")
except Exception as e:
    logger.error(f"Error loading disease model: {e}")

# ─── Cache CSV Stats at Startup ───────────────────────────────────────────────
_stats_cache = {}

def _load_stats_cache():
    try:
        data_path = os.path.join(BASE_DIR, "Testing.csv")
        df = pd.read_csv(data_path)
        crop_yields = df.groupby('Crop')['Crop Yield'].mean().sort_values(ascending=False).head(5).to_dict()
        fert_dist = df['Suitable Fertilizer'].value_counts().to_dict()
        _stats_cache.update({
            'top_crops': list(crop_yields.keys()),
            'top_yields': [round(v, 2) for v in crop_yields.values()],
            'fert_labels': list(fert_dist.keys()),
            'fert_values': list(fert_dist.values())
        })
        logger.info("Stats cache loaded from CSV.")
    except Exception as e:
        logger.error(f"Failed to load stats cache: {e}")

with app.app_context():
    _load_stats_cache()

# ─── Translation Dictionary ───────────────────────────────────────────────────
translations = {
    'mr': {
        'n': 'नायट्रोजन', 'p': 'फॉस्फरस', 'k': 'पोटॅशियम',
        'cost': 'अंदाजित एकूण खर्च',
        'weather_warn': '⚠️ गंभीर हवामान चेतावणी: पुढील ४८ तासांत मुसळधार पाऊस अपेक्षित आहे.',
        'practices_base': 'समान वितरण सुनिश्चित करा आणि बियाण्यांशी थेट संपर्क टाळा.',
        'shopping_title': '🛒 खर्च अनुकूलन (लिनियर प्रोग्रामिंग)'
    },
    'hi': {
        'n': 'नाइट्रोजन', 'p': 'फास्फोरस', 'k': 'पोटैशियम',
        'cost': 'अनुमानित कुल लागत',
        'weather_warn': '⚠️ गंभीर मौसम चेतावनी: अगले 48 घंटों में भारी बारिश होने की उम्मीद है।',
        'practices_base': 'समान वितरण सुनिश्चित करें और बीजों के सीधे संपर्क से बचें।',
        'shopping_title': '🛒 लागत अनुकूलन (रेखीय प्रोग्रामिंग)'
    },
    'en': {
        'n': 'Nitrogen', 'p': 'Phosphorus', 'k': 'Potassium',
        'cost': 'Estimated Total Cost',
        'weather_warn': '⚠️ CRITICAL WEATHER WARNING: Heavy rain expected in 48 hours. Delay fertilizer application.',
        'practices_base': 'Ensure even distribution and avoid direct contact with seeds. Incorporate well into the root zone before planting.',
        'shopping_title': '🛒 Cost Optimization (Linear Programming)'
    }
}

# 38 PlantVillage Classes
PLANT_CLASSES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy',
    'Corn Cercospora', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Grape Healthy',
    'Orange Huanglongbing', 'Peach Bacterial Spot', 'Peach Healthy',
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Raspberry Healthy', 'Soybean Healthy', 'Squash Powdery Mildew',
    'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Tomato Healthy'
]

# ─── Helper Functions ─────────────────────────────────────────────────────────
def allowed_file(filename):
    """Check file extension against whitelist."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_image_bytes(file_bytes):
    """
    Fix #6: Magic-byte validation — verify actual image content, not just extension.
    Returns the re-opened PIL Image on success, raises ValueError on failure.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Raises if not a valid image format
        # Must reopen after verify() — PIL closes the stream
        return Image.open(io.BytesIO(file_bytes))
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"Invalid image content: {e}")

def optimize_fertilizer_cost(n_req, p_req, k_req):
    bounds = [(0, None), (0, None), (0, None)]
    res = opt.linprog(
        FERT_COST_PER_KG,
        A_ub=NUTRIENT_COEFFICIENTS,
        b_ub=[-n_req, -p_req, -k_req],
        bounds=bounds,
        method='highs'
    )
    if res.success:
        return {
            'Urea_kg': round(res.x[0], 2),
            'DAP_kg': round(res.x[1], 2),
            'MOP_kg': round(res.x[2], 2),
            'Total_Cost_INR': round(res.fun, 2)
        }
    return None

def generate_shap_plot(df):
    try:
        rf_model = fertilizer_model.named_steps['regressor'].estimators_[0]
        preprocessor = fertilizer_model.named_steps['preprocessor']
        X_transformed = preprocessor.transform(df)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_transformed)
        feature_names = preprocessor.get_feature_names_out()
        clean_names = []
        for name in feature_names:
            clean = name.split('__')[-1]
            if clean.startswith('Type of soil_'):
                clean = 'Soil Type: ' + clean.replace('Type of soil_', '')
            clean = clean.replace('_', ' ')
            clean_names.append(clean)
        vals = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(list(zip(clean_names, vals)), columns=['Feature', 'Importance'])
        shap_df = shap_df.sort_values(by='Importance', ascending=True).tail(5)
        plt.figure(figsize=(8, 4))
        plt.barh(shap_df['Feature'], shap_df['Importance'], color='#10b981')
        plt.title('SHAP Feature Importance (Nitrogen Target)', color='white')
        plt.gca().set_facecolor('#1a1c23')
        plt.gcf().set_facecolor('#1a1c23')
        plt.gca().tick_params(colors='white')
        plt.tight_layout()
        # Fix #5: Unique filename per request — prevents concurrent-user data leaks
        unique_id = uuid.uuid4().hex[:10]
        plot_filename = f'shap_{unique_id}.png'
        plot_path = os.path.join(STATIC_IMG_DIR, plot_filename)
        plt.savefig(plot_path, transparent=True)
        plt.close()
        return f'/static/img/{plot_filename}'
    except Exception as e:
        logger.error(f"SHAP plot generation failed: {e}")
        return None

def _find_last_conv_layer(model):
    """
    Fix #8: Recursively search for the last Conv2D layer, handling nested models
    like MobileNetV2 (which is itself a Model inside a Sequential).
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # Recurse into sub-models
            result = _find_last_conv_layer(layer)
            if result:
                return result
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def generate_gradcam(img_array, model):
    try:
        last_conv_layer_name = _find_last_conv_layer(model)
        if not last_conv_layer_name:
            logger.warning("Grad-CAM: No Conv2D layer found in model.")
            return None

        # Build a sub-model that outputs the conv layer and the final prediction
        # Works on both flat Sequential and nested models (MobileNetV2)
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer_name).output, model.output]
            )
        except ValueError:
            # Layer is in a sub-model — find and use the sub-model
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    try:
                        inner_output = layer.get_layer(last_conv_layer_name).output
                        grad_model = tf.keras.models.Model(
                            [model.inputs],
                            [inner_output, model.output]
                        )
                        break
                    except ValueError:
                        continue
            else:
                return None

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        img = img_array[0].copy()
        # Denormalize: handle both [0,1] and [-1,1] ranges
        if img.min() < 0:
            img = (img + 1.0) * 127.5
        else:
            img = img * 255.0

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = heatmap_colored * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

        # Fix #5: Unique filename per request
        unique_id = uuid.uuid4().hex[:10]
        cam_filename = f'gradcam_{unique_id}.jpg'
        cam_path = os.path.join(STATIC_IMG_DIR, cam_filename)
        cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        return f'/static/img/{cam_filename}'
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        return None

# ─── IoT Sensor Mock Route ────────────────────────────────────────────────────
@app.route('/api/v1/sensors/read', methods=['GET'])
@login_required
@limiter.limit("30 per minute")
def read_sensors():
    return jsonify({
        'ph': round(np.random.uniform(5.5, 7.5), 1),
        'temp': round(np.random.uniform(20.0, 35.0), 1),
        'rainfall': round(np.random.uniform(500, 1500), 1),
        'soil_type': np.random.choice(["Clayey soil", "Loamy soil", "Sandy soil"])
    })

# ─── Auth Routes ─────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.')
            return render_template('login.html')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            # Fix #4: Never log raw username (PII). Log anonymized hash + IP.
            uid_hash = hashlib.sha256(username.lower().encode()).hexdigest()[:12]
            logger.warning(
                f"Failed login attempt | user_hash={uid_hash} | ip={request.remote_addr}"
            )
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute", methods=["POST"])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.')
            return render_template('register.html')
        if len(username) > 50:
            flash('Username must be 50 characters or fewer.')
            return render_template('register.html')
        if len(password) < 8:
            flash('Password must be at least 8 characters.')
            return render_template('register.html')
        if username.lower() == 'admin':
            flash('This username is reserved. Please choose another.')
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        new_user = User(username, hashed)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    history = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('profile.html', history=history)

# ─── Main Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fertilizer')
@login_required
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/seed')
@login_required
def seed():
    return render_template('seed.html')

@app.route('/disease')
@login_required
def disease():
    return render_template('disease.html')

@app.route('/api/docs')
@login_required
def api_docs():
    if not current_user.is_admin:
        flash('Access Denied: Administrator privileges required.')
        return redirect(url_for('home'))
    return render_template('api_docs.html')

# ─── Health Check ─────────────────────────────────────────────────────────────
# Fix #3: Strip internal model state from public health response
@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# ─── Stats API ─────────────────────────────────────────────────────────────────
# Fix #7: Added rate limit to public stats endpoint
@app.route('/api/stats')
@limiter.limit("60 per minute")
def api_stats():
    if _stats_cache:
        return jsonify(_stats_cache)
    return jsonify({'error': 'Stats unavailable.'}), 503

# ─── Fertilizer Prediction ─────────────────────────────────────────────────────
@app.route('/api/v1/predict/fertilizer', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
@csrf.exempt
def api_predict_fertilizer():
    if fertilizer_model is None:
        return jsonify({'error': 'Fertilizer model is currently unavailable.'}), 503
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON payload.'}), 400

        lang = data.get('Language', 'en')
        t = translations.get(lang, translations['en'])
        heavy_rain = bool(data.get('heavy_rain', False))

        df = pd.DataFrame([{
            'State': str(data.get('State', ''))[:50],
            'Year': int(data.get('Year', 2023)),
            'Season': str(data.get('Season', ''))[:20],
            'Crop': str(data.get('Crop', ''))[:50],
            'Area': float(data.get('Area', 0)),
            'Production': float(data.get('Production', 0)),
            'Rainfall': float(data.get('Rainfall', 0)),
            'avg_temp': float(data.get('avg_temp', 25)),
            'PH Value of Soil': float(data.get('PH_Value_of_Soil', 7.0)),
            'Type of soil': str(data.get('Type_of_soil', ''))[:30]
        }])

        pred = fertilizer_model.predict(df)[0]
        n_req, p_req, k_req = round(pred[0], 1), round(pred[1], 1), round(pred[2], 1)
        shap_url = generate_shap_plot(df)
        opt_res = optimize_fertilizer_cost(n_req, p_req, k_req)

        prediction_str = f"N: {n_req} kg/ha | P: {p_req} kg/ha | K: {k_req} kg/ha"
        reasoning = f"{t['n']}: {n_req} kg/ha, {t['p']}: {p_req} kg/ha, {t['k']}: {k_req} kg/ha."

        practices = t['practices_base'] + "\n\n"
        if heavy_rain:
            practices = t['weather_warn'] + "\n\n" + practices
        if opt_res:
            practices += (
                f"\n{t['shopping_title']}:\n"
                f"Urea: {opt_res['Urea_kg']} kg\n"
                f"DAP: {opt_res['DAP_kg']} kg\n"
                f"MOP: {opt_res['MOP_kg']} kg\n"
                f"{t['cost']}: ₹{opt_res['Total_Cost_INR']}"
            )

        hist = PredictionHistory(
            current_user.id, 'Fertilizer',
            str(data.get('Crop')) + ' / ' + str(data.get('State')),
            prediction_str
        )
        db.session.add(hist)
        db.session.commit()

        return jsonify({
            'prediction': prediction_str,
            'reasoning': reasoning,
            'practices': practices,
            'shap_plot_url': shap_url
        })
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid input in fertilizer prediction: {e}")
        return jsonify({'error': 'Invalid input data. Please check all fields.'}), 400
    except Exception as e:
        logger.error(f"Fertilizer prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

# ─── Seed Prediction ──────────────────────────────────────────────────────────
@app.route('/api/v1/predict/seed', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
@csrf.exempt
def api_predict_seed():
    if seed_model is None:
        return jsonify({'error': 'Seed model is currently unavailable.'}), 503
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only JPG, PNG, and WEBP are allowed.'}), 400

        file_bytes = file.read()

        # Fix #6: Verify actual image bytes, not just extension
        try:
            img = verify_image_bytes(file_bytes)
        except ValueError:
            return jsonify({'error': 'Uploaded file is not a valid image.'}), 400

        img = img.resize((128, 128))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array.reshape(1, 128, 128, 3)  # EfficientNetB0 expects [0,255]

        prediction = seed_model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        # Get the actual class name and interpret it
        if SEED_CLASS_NAMES and predicted_class < len(SEED_CLASS_NAMES):
            raw_class = SEED_CLASS_NAMES[predicted_class]
        else:
            raw_class = ['Average', 'High', 'Poor'][predicted_class % 3]

        result, reasoning, practices = _interpret_seed_class(raw_class)

        # Flag low confidence so the UI can show a caution notice
        low_confidence = confidence < 60.0
        if low_confidence:
            practices = "⚠️ Low confidence prediction — consider re-uploading a clearer, well-lit image of the seeds. " + practices

        gradcam_url = generate_gradcam(img_array, seed_model)

        hist = PredictionHistory(current_user.id, 'Seed', 'Image Upload', result)
        db.session.add(hist)
        db.session.commit()

        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'reasoning': reasoning,
            'practices': practices,
            'gradcam_url': gradcam_url,
            'low_confidence': low_confidence,
        })
    except Exception as e:
        logger.error(f"Seed prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during analysis.'}), 500

# ─── Crop Disease Prediction ───────────────────────────────────────────────────
@app.route('/api/v1/predict/disease', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
@csrf.exempt
def api_predict_disease():
    if disease_model is None:
        return jsonify({'error': 'Disease model is currently unavailable.'}), 503
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only JPG, PNG, and WEBP are allowed.'}), 400

        file_bytes = file.read()

        # Fix #6: Verify actual image bytes
        try:
            img = verify_image_bytes(file_bytes)
        except ValueError:
            return jsonify({'error': 'Uploaded file is not a valid image.'}), 400

        img = img.resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1] for MobileNetV2
        img_array = img_array.reshape(1, 224, 224, 3)

        prediction = disease_model.predict(img_array)
        predicted_class_idx = int(np.argmax(prediction))
        if predicted_class_idx >= len(PLANT_CLASSES):
            predicted_class_idx = 0

        result = PLANT_CLASSES[predicted_class_idx]
        confidence = float(np.max(prediction)) * 100
        gradcam_url = generate_gradcam(img_array, disease_model)

        if "Healthy" in result:
            reasoning = "The neural network detected uniform leaf morphology with healthy chlorophyll patterns and no structural decay."
            practices = "Maintain regular watering and standard nutrient application. No chemical intervention required."
        else:
            reasoning = "Visual analysis indicates structural degradation, discoloration, or pathogenic patterns consistent with the identified disease."
            practices = "Isolate the affected plant immediately. Apply the appropriate fungicide/bactericide and prune infected areas to prevent spread."

        hist = PredictionHistory(current_user.id, 'Disease', 'Leaf Image Upload', result)
        db.session.add(hist)
        db.session.commit()

        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'reasoning': reasoning,
            'practices': practices,
            'gradcam_url': gradcam_url
        })
    except Exception as e:
        logger.error(f"Disease prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during disease analysis.'}), 500

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, port=5000)