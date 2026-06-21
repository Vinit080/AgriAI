from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    # Issue #3 Fix: Role-based access control via DB column, not username string comparison
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    farms = db.relationship('Farm', backref='owner', lazy=True)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

    def __init__(self, username, password_hash, is_admin=False):
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin

class Farm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100))
    area_hectares = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)

    def __init__(self, name, location, area_hectares, user_id):
        self.name = name
        self.location = location
        self.area_hectares = area_hectares
        self.user_id = user_id

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    prediction_type = db.Column(db.String(20))  # 'Fertilizer', 'Seed', or 'Disease'
    input_data = db.Column(db.Text)  # Sanitized summary of inputs
    result = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, prediction_type, input_data, result):
        self.user_id = user_id
        self.prediction_type = prediction_type
        # Truncate to prevent DB pollution from large payloads
        self.input_data = str(input_data)[:500]
        self.result = str(result)[:200]
