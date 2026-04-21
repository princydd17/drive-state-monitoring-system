#!/usr/bin/env python3
"""
AI Detector - Machine learning-based drowsiness detection
Provides Random Forest, SVM, and ensemble detection models
"""

import numpy as np
import joblib
import json
import os
import sys
import time
from typing import Tuple, List, Dict, Optional, Any
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DriverAIDetector:
    """
    AI-based driver drowsiness detector using machine learning models.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize AI detector.
        
        Args:
            model_path: Path to trained model file
        """
        if model_path is None:
            # Look for model in AI models directory
            ai_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(ai_dir, "eye_predictor.dat")
        
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.model_version = "legacy"
        self.feature_names = [
            'left_ear', 'right_ear', 'avg_ear', 'ear_variance', 'blink_frequency',
            'eye_closure_duration', 'head_pitch', 'head_yaw', 'head_roll',
            'mar', 'yawn_frequency'
        ]
        
        # Load or create model
        self._load_or_create_model()
        
        # Configuration
        self.config = {
            'confidence_threshold': 0.7,
            'prediction_window': 10,
            'feature_scaling': True,
            'ensemble_voting': True
        }
        
        # Prediction history
        self.prediction_history = []
    
    def _load_or_create_model(self):
        """Load existing model or create a new one."""
        try:
            if os.path.exists(self.model_path):
                # Load existing model
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.model = model_data['model']
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names', self.feature_names)
                else:
                    # Backward support for direct serialized estimators/pipelines.
                    self.model = model_data
                    self.scaler = None

                # Infer model version from metadata if available.
                model_dir = os.path.dirname(self.model_path)
                metadata_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        self.model_version = metadata.get("version", os.path.basename(model_dir))
                    except Exception:
                        self.model_version = os.path.basename(model_dir) or "legacy"
                else:
                    self.model_version = os.path.basename(model_dir) or "legacy"
                print(f"Loaded AI model from {self.model_path}")
            else:
                # Create new model
                self._create_new_model()
                print("Created new AI model")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new machine learning model."""
        try:
            # Import sklearn components here to avoid import errors
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Create Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Create scaler
            self.scaler = StandardScaler()
            
            # Train with synthetic data (in real application, use real training data)
            self._train_with_synthetic_data()
            
            # Save model
            self._save_model()
            
        except Exception as e:
            print(f"Error creating model: {e}")
    
    def _train_with_synthetic_data(self):
        """Train model with synthetic data for demonstration."""
        try:
            # Import sklearn components here
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Generate synthetic training data
            n_samples = 1000
            
            # Generate features
            X = np.random.randn(n_samples, len(self.feature_names))
            
            # Generate labels (0: alert, 1: drowsy)
            # Higher EAR values and lower head movement indicate alert state
            alert_conditions = (
                (X[:, 2] > 0.25) &  # avg_ear > 0.25
                (np.abs(X[:, 6]) < 10) &  # head_pitch < 10
                (np.abs(X[:, 7]) < 10)    # head_yaw < 10
            )
            
            y = np.where(alert_conditions, 0, 1)
            
            # Add some noise
            noise_mask = np.random.random(n_samples) < 0.1
            y[noise_mask] = 1 - y[noise_mask]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            if self.config['feature_scaling']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model trained with synthetic data. Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error training with synthetic data: {e}")
    
    def _save_model(self):
        """Save trained model to file."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config
            }
            
            joblib.dump(model_data, self.model_path)
            print(f"Model saved to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """
        Predict drowsiness from features.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple[bool, float, str]: (is_drowsy, confidence, severity_level)
        """
        if self.model is None:
            return False, 0.0, "none"
        
        try:
            # Ensure features have correct shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features when external scaler exists.
            if self.config['feature_scaling'] and self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get confidence
            confidence = max(probabilities)
            
            # Determine if drowsy
            is_drowsy = prediction == 1 and confidence > self.config['confidence_threshold']
            
            # Determine severity level
            severity_level = self._determine_severity(features[0], confidence)
            
            # Update prediction history
            self._update_prediction_history(is_drowsy, confidence, severity_level)
            
            return is_drowsy, confidence, severity_level
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0, "none"
    
    def _determine_severity(self, features: np.ndarray, confidence: float) -> str:
        """
        Determine drowsiness severity level.
        
        Args:
            features: Feature vector
            confidence: Prediction confidence
            
        Returns:
            str: Severity level
        """
        # Extract key features
        avg_ear = features[2] if len(features) > 2 else 0.25
        eye_closure_duration = features[5] if len(features) > 5 else 0.0
        head_movement = np.sqrt(features[6]**2 + features[7]**2) if len(features) > 7 else 0.0
        
        # Determine severity based on features and confidence
        if avg_ear < 0.15 and eye_closure_duration > 2.0 and confidence > 0.9:
            return "high"
        elif avg_ear < 0.20 and eye_closure_duration > 1.0 and confidence > 0.8:
            return "medium"
        elif avg_ear < 0.25 and confidence > 0.7:
            return "low"
        else:
            return "none"
    
    def _update_prediction_history(self, is_drowsy: bool, confidence: float, severity: str):
        """Update prediction history."""
        prediction_info = {
            'is_drowsy': is_drowsy,
            'confidence': confidence,
            'severity': severity,
            'timestamp': time.time()
        }
        
        self.prediction_history.append(prediction_info)
        
        # Keep history manageable
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        if not self.prediction_history:
            return {}
        
        # Calculate statistics
        total_predictions = len(self.prediction_history)
        drowsy_predictions = sum(1 for p in self.prediction_history if p['is_drowsy'])
        avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
        
        # Severity distribution
        severity_counts = {}
        for p in self.prediction_history:
            severity = p['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_predictions': total_predictions,
            'drowsy_predictions': drowsy_predictions,
            'drowsiness_rate': drowsy_predictions / total_predictions if total_predictions > 0 else 0.0,
            'average_confidence': avg_confidence,
            'severity_distribution': severity_counts
        }
    
    def retrain_model(self, new_data: np.ndarray, new_labels: np.ndarray):
        """
        Retrain model with new data.
        
        Args:
            new_data: New training features
            new_labels: New training labels
        """
        try:
            # Scale new data
            if self.config['feature_scaling'] and self.scaler is not None:
                new_data_scaled = self.scaler.transform(new_data)
            else:
                new_data_scaled = new_data
            
            # Retrain model
            self.model.fit(new_data_scaled, new_labels)
            
            # Save updated model
            self._save_model()
            
            print("Model retrained successfully")
            
        except Exception as e:
            print(f"Error retraining model: {e}")

class MultiDetector:
    """
    Ensemble detector combining multiple AI models.
    """
    
    def __init__(self):
        """Initialize multi-detector."""
        self.models = {}
        self.weights = {}
        self.scalers = {}
        
        # Initialize different models
        self._initialize_models()
        
        # Configuration
        self.config = {
            'voting_method': 'weighted',  # 'weighted', 'majority', 'average'
            'confidence_threshold': 0.6,
            'min_models': 2
        }
    
    def _initialize_models(self):
        """Initialize different machine learning models."""
        try:
            # Import sklearn components here
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.weights['random_forest'] = 0.4
            
            # SVM
            self.models['svm'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
            self.weights['svm'] = 0.3
            
            # Additional models can be added here
            # self.models['neural_network'] = MLPClassifier(...)
            # self.weights['neural_network'] = 0.3
            
            # Initialize scalers
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            print("Multi-detector models initialized")
            
        except Exception as e:
            print(f"Error initializing multi-detector models: {e}")
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all models with data.
        
        Args:
            X: Training features
            y: Training labels
        """
        try:
            for model_name, model in self.models.items():
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
                print(f"Trained {model_name}")
            
            print("All models trained successfully")
            
        except Exception as e:
            print(f"Error training models: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """
        Make ensemble prediction.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple[bool, float, str]: (is_drowsy, confidence, severity_level)
        """
        if not self.models:
            return False, 0.0, "none"
        
        try:
            predictions = []
            confidences = []
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                # Scale features
                X_scaled = self.scalers[model_name].transform(features.reshape(1, -1))
                
                # Get prediction and probability
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                
                predictions.append(pred)
                confidences.append(max(prob))
            
            # Combine predictions based on voting method
            if self.config['voting_method'] == 'weighted':
                # Weighted voting
                weighted_score = 0.0
                total_weight = 0.0
                
                for i, model_name in enumerate(self.models.keys()):
                    weight = self.weights[model_name]
                    weighted_score += predictions[i] * weight * confidences[i]
                    total_weight += weight
                
                if total_weight > 0:
                    final_score = weighted_score / total_weight
                    is_drowsy = final_score > 0.5
                    confidence = final_score if is_drowsy else 1.0 - final_score
                else:
                    is_drowsy = False
                    confidence = 0.0
            
            elif self.config['voting_method'] == 'majority':
                # Majority voting
                drowsy_votes = sum(predictions)
                total_votes = len(predictions)
                is_drowsy = drowsy_votes > total_votes / 2
                confidence = drowsy_votes / total_votes if is_drowsy else (total_votes - drowsy_votes) / total_votes
            
            else:  # average
                # Average confidence
                avg_prediction = np.mean(predictions)
                avg_confidence = np.mean(confidences)
                is_drowsy = avg_prediction > 0.5
                confidence = avg_confidence
            
            # Apply confidence threshold
            if confidence < self.config['confidence_threshold']:
                is_drowsy = False
            
            # Determine severity
            severity_level = self._determine_ensemble_severity(features, confidence, predictions)
            
            return is_drowsy, confidence, severity_level
            
        except Exception as e:
            print(f"Multi-detector prediction error: {e}")
            return False, 0.0, "none"
    
    def _determine_ensemble_severity(self, features: np.ndarray, confidence: float, predictions: List[int]) -> str:
        """
        Determine severity based on ensemble prediction.
        
        Args:
            features: Feature vector
            confidence: Ensemble confidence
            predictions: Individual model predictions
            
        Returns:
            str: Severity level
        """
        # Extract key features
        avg_ear = features[2] if len(features) > 2 else 0.25
        eye_closure_duration = features[5] if len(features) > 5 else 0.0
        
        # Count positive predictions
        positive_votes = sum(predictions)
        total_votes = len(predictions)
        agreement_rate = positive_votes / total_votes if total_votes > 0 else 0.0
        
        # Determine severity
        if avg_ear < 0.15 and eye_closure_duration > 2.0 and agreement_rate > 0.8 and confidence > 0.9:
            return "high"
        elif avg_ear < 0.20 and eye_closure_duration > 1.0 and agreement_rate > 0.6 and confidence > 0.8:
            return "medium"
        elif avg_ear < 0.25 and agreement_rate > 0.5 and confidence > 0.7:
            return "low"
        else:
            return "none"
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        performance = {}
        
        for model_name, model in self.models.items():
            performance[model_name] = {
                'type': type(model).__name__,
                'weight': self.weights.get(model_name, 0.0),
                'parameters': model.get_params() if hasattr(model, 'get_params') else {}
            }
        
        return performance

def create_multi_detector() -> MultiDetector:
    """
    Create a multi-detector instance.
    
    Returns:
        MultiDetector: Initialized multi-detector
    """
    return MultiDetector() 