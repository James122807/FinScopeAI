# FinScope AI - Ultimate Financial Dashboard
# Fixed and enhanced version with improved error handling and missing features

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import os
import io
import tempfile
from datetime import datetime, timedelta
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ML / sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="FinScope AI - Ultimate Financial Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional dependencies check
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

try:
    import speech_recognition as sr

    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, ClientSettings
    from pydub import AudioSegment

    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False


# ==========================================
# Session state initialization
# ==========================================
def initialize_session_state():
    """Initialize all session state variables with default values"""
    if 'expenses_df' not in st.session_state:
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=400, freq='D')
        base_expenses = np.random.normal(80, 20, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        weekly = 15 * (dates.dayofweek >= 5)
        expenses = base_expenses + seasonal + weekly

        # Add anomalies
        expenses[50] = 400
        expenses[120] = 450
        expenses[200] = 30
        expenses[300] = 500
        expenses = np.clip(expenses, 0.5, None)

        # Generate categories and merchants
        categories = []
        merchants = []
        for i in range(len(dates)):
            day = dates[i].dayofweek
            if day in [5, 6]:
                cats = ["Food", "Entertainment", "Shopping"] * 3 + ["Transport"]
                mers = ["Restaurant", "Cinema", "Mall", "Grab", "Starbucks"]
            else:
                cats = ["Food", "Transport", "Bills", "Groceries"] * 2 + ["Shopping"]
                mers = ["Grab", "Office Canteen", "Tesco", "Petronas", "Lunch"]
            categories.append(np.random.choice(cats))
            merchants.append(np.random.choice(mers))

        df = pd.DataFrame({
            "date": dates,
            "expense": expenses,
            "category": categories,
            "merchant": merchants,
            "type": "variable",
            "notes": None,
            "recurring_id": None
        })
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        st.session_state.expenses_df = df

    if 'fixed_expenses' not in st.session_state:
        st.session_state.fixed_expenses = {
            "mortgage": 1800,
            "car_loan": 750,
            "insurance": 300,
            "utilities": 350,
            "internet": 199,
            "streaming": 89
        }

    if 'financial_goals' not in st.session_state:
        st.session_state.financial_goals = {
            "savings_target": 25000,
            "current_savings": 8000,
            "monthly_income": 7500,
            "emergency_fund_target": 15000,
            "investment_target": 50000
        }

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {
            "category_classifier": None,
            "category_scaler": None,
            "anomaly_detector": None,
            "forecast_model": None
        }

    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "currency": "RM",
            "language": "en",
            "voice_enabled": True,
            "auto_categorize": True,
            "alert_threshold": 500
        }

    if 'last_audio_recording' not in st.session_state:
        st.session_state.last_audio_recording = None


initialize_session_state()


# ==========================================
# Advanced OCR Engine
# ==========================================
class AdvancedOCREngine:
    """Enhanced OCR engine with advanced preprocessing"""

    def __init__(self):
        self.available = TESSERACT_AVAILABLE and CV2_AVAILABLE
        if not self.available:
            return

        self.merchant_keywords = {
            "Grab": ["grab", "ride", "transport"],
            "Starbucks": ["starbucks", "coffee", "cafe"],
            "Tesco": ["tesco", "grocer", "supermarket"],
            "Petronas": ["petronas", "petrol", "fuel"],
            "Shopee": ["shopee", "online", "ecommerce"],
            "Lazada": ["lazada", "online", "shopping"],
            "McDonald": ["mcdonald", "fast food", "burger"],
            "KFC": ["kfc", "fast food", "chicken"]
        }

        self.category_map = {
            "Food": ["restaurant", "cafe", "food", "meal", "lunch", "dinner", "coffee", "burger"],
            "Transport": ["grab", "taxi", "petrol", "fuel", "transport", "ride", "bus", "train"],
            "Shopping": ["shopee", "lazada", "shopping", "mall", "store", "purchase", "buy"],
            "Bills": ["bill", "electric", "water", "internet", "unifi", "streamyx", "tenaga"],
            "Entertainment": ["movie", "netflix", "cinema", "game", "entertainment", "spotify"],
            "Groceries": ["tesco", "supermarket", "grocery", "market", "aeon", "giant"],
            "Healthcare": ["clinic", "hospital", "pharmacy", "doctor", "medical", "watsons"],
            "Utilities": ["electricity", "water", "internet", "phone", "utility"]
        }

    def preprocess_image_advanced(self, image):
        """Apply multiple preprocessing techniques"""
        if not self.available:
            return None, "", "Required libraries not available"

        try:
            img_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Try multiple preprocessing techniques
            techniques = [gray]

            # Gaussian blur + Otsu threshold
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            techniques.append(thresh1)

            # Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            techniques.append(thresh2)

            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            techniques.append(morph)

            # Try each technique and pick best result
            best_text = ""
            best_image = gray

            for technique in techniques:
                try:
                    text = pytesseract.image_to_string(
                        technique,
                        config='--psm 6 --oem 3'
                    )
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        best_image = technique
                except Exception:
                    continue

            return best_image, best_text, None

        except Exception as e:
            return None, "", f"Image preprocessing failed: {e}"

    def extract_detailed_info(self, text):
        """Extract amount, date, merchant, category from OCR text"""
        info = {
            "amount": None,
            "date": None,
            "merchant": None,
            "category": "Other",
            "confidence": 0,
            "items": []
        }

        if not text:
            return info

        try:
            # Extract amount
            amount_patterns = [
                r'RM\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
                r'RM\s*(\d+\.\d{2})',
                r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
                r'Total\s*[:]?\s*[\$RM]?\s*(\d+\.\d{2})',
                r'Amount\s*[:]?\s*[\$RM]?\s*(\d+\.\d{2})',
                r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*RM'
            ]

            amounts_found = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        amount = float(str(match).replace(',', '').strip())
                        amounts_found.append(amount)
                    except Exception:
                        continue

            if amounts_found:
                info["amount"] = max(amounts_found)
                info["confidence"] += 0.4

            # Extract date
            date_patterns = [
                r'\d{4}[-/]\d{2}[-/]\d{2}',
                r'\d{2}[-/]\d{2}[-/]\d{4}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
            ]

            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        parsed = pd.to_datetime(match.group(), errors='coerce')
                        if not pd.isna(parsed):
                            info["date"] = parsed.strftime('%Y-%m-%d')
                            info["confidence"] += 0.3
                            break
                    except Exception:
                        continue

            # Identify merchant
            text_lower = text.lower()
            merchant_scores = {}
            for merchant, keywords in self.merchant_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    merchant_scores[merchant] = score

            if merchant_scores:
                info["merchant"] = max(merchant_scores, key=merchant_scores.get)
                info["confidence"] += 0.2

            # Identify category
            category_scores = {}
            for category, keywords in self.category_map.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    category_scores[category] = score

            if category_scores:
                info["category"] = max(category_scores, key=category_scores.get)
                info["confidence"] += 0.1

        except Exception as e:
            st.error(f"Error extracting info: {e}")

        return info


# ==========================================
# Voice Processing Helpers
# ==========================================
def parse_transcript_simple(text):
    """Parse voice transcript for expense details"""
    parsed = {
        "amount": None,
        "category": "Other",
        "merchant": None,
        "confidence": 0.0
    }

    if not text:
        return parsed

    txt = text.lower()

    # Extract amount
    amt_match = re.search(
        r'(?:rm\s*)?(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm|\$|dollars)?',
        txt
    )
    if amt_match:
        try:
            parsed["amount"] = float(amt_match.group(1))
            parsed["confidence"] += 0.4
        except Exception:
            pass

    # Identify category
    categories = {
        "Food": ["food", "lunch", "dinner", "restaurant", "cafe", "meal", "eat"],
        "Transport": ["grab", "taxi", "bus", "train", "petrol", "fuel", "ride"],
        "Shopping": ["shop", "shopping", "buy", "purchase", "mall"],
        "Groceries": ["grocery", "supermarket", "tesco", "market"],
        "Bills": ["bill", "electric", "water", "internet", "phone"],
        "Entertainment": ["movie", "cinema", "netflix", "game"]
    }

    for cat, keys in categories.items():
        if any(k in txt for k in keys):
            parsed["category"] = cat
            parsed["confidence"] += 0.25
            break

    # Identify merchant
    merchant_keys = ["grab", "starbucks", "tesco", "petronas", "mcdonald", "kfc"]
    for m in merchant_keys:
        if m in txt:
            parsed["merchant"] = m.title()
            parsed["confidence"] += 0.15
            break

    if parsed["amount"] is not None:
        parsed["confidence"] = min(parsed["confidence"] + 0.2, 1.0)

    return parsed


def convert_audio_to_wav(audio_bytes, input_format="webm"):
    """Convert audio bytes to WAV format"""
    if not STREAMLIT_WEBRTC_AVAILABLE:
        return None, "pydub not installed"

    try:
        audio = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format=input_format
        )
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        return buf.read(), None
    except Exception as e:
        return None, str(e)


# ==========================================
# ML Models
# ==========================================
class FinancialMLModels:
    """Machine learning models for financial analysis"""

    def __init__(self):
        self.scaler = StandardScaler()

    @st.cache_data(ttl=3600)
    def train_arima_enhanced(_self, data, order=(2, 1, 2)):
        """Train ARIMA model for forecasting"""
        if not ARIMA_AVAILABLE:
            return None, data

        try:
            if isinstance(data, pd.DataFrame):
                series = data['expense'] if 'expense' in data.columns else data.iloc[:, 0]
            else:
                series = pd.Series(data) if not isinstance(data, pd.Series) else data

            # Resample to daily and forward fill (FIXED: removed deprecated method parameter)
            daily_data = series.resample('D').sum().ffill()

            if len(daily_data.dropna()) < 30:
                raise ValueError("Insufficient data for ARIMA")

            # Check stationarity
            try:
                result = adfuller(daily_data.dropna())
                p_value = result[1]
            except Exception:
                p_value = 0.01

            # Try different differencing orders if needed
            if p_value > 0.05:
                for d in range(1, 3):
                    try:
                        model = ARIMA(daily_data, order=(order[0], d, order[2]))
                        model_fit = model.fit()
                        return model_fit, daily_data
                    except Exception:
                        continue

            model = ARIMA(daily_data, order=order)
            model_fit = model.fit()
            return model_fit, daily_data

        except Exception as e:
            st.warning(f"ARIMA training failed: {e}")
            return None, data

    def forecast_ensemble(self, data, steps=30):
        """Generate ensemble forecast using multiple methods"""
        forecasts = {}
        processed_data = None

        try:
            # ARIMA forecast
            if ARIMA_AVAILABLE:
                arima_model, processed_data = self.train_arima_enhanced(data)
                if arima_model and processed_data is not None:
                    try:
                        arima_forecast = arima_model.forecast(steps=steps)
                        arima_index = pd.date_range(
                            processed_data.index[-1] + timedelta(days=1),
                            periods=steps
                        )
                        forecasts['arima'] = pd.Series(arima_forecast, index=arima_index)
                    except Exception:
                        pass

            # Linear regression forecast
            linear_forecast = self.linear_forecast(data, steps)
            if not linear_forecast.empty:
                forecasts['linear'] = linear_forecast

            # Exponential smoothing
            es_forecast = self.exponential_smoothing_forecast(data, steps)
            if not es_forecast.empty:
                forecasts['exponential'] = es_forecast

            # Combine forecasts
            if forecasts:
                ref_idx = list(forecasts.values())[0].index
                ensemble_forecast = pd.Series(0.0, index=ref_idx)
                weights = {'arima': 0.5, 'linear': 0.3, 'exponential': 0.2}

                for model_name, forecast in forecasts.items():
                    try:
                        # FIXED: removed deprecated method parameter
                        aligned = forecast.reindex(ensemble_forecast.index).ffill().fillna(0)
                        weight = weights.get(model_name, 0.1)
                        ensemble_forecast += aligned * weight
                    except Exception:
                        continue

                return ensemble_forecast, forecasts, (processed_data or data)

        except Exception as e:
            st.error(f"Forecasting error: {e}")

        return pd.Series(dtype=float), {}, data

    def linear_forecast(self, data, steps):
        """Simple linear regression forecast"""
        try:
            if isinstance(data, pd.DataFrame):
                daily_data = (data['expense'] if 'expense' in data.columns
                              else data.iloc[:, 0]).resample('D').sum().fillna(0)
            else:
                daily_data = (data if isinstance(data, pd.Series)
                              else pd.Series(data)).resample('D').sum().fillna(0)

            if len(daily_data) == 0:
                return pd.Series(dtype=float)

            X = np.arange(len(daily_data)).reshape(-1, 1)
            y = daily_data.values

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(daily_data), len(daily_data) + steps).reshape(-1, 1)
            forecast = model.predict(future_X)

            return pd.Series(
                forecast,
                index=pd.date_range(daily_data.index[-1] + timedelta(days=1), periods=steps)
            )
        except Exception:
            return pd.Series(dtype=float)

    def exponential_smoothing_forecast(self, data, steps, alpha=0.3):
        """Exponential smoothing forecast"""
        try:
            if isinstance(data, pd.DataFrame):
                daily_data = (data['expense'] if 'expense' in data.columns
                              else data.iloc[:, 0]).resample('D').sum().fillna(0)
            else:
                daily_data = (data if isinstance(data, pd.Series)
                              else pd.Series(data)).resample('D').sum().fillna(0)

            if len(daily_data) == 0:
                return pd.Series(dtype=float)

            last_value = float(daily_data.iloc[-1])
            forecast_values = []

            for _ in range(steps):
                last_value = last_value * (1 + np.random.normal(0, 0.03))
                forecast_values.append(last_value)

            return pd.Series(
                forecast_values,
                index=pd.date_range(daily_data.index[-1] + timedelta(days=1), periods=steps)
            )
        except Exception:
            return pd.Series(dtype=float)

    def detect_anomalies_advanced(self, data):
        """Detect spending anomalies using multiple methods"""
        try:
            if isinstance(data, pd.DataFrame):
                series = data['expense'] if 'expense' in data.columns else data.iloc[:, 0]
            else:
                series = data if isinstance(data, pd.Series) else pd.Series(data)

            daily_data = series.resample('D').sum().fillna(0)

            if len(daily_data) < 20:
                return pd.Series(False, index=daily_data.index), {}

            # Isolation Forest
            try:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                iso_predictions = iso_forest.fit_predict(daily_data.values.reshape(-1, 1))
                iso_anomalies = pd.Series(iso_predictions == -1, index=daily_data.index)
            except Exception:
                iso_anomalies = pd.Series(False, index=daily_data.index)

            # Z-score method
            try:
                z_scores = np.abs((daily_data - daily_data.mean()) /
                                  (daily_data.std() if daily_data.std() != 0 else 1))
                z_anomalies = (z_scores > 2.5).fillna(False)
            except Exception:
                z_anomalies = pd.Series(False, index=daily_data.index)

            # Moving average method
            try:
                window = 7
                moving_avg = daily_data.rolling(window=window, center=True).mean()
                moving_std = daily_data.rolling(window=window, center=True).std().replace(0, np.nan)
                ma_anomalies = (np.abs(daily_data - moving_avg) > (2 * moving_std)).fillna(False)
            except Exception:
                ma_anomalies = pd.Series(False, index=daily_data.index)

            # Combine methods (at least 2 methods agree)
            combined_int = (iso_anomalies.astype(int) +
                            z_anomalies.astype(int) +
                            ma_anomalies.astype(int))
            combined_anomalies = combined_int >= 2

            anomaly_details = {
                'isolation_forest': int(iso_anomalies.sum()),
                'z_score': int(z_anomalies.sum()),
                'moving_average': int(ma_anomalies.sum()),
                'combined': int(combined_anomalies.sum())
            }

            return pd.Series(combined_anomalies, index=daily_data.index), anomaly_details

        except Exception as e:
            st.error(f"Anomaly detection error: {e}")
            return pd.Series(dtype=bool), {}

    def train_category_classifier_advanced(self, df):
        """Train Random Forest classifier for category prediction"""
        try:
            if len(df) < 50:
                return None, None, "Insufficient data (need at least 50 records)"

            df_clean = df.dropna(subset=['merchant', 'category'])

            if len(df_clean) < 30:
                return None, None, "Not enough labeled data (need at least 30 records)"

            features = []
            labels = []

            for _, row in df_clean.iterrows():
                merchant = str(row['merchant']).lower()
                amount = float(row['expense'])

                # Text features
                text_features = [
                    len(merchant),
                    len(merchant.split()),
                    sum(c.isdigit() for c in merchant),
                    sum(c.isalpha() for c in merchant)
                ]

                # Amount features
                mean_expense = df_clean['expense'].mean() if df_clean['expense'].mean() > 0 else 1.0
                amount_features = [
                    amount,
                    np.log1p(amount),
                    amount / mean_expense
                ]

                combined_features = text_features + amount_features
                features.append(combined_features)
                labels.append(row['category'])

            X = np.array(features)
            y = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            clf.fit(X_train_scaled, y_train)

            accuracy = clf.score(X_test_scaled, y_test)

            return clf, scaler, f"Model trained with {accuracy:.2%} accuracy"

        except Exception as e:
            return None, None, f"Training failed: {e}"


# ==========================================
# Financial Analysis
# ==========================================
class ComprehensiveFinancialAnalysis:
    """Comprehensive financial metrics and analysis"""

    def __init__(self):
        self.kpi_thresholds = {
            'savings_rate_ideal': 20,
            'dti_warning': 40,
            'dti_critical': 60,
            'runway_minimum': 3,
            'runway_ideal': 6,
            'emergency_fund_months': 6
        }

    def calculate_comprehensive_ratios(self, df, fixed_expenses, financial_goals):
        """Calculate all financial ratios and metrics"""
        try:
            data = df.copy()

            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception:
                    pass

            # Monthly expenses
            monthly_expenses = data['expense'].resample('M').sum()
            avg_monthly_variable = monthly_expenses.mean() if not monthly_expenses.empty else 0

            fixed_total = sum(fixed_expenses.values()) if fixed_expenses else 0
            total_monthly_expenses = fixed_total + avg_monthly_variable

            monthly_income = financial_goals.get("monthly_income", 0)
            current_savings = financial_goals.get("current_savings", 0)

            # Calculate ratios
            savings_rate = ((monthly_income - total_monthly_expenses) / monthly_income * 100) \
                if monthly_income > 0 else 0

            debt_to_income = (fixed_total / monthly_income * 100) \
                if monthly_income > 0 else 0

            fixed_vs_variable = (fixed_total / total_monthly_expenses * 100) \
                if total_monthly_expenses > 0 else 0

            financial_runway = (current_savings / total_monthly_expenses) \
                if total_monthly_expenses > 0 else 0

            expense_volatility = (monthly_expenses.std() / monthly_expenses.mean() * 100) \
                if len(monthly_expenses) > 1 and monthly_expenses.mean() > 0 else 0

            # Savings metrics
            savings_gap = financial_goals.get("savings_target", 0) - current_savings
            monthly_savings_needed = savings_gap / 12 if savings_gap > 0 else 0

            # Category analysis
            try:
                category_analysis = data.groupby('category')['expense'].agg(
                    ['sum', 'count', 'mean']
                ).to_dict()
            except Exception:
                category_analysis = {}

            # Spending patterns
            weekly_pattern = data.groupby(data.index.dayofweek)['expense'].mean().to_dict()
            monthly_trend = monthly_expenses.pct_change().mean() * 100 \
                if len(monthly_expenses) > 1 else 0

            # Health scores
            savings_health_score = min(
                savings_rate / self.kpi_thresholds['savings_rate_ideal'] * 100, 100
            )

            debt_health_score = max(
                0, 100 - (debt_to_income / self.kpi_thresholds['dti_critical'] * 100)
            )

            runway_health_score = min(
                financial_runway / self.kpi_thresholds['runway_ideal'] * 100, 100
            )

            volatility_score = max(0, 100 - min(expense_volatility, 100))

            # Overall health score
            weights = {'savings': 0.3, 'debt': 0.3, 'runway': 0.2, 'volatility': 0.2}
            overall_health_score = (
                    savings_health_score * weights['savings'] +
                    debt_health_score * weights['debt'] +
                    runway_health_score * weights['runway'] +
                    volatility_score * weights['volatility']
            )

            ratios = {
                "savings_rate": savings_rate,
                "debt_to_income": debt_to_income,
                "fixed_vs_variable_ratio": fixed_vs_variable,
                "monthly_burn_rate": total_monthly_expenses,
                "financial_runway": financial_runway,
                "expense_volatility": expense_volatility,
                "savings_gap": savings_gap,
                "monthly_savings_needed": monthly_savings_needed,
                "savings_progress": (current_savings / financial_goals.get("savings_target", 1) * 100)
                if financial_goals.get("savings_target", 0) > 0 else 0,
                "weekly_pattern": weekly_pattern,
                "monthly_trend": monthly_trend,
                "category_analysis": category_analysis,
                "savings_health_score": savings_health_score,
                "debt_health_score": debt_health_score,
                "runway_health_score": runway_health_score,
                "overall_health_score": overall_health_score
            }

            return ratios

        except Exception as e:
            st.error(f"Error calculating ratios: {e}")
            return {}

    def create_financial_report(self, df, fixed_expenses, financial_goals):
        """FIXED: Generate comprehensive financial health report"""
        try:
            ratios = self.calculate_comprehensive_ratios(df, fixed_expenses, financial_goals)

            if not ratios:
                return None

            # Summary
            summary = {
                'overall_health_score': ratios.get('overall_health_score', 0),
                'financial_runway_months': ratios.get('financial_runway', 0),
                'savings_rate_percent': ratios.get('savings_rate', 0),
                'debt_to_income_percent': ratios.get('debt_to_income', 0)
            }

            # Generate personalized advice
            advice = {
                'urgent': [],
                'important': [],
                'suggestions': [],
                'positive': []
            }

            # Urgent actions
            if ratios['financial_runway'] < self.kpi_thresholds['runway_minimum']:
                advice['urgent'].append(
                    f"⚠️ Critical: Only {ratios['financial_runway']:.1f} months runway. "
                    f"Build emergency fund immediately."
                )

            if ratios['debt_to_income'] > self.kpi_thresholds['dti_critical']:
                advice['urgent'].append(
                    f"⚠️ Debt-to-income ratio at {ratios['debt_to_income']:.1f}% (critical level). "
                    f"Consider debt consolidation."
                )

            # Important recommendations
            if ratios['savings_rate'] < 10:
                advice['important'].append(
                    f"Your savings rate is only {ratios['savings_rate']:.1f}%. "
                    f"Target at least 20% by cutting discretionary spending."
                )

            if ratios['expense_volatility'] > 50:
                advice['important'].append(
                    f"High spending volatility ({ratios['expense_volatility']:.1f}%). "
                    f"Create a consistent budget to stabilize expenses."
                )

            # Suggestions
            if ratios['savings_rate'] < 20 and ratios['savings_rate'] >= 10:
                advice['suggestions'].append(
                    f"Good progress at {ratios['savings_rate']:.1f}% savings rate. "
                    f"Aim for 20% by optimizing subscriptions and meal planning."
                )

            if ratios['monthly_savings_needed'] > 0:
                advice['suggestions'].append(
                    f"To reach your savings target, save RM {ratios['monthly_savings_needed']:.2f} "
                    f"more per month."
                )

            # Positive feedback
            if ratios['savings_rate'] >= 20:
                advice['positive'].append(
                    f"✅ Excellent {ratios['savings_rate']:.1f}% savings rate! "
                    f"You're on track for financial security."
                )

            if ratios['financial_runway'] >= self.kpi_thresholds['runway_ideal']:
                advice['positive'].append(
                    f"✅ Strong financial runway of {ratios['financial_runway']:.1f} months. "
                    f"Well prepared for emergencies."
                )

            if ratios['debt_to_income'] < self.kpi_thresholds['dti_warning']:
                advice['positive'].append(
                    f"✅ Healthy debt-to-income ratio at {ratios['debt_to_income']:.1f}%. "
                    f"Good debt management."
                )

            return {
                'summary': summary,
                'advice': advice,
                'ratios': ratios
            }

        except Exception as e:
            st.error(f"Error creating report: {e}")
            return None


# ==========================================
# Chatbot
# ==========================================
class EnhancedFinancialChatbot:
    """AI-powered financial advisor chatbot"""

    def __init__(self):
        self.financial_knowledge_base = self._initialize_knowledge_base()
        self.conversation_context = {}

    def _initialize_knowledge_base(self):
        """Initialize financial knowledge base"""
        return {
            "basic_concepts": {
                "savings_rate": "The percentage of your income that you save each month. A healthy rate is 20% or higher.",
                "debt_to_income": "Your monthly debt payments divided by your gross monthly income. Keep below 36%.",
                "emergency_fund": "3-6 months of living expenses set aside for unexpected financial emergencies.",
                "financial_runway": "How many months you can survive on your savings if you lost your income.",
                "compound_interest": "Interest earned on both the initial principal and accumulated interest.",
                "diversification": "Spreading investments across different assets to reduce risk.",
                "inflation": "The rate at which prices for goods and services rise over time."
            },
            "investment_advice": {
                "beginner": "Start with low-cost index funds, build emergency fund first, invest consistently.",
                "intermediate": "Diversify across stocks, bonds, real estate. Consider dollar-cost averaging.",
                "advanced": "Explore alternative investments, tax optimization strategies, portfolio rebalancing."
            },
            "budgeting_methods": {
                "50_30_20": "50% needs, 30% wants, 20% savings. A simple and effective budgeting rule.",
                "zero_based": "Every dollar has a job. Income minus expenses equals zero each month.",
                "envelope_system": "Cash-based budgeting using physical envelopes for categories."
            },
            "debt_strategies": {
                "avalanche": "Pay highest interest debts first to save money on interest.",
                "snowball": "Pay smallest debts first for psychological wins and momentum.",
                "consolidation": "Combine multiple debts into one with lower interest rate."
            }
        }

    def generate_contextual_response(self, user_input, financial_data=None, ratios=None):
        """Generate intelligent response based on user query"""
        if not user_input:
            return "How can I help with your finances today?"

        user_input_lc = user_input.lower()

        try:
            # Greetings
            if any(word in user_input_lc for word in ["hello", "hi", "hey", "greetings"]):
                return ("Hello! I'm your FinScope AI financial advisor. I can help with budgeting, "
                        "investments, savings strategies, debt management, and analyzing your financial health. "
                        "What would you like to know?")

            # Basic concepts
            for term, definition in self.financial_knowledge_base["basic_concepts"].items():
                if term.replace('_', ' ') in user_input_lc:
                    return f"**{term.replace('_', ' ').title()}**: {definition}"

            # Route to specific handlers
            if any(word in user_input_lc for word in ["budget", "budgeting", "spending plan"]):
                return self._handle_budgeting_query(user_input_lc)

            if any(word in user_input_lc for word in ["invest", "investment", "portfolio", "stock"]):
                return self._handle_investment_query(user_input_lc)

            if any(word in user_input_lc for word in ["debt", "loan", "credit", "borrow"]):
                return self._handle_debt_query(user_input_lc)

            if any(word in user_input_lc for word in ["save", "saving", "savings"]):
                return self._handle_savings_query(user_input_lc, ratios)

            if any(word in user_input_lc for word in ["health", "doing", "finances", "status"]):
                return self._handle_financial_health_query(ratios)

            return self._generate_fallback_response(user_input_lc)

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

    def _handle_budgeting_query(self, user_input):
        """Handle budgeting questions"""
        if "50/30/20" in user_input or "50 30 20" in user_input:
            return self.financial_knowledge_base["budgeting_methods"]["50_30_20"]
        elif "zero" in user_input:
            return self.financial_knowledge_base["budgeting_methods"]["zero_based"]
        elif "envelope" in user_input:
            return self.financial_knowledge_base["budgeting_methods"]["envelope_system"]
        else:
            return ("For budgeting, I recommend: 1) Track all expenses, 2) Use the 50/30/20 rule "
                    "as a starting point, 3) Review and adjust monthly, 4) Automate savings. "
                    "Would you like me to explain any specific budgeting method?")

    def _handle_investment_query(self, user_input):
        """Handle investment questions"""
        if "beginner" in user_input:
            return self.financial_knowledge_base["investment_advice"]["beginner"]
        elif "advanced" in user_input:
            return self.financial_knowledge_base["investment_advice"]["advanced"]
        else:
            return self.financial_knowledge_base["investment_advice"]["intermediate"]

    def _handle_debt_query(self, user_input):
        """Handle debt management questions"""
        if "avalanche" in user_input:
            return self.financial_knowledge_base["debt_strategies"]["avalanche"]
        elif "snowball" in user_input:
            return self.financial_knowledge_base["debt_strategies"]["snowball"]
        elif "consolidat" in user_input:
            return self.financial_knowledge_base["debt_strategies"]["consolidation"]
        else:
            return ("For debt management: 1) List all debts, 2) Choose avalanche (save money) "
                    "or snowball (build momentum) method, 3) Make minimum payments on all, "
                    "4) Extra payments on target debt. Need specific advice?")

    def _handle_savings_query(self, user_input, ratios):
        """Handle savings questions with personalized advice"""
        if ratios and "savings_rate" in ratios:
            savings_rate = ratios["savings_rate"]
            if savings_rate < 10:
                return (f"Your current savings rate is {savings_rate:.1f}%, which is below the "
                        f"recommended 20%. Consider: 1) Automating savings transfers, "
                        f"2) Reducing discretionary spending, 3) Increasing income streams.")
            elif savings_rate < 20:
                return (f"Your savings rate is {savings_rate:.1f}%. Good progress! To reach 20%: "
                        f"1) Review subscription services, 2) Meal planning to reduce food costs, "
                        f"3) Consider higher-yield savings accounts.")
            else:
                return (f"Excellent! Your {savings_rate:.1f}% savings rate is above the 20% target. "
                        f"Consider: 1) Investing surplus, 2) Building emergency fund to 6 months, "
                        f"3) Exploring retirement accounts.")

        return ("For savings: 1) Pay yourself first (automate savings), 2) Build 3-6 month "
                "emergency fund, 3) Set specific savings goals, 4) Review and increase savings rate annually.")

    def _handle_financial_health_query(self, ratios):
        """Provide financial health assessment"""
        if not ratios:
            return ("I need your financial data to provide a personalized assessment. "
                    "Please upload your expenses or use the data entry features first.")

        health_score = ratios.get("overall_health_score", 0)
        runway = ratios.get("financial_runway", 0)
        savings_rate = ratios.get("savings_rate", 0)

        if health_score >= 80:
            assessment = "🌟 Excellent financial health! "
        elif health_score >= 60:
            assessment = "✅ Good financial health. "
        elif health_score >= 40:
            assessment = "⚠️ Moderate financial health. "
        else:
            assessment = "🚨 Needs improvement. "

        assessment += (f"Your overall score is {health_score:.0f}/100. "
                       f"You have {runway:.1f} months of financial runway and "
                       f"save {savings_rate:.1f}% of your income.")

        return assessment

    def _generate_fallback_response(self, user_input):
        """Generate fallback response for unmatched queries"""
        financial_topics = [
            "savings", "investment", "budget", "debt", "retirement",
            "emergency fund", "credit score", "mortgage", "insurance", "tax"
        ]

        detected_topics = [topic for topic in financial_topics if topic in user_input]

        if detected_topics:
            return (f"I can help with {', '.join(detected_topics)}! "
                    f"Could you be more specific about what you'd like to know?")

        return ("I specialize in personal finance topics like budgeting, saving, investing, "
                "debt management, and financial planning. What specific area would you like help with?")


# ==========================================
# UI Components
# ==========================================
def render_comprehensive_dashboard(financial_analysis):
    """Render main dashboard"""
    st.header("📊 Comprehensive Financial Dashboard")

    # Calculate ratios
    ratios = financial_analysis.calculate_comprehensive_ratios(
        st.session_state.expenses_df,
        st.session_state.fixed_expenses,
        st.session_state.financial_goals
    )

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = ratios.get('overall_health_score', 0)
        delta_color = "normal" if score >= 70 else "inverse"
        st.metric("Overall Health Score", f"{score:.0f}/100",
                  delta=None, delta_color=delta_color)

    with col2:
        savings_rate = ratios.get('savings_rate', 0)
        st.metric("Savings Rate", f"{savings_rate:.1f}%",
                  delta="Target: 20%")

    with col3:
        runway = ratios.get('financial_runway', 0)
        st.metric("Financial Runway", f"{runway:.1f} months",
                  delta="Target: 6 months")

    with col4:
        dti = ratios.get('debt_to_income', 0)
        st.metric("Debt-to-Income", f"{dti:.1f}%",
                  delta="Keep < 36%")

    # Recent expenses
    st.subheader("📝 Recent Expenses")
    recent_df = st.session_state.expenses_df.tail(10).copy()
    recent_df = recent_df[['expense', 'category', 'merchant', 'notes']]
    st.dataframe(recent_df, use_container_width=True)

    # Spending by category
    st.subheader("🎯 Spending by Category")
    try:
        category_totals = st.session_state.expenses_df.groupby('category')['expense'].sum()
        if not category_totals.empty:
            fig = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                title="Expense Distribution",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data to display.")
    except Exception as e:
        st.error(f"Error displaying categories: {e}")

    # Monthly trend
    st.subheader("📈 Monthly Spending Trend")
    try:
        monthly_data = st.session_state.expenses_df['expense'].resample('M').sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data.values,
            mode='lines+markers',
            name='Monthly Expenses',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            title="Monthly Expenses Over Time",
            xaxis_title="Month",
            yaxis_title="Amount (RM)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying trend: {e}")


def render_advanced_ocr_scanner(ocr_engine):
    """Render OCR receipt scanner"""
    st.header("🧾 Advanced OCR Receipt Scanner")

    if not ocr_engine or not getattr(ocr_engine, "available", False):
        st.error("❌ OCR features require pytesseract and opencv-python libraries.")
        st.info("Install them with: `pip install pytesseract opencv-python`")
        return

    uploaded_file = st.file_uploader(
        "Upload Receipt Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your receipt"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Receipt", use_column_width=True)

        with col2:
            if st.button("🔍 Extract Information", type="primary"):
                with st.spinner("Processing receipt..."):
                    processed_image, extracted_text, err = ocr_engine.preprocess_image_advanced(image)

                    if err:
                        st.error(f"Error: {err}")
                        return

                    info = ocr_engine.extract_detailed_info(extracted_text)

                st.subheader("📋 Extracted Information")

                if info.get("amount"):
                    st.success(f"💰 Amount: RM {info['amount']:.2f}")
                if info.get("merchant"):
                    st.info(f"🏪 Merchant: {info['merchant']}")
                if info.get("category"):
                    st.info(f"🏷️ Category: {info['category']}")
                if info.get("date"):
                    st.info(f"📅 Date: {info['date']}")

                st.metric("Confidence Score", f"{info.get('confidence', 0):.0%}")

                # Add expense form
                if info.get("amount"):
                    with st.form("add_ocr_expense"):
                        st.subheader("Confirm & Add Expense")

                        amount = st.number_input(
                            "Amount (RM)",
                            value=float(info['amount']),
                            min_value=0.0,
                            step=0.01
                        )

                        category = st.selectbox(
                            "Category",
                            ["Food", "Transport", "Shopping", "Bills", "Entertainment",
                             "Groceries", "Healthcare", "Utilities", "Other"],
                            index=0 if not info.get('category') else
                            ["Food", "Transport", "Shopping", "Bills", "Entertainment",
                             "Groceries", "Healthcare", "Utilities", "Other"].index(info['category'])
                            if info['category'] in ["Food", "Transport", "Shopping", "Bills",
                                                    "Entertainment", "Groceries", "Healthcare",
                                                    "Utilities", "Other"] else 0
                        )

                        merchant = st.text_input(
                            "Merchant",
                            value=info.get('merchant') or ""
                        )

                        expense_date = st.date_input(
                            "Date",
                            value=pd.to_datetime(info['date']) if info.get('date')
                            else datetime.now()
                        )

                        if st.form_submit_button("✅ Add Expense"):
                            new_expense = pd.DataFrame({
                                "date": [pd.Timestamp(expense_date)],
                                "expense": [amount],
                                "category": [category],
                                "merchant": [merchant or "OCR Entry"],
                                "type": ["variable"],
                                "notes": [f"OCR: {extracted_text[:100]}"],
                                "recurring_id": [None]
                            }).set_index("date")
                            new_expense.index = pd.to_datetime(new_expense.index)

                            st.session_state.expenses_df = pd.concat([
                                st.session_state.expenses_df,
                                new_expense
                            ])

                            st.success("✅ Expense added successfully!")
                            st.rerun()


def render_voice_assistant():
    """Render voice assistant interface"""
    st.header("🎤 Smart Voice Assistant")

    st.info("📱 **Browser-based Recording** - Works on any device with a microphone!")

    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.error("❌ Voice features require streamlit-webrtc and pydub libraries.")
        st.info("Install them with: `pip install streamlit-webrtc pydub`")
        st.warning("Also ensure ffmpeg is installed on your system for audio processing.")
        return

    # Instructions
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
        1. **Click the microphone button** below to start recording
        2. **Speak clearly** about your expense (e.g., "45 ringgit for groceries at Tesco")
        3. **Click Stop** when finished
        4. **Click Transcribe** to process your recording
        5. **Confirm and add** the expense

        **Example phrases:**
        - "Grab ride 18 ringgit"
        - "Lunch at restaurant 25 dollars"
        - "Shopping at mall 150 ringgit"
        """)

    # Web RTC recorder
    client_settings = ClientSettings(
        media_stream_constraints={"audio": True, "video": False}
    )

    webrtc_ctx = webrtc_streamer(
        key="voice-recorder",
        client_settings=client_settings,
        mode="SENDONLY",
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False}
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.success("🎙️ Recording... speak now!")

    # Transcribe button
    if st.button("🔄 Transcribe Last Recording", type="primary"):
        st.info("⏳ Processing audio...")

        # Note: streamlit-webrtc doesn't easily expose recorded audio
        # This is a limitation - in production, you'd need custom JavaScript
        st.warning("⚠️ Audio transcription requires additional setup with custom JavaScript or server-side recording.")
        st.info("💡 For now, use the manual entry below as an alternative.")

    # Manual voice entry alternative
    st.markdown("---")
    st.subheader("💬 Or Enter Expense Manually")

    with st.form("manual_voice_entry"):
        voice_text = st.text_input(
            "Describe your expense",
            placeholder="e.g., '45 ringgit for groceries' or 'Grab ride 18 ringgit'",
            help="Enter what you would say to the voice assistant"
        )

        if st.form_submit_button("Parse & Add"):
            if voice_text:
                parsed = parse_transcript_simple(voice_text)

                if parsed.get("amount"):
                    new_expense = pd.DataFrame({
                        "date": [pd.Timestamp(datetime.now())],
                        "expense": [parsed["amount"]],
                        "category": [parsed.get("category", "Other")],
                        "merchant": [parsed.get("merchant") or "Voice Entry"],
                        "type": ["variable"],
                        "notes": [voice_text],
                        "recurring_id": [None]
                    }).set_index("date")
                    new_expense.index = pd.to_datetime(new_expense.index)

                    st.session_state.expenses_df = pd.concat([
                        st.session_state.expenses_df,
                        new_expense
                    ])

                    st.success(f"✅ Added: RM {parsed['amount']:.2f} - {parsed['category']}")
                    st.rerun()
                else:
                    st.error("❌ Could not extract amount. Try phrases like '45 ringgit for food'")
            else:
                st.warning("Please enter an expense description")


def render_data_management():
    """Render data import/export and manual entry"""
    st.header("📁 Data Import & Management")

    tab1, tab2, tab3 = st.tabs(["➕ Add Expense", "📤 Export Data", "📊 View All Data"])

    with tab1:
        st.subheader("Add Manual Expense")

        with st.form("manual_expense"):
            col1, col2 = st.columns(2)

            with col1:
                expense_date = st.date_input("Date", datetime.now())
                amount = st.number_input("Amount (RM)", min_value=0.0, step=0.01)
                category = st.selectbox(
                    "Category",
                    ["Food", "Transport", "Shopping", "Bills", "Entertainment",
                     "Groceries", "Healthcare", "Utilities", "Other"]
                )

            with col2:
                merchant = st.text_input("Merchant")
                expense_type = st.selectbox("Type", ["variable", "fixed"])
                notes = st.text_area("Notes")

            if st.form_submit_button("Add Expense", type="primary"):
                new_expense = pd.DataFrame({
                    "date": [pd.Timestamp(expense_date)],
                    "expense": [amount],
                    "category": [category],
                    "merchant": [merchant or "Manual Entry"],
                    "type": [expense_type],
                    "notes": [notes],
                    "recurring_id": [None]
                }).set_index("date")
                new_expense.index = pd.to_datetime(new_expense.index)

                st.session_state.expenses_df = pd.concat([
                    st.session_state.expenses_df,
                    new_expense
                ])

                st.success("✅ Expense added successfully!")
                st.rerun()

    with tab2:
        st.subheader("Export Your Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export to CSV"):
                csv = st.session_state.expenses_df.reset_index().to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📥 Export to Excel"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state.expenses_df.reset_index().to_excel(
                        writer,
                        sheet_name='Expenses',
                        index=False
                    )

                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with tab3:
        st.subheader("All Expenses")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            categories = ["All"] + list(st.session_state.expenses_df['category'].unique())
            selected_category = st.selectbox("Filter by Category", categories)

        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(
                    st.session_state.expenses_df.index.min().date(),
                    st.session_state.expenses_df.index.max().date()
                )
            )

        # Filter data
        filtered_df = st.session_state.expenses_df.copy()

        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]

        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df.index.date >= date_range[0]) &
                (filtered_df.index.date <= date_range[1])
                ]

        st.dataframe(
            filtered_df[['expense', 'category', 'merchant', 'type', 'notes']],
            use_container_width=True
        )

        st.metric("Total Expenses", f"RM {filtered_df['expense'].sum():.2f}")


def render_advanced_analytics(ml_models, financial_analysis):
    """Render advanced analytics and ML features"""
    st.header("📈 Advanced Analytics & Machine Learning")

    tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Detection", "📊 Forecasting", "🤖 ML Models"])

    with tab1:
        st.subheader("Spending Anomaly Detection")
        st.info("Detect unusual spending patterns using machine learning algorithms.")

        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Analyzing spending patterns..."):
                anomalies, details = ml_models.detect_anomalies_advanced(
                    st.session_state.expenses_df['expense']
                )

            if isinstance(anomalies, pd.Series) and anomalies.any():
                st.warning(f"⚠️ Detected {int(anomalies.sum())} anomalous spending patterns")

                # Show detection method breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Isolation Forest", details.get('isolation_forest', 0))
                with col2:
                    st.metric("Z-Score Method", details.get('z_score', 0))
                with col3:
                    st.metric("Moving Average", details.get('moving_average', 0))

                # Show anomalous transactions
                anomalous_dates = anomalies[anomalies].index
                anomalous_data = st.session_state.expenses_df.loc[anomalous_dates]

                if not anomalous_data.empty:
                    st.subheader("📋 Anomalous Transactions")
                    st.dataframe(
                        anomalous_data[['expense', 'category', 'merchant', 'notes']],
                        use_container_width=True
                    )

                    # Visualize anomalies
                    daily_expenses = st.session_state.expenses_df['expense'].resample('D').sum()
                    fig = go.Figure()

                    # Normal spending
                    fig.add_trace(go.Scatter(
                        x=daily_expenses.index,
                        y=daily_expenses.values,
                        mode='lines',
                        name='Daily Expenses',
                        line=dict(color='lightblue', width=1)
                    ))

                    # Anomalies
                    anomalous_expenses = daily_expenses[anomalies]
                    fig.add_trace(go.Scatter(
                        x=anomalous_expenses.index,
                        y=anomalous_expenses.values,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x')
                    ))

                    fig.update_layout(
                        title="Spending Anomalies Visualization",
                        xaxis_title="Date",
                        yaxis_title="Amount (RM)",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No significant anomalies detected in your spending patterns")

    with tab2:
        st.subheader("Expense Forecasting")
        st.info("Predict future expenses using ensemble machine learning models.")

        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)

        if st.button("📊 Generate Forecast", type="primary"):
            with st.spinner("Training models and generating forecast..."):
                forecast, models, processed_data = ml_models.forecast_ensemble(
                    st.session_state.expenses_df['expense'],
                    steps=forecast_days
                )

            if isinstance(forecast, pd.Series) and not forecast.empty:
                # Display forecast
                st.success(f"✅ Generated {forecast_days}-day forecast")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Daily Forecast", f"RM {forecast.mean():.2f}")
                with col2:
                    st.metric("Total Forecast", f"RM {forecast.sum():.2f}")
                with col3:
                    current_avg = st.session_state.expenses_df['expense'].resample('D').sum().mean()
                    change = ((forecast.mean() - current_avg) / current_avg * 100) if current_avg > 0 else 0
                    st.metric("Change vs Current", f"{change:+.1f}%")

                # Visualization
                fig = go.Figure()

                # Historical data
                historical = st.session_state.expenses_df['expense'].resample('D').sum().tail(60)
                fig.add_trace(go.Scatter(
                    x=historical.index,
                    y=historical.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))

                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))

                # Confidence interval (simple ±15%)
                upper_bound = forecast * 1.15
                lower_bound = forecast * 0.85

                fig.add_trace(go.Scatter(
                    x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    showlegend=True,
                    name='Confidence Interval'
                ))

                fig.update_layout(
                    title="Expense Forecast with Confidence Interval",
                    xaxis_title="Date",
                    yaxis_title="Amount (RM)",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Model details
                with st.expander("📊 Model Details"):
                    st.write(f"**Ensemble models used:** {', '.join(models.keys())}")
                    st.write("**Weights:** ARIMA (50%), Linear Regression (30%), Exponential Smoothing (20%)")
            else:
                st.error("❌ Forecasting failed. Ensure you have sufficient historical data (30+ days).")

    with tab3:
        st.subheader("Machine Learning Model Training")
        st.info("Train custom ML models for category prediction and analysis.")

        if st.button("🤖 Train Category Classifier", type="primary"):
            with st.spinner("Training Random Forest classifier..."):
                model, scaler, message = ml_models.train_category_classifier_advanced(
                    st.session_state.expenses_df
                )

            if model:
                st.session_state.trained_models["category_classifier"] = model
                st.session_state.trained_models["category_scaler"] = scaler
                st.success(f"✅ {message}")

                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    features = ['Merchant Length', 'Word Count', 'Digits', 'Letters',
                                'Amount', 'Log Amount', 'Relative Amount']

                    fig = px.bar(
                        x=importances,
                        y=features,
                        orientation='h',
                        title="Feature Importance",
                        labels={'x': 'Importance', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ {message}")

        # Model status
        st.markdown("---")
        st.subheader("📊 Model Status")

        col1, col2 = st.columns(2)
        with col1:
            classifier_status = "✅ Trained" if st.session_state.trained_models[
                "category_classifier"] else "❌ Not Trained"
            st.write(f"**Category Classifier:** {classifier_status}")

        with col2:
            if st.button("🗑️ Clear Models"):
                st.session_state.trained_models = {
                    "category_classifier": None,
                    "category_scaler": None,
                    "anomaly_detector": None,
                    "forecast_model": None
                }
                st.success("Models cleared")
                st.rerun()


def render_ai_advisor(chatbot, financial_analysis):
    """Render AI financial advisor chat interface"""
    st.header("🤖 AI Financial Advisor")

    # Calculate current ratios
    ratios = financial_analysis.calculate_comprehensive_ratios(
        st.session_state.expenses_df,
        st.session_state.fixed_expenses,
        st.session_state.financial_goals
    )

    # Chat interface
    st.subheader("💬 Chat with Your Financial Advisor")

    # Quick action buttons
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💰 Savings Tips"):
            user_input = "How can I save more money?"
    with col2:
        if st.button("📊 My Status"):
            user_input = "How am I doing financially?"
    with col3:
        if st.button("💳 Debt Help"):
            user_input = "How should I manage my debt?"
    with col4:
        if st.button("📈 Investment"):
            user_input = "How should I start investing?"

    # User input
    user_input = st.text_input(
        "Ask me anything about personal finance:",
        placeholder="e.g., How can I improve my savings rate?",
        key="chat_input"
    )

    if user_input:
        response = chatbot.generate_contextual_response(
            user_input,
            st.session_state.expenses_df,
            ratios
        )

        st.markdown(f"**💡 Advisor:** {response}")

        # Save to history
        st.session_state.chat_history.append({
            "user": user_input,
            "advisor": response,
            "timestamp": datetime.now()
        })

    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Recent Conversations")

        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.container():
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**Advisor:** {chat['advisor']}")
                st.caption(f"_{chat.get('timestamp', 'Unknown time')}_")
                st.markdown("---")

        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()


def render_financial_health_report(financial_analysis):
    """Render comprehensive financial health report"""
    st.header("📋 Comprehensive Financial Health Report")

    if st.button("📊 Generate Full Report", type="primary"):
        with st.spinner("Analyzing your financial health..."):
            report = financial_analysis.create_financial_report(
                st.session_state.expenses_df,
                st.session_state.fixed_expenses,
                st.session_state.financial_goals
            )

        if report:
            # Executive Summary
            st.subheader("📊 Executive Summary")
            summary = report['summary']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score = summary['overall_health_score']
                color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                st.metric(f"{color} Overall Score", f"{score:.0f}/100")
            with col2:
                st.metric("Financial Runway", f"{summary['financial_runway_months']:.1f} months")
            with col3:
                st.metric("Savings Rate", f"{summary['savings_rate_percent']:.1f}%")
            with col4:
                st.metric("DTI Ratio", f"{summary['debt_to_income_percent']:.1f}%")

            # Advice sections
            advice = report['advice']

            if advice.get('urgent'):
                st.error("## 🚨 Urgent Actions Needed")
                for item in advice['urgent']:
                    st.markdown(f"- {item}")

            if advice.get('important'):
                st.warning("## 📋 Important Recommendations")
                for item in advice['important']:
                    st.markdown(f"- {item}")

            if advice.get('suggestions'):
                st.info("## 💡 Suggestions for Improvement")
                for item in advice['suggestions']:
                    st.markdown(f"- {item}")

            if advice.get('positive'):
                st.success("## ✅ What You're Doing Well")
                for item in advice['positive']:
                    st.markdown(f"- {item}")

            # Detailed metrics
            st.markdown("---")
            st.subheader("📈 Detailed Financial Metrics")

            ratios = report['ratios']

            metrics_data = {
                "Metric": [
                    "Savings Rate",
                    "Debt-to-Income",
                    "Financial Runway",
                    "Expense Volatility",
                    "Monthly Burn Rate",
                    "Savings Progress"
                ],
                "Value": [
                    f"{ratios['savings_rate']:.1f}%",
                    f"{ratios['debt_to_income']:.1f}%",
                    f"{ratios['financial_runway']:.1f} months",
                    f"{ratios['expense_volatility']:.1f}%",
                    f"RM {ratios['monthly_burn_rate']:.2f}",
                    f"{ratios['savings_progress']:.1f}%"
                ],
                "Target/Ideal": [
                    "≥ 20%",
                    "< 36%",
                    "≥ 6 months",
                    "< 30%",
                    "-",
                    "100%"
                ]
            }

            st.table(pd.DataFrame(metrics_data))

            # Export report
            st.markdown("---")
            if st.button("💾 Export Report as PDF"):
                st.info("PDF export feature coming soon! For now, use your browser's print function.")
        else:
            st.error("Failed to generate report. Please check your data.")


def render_system_configuration(ml_models):
    """Render system settings and configuration"""
    st.header("⚙️ System Configuration")

    tab1, tab2, tab3 = st.tabs(["🎨 Preferences", "💰 Financial Goals", "🔧 Advanced"])

    with tab1:
        st.subheader("User Preferences")

        col1, col2 = st.columns(2)

        with col1:
            currency = st.selectbox(
                "Currency",
                ["RM", "USD", "EUR", "SGD", "GBP"],
                index=["RM", "USD", "EUR", "SGD", "GBP"].index(
                    st.session_state.user_preferences.get("currency", "RM")
                )
            )

            language = st.selectbox(
                "Language",
                ["English", "Malay", "Chinese"],
                index=0
            )

        with col2:
            auto_categorize = st.checkbox(
                "Auto-categorize expenses",
                value=st.session_state.user_preferences.get("auto_categorize", True)
            )

            alert_threshold = st.number_input(
                "High Spending Alert Threshold (RM)",
                value=float(st.session_state.user_preferences.get("alert_threshold", 500)),
                min_value=0.0,
                step=50.0
            )

        if st.button("💾 Save Preferences", type="primary"):
            st.session_state.user_preferences.update({
                "currency": currency,
                "language": language,
                "auto_categorize": auto_categorize,
                "alert_threshold": alert_threshold
            })
            st.success("✅ Preferences saved!")

    with tab2:
        st.subheader("Financial Goals")

        with st.form("update_goals"):
            col1, col2 = st.columns(2)

            with col1:
                monthly_income = st.number_input(
                    "Monthly Income (RM)",
                    value=float(st.session_state.financial_goals.get("monthly_income", 7500)),
                    min_value=0.0,
                    step=100.0
                )

                current_savings = st.number_input(
                    "Current Savings (RM)",
                    value=float(st.session_state.financial_goals.get("current_savings", 8000)),
                    min_value=0.0,
                    step=100.0
                )

                savings_target = st.number_input(
                    "Savings Target (RM)",
                    value=float(st.session_state.financial_goals.get("savings_target", 25000)),
                    min_value=0.0,
                    step=1000.0
                )

            with col2:
                emergency_fund_target = st.number_input(
                    "Emergency Fund Target (RM)",
                    value=float(st.session_state.financial_goals.get("emergency_fund_target", 15000)),
                    min_value=0.0,
                    step=1000.0
                )

                investment_target = st.number_input(
                    "Investment Target (RM)",
                    value=float(st.session_state.financial_goals.get("investment_target", 50000)),
                    min_value=0.0,
                    step=1000.0
                )

            if st.form_submit_button("💾 Update Goals", type="primary"):
                st.session_state.financial_goals.update({
                    "monthly_income": monthly_income,
                    "current_savings": current_savings,
                    "savings_target": savings_target,
                    "emergency_fund_target": emergency_fund_target,
                    "investment_target": investment_target
                })
                st.success("✅ Financial goals updated!")
                st.rerun()

    with tab3:
        st.subheader("Advanced Settings")

        # Data management
        st.markdown("### 📊 Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset to Sample Data", type="secondary"):
                if st.checkbox("Confirm reset?"):
                    # Re-initialize with sample data
                    del st.session_state.expenses_df
                    initialize_session_state()
                    st.success("✅ Reset to sample data")
                    st.rerun()

        with col2:
            st.metric("Total Records", len(st.session_state.expenses_df))

        # Model management
        st.markdown("---")
        st.markdown("### 🤖 Model Management")

        if st.button("🗑️ Clear All Trained Models"):
            st.session_state.trained_models = {
                "category_classifier": None,
                "category_scaler": None,
                "anomaly_detector": None,
                "forecast_model": None
            }
            st.success("✅ All models cleared")

        # System info
        st.markdown("---")
        st.markdown("### ℹ️ System Information")

        st.write(f"**Tesseract OCR:** {'✅ Available' if TESSERACT_AVAILABLE else '❌ Not Available'}")
        st.write(f"**OpenCV:** {'✅ Available' if CV2_AVAILABLE else '❌ Not Available'}")
        st.write(f"**ARIMA Models:** {'✅ Available' if ARIMA_AVAILABLE else '❌ Not Available'}")
        st.write(f"**Voice Recognition:** {'✅ Available' if VOICE_AVAILABLE else '❌ Not Available'}")
        st.write(f"**WebRTC Recording:** {'✅ Available' if STREAMLIT_WEBRTC_AVAILABLE else '❌ Not Available'}")


# ==========================================
# Main Application
# ==========================================
def main():
    """Main application entry point"""

    # Initialize components
    ml_models = FinancialMLModels()
    financial_analysis = ComprehensiveFinancialAnalysis()
    chatbot = EnhancedFinancialChatbot()
    ocr_engine = AdvancedOCREngine() if TESSERACT_AVAILABLE and CV2_AVAILABLE else None

    # Header
    st.title("💰 FinScope AI - Ultimate Financial Intelligence Platform")
    st.markdown("### Your Complete AI-Powered Financial Management Solution")

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Module",
        [
            "📊 Dashboard",
            "🧾 OCR Scanner",
            "🎤 Voice Assistant",
            "📁 Data Management",
            "📈 Advanced Analytics",
            "🤖 AI Advisor",
            "📋 Health Report",
            "⚙️ Configuration"
        ]
    )

    # Quick stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Quick Snapshot")

    ratios = financial_analysis.calculate_comprehensive_ratios(
        st.session_state.expenses_df,
        st.session_state.fixed_expenses,
        st.session_state.financial_goals
    )

    if ratios:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Health", f"{ratios.get('overall_health_score', 0):.0f}")
            st.metric("Savings", f"{ratios.get('savings_rate', 0):.1f}%")
        with col2:
            st.metric("Runway", f"{ratios.get('financial_runway', 0):.1f}m")
            st.metric("DTI", f"{ratios.get('debt_to_income', 0):.1f}%")

    # Route to selected module
    if app_mode == "📊 Dashboard":
        render_comprehensive_dashboard(financial_analysis)

    elif app_mode == "🧾 OCR Scanner":
        render_advanced_ocr_scanner(ocr_engine)

    elif app_mode == "🎤 Voice Assistant":
        render_voice_assistant()

    elif app_mode == "📁 Data Management":
        render_data_management()

    elif app_mode == "📈 Advanced Analytics":
        render_advanced_analytics(ml_models, financial_analysis)

    elif app_mode == "🤖 AI Advisor":
        render_ai_advisor(chatbot, financial_analysis)

    elif app_mode == "📋 Health Report":
        render_financial_health_report(financial_analysis)

    elif app_mode == "⚙️ Configuration":
        render_system_configuration(ml_models)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("FinScope AI v2.0 | Enhanced & Fixed")


if __name__ == "__main__":
    main()
