import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Transaction Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .fraud-alert {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .safe-alert {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .model-card {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionSystem:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.X_test = None
        self.y_test = None
        
    def load_and_preprocess_data(self, file_path="AIML Dataset3.csv"):
        """Load and preprocess the dataset"""
        try:
            if not os.path.exists(file_path):
                st.error(f"Dataset file not found: {file_path}")
                return None
            
            # Load data
            data = pd.read_csv(file_path)
            st.success(f"Dataset loaded successfully: {len(data):,} transactions")
            
            # Display basic info
            st.write(f"Dataset shape: {data.shape}")
            st.write("Columns:", list(data.columns))
            
            return data
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None
    
    def prepare_features(self, data):
        """Prepare features for model training"""
        # Drop irrelevant columns
        X = data.drop(columns=["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"])
        y = data["isFraud"]
        
        # Encode categorical column "type" using one-hot encoding
        X = pd.get_dummies(X, columns=["type"], drop_first=True)
        
        # Store feature columns for later use
        self.feature_columns = X.columns
        
        return X, y
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store model
        self.models['logistic_regression'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'name': 'Logistic Regression'
        }
        
        return model, y_pred, y_prob
    
    def train_isolation_forest(self, X_train, X_test, y_train, y_test):
        """Train Isolation Forest model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate contamination rate
        contamination_rate = y_train.mean()
        
        # Train model
        model = IsolationForest(contamination=contamination_rate, random_state=42)
        model.fit(X_train_scaled)
        
        # Predictions
        y_pred_if = model.predict(X_test_scaled)
        # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0 (not fraud)
        y_pred = [1 if p == -1 else 0 for p in y_pred_if]
        # Get anomaly scores for ROC-AUC
        y_scores = -model.decision_function(X_test_scaled)
        
        # Store model
        self.models['isolation_forest'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_scores,
            'name': 'Isolation Forest'
        }
        
        return model, y_pred, y_scores
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        # Scale features (optional for RF, but keeping for consistency)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store model
        self.models['random_forest'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'name': 'Random Forest'
        }
        
        return model, y_pred, y_prob
    
    def train_all_models(self, data):
        """Train all three models"""
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        progress_bar = st.progress(0)
        
        # Train Logistic Regression
        st.write("Training Logistic Regression...")
        self.train_logistic_regression(X_train, X_test, y_train, y_test)
        progress_bar.progress(33)
        
        # Train Isolation Forest
        st.write("Training Isolation Forest...")
        self.train_isolation_forest(X_train, X_test, y_train, y_test)
        progress_bar.progress(66)
        
        # Train Random Forest
        st.write("Training Random Forest...")
        self.train_random_forest(X_train, X_test, y_train, y_test)
        progress_bar.progress(100)
        
        st.success("All models trained successfully!")
        
        return X_train, X_test, y_train, y_test
    
    def predict_single_transaction(self, model_name, transaction_data):
        """Predict fraud for a single transaction"""
        if model_name not in self.models:
            return None, None
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Prepare input data
        input_df = pd.DataFrame([transaction_data])
        
        # One-hot encode the type column
        input_df = pd.get_dummies(input_df, columns=["type"], drop_first=True)
        
        # Align columns with training data
        input_df = input_df.reindex(columns=self.feature_columns, fill_value=0)
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        if model_name == 'isolation_forest':
            prediction = model.predict(input_scaled)[0]
            score = -model.decision_function(input_scaled)[0]
            is_fraud = prediction == -1
            probability = score
        else:
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0, 1]
            is_fraud = prediction == 1
        
        return is_fraud, probability
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        try:
            # Save individual models
            for model_name, model_info in self.models.items():
                joblib.dump(model_info['model'], f"{model_name}_model.pkl")
            
            # Save scaler and feature columns
            joblib.dump(self.scaler, "scaler.pkl")
            joblib.dump(self.feature_columns, "feature_columns.pkl")
            
            st.success("All models and preprocessing objects saved successfully!")
            
        except Exception as e:
            st.error(f"Error saving models: {str(e)}")

# Initialize the system
@st.cache_resource
def get_fraud_system():
    return FraudDetectionSystem()

def main():
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection System</h1>', unsafe_allow_html=True)
    
    fraud_system = get_fraud_system()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Dataset configuration
    st.sidebar.markdown("### 📁 Dataset Configuration")
    dataset_path = st.sidebar.text_input(
        "Dataset Path:", 
        value="AIML Dataset3.csv",
        help="Path to your CSV dataset file"
    )
    
    # Load and train models
    if st.sidebar.button("🚀 Load Data & Train All Models", type="primary"):
        with st.spinner("Loading dataset and training models..."):
            # Load data
            data = fraud_system.load_and_preprocess_data(dataset_path)
            
            if data is not None:
                # Store data in session state
                st.session_state['data'] = data
                st.session_state['data_loaded'] = True
                
                # Train all models
                X_train, X_test, y_train, y_test = fraud_system.train_all_models(data)
                
                # Store training results
                st.session_state['models_trained'] = True
                st.session_state['fraud_system'] = fraud_system
    
    # Save models option
    if st.session_state.get('models_trained', False):
        if st.sidebar.button("💾 Save Trained Models"):
            fraud_system.save_models()
    
    # Main content
    if not st.session_state.get('data_loaded', False):
        # Welcome screen
        st.markdown("""
        ## Welcome to the Fraud Detection System
        
        This application trains and compares three different machine learning models for fraud detection:
        
        ### 📊 Models Available:
        - **Logistic Regression**: Linear model with balanced class weights
        - **Isolation Forest**: Unsupervised anomaly detection
        - **Random Forest**: Ensemble method with 200 trees
        
        ### 🚀 Getting Started:
        1. Ensure your dataset file is available (default: "AIML Dataset3.csv")
        2. Click "Load Data & Train All Models" in the sidebar
        3. Compare model performances and make predictions
        
        ### 📋 Expected Dataset Format:
        Your CSV should contain these columns:
        - `step`: Time step
        - `type`: Transaction type 
        - `amount`: Transaction amount
        - `nameOrig`: Origin account (will be dropped)
        - `oldbalanceOrg`: Origin old balance
        - `newbalanceOrig`: Origin new balance  
        - `nameDest`: Destination account (will be dropped)
        - `oldbalanceDest`: Destination old balance
        - `newbalanceDest`: Destination new balance
        - `isFraud`: Target variable (0 = legitimate, 1 = fraud)
        - `isFlaggedFraud`: System flag (will be dropped)
        """)
        
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Comparison", "🔍 Single Prediction", "📈 Model Details"])
    
    # Tab 1: Data Overview
    with tab1:
        if 'data' in st.session_state:
            data = st.session_state['data']
            
            st.header("📊 Dataset Overview")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Transactions</h3>
                    <h2>{len(data):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fraud_count = data['isFraud'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fraudulent</h3>
                    <h2>{fraud_count:,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                fraud_rate = (fraud_count / len(data)) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fraud Rate</h3>
                    <h2>{fraud_rate:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_amount = data['amount'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Average Amount</h3>
                    <h2>${avg_amount:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Transaction type distribution
                type_counts = data['type'].value_counts()
                fig_type = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title='Transaction Types Distribution'
                )
                st.plotly_chart(fig_type, use_container_width=True)
            
            with col2:
                # Fraud distribution
                fraud_dist = data['isFraud'].value_counts()
                fig_fraud = px.pie(
                    values=fraud_dist.values,
                    names=['Legitimate', 'Fraudulent'],
                    title='Fraud Distribution',
                    color_discrete_map={'Legitimate': '#4CAF50', 'Fraudulent': '#F44336'}
                )
                st.plotly_chart(fig_fraud, use_container_width=True)
            
            # Amount distribution by fraud status
            fig_amount = px.histogram(
                data, 
                x='amount', 
                color='isFraud',
                title='Transaction Amount Distribution by Fraud Status',
                nbins=50,
                color_discrete_map={0: '#4CAF50', 1: '#F44336'}
            )
            fig_amount.update_xaxes(title="Transaction Amount")
            fig_amount.update_yaxes(title="Count")
            st.plotly_chart(fig_amount, use_container_width=True)
            
            # Sample data
            st.subheader("Sample Data")
            st.dataframe(data.head(10), use_container_width=True)
    
    # Tab 2: Model Comparison
    with tab2:
        if st.session_state.get('models_trained', False):
            st.header("🤖 Model Performance Comparison")
            
            fraud_system = st.session_state['fraud_system']
            
            # Model comparison metrics
            comparison_data = []
            
            for model_name, model_info in fraud_system.models.items():
                y_pred = model_info['predictions']
                y_prob = model_info['probabilities']
                
                # Calculate metrics
                report = classification_report(fraud_system.y_test, y_pred, output_dict=True)
                
                if model_name == 'isolation_forest':
                    roc_auc = roc_auc_score(fraud_system.y_test, y_prob)
                else:
                    roc_auc = roc_auc_score(fraud_system.y_test, y_prob)
                
                comparison_data.append({
                    'Model': model_info['name'],
                    'Precision': report['1']['precision'],
                    'Recall': report['1']['recall'],
                    'F1-Score': report['1']['f1-score'],
                    'ROC-AUC': roc_auc
                })
            
            # Display comparison table
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize comparison
            fig_comparison = px.bar(
                comparison_df,
                x='Model',
                y=['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                title='Model Performance Comparison',
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Individual model details
            col1, col2, col3 = st.columns(3)
            
            for i, (model_name, model_info) in enumerate(fraud_system.models.items()):
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>{model_info['name']}</h4>
                        <p><strong>Precision:</strong> {comparison_data[i]['Precision']:.3f}</p>
                        <p><strong>Recall:</strong> {comparison_data[i]['Recall']:.3f}</p>
                        <p><strong>ROC-AUC:</strong> {comparison_data[i]['ROC-AUC']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Please train the models first using the sidebar.")
    
    # Tab 3: Single Prediction
    with tab3:
        if st.session_state.get('models_trained', False):
            st.header("🔍 Single Transaction Prediction")
            
            fraud_system = st.session_state['fraud_system']
            data = st.session_state['data']
            
            # Model selection
            model_choice = st.selectbox(
                "Select Model for Prediction:",
                options=list(fraud_system.models.keys()),
                format_func=lambda x: fraud_system.models[x]['name']
            )
            
            # Input form
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transaction Details")
                step = st.number_input("Step", min_value=0, value=100)
                trans_type = st.selectbox("Transaction Type", options=data['type'].unique())
                amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
                oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=5000.0, format="%.2f")
            
            with col2:
                st.subheader("Balance Information")
                newbalanceOrig = st.number_input("Origin New Balance", min_value=0.0, value=4000.0, format="%.2f")
                oldbalanceDest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0, format="%.2f")
                newbalanceDest = st.number_input("Destination New Balance", min_value=0.0, value=1000.0, format="%.2f")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Predict Fraud", type="primary"):
                    # Prepare transaction data
                    transaction_data = {
                        'step': step,
                        'type': trans_type,
                        'amount': amount,
                        'oldbalanceOrg': oldbalanceOrg,
                        'newbalanceOrig': newbalanceOrig,
                        'oldbalanceDest': oldbalanceDest,
                        'newbalanceDest': newbalanceDest
                    }
                    
                    # Make prediction
                    is_fraud, probability = fraud_system.predict_single_transaction(model_choice, transaction_data)
                    
                    # Display result
                    st.subheader("Prediction Result")
                    if is_fraud:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <h3>⚠️ FRAUDULENT TRANSACTION DETECTED</h3>
                            <p><strong>Model:</strong> {fraud_system.models[model_choice]['name']}</p>
                            <p><strong>Confidence Score:</strong> {probability:.4f}</p>
                            <p><strong>Recommendation:</strong> Block transaction and investigate</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h3>✅ LEGITIMATE TRANSACTION</h3>
                            <p><strong>Model:</strong> {fraud_system.models[model_choice]['name']}</p>
                            <p><strong>Confidence Score:</strong> {probability:.4f}</p>
                            <p><strong>Recommendation:</strong> Process transaction</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # with col2:
            #     if st.button("🎲 Generate Random Transaction"):
            #         # Generate random transaction from dataset
            #         sample = data.sample(1).iloc[0]
            #         st.json({
            #             'step': int(sample['step']),
            #             'type': sample['type'],
            #             'amount': float(sample['amount']),
            #             'oldbalanceOrg': float(sample['oldbalanceOrg']),
            #             'newbalanceOrig': float(sample['newbalanceOrig']),
            #             'oldbalanceDest': float(sample['oldbalanceDest']),
            #             'newbalanceDest': float(sample['newbalanceDest'])
            #         })
            
            with col2:
                if st.button("🎲 Generate Random Transaction"):
                # Sample one row from dataset
                    sample = data.sample(1).iloc[0]

                # Build transaction dict exactly as used by predict_single_transaction
                    transaction_data = {
                        'step': int(sample['step']),
                        'type': sample['type'],
                        'amount': float(sample['amount']),
                        'oldbalanceOrg': float(sample['oldbalanceOrg']),
                        'newbalanceOrig': float(sample['newbalanceOrig']),
                        'oldbalanceDest': float(sample['oldbalanceDest']),
                        'newbalanceDest': float(sample['newbalanceDest'])
                    }

                # Show the generated transaction
                    st.subheader("Generated Transaction")
                    st.json(transaction_data)

                # Predict immediately using selected model_choice (defined earlier in Tab 3)
                    is_fraud, probability = fraud_system.predict_single_transaction(model_choice, transaction_data)

                # Friendly display of the prediction result
                    st.subheader("Prediction for Generated Transaction")
                    if is_fraud is None:
                        st.error("Model not available for prediction. Make sure models are trained and loaded.")
                    else:
                # If IsolationForest, probability is actually anomaly score — label accordingly
                        if model_choice == 'isolation_forest':
                            st.markdown(f"""
                                <div class="fraud-alert">
                                <h3>🚨 Model: {fraud_system.models[model_choice]['name']}</h3>
                                <p><strong>Anomaly Score (higher = more anomalous):</strong> {probability:.4f}</p>
                                <p><strong>Detected as fraud:</strong> {"Yes" if is_fraud else "No"}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                        # For classifiers, probability is class probability for fraud
                            st.markdown(f"""
                                <div class="{'fraud-alert' if is_fraud else 'safe-alert'}">
                                <h3>{'⚠️ FRAUDULENT TRANSACTION DETECTED' if is_fraud else '✅ LEGITIMATE TRANSACTION'}</h3>
                                <p><strong>Model:</strong> {fraud_system.models[model_choice]['name']}</p>
                                <p><strong>Fraud Probability:</strong> {probability:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)

        else:
            st.info("Please train the models first using the sidebar.")
    
    # Tab 4: Model Details
    with tab4:
        if st.session_state.get('models_trained', False):
            st.header("📈 Detailed Model Analysis")
            
            fraud_system = st.session_state['fraud_system']
            
            # Model selection for detailed view
            selected_model = st.selectbox(
                "Select Model for Detailed Analysis:",
                options=list(fraud_system.models.keys()),
                format_func=lambda x: fraud_system.models[x]['name']
            )
            
            model_info = fraud_system.models[selected_model]
            y_pred = model_info['predictions']
            y_prob = model_info['probabilities']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm = confusion_matrix(fraud_system.y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    title=f"Confusion Matrix - {model_info['name']}",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Legitimate', 'Fraudulent'],
                    y=['Legitimate', 'Fraudulent']
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # ROC Curve
                fpr, tpr, _ = roc_curve(fraud_system.y_test, y_prob)
                roc_auc = roc_auc_score(fraud_system.y_test, y_prob)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    line=dict(color='blue', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random Classifier',
                    line=dict(color='red', dash='dash')
                ))
                fig_roc.update_layout(
                    title=f'ROC Curve - {model_info["name"]}',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(fraud_system.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
        else:
            st.info("Please train the models first using the sidebar.")

if __name__ == "__main__":
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'models_trained' not in st.session_state:
        st.session_state['models_trained'] = False
    
    main()