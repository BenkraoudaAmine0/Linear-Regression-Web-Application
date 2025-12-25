import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Linear Regression from Scratch",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .formula {
        font-family: 'Courier New', monospace;
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


class LinearRegressionFromScratch:
    """
    Simple Linear Regression implementation using Least Squares Method.
    """
    
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):
        """Fit the linear regression model using the Least Squares Method."""
        X = np.array(X)
        y = np.array(y)
        
        n = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(y)
        sum_xy = np.sum(X * y)
        sum_x_squared = np.sum(X ** 2)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x_squared - sum_x ** 2
        
        if denominator == 0:
            raise ValueError("Cannot fit: all X values are identical")
        
        self.slope = numerator / denominator
        self.intercept = (sum_y - self.slope * sum_x) / n
        
        return self
    
    def predict(self, X):
        """Make predictions using the fitted linear model."""
        if self.slope is None or self.intercept is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.array(X)
        return self.slope * X + self.intercept
    
    def r_squared(self, X, y):
        """Calculate the coefficient of determination (R²)."""
        y = np.array(y)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def mse(self, X, y):
        """Calculate Mean Squared Error (MSE)."""
        y = np.array(y)
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def get_params(self):
        """Get the fitted parameters."""
        return {
            'slope': self.slope,
            'intercept': self.intercept
        }


def generate_data(n_points=50, slope=2.5, intercept=10, noise=5, random_seed=42):
    """Generate synthetic data for linear regression."""
    np.random.seed(random_seed)
    X = np.linspace(0, 10, n_points)
    y = slope * X + intercept + np.random.normal(0, noise, n_points)
    return X, y


def plot_regression(X, y, model, title="Linear Regression Results"):
    """Create a comprehensive visualization of the regression results."""
    y_pred = model.predict(X)
    r2 = model.r_squared(X, y)
    mse_value = model.mse(X, y)
    params = model.get_params()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Scatter plot with regression line
    ax1.scatter(X, y, color='#3498db', alpha=0.6, s=80, edgecolors='black', 
                linewidth=0.5, label='Data Points')
    ax1.plot(X, y_pred, color='#e74c3c', linewidth=2.5, label='Regression Line', 
             linestyle='--')
    
    # Draw residual lines
    for i in range(len(X)):
        ax1.plot([X[i], X[i]], [y[i], y_pred[i]], color='gray', 
                linestyle=':', alpha=0.4, linewidth=1)
    
    ax1.set_xlabel('X (Independent Variable)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (Dependent Variable)', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add equation and statistics
    equation_text = f"y = {params['slope']:.3f}x + {params['intercept']:.3f}"
    stats_text = f"R² = {r2:.4f}\nMSE = {mse_value:.4f}"
    
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.text(0.05, 0.85, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Residuals plot
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, color='#9b59b6', alpha=0.6, s=80, 
                edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Errors)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, params, r2, mse_value


def plot_polynomial_regression(X, y, degree=2, title="Polynomial Regression Results"):
    """Fit and plot polynomial regression."""
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    poly_func = np.poly1d(coeffs)
    
    y_pred = poly_func(X)
    
    # Sort X for smooth line plotting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_pred_sorted = poly_func(X_sorted)
    
    # Calculate metrics
    ss_res = np.sum((y - y_pred) ** 2)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        
    mse_value = np.mean((y - y_pred) ** 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Scatter plot with regression curve
    ax1.scatter(X, y, color='#3498db', alpha=0.6, s=80, edgecolors='black', 
                linewidth=0.5, label='Data Points')
    ax1.plot(X_sorted, y_pred_sorted, color='#2ecc71', linewidth=2.5, 
             label=f'Polynomial (Deg {degree})', linestyle='-')
    
    ax1.set_xlabel('X (Independent Variable)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (Dependent Variable)', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add equation and statistics
    if degree == 2:
        equation_text = f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}"
    else:
        equation_text = f"Poly Degree {degree}"
        
    stats_text = f"R² = {r2:.4f}\nMSE = {mse_value:.4f}"
    
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.text(0.05, 0.85, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='#dcedc8', alpha=0.8))
    
    # Plot 2: Residuals plot
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, color='#9b59b6', alpha=0.6, s=80, 
                edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Errors)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot (Polynomial)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, coeffs, r2, mse_value


def assess_model_quality(r2, mse, rmse, n_points):
    """
    Assess the quality of the regression model and provide interpretation.
    
    Parameters:
        r2 (float): R-squared score
        mse (float): Mean Squared Error
        rmse (float): Root Mean Squared Error
        n_points (int): Number of data points
    
    Returns:
        dict: Assessment with quality rating, color, icon, interpretation, and recommendations
    """
    # Determine quality based on R² score
    if r2 >= 0.9:
        quality = "Excellent"
        color = "#4caf50"  # Green
        icon = "🌟"
        interpretation = "The model fits the data very well! The predictions are highly accurate."
        recommendations = [
            "✅ Your model is performing excellently!",
            "💡 Consider using this model for predictions",
            "📊 Document your findings and share results"
        ]
    elif r2 >= 0.7:
        quality = "Good"
        color = "#2196f3"  # Blue
        icon = "✅"
        interpretation = "The model shows a strong relationship between variables. Predictions are reliable."
        recommendations = [
            "✅ Your model is performing well",
            "💡 You can use this for predictions with confidence",
            "🔍 Consider collecting more data to potentially improve further"
        ]
    elif r2 >= 0.5:
        quality = "Fair"
        color = "#ff9800"  # Orange
        icon = "⚠️"
        interpretation = "The model shows a moderate relationship. There's room for improvement."
        recommendations = [
            "📊 Try adding more relevant features to your model",
            "🔍 Check for outliers in your data that might be affecting the fit",
            "📈 Consider polynomial regression if the relationship isn't linear",
            "🧹 Clean your data: remove missing values and inconsistencies"
        ]
    elif r2 >= 0.3:
        quality = "Poor"
        color = "#f44336"  # Red
        icon = "❌"
        interpretation = "The model shows weak predictive power. Consider other features or models."
        recommendations = [
            "🔄 Try different features - current ones may not be predictive",
            "📊 Collect more relevant data that better explains the target variable",
            "🤔 The relationship might not be linear - try other models",
            "🔍 Check if there are confounding variables you haven't considered",
            "📈 Consider feature engineering (combining features, transformations)"
        ]
    else:
        quality = "Very Poor"
        color = "#d32f2f"  # Dark Red
        icon = "🚫"
        interpretation = "The model has very weak predictive power. Linear regression may not be suitable."
        recommendations = [
            "🔄 Linear regression is not suitable for this data",
            "📊 The features chosen have little to no relationship with the target",
            "🤔 Try completely different features or a different approach",
            "🔍 Visualize your data to understand the relationship better",
            "💡 Consider non-linear models (polynomial, decision trees, etc.)",
            "📈 Check if the target variable can actually be predicted from available data"
        ]
    
    # Additional insights
    insights = []
    
    if r2 < 0:
        insights.append("⚠️ Negative R²: The model performs worse than a horizontal line (mean)!")
        recommendations.insert(0, "🚨 CRITICAL: Your model is worse than just predicting the average!")
    
    if n_points < 30:
        insights.append(f"📊 Small sample size ({n_points} points): Results may not be reliable.")
        recommendations.append(f"📈 Collect more data - you only have {n_points} points (aim for 100+)")
    elif n_points > 1000:
        insights.append(f"📊 Large sample size ({n_points} points): Results are statistically robust.")
    
    if rmse > 0:
        insights.append(f"📏 Average prediction error (RMSE): ±{rmse:.4f}")
    
    return {
        'quality': quality,
        'color': color,
        'icon': icon,
        'interpretation': interpretation,
        'insights': insights,
        'recommendations': recommendations,
        'r2_percent': r2 * 100 if r2 > 0 else 0
    }


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Linear Regression from Scratch</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Understanding Linear Algebra through Interactive Regression Analysis</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Choose a section:", 
                            ["📚 Introduction", 
                             "🎓 Theory & Formulas", 
                             "🔬 Interactive Demo", 
                             "📊 Dataset Analysis",
                             "💡 Step-by-Step Guide"])
    
    # ==================== INTRODUCTION PAGE ====================
    if page == "📚 Introduction":
        st.markdown('<h2 class="sub-header">Welcome to Linear Regression Learning Platform</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 What You'll Learn</h3>
        <ul>
            <li><strong>Mathematical Foundation:</strong> Understand the least squares method</li>
            <li><strong>Implementation:</strong> Build linear regression from scratch (no ML libraries!)</li>
            <li><strong>Visualization:</strong> See how regression works with interactive plots</li>
            <li><strong>Evaluation:</strong> Learn about R², MSE, and model quality metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 What is Linear Regression?
            
            Linear regression is a statistical method that models the relationship between:
            - **Independent variable (X)**: The input or predictor
            - **Dependent variable (Y)**: The output or response
            
            The goal is to find the best-fit line: **y = ax + b**
            - **a**: Slope (how steep the line is)
            - **b**: Intercept (where the line crosses the y-axis)
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Why It Matters
            
            Linear regression is fundamental in:
            - **Prediction**: Forecasting future values
            - **Trend Analysis**: Understanding relationships
            - **Data Science**: Foundation for advanced ML
            - **Statistics**: Hypothesis testing and inference
            
            It's the building block of machine learning!
            """)
        
        st.markdown("""
        <div class="success-box">
        <h3>✨ Features of This Platform</h3>
        <ul>
            <li>🎮 <strong>Interactive Controls:</strong> Adjust parameters in real-time</li>
            <li>📊 <strong>Visual Learning:</strong> See mathematical concepts come to life</li>
            <li>📁 <strong>Real Datasets:</strong> Work with actual data (Forest Fires, Diabetes)</li>
            <li>🧮 <strong>Step-by-Step:</strong> Follow the algorithm execution</li>
            <li>💾 <strong>Export Results:</strong> Download your analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== THEORY PAGE ====================
    elif page == "🎓 Theory & Formulas":
        st.markdown('<h2 class="sub-header">Mathematical Foundation</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        ### 📐 The Least Squares Method
        
        The least squares method finds the line that minimizes the sum of squared residuals (errors).
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 Objective Function
            
            We want to minimize:
            """)
            st.latex(r"\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
            
            st.markdown(r"""
            Where:
            - $y_i$ = Actual value
            - $\hat{y}_i$ = Predicted value
            - $n$ = Number of data points
            """)
        
        with col2:
            st.markdown("""
            #### 📊 The Linear Model
            
            Our prediction equation:
            """)
            st.latex(r"\hat{y} = ax + b")
            
            st.markdown("""
            Where:
            - $a$ = Slope coefficient
            - $b$ = Y-intercept
            - $x$ = Input variable
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🧮 Formulas (Derived from Calculus)
        
        By taking partial derivatives and setting them to zero, we get:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Slope Formula")
            st.latex(r"a = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
            
            st.markdown(r"""
            **Components:**
            - $n$ = number of points
            - $\sum xy$ = sum of products
            - $\sum x$ = sum of X values
            - $\sum y$ = sum of Y values
            - $\sum x^2$ = sum of X squared
            """)
        
        with col2:
            st.markdown("#### Intercept Formula")
            st.latex(r"b = \frac{\sum y - a\sum x}{n}")
            
            st.markdown(r"""
            **Interpretation:**
            - Calculated after finding slope
            - Ensures line passes through mean point
            - $(\bar{x}, \bar{y})$ lies on the line
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📏 Evaluation Metrics
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### R² Score (Coefficient of Determination)")
            st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
            
            st.markdown(r"""
            Where:
            - $SS_{res} = \sum(y_i - \hat{y}_i)^2$ (Residual sum of squares)
            - $SS_{tot} = \sum(y_i - \bar{y})^2$ (Total sum of squares)
            
            **Interpretation:**
            - R² = 1: Perfect fit
            - R² = 0: No better than mean
            - R² < 0: Worse than mean
            """)
        
        with col2:
            st.markdown("#### Mean Squared Error (MSE)")
            st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
            
            st.markdown("""
            **Characteristics:**
            - Always non-negative
            - Lower is better
            - Same units as Y²
            - Sensitive to outliers
            
            **Related:** RMSE = √MSE (same units as Y)
            """)
    
    # ==================== INTERACTIVE DEMO PAGE ====================
    elif page == "🔬 Interactive Demo":
        st.markdown('<h2 class="sub-header">Interactive Regression Demo</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>🎮 How to Use:</strong> Adjust the sliders below to generate synthetic data with different characteristics. 
        Watch how the regression line adapts to the data!
        </div>
        """, unsafe_allow_html=True)
        
        # Controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_points = st.slider("📊 Number of Points", 10, 200, 50, 10)
        with col2:
            true_slope = st.slider("📈 True Slope", -10.0, 10.0, 2.5, 0.5)
        with col3:
            true_intercept = st.slider("📍 True Intercept", -50.0, 50.0, 10.0, 5.0)
        with col4:
            noise_level = st.slider("🔊 Noise Level", 0.0, 20.0, 5.0, 1.0)
        
        # Generate and fit
        X, y = generate_data(n_points, true_slope, true_intercept, noise_level)
        model = LinearRegressionFromScratch()
        model.fit(X, y)
        
        # Plot
        fig, params, r2, mse_value = plot_regression(X, y, model, 
                                                      f"Synthetic Data (True: y={true_slope}x+{true_intercept})")
        st.pyplot(fig)
        
        # Results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fitted Slope", f"{params['slope']:.4f}", 
                     f"{params['slope'] - true_slope:.4f}")
        with col2:
            st.metric("Fitted Intercept", f"{params['intercept']:.4f}", 
                     f"{params['intercept'] - true_intercept:.4f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}", 
                     f"{r2*100:.2f}% variance")
        with col4:
            st.metric("MSE", f"{mse_value:.4f}", 
                     f"RMSE: {np.sqrt(mse_value):.4f}")
        
        # Explanation
        st.markdown("""
        <div class="success-box">
        <h4>📊 Understanding the Results</h4>
        <ul>
            <li><strong>Fitted vs True:</strong> Compare the estimated parameters with the true values</li>
            <li><strong>R² Score:</strong> Shows how well the model explains the variance (higher is better)</li>
            <li><strong>Residuals:</strong> The vertical lines show prediction errors</li>
            <li><strong>Noise Effect:</strong> More noise → Lower R² → Harder to fit</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== DATASET ANALYSIS PAGE ====================
    elif page == "📊 Dataset Analysis":
        st.markdown('<h2 class="sub-header">Real Dataset Analysis</h2>', 
                    unsafe_allow_html=True)
        
        # Dataset selection
        dataset_choice = st.selectbox("📁 Choose a dataset:", 
                                      ["Forest Fires", "Diabetes", "Upload Your Own"])
        
        if dataset_choice == "Forest Fires":
            try:
                df = pd.read_csv('MiniProjet2/forestfires.csv')
                st.success("✅ Forest Fires dataset loaded successfully!")
                
                st.markdown("""
                <div class="info-box">
                <strong>📖 About this dataset:</strong> Contains data about forest fires in Portugal, 
                including meteorological conditions and burned area.
                </div>
                """, unsafe_allow_html=True)
                
            except FileNotFoundError:
                st.error("❌ Forest fires dataset not found. Please check the file path.")
                return
                
        elif dataset_choice == "Diabetes":
            try:
                df = pd.read_csv('MiniProjet2/diabetes_data.csv')
                
                # Convert categorical columns to numeric (like in MiniProjetDiabet.ipynb)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = pd.Categorical(df[col]).codes
                
                st.success("✅ Diabetes dataset loaded successfully!")
                
                st.markdown("""
                <div class="info-box">
                <strong>📖 About this dataset:</strong> Contains medical data for diabetes prediction, 
                including symptoms and patient characteristics. Categorical columns have been automatically converted to numeric codes.
                </div>
                """, unsafe_allow_html=True)
                
            except FileNotFoundError:
                st.error("❌ Diabetes dataset not found. Please check the file path.")
                return
        
        else:  # Upload Your Own
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                # Convert categorical columns to numeric (similar to Diabetes dataset logic)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = pd.Categorical(df[col]).codes
                        
                st.success("✅ File uploaded successfully!")
            else:
                st.info("👆 Please upload a CSV file to continue")
                return
        
        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        
        # Show data preview
        with st.expander("👁️ View Data Preview"):
            st.dataframe(df.head(10))
        
        # Column selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for regression analysis")
            return
        
        # Regression mode selection
        st.markdown("---")
        st.markdown("### 🎯 Regression Mode")
        regression_mode = st.radio(
            "Choose regression type:",
            ["Simple Regression (2 columns)", "Multiple Regression (All columns vs Target)"],
            horizontal=True
        )
        
        if regression_mode == "Simple Regression (2 columns)":
            # Original simple regression with 2 columns
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("📊 Select X column (Independent Variable):", numeric_cols)
            with col2:
                y_col = st.selectbox("📊 Select Y column (Dependent Variable):", numeric_cols)
            
            if x_col == y_col:
                st.warning("⚠️ Please select different columns for X and Y")
                return
            
            # Perform regression
            X = df[x_col].values
            y = df[y_col].values
            
            # Remove NaN values
            mask = ~(np.isnan(X) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 2:
                st.error("❌ Not enough valid data points after removing NaN values")
                return
            
            model = LinearRegressionFromScratch()
            model.fit(X, y)
            
            # Plot
            fig, params, r2, mse_value = plot_regression(X, y, model, 
                                                          f"{dataset_choice}: {x_col} vs {y_col}")
            st.pyplot(fig)
            
            # Results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Slope (a)", f"{params['slope']:.6f}")
            with col2:
                st.metric("Intercept (b)", f"{params['intercept']:.6f}")
            with col3:
                st.metric("R² Score", f"{r2:.6f}")
            with col4:
                st.metric("RMSE", f"{np.sqrt(mse_value):.6f}")
            
            # Model Quality Assessment
            assessment = assess_model_quality(r2, mse_value, np.sqrt(mse_value), len(X))
            
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color: {assessment['color']}15; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid {assessment['color']};">
                <h3 style="color: {assessment['color']}; margin-top: 0;">{assessment['icon']} Model Quality: {assessment['quality']}</h3>
                <p style="font-size: 1.1rem; margin-bottom: 1rem;"><strong>{assessment['interpretation']}</strong></p>
                <p style="margin-bottom: 0.5rem;"><strong>📊 Variance Explained:</strong> {assessment['r2_percent']:.2f}%</p>
                {''.join([f'<p style="margin: 0.3rem 0;">{insight}</p>' for insight in assessment['insights']])}
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            if assessment['recommendations']:
                st.markdown("### 💡 How to Improve Your Model")
                st.markdown("""
                <div class="info-box">
                <strong>Recommendations based on your results:</strong>
                </div>
                """, unsafe_allow_html=True)
                
                for rec in assessment['recommendations']:
                    st.markdown(f"- {rec}")
            
            # Download results
            results_df = pd.DataFrame({
                'X': X,
                'Y_Actual': y,
                'Y_Predicted': model.predict(X),
                'Residual': y - model.predict(X)
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Results as CSV",
                data=csv,
                file_name=f"regression_results_{x_col}_vs_{y_col}.csv",
                mime="text/csv"
            )
            
            # Check for high error and switch to polynomial
            st.markdown("---")
            st.markdown("### 🔄 Adaptive Model Selection")
            
            # Threshold input
            col_thresh1, col_thresh2, col_thresh3 = st.columns([2, 1, 1])
            with col_thresh1:
                st.markdown("""
                **Check for Linear Model Limitations:**
                If the Mean Squared Error (MSE) is too high, it might indicate a non-linear relationship.
                Set a threshold below. The app will iteratively increase the polynomial degree (up to a limit) until the error is below the threshold.
                """)
            with col_thresh2:
                # Default threshold slightly above current MSE
                default_threshold = float(mse_value * 1.5) if mse_value > 0 else 100.0
                mse_threshold = st.number_input(
                    "MSE Threshold",
                    value=default_threshold,
                    format="%.4f",
                    help="Target MSE. If current error > this, degree increases."
                )
            with col_thresh3:
                max_degree = st.number_input(
                    "Max Degree Limit",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help="Maximum polynomial degree to attempt."
                )
            
            if mse_value > mse_threshold:
                st.warning(f"⚠️ High Error Detected! Linear MSE ({mse_value:.4f}) > Threshold ({mse_threshold:.4f}).")
                
                # Iterative Polynomial Search
                current_degree = 2
                best_mse = mse_value # Start with linear MSE
                best_degree = 1
                best_r2 = r2
                best_fig = None
                best_coeffs = None
                
                search_history = []
                
                placeholder = st.empty()
                
                while current_degree <= max_degree:
                    # Fit current degree
                    # We compute this to check MSE. Ideally we wrap this invalid a function to avoid re-plotting, 
                    # but plot_polynomial_regression returns everything we need.
                    # We'll use a simplified check first or just run the plot function (it's fast enough for small data)
                    coeffs = np.polyfit(X, y, current_degree)
                    poly_func = np.poly1d(coeffs)
                    y_pred_poly = poly_func(X)
                    mse_poly = np.mean((y - y_pred_poly) ** 2)
                    
                    search_history.append({
                        'degree': current_degree, 
                        'mse': mse_poly,
                        'coeffs': coeffs
                    })
                    
                    placeholder.info(f"🔄 Trying Polynomial Degree {current_degree}... MSE: {mse_poly:.4f}")
                    
                    # Update best if better (should be better as degree increases usually)
                    if mse_poly < best_mse:
                        best_mse = mse_poly
                        best_degree = current_degree
                        best_coeffs = coeffs
                        
                    # Check if we met the threshold
                    if mse_poly <= mse_threshold:
                        break
                        
                    current_degree += 1
                
                # Plot the final result (Best Degree Found or Max Reached)
                placeholder.empty()
                
                if best_degree > 1:
                    status_msg = "✅ Found suitable model!" if best_mse <= mse_threshold else "⚠️ Reached max degree limit without meeting threshold."
                    st.success(f"{status_msg} Switching to **Polynomial Degree {best_degree}**.")
                    
                    fig_poly, coeffs_poly, r2_poly, mse_poly = plot_polynomial_regression(
                        X, y, degree=best_degree, 
                        title=f"Polynomial Regression (Degree {best_degree}): {x_col} vs {y_col}"
                    )
                    st.pyplot(fig_poly)
                    
                    # Display Polynomial Results
                    col_p1, col_p2, col_p3 = st.columns(3)
                    
                    with col_p1:
                        st.metric(f"Degree {best_degree} R²", f"{r2_poly:.6f}", 
                                 f"{(r2_poly - r2):.6f} vs Linear",
                                 delta_color="normal")
                    with col_p2:
                        st.metric(f"Degree {best_degree} MSE", f"{mse_poly:.6f}", 
                                 f"{(mse_poly - mse_value):.6f} vs Linear",
                                 delta_color="inverse")
                    with col_p3:
                        st.metric("Total Improvement", 
                                 f"{((mse_value - mse_poly)/mse_value * 100):.2f}%" if mse_value > 0 else "N/A",
                                 "Error Reduction")
                    
                    # Show Detailed Step-by-Step Optimization
                    st.markdown("### 📝 Detailed Optimization Steps")
                    with st.expander("View Step-by-Step Error Correction & Equations", expanded=True):
                        st.markdown("Below shows how the model adapted by increasing the degree to reduce the error.")
                        
                        previous_mse = mse_value
                        
                        for step in search_history:
                            deg = step['degree']
                            a = step['mse']
                            c = step['coeffs']
                            
                            # Create equation string
                            terms = []
                            for power, coeff in enumerate(c[::-1]):
                                if power == 0:
                                    terms.append(f"{coeff:.3f}")
                                elif power == 1:
                                    terms.append(f"{coeff:.3f}x")
                                else:
                                    terms.append(f"{coeff:.3f}x^{power}")
                            eq_str = " + ".join(terms[::-1]).replace("+ -", "- ")
                            
                            # Calculate correction
                            correction = previous_mse - a
                            pct_improvement = (correction / previous_mse * 100) if previous_mse > 0 else 0
                            
                            st.markdown(f"#### 🔹 Degree {deg}")
                            col_s1, col_s2 = st.columns([3, 1])
                            
                            with col_s1:
                                st.markdown(f"**Equation:** ${eq_str}$")
                                st.markdown(f"**MSE:** {a:.6f}")
                            
                            with col_s2:
                                if correction > 0:
                                    st.success(f"Error Reduced by:\n{correction:.6f} \n({pct_improvement:.1f}%)")
                                else:
                                    st.warning("No improvement")
                                    
                            st.markdown("---")
                            previous_mse = a
                            
                        st.caption("The algorithm stops when MSE ≤ Threshold or Max Degree is reached.")
                
                else:
                    st.warning("⚠️ Polynomial regression did not improve the model (MSE did not decrease). The relationship might be purely linear or the data is too noisy.")
                    
                    # Show Search History
                    st.markdown("### 📝 Optimization Attempts")
                    with st.expander("View Search History", expanded=True):
                        st.write("The algorithm tried higher degrees but found no improvement:")
                        if search_history:
                            hist_df = pd.DataFrame(search_history)[['degree', 'mse']]
                            st.dataframe(hist_df.style.format({'mse': '{:.6f}'}))
                        else:
                            st.write("No search history available.")
                
            else:
                st.success(f"✅ Linear MSE ({mse_value:.4f}) is acceptable (<= {mse_threshold:.4f}). No polynomial switch needed.")
            
            # ==================== STEP-BY-STEP CALCULATIONS ====================
            st.markdown("---")
            st.markdown("### 🔍 Step-by-Step Calculations")
            st.markdown("""
            <div class="info-box">
            <strong>📚 Understanding the Math:</strong> See how the regression parameters were calculated step by step.
            </div>
            """, unsafe_allow_html=True)
            
            # Step 1: Calculate Necessary Sums
            st.markdown("#### Step 1️⃣: Calculate Necessary Sums")
            
            n = len(X)
            sum_x = np.sum(X)
            sum_y = np.sum(y)
            sum_xy = np.sum(X * y)
            sum_x2 = np.sum(X ** 2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"""
n = {n}
Σx = {sum_x}
Σy = {sum_y}
                """)
            with col2:
                st.code(f"""
Σ(xy) = {sum_xy}
Σ(x²) = {sum_x2}
                """)
            
            # Step 2: Calculate Slope (a)
            st.markdown("---")
            st.markdown("#### Step 2️⃣: Calculate Slope (a)")
            
            st.latex(r"a = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
            
            a_numerator = n * sum_xy - sum_x * sum_y
            a_denominator = n * sum_x2 - sum_x**2
            a = params['slope']
            
            st.code(f"""
Numerator = n×Σ(xy) - Σx×Σy
          = {n}×{sum_xy} - {sum_x}×{sum_y}
          = {a_numerator}

Denominator = n×Σ(x²) - (Σx)²
            = {n}×{sum_x2} - {sum_x}²
            = {a_denominator}

Slope (a) = {a_numerator} / {a_denominator} = {a:.6f}
            """)
            
            # Step 3: Calculate Intercept (b)
            st.markdown("---")
            st.markdown("#### Step 3️⃣: Calculate Intercept (b)")
            
            st.latex(r"b = \frac{\sum y - a\sum x}{n}")
            
            b = params['intercept']
            
            st.code(f"""
b = (Σy - a×Σx) / n
  = ({sum_y} - {a:.6f}×{sum_x}) / {n}
  = {b:.6f}
            """)
            
            # Step 4: Make Predictions
            st.markdown("---")
            st.markdown("#### Step 4️⃣: Make Predictions")
            
            y_pred = model.predict(X)
            
            # Show first 10 predictions
            pred_df = pd.DataFrame({
                'X': X[:10],
                'Y_Actual': y[:10],
                'Y_Predicted': y_pred[:10],
                'Residual': (y - y_pred)[:10]
            })
            st.dataframe(pred_df)
            st.info(f"📊 Showing first 10 predictions out of {len(X)} total data points")
            
            # Step 5: Evaluate the Model
            st.markdown("---")
            st.markdown("#### Step 5️⃣: Evaluate the Model")
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_calc = 1 - (ss_res / ss_tot)
            mse_calc = np.mean((y - y_pred) ** 2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"""
SS_res = Σ(y - ŷ)² = {ss_res:.6f}
SS_tot = Σ(y - ȳ)² = {ss_tot:.6f}

R² = 1 - (SS_res / SS_tot)
   = 1 - ({ss_res:.6f} / {ss_tot:.6f})
   = {r2_calc:.6f}
                """)
            with col2:
                st.code(f"""
MSE = (1/n) × Σ(y - ŷ)²
    = (1/{n}) × {ss_res:.6f}
    = {mse_calc:.6f}

RMSE = √MSE = {np.sqrt(mse_calc):.6f}
                """)
            
        else:  # Multiple Regression
            st.info("📊 **Multiple Regression Mode**: Select the target column (Y), and all other numeric columns will be used as features (X)")
            
            # Select target column
            y_col = st.selectbox("🎯 Select Target Column (Y):", numeric_cols)
            
            # Get all other columns as features
            x_cols = [col for col in numeric_cols if col != y_col]
            
            if len(x_cols) == 0:
                st.warning("⚠️ Need at least one feature column (X) after selecting target")
                return
            
            st.success(f"✅ Using {len(x_cols)} feature columns: {', '.join(x_cols)}")
            
            # Prepare data
            y = df[y_col].values
            
            # For multiple regression, we'll perform regression for each X column separately
            st.markdown("---")
            st.markdown(f"### 📈 Individual Regressions: Each Feature vs {y_col}")
            
            # Create tabs for each feature
            tabs = st.tabs(x_cols)
            
            for idx, (tab, x_col) in enumerate(zip(tabs, x_cols)):
                with tab:
                    X = df[x_col].values
                    
                    # Remove NaN values
                    mask = ~(np.isnan(X) | np.isnan(y))
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) < 2:
                        st.error(f"❌ Not enough valid data points for {x_col}")
                        continue
                    
                    # Fit model
                    model = LinearRegressionFromScratch()
                    model.fit(X_clean, y_clean)
                    
                    # Plot
                    fig, params, r2, mse_value = plot_regression(
                        X_clean, y_clean, model, 
                        f"{x_col} vs {y_col}"
                    )
                    st.pyplot(fig)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Slope", f"{params['slope']:.6f}")
                    with col2:
                        st.metric("Intercept", f"{params['intercept']:.6f}")
                    with col3:
                        st.metric("R²", f"{r2:.6f}")
                    with col4:
                        st.metric("RMSE", f"{np.sqrt(mse_value):.6f}")
                    
                    # Model Quality Assessment for this feature
                    assessment = assess_model_quality(r2, mse_value, np.sqrt(mse_value), len(X_clean))
                    
                    st.markdown(f"""
                    <div style="background-color: {assessment['color']}15; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid {assessment['color']}; margin-top: 1rem;">
                        <h4 style="color: {assessment['color']}; margin-top: 0;">{assessment['icon']} {assessment['quality']}</h4>
                        <p style="margin-bottom: 0.5rem;">{assessment['interpretation']}</p>
                        <p style="margin: 0;"><strong>Variance Explained:</strong> {assessment['r2_percent']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show recommendations if model needs improvement
                    if r2 < 0.7:
                        with st.expander("💡 How to Improve This Feature"):
                            for rec in assessment['recommendations']:
                                st.markdown(f"- {rec}")
            
            # Summary comparison
            st.markdown("---")
            st.markdown("### 📊 Summary: All Features Comparison")
            
            summary_data = []
            for x_col in x_cols:
                X = df[x_col].values
                mask = ~(np.isnan(X) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) >= 2:
                    model = LinearRegressionFromScratch()
                    model.fit(X_clean, y_clean)
                    params = model.get_params()
                    r2 = model.r_squared(X_clean, y_clean)
                    mse_val = model.mse(X_clean, y_clean)
                    
                    # Get quality assessment
                    assessment = assess_model_quality(r2, mse_val, np.sqrt(mse_val), len(X_clean))
                    
                    summary_data.append({
                        'Feature': x_col,
                        'Quality': f"{assessment['icon']} {assessment['quality']}",
                        'R²': f"{r2:.4f}",
                        'Slope': f"{params['slope']:.4f}",
                        'Intercept': f"{params['intercept']:.4f}",
                        'RMSE': f"{np.sqrt(mse_val):.4f}",
                        'Data Points': len(X_clean)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Download summary
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Summary as CSV",
                    data=csv_summary,
                    file_name=f"multiple_regression_summary_{y_col}.csv",
                    mime="text/csv"
                )
    
    # ==================== STEP-BY-STEP GUIDE PAGE ====================
    elif page == "💡 Step-by-Step Guide":
        st.markdown('<h2 class="sub-header">Step-by-Step Implementation Guide</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>🎯 Learning Objective:</strong> Understand how to implement linear regression from scratch, 
        step by step, without using ML libraries.
        </div>
        """, unsafe_allow_html=True)
        
        # Example data
        st.markdown("### 📊 Example Data")
        X_example = np.array([1, 2, 3, 4, 5])
        y_example = np.array([2, 4, 5, 4, 5])
        
        example_df = pd.DataFrame({'X': X_example, 'Y': y_example})
        st.dataframe(example_df)
        
        # Step 1
        st.markdown("---")
        st.markdown("### Step 1️⃣: Calculate Necessary Sums")
        
        n = len(X_example)
        sum_x = np.sum(X_example)
        sum_y = np.sum(y_example)
        sum_xy = np.sum(X_example * y_example)
        sum_x2 = np.sum(X_example ** 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"""
n = {n}
Σx = {sum_x}
Σy = {sum_y}
            """)
        with col2:
            st.code(f"""
Σ(xy) = {sum_xy}
Σ(x²) = {sum_x2}
            """)
        
        # Step 2
        st.markdown("---")
        st.markdown("### Step 2️⃣: Calculate Slope (a)")
        
        st.latex(r"a = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
        
        a_numerator = n * sum_xy - sum_x * sum_y
        a_denominator = n * sum_x2 - sum_x**2
        a = a_numerator / a_denominator
        
        st.code(f"""
Numerator = n×Σ(xy) - Σx×Σy
          = {n}×{sum_xy} - {sum_x}×{sum_y}
          = {a_numerator}

Denominator = n×Σ(x²) - (Σx)²
            = {n}×{sum_x2} - {sum_x}²
            = {a_denominator}

Slope (a) = {a_numerator} / {a_denominator} = {a:.6f}
        """)
        
        # Step 3
        st.markdown("---")
        st.markdown("### Step 3️⃣: Calculate Intercept (b)")
        
        st.latex(r"b = \frac{\sum y - a\sum x}{n}")
        
        b = (sum_y - a * sum_x) / n
        
        st.code(f"""
b = (Σy - a×Σx) / n
  = ({sum_y} - {a:.6f}×{sum_x}) / {n}
  = {b:.6f}
        """)
        
        # Step 4
        st.markdown("---")
        st.markdown("### Step 4️⃣: Make Predictions")
        
        y_pred = a * X_example + b
        
        pred_df = pd.DataFrame({
            'X': X_example,
            'Y_Actual': y_example,
            'Y_Predicted': y_pred,
            'Residual': y_example - y_pred
        })
        st.dataframe(pred_df)
        
        # Step 5
        st.markdown("---")
        st.markdown("### Step 5️⃣: Evaluate the Model")
        
        ss_res = np.sum((y_example - y_pred) ** 2)
        ss_tot = np.sum((y_example - np.mean(y_example)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        mse = np.mean((y_example - y_pred) ** 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"""
SS_res = Σ(y - ŷ)² = {ss_res:.6f}
SS_tot = Σ(y - ȳ)² = {ss_tot:.6f}

R² = 1 - (SS_res / SS_tot)
   = 1 - ({ss_res:.6f} / {ss_tot:.6f})
   = {r2:.6f}
            """)
        with col2:
            st.code(f"""
MSE = (1/n) × Σ(y - ŷ)²
    = (1/{n}) × {ss_res:.6f}
    = {mse:.6f}

RMSE = √MSE = {np.sqrt(mse):.6f}
            """)
        
        # Visualization
        st.markdown("---")
        st.markdown("### Step 6️⃣: Visualize Results")
        
        model_example = LinearRegressionFromScratch()
        model_example.fit(X_example, y_example)
        fig, _, _, _ = plot_regression(X_example, y_example, model_example, 
                                       "Step-by-Step Example")
        st.pyplot(fig)
        
        st.markdown("""
        <div class="success-box">
        <h4>🎉 Congratulations!</h4>
        You've successfully implemented linear regression from scratch! 
        You now understand:
        <ul>
            <li>✅ How to calculate slope and intercept using least squares</li>
            <li>✅ How to make predictions with the fitted model</li>
            <li>✅ How to evaluate model performance with R² and MSE</li>
            <li>✅ How to visualize regression results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Linear Regression from Scratch - Mini Project 2</strong></p>
        <p>Understanding Linear Algebra through Interactive Learning</p>
        <p>Built with ❤️ using Streamlit | No ML Libraries Used (Only NumPy for arrays)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
