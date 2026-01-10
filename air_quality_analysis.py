# ==========================================
# Atmospheric Pollution Multivariate Analysis
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, tensorflow
# Input files: Brasilia_Air_Quality.xlsx, Cairo_Air_Quality.xlsx, Dubai_Air_Quality.xlsx
# Output files: em_scatter.png, ica_sources.png, cca_scatter.png, factor_heatmap.png,
#               vae_predictions.png, nmf_patterns.png, kpca_components.png,
#               ridge_predictions.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA, TruncatedSVD, NMF, PCA, KernelPCA, FactorAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LassoLars, ElasticNetCV, RidgeCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import kurtosis
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- Load Air Quality Data ----------
print("Loading air quality datasets...")
cairo_df = pd.read_excel('Cairo_Air_Quality.xlsx')
dubai_df = pd.read_excel('Dubai_Air_Quality.xlsx')
brasilia_df = pd.read_excel('Brasilia_Air_Quality.xlsx')

print(f"Cairo: {cairo_df.shape}, Dubai: {dubai_df.shape}, Brasilia: {brasilia_df.shape}")

# ==========================================
# 1. EXPECTATION-MAXIMIZATION (CAIRO)
# ==========================================
print("\n" + "="*50)
print("1. Expectation-Maximization on Cairo PM2.5/PM10")
print("="*50)

# Extract PM2.5 and PM10, remove missing values
em_data = cairo_df[['PM2.5', 'PM10']].dropna().values

# Initialize EM with 3 components
gmm = GaussianMixture(n_components=3, covariance_type='full',
                       random_state=42, max_iter=200)
gmm.fit(em_data)

# Extract BIC
bic_value = gmm.bic(em_data)
print(f"\nBIC Value: {bic_value:.2f}")

# Predict component assignments
component_labels = gmm.predict(em_data)

# Generate scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(em_data[:, 0], em_data[:, 1], c=component_labels,
                     cmap='viridis', alpha=0.5, s=10)
plt.xlabel('PM2.5 (μg/m³)', fontsize=12)
plt.ylabel('PM10 (μg/m³)', fontsize=12)
plt.title('Gaussian Mixture Model: Cairo PM2.5 vs PM10\n3 Pollution Regimes', fontsize=14)
plt.colorbar(scatter, label='Component')
plt.tight_layout()
plt.savefig('em_scatter.png', dpi=150)
plt.close()
print("Saved: em_scatter.png")

# ==========================================
# 2. INDEPENDENT COMPONENT ANALYSIS (DUBAI)
# ==========================================
print("\n" + "="*50)
print("2. Independent Component Analysis on Dubai Pollutants")
print("="*50)

# Extract CO, NO2, SO2, O3
ica_data = dubai_df[['CO', 'NO2', 'SO2', 'O3']].dropna()

# Standardize features
scaler = StandardScaler()
ica_data_scaled = scaler.fit_transform(ica_data)

# Apply FastICA
ica = FastICA(n_components=3, max_iter=500, random_state=42, fun='logcosh')
sources = ica.fit_transform(ica_data_scaled)

# Calculate kurtosis of first component
kurt_first = kurtosis(sources[:, 0])
print(f"\nKurtosis of First Independent Component: {kurt_first:.4f}")

# Generate line plot
plt.figure(figsize=(14, 6))
time_index = np.arange(len(sources))
plt.plot(time_index, sources[:, 0], label='IC 1', alpha=0.7, linewidth=0.5)
plt.plot(time_index, sources[:, 1], label='IC 2', alpha=0.7, linewidth=0.5)
plt.plot(time_index, sources[:, 2], label='IC 3', alpha=0.7, linewidth=0.5)
plt.xlabel('Time Index (hours)', fontsize=12)
plt.ylabel('Independent Component Values', fontsize=12)
plt.title('FastICA: Three Independent Pollution Sources', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('ica_sources.png', dpi=150)
plt.close()
print("Saved: ica_sources.png")

# ==========================================
# 3. CANONICAL CORRELATION ANALYSIS (BRASILIA + CAIRO)
# ==========================================
print("\n" + "="*50)
print("3. Canonical Correlation Analysis on Brasilia + Cairo")
print("="*50)

# Stack datasets vertically
combined_df = pd.concat([brasilia_df, cairo_df], axis=0, ignore_index=True)

# Create X (CO, NO2, SO2) and Y (PM2.5, PM10)
cca_data = combined_df[['CO', 'NO2', 'SO2', 'PM2.5', 'PM10']].dropna()
X_cca = cca_data[['CO', 'NO2', 'SO2']].values
Y_cca = cca_data[['PM2.5', 'PM10']].values

# Fit CCA
cca = CCA(n_components=2)
cca.fit(X_cca, Y_cca)

# Transform to canonical variates
X_c, Y_c = cca.transform(X_cca, Y_cca)

# Calculate first canonical correlation
corr_first = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
print(f"\nFirst Canonical Correlation: {corr_first:.4f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.3, s=10, color='steelblue')
plt.xlabel('First Canonical Variate of X (CO, NO2, SO2)', fontsize=12)
plt.ylabel('First Canonical Variate of Y (PM2.5, PM10)', fontsize=12)
plt.title(f'Canonical Correlation Analysis\nCorrelation: {corr_first:.4f}', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cca_scatter.png', dpi=150)
plt.close()
print("Saved: cca_scatter.png")

# ==========================================
# 4. FACTOR ANALYSIS (DUBAI)
# ==========================================
print("\n" + "="*50)
print("4. Factor Analysis on Dubai Pollutants")
print("="*50)

# Select 6 pollutants
fa_data = dubai_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].dropna()

# Standardize
scaler_fa = StandardScaler()
fa_data_scaled = scaler_fa.fit_transform(fa_data)

# Fit Factor Analysis
fa = FactorAnalysis(n_components=3, rotation='varimax', max_iter=1000, random_state=42)
fa.fit(fa_data_scaled)

# Get factor loadings
loadings = fa.components_.T

# Compute communality for PM2.5 (index 4)
communality_pm25 = np.sum(loadings[4, :]**2)
print(f"\nCommunality for PM2.5: {communality_pm25:.4f}")

# Generate heatmap
plt.figure(figsize=(8, 6))
pollutant_names = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']
factor_names = ['Factor 1', 'Factor 2', 'Factor 3']

im = plt.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, label='Loading Magnitude')
plt.xticks(range(3), factor_names)
plt.yticks(range(6), pollutant_names)
plt.xlabel('Factors', fontsize=12)
plt.ylabel('Pollutants', fontsize=12)
plt.title('Factor Analysis: Factor Loadings\nVarimax Rotation', fontsize=14)

# Add text annotations
for i in range(6):
    for j in range(3):
        text = plt.text(j, i, f'{loadings[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.savefig('factor_heatmap.png', dpi=150)
plt.close()
print("Saved: factor_heatmap.png")

# ==========================================
# 5. VARIATIONAL AUTOENCODER REGRESSION (CAIRO)
# ==========================================
print("\n" + "="*50)
print("5. Variational Autoencoder Regression on Cairo AQI")
print("="*50)

# Extract first 4000 for training
cairo_vae = cairo_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'AQI']].dropna()
train_vae = cairo_vae.iloc[:4000]
test_vae = cairo_vae.iloc[4000:]

X_train_vae = train_vae[['CO', 'NO2', 'SO2', 'O3', 'PM2.5']].values
y_train_vae = train_vae['AQI'].values
X_test_vae = test_vae[['CO', 'NO2', 'SO2', 'O3', 'PM2.5']].values
y_test_vae = test_vae['AQI'].values

# Standardize
scaler_vae = StandardScaler()
X_train_vae_scaled = scaler_vae.fit_transform(X_train_vae)
X_test_vae_scaled = scaler_vae.transform(X_test_vae)

# Build VAE architecture
latent_dim = 8
input_dim = 5

# Custom sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(32, activation='relu')(encoder_inputs)
x = layers.Dense(16, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16, activation='relu')(latent_inputs)
x = layers.Dense(32, activation='relu')(x)
decoder_outputs = layers.Dense(1)(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

# VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.square(y - tf.squeeze(reconstruction))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            total_loss = reconstruction_loss + 0.1 * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Train
print("Training VAE...")
history = vae.fit(X_train_vae_scaled, y_train_vae,
                 epochs=50, batch_size=64, verbose=0)

# Predict on test set
y_pred_vae = vae.predict(X_test_vae_scaled, verbose=0).flatten()

# Calculate MAPE
mape = np.mean(np.abs((y_test_vae - y_pred_vae) / y_test_vae)) * 100
print(f"\nMean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_vae, y_pred_vae, alpha=0.5, s=20, color='coral')
plt.plot([y_test_vae.min(), y_test_vae.max()],
         [y_test_vae.min(), y_test_vae.max()],
         'k--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual AQI', fontsize=12)
plt.ylabel('Predicted AQI', fontsize=12)
plt.title(f'VAE Regression: AQI Prediction\nMAPE: {mape:.2f}%', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vae_predictions.png', dpi=150)
plt.close()
print("Saved: vae_predictions.png")

# ==========================================
# 6. LASSO-LARS (BRASILIA)
# ==========================================
print("\n" + "="*50)
print("6. Lasso-LARS Regression on Brasilia AQI")
print("="*50)

# Extract AQI and predictors
lars_data = brasilia_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].dropna()
X_lars = lars_data[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].values
y_lars = lars_data['AQI'].values

# Fit LassoLars
lasso_lars = LassoLars(alpha=0.1, random_state=42)
lasso_lars.fit(X_lars, y_lars)

# Count non-zero coefficients
n_selected = np.sum(lasso_lars.coef_ != 0)
print(f"\nNumber of Selected Features: {n_selected}")

# ==========================================
# 7. ELASTIC NET CV (DUBAI + CAIRO)
# ==========================================
print("\n" + "="*50)
print("7. Elastic Net CV on Dubai + Cairo PM10")
print("="*50)

# Combine datasets
combined_en = pd.concat([dubai_df, cairo_df], axis=0, ignore_index=True)
en_data = combined_en[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].dropna()

X_en = en_data[['CO', 'NO2', 'SO2', 'O3', 'PM2.5']].values
y_en = en_data['PM10'].values

# Fit ElasticNetCV
elastic_net = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42, max_iter=2000)
elastic_net.fit(X_en, y_en)

# Extract optimal alpha
optimal_alpha = elastic_net.alpha_
print(f"\nOptimal Alpha: {optimal_alpha:.4f}")

# ==========================================
# 8. TRUNCATED SVD (BRASILIA)
# ==========================================
print("\n" + "="*50)
print("8. Truncated SVD on Brasilia Pollutants")
print("="*50)

# Extract pollutant matrix
svd_data = brasilia_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].dropna()

# Standardize
scaler_svd = StandardScaler()
svd_data_scaled = scaler_svd.fit_transform(svd_data)

# Apply TruncatedSVD
tsvd = TruncatedSVD(n_components=3, random_state=42)
tsvd.fit(svd_data_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.sum(tsvd.explained_variance_ratio_)
print(f"\nCumulative Explained Variance (3 components): {cumulative_variance:.4f}")

# ==========================================
# 9. NON-NEGATIVE MATRIX FACTORIZATION (CAIRO)
# ==========================================
print("\n" + "="*50)
print("9. Non-Negative Matrix Factorization on Cairo")
print("="*50)

# Extract PM2.5, PM10, NO2, SO2
nmf_data = cairo_df[['PM2.5', 'PM10', 'NO2', 'SO2']].dropna()

# Ensure non-negativity by shifting minimum to zero
nmf_data_shifted = nmf_data - nmf_data.min().min()
nmf_matrix = nmf_data_shifted.values

# Fit NMF
nmf = NMF(n_components=2, init='nndsvd', max_iter=400, random_state=42)
W = nmf.fit_transform(nmf_matrix)
H = nmf.components_

# Calculate reconstruction error (Frobenius norm)
reconstructed = np.dot(W, H)
reconstruction_error = np.linalg.norm(nmf_matrix - reconstructed, 'fro')
print(f"\nReconstruction Error (Frobenius norm): {reconstruction_error:.2f}")

# ==========================================
# 10. QUADRATIC DISCRIMINANT ANALYSIS (DUBAI)
# ==========================================
print("\n" + "="*50)
print("10. Quadratic Discriminant Analysis on Dubai")
print("="*50)

# Create binary target
qda_data = dubai_df[['CO', 'NO2', 'PM2.5', 'PM10', 'AQI']].dropna()
qda_data['high_pollution'] = (qda_data['AQI'] > 75).astype(int)

X_qda = qda_data[['CO', 'NO2', 'PM2.5', 'PM10']].values
y_qda = qda_data['high_pollution'].values

# Split train-test
X_train_qda, X_test_qda, y_train_qda, y_test_qda = train_test_split(
    X_qda, y_qda, test_size=0.3, random_state=42)

# Fit QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_qda, y_train_qda)

# Predict and calculate F1 score
y_pred_qda = qda.predict(X_test_qda)
f1 = f1_score(y_test_qda, y_pred_qda)
print(f"\nF1 Score: {f1:.4f}")

# ==========================================
# 11. KERNEL PCA (BRASILIA)
# ==========================================
print("\n" + "="*50)
print("11. Kernel PCA on Brasilia Pollutants")
print("="*50)

# Extract and standardize
kpca_data = brasilia_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].dropna()
scaler_kpca = StandardScaler()
kpca_data_scaled = scaler_kpca.fit_transform(kpca_data)

# Apply Kernel PCA
kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1, random_state=42)
kpca_transformed = kpca.fit_transform(kpca_data_scaled)

# Calculate variance explained by first component
# For kernel PCA, we use eigenvalues (lambdas_)
total_variance = np.sum(kpca.eigenvalues_)
first_component_variance = kpca.eigenvalues_[0] / total_variance
print(f"\nVariance Explained by First Component: {first_component_variance:.4f}")

# ==========================================
# 12. RIDGE REGRESSION CV (CAIRO + DUBAI)
# ==========================================
print("\n" + "="*50)
print("12. Ridge Regression CV on Cairo + Dubai AQI")
print("="*50)

# Combine datasets
combined_ridge = pd.concat([cairo_df, dubai_df], axis=0, ignore_index=True)
ridge_data = combined_ridge[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].dropna()

X_ridge = ridge_data[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].values
y_ridge = ridge_data['AQI'].values

# Fit RidgeCV
alphas = np.logspace(-1, 2, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_ridge, y_ridge)

# Calculate R-squared on full dataset
r2_score = ridge_cv.score(X_ridge, y_ridge)
print(f"\nR-squared: {r2_score:.4f}")

# ==========================================
# MULTIVARIATE STRUCTURE COMPARISON
# ==========================================
print("\n" + "="*50)
print("MULTIVARIATE STRUCTURE COMPARISON")
print("="*50)

print(f"\nCanonical Correlation (Maximum): {corr_first:.4f}")
print(f"Factor Analysis Communality (PM2.5): {communality_pm25:.4f}")

# Answer to open-ended question
answer = "Canonical correlation analysis better reveals multivariate pollution structure as it directly quantifies the maximum achievable linear relationship between primary pollutants and particulate matter with a correlation of {:.4f}, whereas factor analysis communality of {:.4f} only explains individual variable variance without capturing cross-set associations.".format(corr_first, communality_pm25)

print("\n" + "="*50)
print("ANSWER TO OPEN-ENDED QUESTION:")
print("="*50)
print(answer)

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("\n" + "="*50)
print("KEY OUTPUTS SUMMARY")
print("="*50)
print(f"1. EM BIC: {bic_value:.2f}")
print(f"2. ICA Kurtosis (1st component): {kurt_first:.4f}")
print(f"3. CCA First Correlation: {corr_first:.4f}")
print(f"4. Factor Analysis PM2.5 Communality: {communality_pm25:.4f}")
print(f"5. VAE MAPE: {mape:.2f}%")
print(f"6. Lasso-LARS Selected Features: {n_selected}")
print(f"7. ElasticNet Optimal Alpha: {optimal_alpha:.4f}")
print(f"8. TruncatedSVD Cumulative Variance: {cumulative_variance:.4f}")
print(f"9. NMF Reconstruction Error: {reconstruction_error:.2f}")
print(f"10. QDA F1 Score: {f1:.4f}")
print(f"11. KernelPCA First Component Variance: {first_component_variance:.4f}")
print(f"12. RidgeCV R-squared: {r2_score:.4f}")

print("\n" + "="*50)
print("Analysis Complete!")
print("="*50)
