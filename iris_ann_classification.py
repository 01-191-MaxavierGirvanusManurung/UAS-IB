"""
Eksplorasi dan Klasifikasi Dataset Iris menggunakan Artificial Neural Network (ANN)
Dataset: Iris Flower Dataset
Tipe Klasifikasi: Binary-Class dan Multi-Class
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IrisANNClassifier:
    """
    Class untuk eksplorasi dan klasifikasi dataset Iris menggunakan ANN
    """
    
    def __init__(self, data_path='IRIS.csv'):
        """
        Initialize classifier dengan path ke dataset
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_explore_data(self):
        """
        Load dataset dan lakukan eksplorasi data awal
        """
        print("="*70)
        print("LOADING DAN EKSPLORASI DATA")
        print("="*70)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        print("\n1. Informasi Dataset:")
        print(f"   - Jumlah baris: {self.df.shape[0]}")
        print(f"   - Jumlah kolom: {self.df.shape[1]}")
        print(f"   - Kolom: {list(self.df.columns)}")
        
        print("\n2. Lima baris pertama dataset:")
        print(self.df.head())
        
        print("\n3. Informasi tipe data:")
        print(self.df.info())
        
        print("\n4. Statistik deskriptif:")
        print(self.df.describe())
        
        print("\n5. Distribusi kelas:")
        print(self.df['species'].value_counts())
        
        print("\n6. Cek missing values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def visualize_data(self):
        """
        Visualisasi dataset untuk memahami distribusi dan korelasi
        """
        print("\n" + "="*70)
        print("VISUALISASI DATA")
        print("="*70)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Distribusi fitur
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribusi Fitur-fitur Dataset Iris', fontsize=16, fontweight='bold')
        
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
        
        for idx, (feature, color) in enumerate(zip(features, colors)):
            ax = axes[idx//2, idx%2]
            self.df[feature].hist(bins=20, ax=ax, color=color, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribusi {feature}', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frekuensi')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('iris_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Grafik distribusi fitur disimpan sebagai 'iris_distribution.png'")
        
        # 2. Boxplot untuk melihat outlier
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Boxplot Fitur-fitur Dataset Iris', fontsize=16, fontweight='bold')
        
        for idx, (feature, color) in enumerate(zip(features, colors)):
            ax = axes[idx//2, idx%2]
            self.df.boxplot(column=feature, by='species', ax=ax)
            ax.set_title(f'Boxplot {feature} per Species')
            ax.set_xlabel('Species')
            ax.set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig('iris_boxplot.png', dpi=300, bbox_inches='tight')
        print("✓ Grafik boxplot disimpan sebagai 'iris_boxplot.png'")
        
        # 3. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap Fitur-fitur Iris', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap disimpan sebagai 'iris_correlation.png'")
        
        # 4. Pairplot
        plt.figure(figsize=(12, 10))
        pairplot_fig = sns.pairplot(self.df, hue='species', 
                                    diag_kind='kde', 
                                    markers=['o', 's', 'D'],
                                    palette='Set2')
        pairplot_fig.fig.suptitle('Pairplot Dataset Iris', y=1.02, fontsize=16, fontweight='bold')
        plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
        print("✓ Pairplot disimpan sebagai 'iris_pairplot.png'")
        
        plt.close('all')
        print("\n✓ Semua visualisasi berhasil dibuat!")
    
    def prepare_data_multiclass(self, test_size=0.2):
        """
        Prepare data untuk multi-class classification (3 kelas)
        """
        print("\n" + "="*70)
        print("PERSIAPAN DATA: MULTI-CLASS CLASSIFICATION")
        print("="*70)
        
        # Pisahkan fitur dan target
        self.X = self.df.drop('species', axis=1).values
        y_encoded = self.label_encoder.fit_transform(self.df['species'])
        self.y = to_categorical(y_encoded, num_classes=3)
        
        print(f"\n✓ Jumlah kelas: 3 (Iris-setosa, Iris-versicolor, Iris-virginica)")
        print(f"✓ Shape fitur (X): {self.X.shape}")
        print(f"✓ Shape target (y): {self.y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Standardize fitur
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Data training: {X_train_scaled.shape[0]} samples")
        print(f"✓ Data testing: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def prepare_data_binary(self, test_size=0.2):
        """
        Prepare data untuk binary classification (Iris-setosa vs Non-setosa)
        """
        print("\n" + "="*70)
        print("PERSIAPAN DATA: BINARY CLASSIFICATION")
        print("="*70)
        print("(Iris-setosa vs Non-setosa)")
        
        # Pisahkan fitur dan target
        self.X = self.df.drop('species', axis=1).values
        # Binary: 1 jika Iris-setosa, 0 jika lainnya
        y_binary = (self.df['species'] == 'Iris-setosa').astype(int).values
        
        print(f"\n✓ Kelas 1 (Iris-setosa): {np.sum(y_binary)} samples")
        print(f"✓ Kelas 0 (Non-setosa): {len(y_binary) - np.sum(y_binary)} samples")
        print(f"✓ Shape fitur (X): {self.X.shape}")
        print(f"✓ Shape target (y): {y_binary.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        # Standardize fitur
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Data training: {X_train_scaled.shape[0]} samples")
        print(f"✓ Data testing: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_ann_multiclass(self, input_dim=4, hidden_layers=[16, 8], dropout_rate=0.2):
        """
        Build ANN model untuk multi-class classification
        
        Args:
            input_dim: Jumlah fitur input
            hidden_layers: List jumlah neuron di setiap hidden layer
            dropout_rate: Dropout rate untuk regularisasi
        """
        print("\n" + "="*70)
        print("MEMBANGUN MODEL ANN: MULTI-CLASS CLASSIFICATION")
        print("="*70)
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for idx, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', 
                                  name=f'hidden_layer_{idx+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{idx+1}'))
        
        # Output layer (3 kelas dengan softmax)
        model.add(layers.Dense(3, activation='softmax', name='output_layer'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n✓ Model Multi-class ANN berhasil dibuat!")
        print("\nArsitektur Model:")
        model.summary()
        
        return model
    
    def build_ann_binary(self, input_dim=4, hidden_layers=[12, 6], dropout_rate=0.2):
        """
        Build ANN model untuk binary classification
        
        Args:
            input_dim: Jumlah fitur input
            hidden_layers: List jumlah neuron di setiap hidden layer
            dropout_rate: Dropout rate untuk regularisasi
        """
        print("\n" + "="*70)
        print("MEMBANGUN MODEL ANN: BINARY CLASSIFICATION")
        print("="*70)
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for idx, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', 
                                  name=f'hidden_layer_{idx+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{idx+1}'))
        
        # Output layer (1 neuron dengan sigmoid untuk binary)
        model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n✓ Model Binary ANN berhasil dibuat!")
        print("\nArsitektur Model:")
        model.summary()
        
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, 
                   epochs=100, batch_size=16, verbose=1):
        """
        Train ANN model dengan early stopping dan model checkpoint
        """
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        print("\n✓ Training selesai!")
        
        return history
    
    def plot_training_history(self, history, title_prefix=''):
        """
        Plot training history (loss dan accuracy)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{title_prefix} Training History', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{title_prefix.lower().replace(" ", "_")}_training_history.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history plot disimpan sebagai '{filename}'")
        plt.close()
    
    def evaluate_model(self, model, X_test, y_test, model_type='multiclass'):
        """
        Evaluate model dan tampilkan metrik evaluasi
        """
        print("\n" + "="*70)
        print("EVALUASI MODEL")
        print("="*70)
        
        # Prediksi
        y_pred_proba = model.predict(X_test, verbose=0)
        
        if model_type == 'multiclass':
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        else:  # binary
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = y_test
            target_names = ['Non-setosa', 'Iris-setosa']
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_type.capitalize()} Classification', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.tight_layout()
        filename = f'{model_type}_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix disimpan sebagai '{filename}'")
        plt.close()
        
        return accuracy, y_pred, y_true
    
    def compare_architectures_multiclass(self, X_train, X_test, y_train, y_test):
        """
        Eksplorasi berbagai arsitektur ANN untuk multi-class classification
        """
        print("\n" + "="*70)
        print("EKSPLORASI BERBAGAI ARSITEKTUR ANN (MULTI-CLASS)")
        print("="*70)
        
        architectures = [
            {'name': 'Shallow (1 layer)', 'layers': [8], 'dropout': 0.2},
            {'name': 'Medium (2 layers)', 'layers': [16, 8], 'dropout': 0.2},
            {'name': 'Deep (3 layers)', 'layers': [32, 16, 8], 'dropout': 0.3},
            {'name': 'Wide (2 layers)', 'layers': [64, 32], 'dropout': 0.3},
        ]
        
        results = []
        
        for arch in architectures:
            print(f"\n{'='*70}")
            print(f"Testing: {arch['name']}")
            print(f"Layers: {arch['layers']}, Dropout: {arch['dropout']}")
            print(f"{'='*70}")
            
            # Build model
            model = self.build_ann_multiclass(
                hidden_layers=arch['layers'],
                dropout_rate=arch['dropout']
            )
            
            # Train model
            history = self.train_model(
                model, X_train, y_train, X_test, y_test,
                epochs=100, batch_size=16, verbose=0
            )
            
            # Evaluate
            accuracy, _, _ = self.evaluate_model(model, X_test, y_test, 'multiclass')
            
            results.append({
                'Architecture': arch['name'],
                'Layers': str(arch['layers']),
                'Dropout': arch['dropout'],
                'Accuracy': accuracy,
                'Final Train Loss': history.history['loss'][-1],
                'Final Val Loss': history.history['val_loss'][-1]
            })
        
        # Summary
        results_df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("SUMMARY PERBANDINGAN ARSITEKTUR")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('multiclass_architecture_comparison.csv', index=False)
        print("\n✓ Hasil perbandingan disimpan ke 'multiclass_architecture_comparison.csv'")
        
        return results_df


def main():
    """
    Main function untuk menjalankan seluruh pipeline
    """
    print("\n" + "="*70)
    print("IRIS FLOWER CLASSIFICATION MENGGUNAKAN ARTIFICIAL NEURAL NETWORK")
    print("="*70)
    print("Dataset: IRIS.csv")
    print("Metode: Binary Classification & Multi-Class Classification")
    print("="*70)
    
    # Initialize classifier
    classifier = IrisANNClassifier(data_path='IRIS.csv')
    
    # 1. Load dan eksplorasi data
    classifier.load_and_explore_data()
    
    # 2. Visualisasi data
    classifier.visualize_data()
    
    # ========================================================================
    # MULTI-CLASS CLASSIFICATION (3 kelas)
    # ========================================================================
    print("\n\n" + "="*70)
    print("PART 1: MULTI-CLASS CLASSIFICATION (3 KELAS)")
    print("="*70)
    
    # Prepare data
    X_train_mc, X_test_mc, y_train_mc, y_test_mc = classifier.prepare_data_multiclass()
    
    # Build model
    model_mc = classifier.build_ann_multiclass(hidden_layers=[16, 8], dropout_rate=0.2)
    
    # Train model
    history_mc = classifier.train_model(
        model_mc, X_train_mc, y_train_mc, X_test_mc, y_test_mc,
        epochs=150, batch_size=16
    )
    
    # Plot training history
    classifier.plot_training_history(history_mc, 'Multi-Class')
    
    # Evaluate model
    acc_mc, _, _ = classifier.evaluate_model(model_mc, X_test_mc, y_test_mc, 'multiclass')
    
    # Eksplorasi arsitektur
    print("\n\nMelakukan eksplorasi berbagai arsitektur ANN...")
    results_mc = classifier.compare_architectures_multiclass(
        X_train_mc, X_test_mc, y_train_mc, y_test_mc
    )
    
    # ========================================================================
    # BINARY CLASSIFICATION (Iris-setosa vs Non-setosa)
    # ========================================================================
    print("\n\n" + "="*70)
    print("PART 2: BINARY CLASSIFICATION (IRIS-SETOSA VS NON-SETOSA)")
    print("="*70)
    
    # Prepare data
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = classifier.prepare_data_binary()
    
    # Build model
    model_bc = classifier.build_ann_binary(hidden_layers=[12, 6], dropout_rate=0.2)
    
    # Train model
    history_bc = classifier.train_model(
        model_bc, X_train_bc, y_train_bc, X_test_bc, y_test_bc,
        epochs=150, batch_size=16
    )
    
    # Plot training history
    classifier.plot_training_history(history_bc, 'Binary')
    
    # Evaluate model
    acc_bc, _, _ = classifier.evaluate_model(model_bc, X_test_bc, y_test_bc, 'binary')
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "="*70)
    print("RINGKASAN HASIL AKHIR")
    print("="*70)
    print(f"\n✓ Multi-Class Classification Accuracy: {acc_mc:.4f} ({acc_mc*100:.2f}%)")
    print(f"✓ Binary Classification Accuracy: {acc_bc:.4f} ({acc_bc*100:.2f}%)")
    
    print("\n✓ Semua file hasil telah disimpan:")
    print("  - iris_distribution.png")
    print("  - iris_boxplot.png")
    print("  - iris_correlation.png")
    print("  - iris_pairplot.png")
    print("  - multi-class_training_history.png")
    print("  - multiclass_confusion_matrix.png")
    print("  - multiclass_architecture_comparison.csv")
    print("  - binary_training_history.png")
    print("  - binary_confusion_matrix.png")
    
    print("\n" + "="*70)
    print("PROGRAM SELESAI!")
    print("="*70)


if __name__ == "__main__":
    main()
