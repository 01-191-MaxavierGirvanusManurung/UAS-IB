# Eksplorasi Dataset Iris dengan Artificial Neural Network (ANN)

## Deskripsi
Project ini melakukan eksplorasi dan klasifikasi pada dataset Iris Flower menggunakan Artificial Neural Network (ANN). Program ini mendukung **dua jenis klasifikasi**:
1. **Multi-Class Classification**: Mengklasifikasikan 3 jenis bunga iris (Iris-setosa, Iris-versicolor, Iris-virginica)
2. **Binary Classification**: Mengklasifikasikan Iris-setosa vs Non-setosa

## Fitur Utama

### 1. Eksplorasi Data
- Loading dan analisis data
- Statistik deskriptif
- Distribusi kelas
- Deteksi missing values

### 2. Visualisasi
- Distribusi fitur (histogram)
- Boxplot untuk deteksi outlier
- Correlation heatmap
- Pairplot untuk melihat hubungan antar fitur

### 3. Model ANN

#### Multi-Class Classification
- Input Layer: 4 fitur (sepal_length, sepal_width, petal_length, petal_width)
- Hidden Layers: Customizable (default: 16, 8 neurons)
- Output Layer: 3 neurons dengan aktivasi softmax
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

#### Binary Classification
- Input Layer: 4 fitur
- Hidden Layers: Customizable (default: 12, 6 neurons)
- Output Layer: 1 neuron dengan aktivasi sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam

### 4. Modifikasi Model
- Dropout regularization untuk mencegah overfitting
- Early stopping untuk menghentikan training jika tidak ada improvement
- Learning rate reduction untuk optimasi yang lebih baik
- Eksplorasi berbagai arsitektur (shallow, medium, deep, wide)

### 5. Evaluasi
- Accuracy score
- Classification report (precision, recall, f1-score)
- Confusion matrix
- Training history plots

## Instalasi

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Verifikasi Dataset
Pastikan file `IRIS.csv` ada di folder yang sama dengan script.

## Cara Menjalankan

### Menjalankan Program Lengkap
```powershell
python iris_ann_classification.py
```

Program akan otomatis:
1. Load dan eksplorasi dataset
2. Membuat visualisasi
3. Melatih model Multi-Class Classification
4. Melakukan eksplorasi berbagai arsitektur ANN
5. Melatih model Binary Classification
6. Mengevaluasi kedua model
7. Menyimpan semua hasil

## Output yang Dihasilkan

### File Visualisasi
- `iris_distribution.png` - Distribusi setiap fitur
- `iris_boxplot.png` - Boxplot per species
- `iris_correlation.png` - Correlation heatmap
- `iris_pairplot.png` - Pairplot semua fitur
- `multi-class_training_history.png` - Training history multi-class
- `binary_training_history.png` - Training history binary
- `multiclass_confusion_matrix.png` - Confusion matrix multi-class
- `binary_confusion_matrix.png` - Confusion matrix binary

### File CSV
- `multiclass_architecture_comparison.csv` - Hasil perbandingan berbagai arsitektur

## Struktur Code

```
iris_ann_classification.py
│
├── IrisANNClassifier (Class)
│   ├── load_and_explore_data()       # Load dan eksplorasi data
│   ├── visualize_data()               # Membuat visualisasi
│   ├── prepare_data_multiclass()      # Prepare data untuk multi-class
│   ├── prepare_data_binary()          # Prepare data untuk binary
│   ├── build_ann_multiclass()         # Build model ANN multi-class
│   ├── build_ann_binary()             # Build model ANN binary
│   ├── train_model()                  # Training model dengan callbacks
│   ├── plot_training_history()        # Plot training history
│   ├── evaluate_model()               # Evaluasi model
│   └── compare_architectures_multiclass() # Eksplorasi arsitektur
│
└── main()                             # Main function
```

## Modifikasi yang Dilakukan pada Model ANN

### 1. Regularisasi
- **Dropout Layers**: Menambahkan dropout setelah setiap hidden layer untuk mencegah overfitting
- **Dropout Rate**: Customizable (default 0.2 untuk binary, 0.2-0.3 untuk multi-class)

### 2. Optimasi Training
- **Early Stopping**: Monitoring validation loss dengan patience 15 epochs
- **Reduce Learning Rate**: Mengurangi learning rate saat validation loss plateau
- **Batch Size**: Default 16 (bisa disesuaikan)
- **Epochs**: Maximum 150 (akan stop lebih awal jika konvergen)

### 3. Arsitektur yang Dieksplor
1. **Shallow Network**: 1 hidden layer (8 neurons)
2. **Medium Network**: 2 hidden layers (16, 8 neurons)
3. **Deep Network**: 3 hidden layers (32, 16, 8 neurons)
4. **Wide Network**: 2 hidden layers (64, 32 neurons)

### 4. Preprocessing
- **Standardization**: Menggunakan StandardScaler untuk normalisasi fitur
- **Label Encoding**: Untuk multi-class menggunakan one-hot encoding
- **Stratified Split**: Memastikan distribusi kelas seimbang di train dan test set

## Metrik Evaluasi

### Multi-Class Classification
- Accuracy
- Precision, Recall, F1-Score per class
- Confusion Matrix 3x3

### Binary Classification
- Accuracy
- Precision, Recall, F1-Score per class
- Confusion Matrix 2x2

## Dependencies
- **numpy**: Operasi numerik
- **pandas**: Manipulasi data
- **matplotlib**: Visualisasi
- **seaborn**: Visualisasi statistik
- **scikit-learn**: Preprocessing dan evaluasi
- **tensorflow**: Deep learning framework untuk ANN

## Tips Penggunaan

### Modifikasi Arsitektur
Untuk mencoba arsitektur lain, edit bagian ini di `main()`:
```python
# Multi-class
model_mc = classifier.build_ann_multiclass(
    hidden_layers=[32, 16, 8],  # Ubah jumlah layer dan neurons
    dropout_rate=0.3             # Ubah dropout rate
)

# Binary
model_bc = classifier.build_ann_binary(
    hidden_layers=[24, 12],      # Ubah jumlah layer dan neurons
    dropout_rate=0.25            # Ubah dropout rate
)
```

### Modifikasi Hyperparameters
```python
history = classifier.train_model(
    model, X_train, y_train, X_test, y_test,
    epochs=200,      # Ubah jumlah epochs
    batch_size=32    # Ubah batch size
)
```

## Hasil yang Diharapkan
- Multi-Class Accuracy: ~95-98%
- Binary Classification Accuracy: ~100% (karena Iris-setosa sangat berbeda dari yang lain)

## Troubleshooting

### Import Error
```powershell
pip install --upgrade tensorflow scikit-learn matplotlib seaborn pandas numpy
```

### Memory Error
Kurangi batch size atau ukuran network:
```python
batch_size=8
hidden_layers=[8, 4]
```

## Author
Dibuat untuk eksplorasi klasifikasi Iris Flower Dataset menggunakan ANN

## License
Educational purposes
