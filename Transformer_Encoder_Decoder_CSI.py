"""
Transformer Encodeur-Décodeur pour données CSI (Channel State Information)
Application: Reconnaissance d'Activité Humaine (HAR)

Ce modèle utilise une architecture Transformer complète avec:
- Encodeur: Traite les séquences CSI d'entrée
- Décodeur: Génère les prédictions de classification

Auteur: Assistant IA
Date: Janvier 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration et Hyperparamètres
# =============================================================================

class Config:
    """Configuration du modèle et de l'entraînement"""
    # Données
    SEQUENCE_LENGTH = 300  # Longueur des séquences temporelles
    NUM_FEATURES = 52      # Nombre de sous-porteuses CSI (à ajuster selon vos données)
    NUM_CLASSES = 7        # Nombre de classes d'activités
    
    # Architecture Transformer
    D_MODEL = 128          # Dimension du modèle
    NUM_HEADS = 8          # Nombre de têtes d'attention
    NUM_ENCODER_LAYERS = 4 # Nombre de couches encodeur
    NUM_DECODER_LAYERS = 4 # Nombre de couches décodeur
    DFF = 512              # Dimension du feed-forward
    DROPOUT_RATE = 0.1     # Taux de dropout
    
    # Entraînement
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 4000
    
    # Classes d'activités
    ACTIVITIES = ['bend', 'fall', 'lie down', 'run', 'sitdown', 'standup', 'walk']


# =============================================================================
# Encodage Positionnel
# =============================================================================

class PositionalEncoding(layers.Layer):
    """
    Encodage positionnel sinusoïdal pour le Transformer.
    Ajoute des informations sur la position des éléments dans la séquence.
    """
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Créer l'encodage positionnel
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.positional_encoding = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.positional_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config


# =============================================================================
# Couche Multi-Head Attention
# =============================================================================

class MultiHeadAttention(layers.Layer):
    """
    Couche d'attention multi-têtes.
    Permet au modèle de se concentrer sur différentes parties de la séquence.
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Divise la dernière dimension en (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calcule l'attention par produit scalaire"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Mise à l'échelle
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Appliquer le masque si présent
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax pour obtenir les poids d'attention
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForwardNetwork(layers.Layer):
    """Réseau feed-forward avec deux couches denses"""
    def __init__(self, d_model, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dense1 = layers.Dense(dff, activation='gelu')
        self.dense2 = layers.Dense(d_model)
    
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff
        })
        return config


# =============================================================================
# Couche Encodeur
# =============================================================================

class EncoderLayer(layers.Layer):
    """
    Une couche de l'encodeur Transformer.
    Contient: Self-Attention + Feed-Forward + Normalisation + Connexions résiduelles
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        # Self-attention avec connexion résiduelle et normalisation
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward avec connexion résiduelle et normalisation
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# Couche Décodeur
# =============================================================================

class DecoderLayer(layers.Layer):
    """
    Une couche du décodeur Transformer.
    Contient: Self-Attention + Cross-Attention + Feed-Forward
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-attention
        self.ffn = FeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        # Self-attention du décodeur (avec masque look-ahead pour l'auto-régression)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        # Cross-attention (requête du décodeur, clé/valeur de l'encodeur)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed-forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3, attn_weights_block1, attn_weights_block2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# Encodeur Complet
# =============================================================================

class Encoder(layers.Layer):
    """
    Encodeur complet du Transformer.
    Transforme les séquences CSI en représentations latentes.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 input_dim, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_dim = input_dim
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        
        # Projection linéaire des features CSI vers d_model
        self.input_projection = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate) 
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        # x shape: (batch_size, seq_len, num_features)
        
        # Projection vers d_model dimensions
        x = self.input_projection(x)
        
        # Mise à l'échelle par sqrt(d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Ajout de l'encodage positionnel
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Passage à travers les couches de l'encodeur
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        
        return x  # (batch_size, seq_len, d_model)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'input_dim': self.input_dim,
            'max_len': self.max_len,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# Décodeur Complet
# =============================================================================

class Decoder(layers.Layer):
    """
    Décodeur complet du Transformer.
    Utilise les représentations de l'encodeur pour générer les prédictions.
    """
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_dim, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.target_dim = target_dim
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        
        self.target_embedding = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        attention_weights = {}
        
        # Embedding et encodage positionnel
        x = self.target_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Passage à travers les couches du décodeur
        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(
                x, enc_output, 
                training=training, 
                look_ahead_mask=look_ahead_mask, 
                padding_mask=padding_mask
            )
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        return x, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'target_dim': self.target_dim,
            'max_len': self.max_len,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# Modèle Transformer Encodeur-Décodeur Complet
# =============================================================================

class TransformerEncoderDecoder(keras.Model):
    """
    Modèle Transformer Encodeur-Décodeur complet pour la classification CSI.
    
    Pour la classification d'activités, on utilise:
    - L'encodeur pour traiter les séquences CSI
    - Le décodeur pour raffiner les représentations
    - Une tête de classification pour la prédiction finale
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Encodeur
        self.encoder = Encoder(
            num_layers=config.NUM_ENCODER_LAYERS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dff=config.DFF,
            input_dim=config.NUM_FEATURES,
            max_len=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE
        )
        
        # Décodeur
        self.decoder = Decoder(
            num_layers=config.NUM_DECODER_LAYERS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dff=config.DFF,
            target_dim=config.D_MODEL,
            max_len=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE
        )
        
        # Token de requête appris pour la classification (style BERT [CLS])
        self.query_token = self.add_weight(
            name='query_token',
            shape=(1, 1, config.D_MODEL),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Tête de classification
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = keras.Sequential([
            layers.Dense(config.DFF, activation='gelu'),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(config.DFF // 2, activation='gelu'),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(config.NUM_CLASSES, activation='softmax')
        ])
    
    def create_masks(self, seq_len):
        """Crée les masques pour le décodeur"""
        # Masque look-ahead pour empêcher de regarder les positions futures
        look_ahead_mask = 1 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        return look_ahead_mask
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, seq_len, num_features)
        batch_size = tf.shape(inputs)[0]
        
        # Passage à travers l'encodeur
        enc_output = self.encoder(inputs, training=training)
        # enc_output shape: (batch_size, seq_len, d_model)
        
        # Utiliser le token de requête répété pour chaque échantillon du batch
        # comme entrée initiale du décodeur
        query_tokens = tf.tile(self.query_token, [batch_size, 1, 1])
        
        # Pour la classification, on peut utiliser enc_output comme entrée du décodeur
        # avec le mécanisme de cross-attention
        dec_output, attention_weights = self.decoder(
            query_tokens, enc_output, training=training
        )
        # dec_output shape: (batch_size, 1, d_model)
        
        # Combinaison des sorties encodeur et décodeur
        # Option 1: Utiliser uniquement la sortie du décodeur
        # combined = tf.squeeze(dec_output, axis=1)
        
        # Option 2: Pooling sur l'encodeur + sortie décodeur (meilleure performance)
        enc_pooled = self.global_pool(enc_output)
        dec_squeezed = tf.squeeze(dec_output, axis=1)
        combined = enc_pooled + dec_squeezed
        
        # Classification finale
        output = self.classifier(combined)
        
        return output
    
    def get_attention_weights(self, inputs):
        """Récupère les poids d'attention pour la visualisation"""
        batch_size = tf.shape(inputs)[0]
        enc_output = self.encoder(inputs, training=False)
        query_tokens = tf.tile(self.query_token, [batch_size, 1, 1])
        _, attention_weights = self.decoder(query_tokens, enc_output, training=False)
        return attention_weights
    
    def get_config(self):
        """Retourne la configuration pour la sérialisation"""
        return {
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'num_features': self.config.NUM_FEATURES,
            'num_classes': self.config.NUM_CLASSES,
            'd_model': self.config.D_MODEL,
            'num_heads': self.config.NUM_HEADS,
            'num_encoder_layers': self.config.NUM_ENCODER_LAYERS,
            'num_decoder_layers': self.config.NUM_DECODER_LAYERS,
            'dff': self.config.DFF,
            'dropout_rate': self.config.DROPOUT_RATE,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        """Reconstruit le modèle depuis la configuration"""
        config = Config()
        config.SEQUENCE_LENGTH = config_dict['sequence_length']
        config.NUM_FEATURES = config_dict['num_features']
        config.NUM_CLASSES = config_dict['num_classes']
        config.D_MODEL = config_dict['d_model']
        config.NUM_HEADS = config_dict['num_heads']
        config.NUM_ENCODER_LAYERS = config_dict['num_encoder_layers']
        config.NUM_DECODER_LAYERS = config_dict['num_decoder_layers']
        config.DFF = config_dict['dff']
        config.DROPOUT_RATE = config_dict['dropout_rate']
        return cls(config)


# =============================================================================
# Scheduler de Learning Rate avec Warmup
# =============================================================================

class TransformerLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule avec warmup comme dans le papier "Attention is All You Need"
    """
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            'd_model': int(self.d_model.numpy()),
            'warmup_steps': self.warmup_steps
        }


# =============================================================================
# Chargement et Prétraitement des Données CSI
# =============================================================================

def load_csi_data(data_dir, activities, sequence_length=100):
    """
    Charge les données CSI depuis les fichiers .npz ou .csv
    
    Args:
        data_dir: Répertoire contenant les données
        activities: Liste des activités à charger
        sequence_length: Longueur des séquences à extraire
    
    Returns:
        X: Données CSI (samples, sequence_length, num_features)
        y: Labels des activités
    """
    X_list = []
    y_list = []
    
    for activity in activities:
        # Essayer de charger depuis .npz
        npz_file = os.path.join(data_dir, f'X_{activity}.npz')
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            X_activity = data['arr_0'] if 'arr_0' in data.files else data[data.files[0]]
            
            # S'assurer que les données ont la bonne forme
            if len(X_activity.shape) == 2:
                # Reshape en séquences si nécessaire
                num_samples = X_activity.shape[0] // sequence_length
                X_activity = X_activity[:num_samples * sequence_length].reshape(
                    num_samples, sequence_length, -1
                )
            
            X_list.append(X_activity)
            y_list.extend([activity] * X_activity.shape[0])
            print(f"Chargé {activity}: {X_activity.shape}")
        else:
            # Charger depuis les fichiers CSV
            csv_pattern = os.path.join(data_dir, activity, '*.csv')
            csv_files = glob.glob(csv_pattern)
            
            activity_data = []
            for csv_file in csv_files:
                try:
                    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
                    if len(data) >= sequence_length:
                        # Extraire des fenêtres de la séquence
                        num_windows = len(data) // sequence_length
                        for i in range(num_windows):
                            window = data[i*sequence_length:(i+1)*sequence_length]
                            activity_data.append(window)
                            y_list.append(activity)
                except Exception as e:
                    print(f"Erreur lors du chargement de {csv_file}: {e}")
            
            if activity_data:
                X_list.append(np.stack(activity_data))
    
    # Concaténer tous les arrays le long de l'axe 0
    X = np.concatenate(X_list, axis=0)
    
    # Encoder les labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_list)
    
    return X, y, label_encoder


def preprocess_csi_data(X, normalize=True):
    """
    Prétraite les données CSI
    
    Args:
        X: Données brutes (samples, seq_len, features)
        normalize: Si True, normalise les données
    
    Returns:
        X_processed: Données prétraitées
    """
    X_processed = X.astype(np.float32)
    
    if normalize:
        # Normalisation par échantillon (Z-score)
        mean = np.mean(X_processed, axis=(1, 2), keepdims=True)
        std = np.std(X_processed, axis=(1, 2), keepdims=True) + 1e-8
        X_processed = (X_processed - mean) / std
    
    return X_processed


# =============================================================================
# Entraînement et Évaluation
# =============================================================================

def create_model(config):
    """Crée et compile le modèle"""
    model = TransformerEncoderDecoder(config)
    
    # Learning rate schedule
    lr_schedule = TransformerLRSchedule(config.D_MODEL, config.WARMUP_STEPS)
    
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, config, save_path='transformer_encoder_decoder_best.keras'):
    """Entraîne le modèle avec early stopping et sauvegarde du meilleur modèle"""
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs/transformer_enc_dec',
            histogram_freq=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, label_encoder):
    """Évalue le modèle et affiche les métriques"""
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Rapport de classification
    class_names = label_encoder.classes_
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION")
    print("="*60)
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion - Transformer Encodeur-Décodeur CSI')
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie classe')
    plt.tight_layout()
    plt.savefig('confusion_matrix_transformer_enc_dec.png', dpi=150)
    plt.show()
    
    return y_pred_classes


def plot_training_history(history):
    """Affiche les courbes d'entraînement"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Évolution de la Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Évolution de l\'Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_transformer_enc_dec.png', dpi=150)
    plt.show()


# =============================================================================
# Visualisation des Attentions
# =============================================================================

def visualize_attention(model, X_sample, sample_idx=0):
    """Visualise les poids d'attention pour un échantillon"""
    
    attention_weights = model.get_attention_weights(X_sample[sample_idx:sample_idx+1])
    
    # Visualiser l'attention du décodeur vers l'encodeur (cross-attention)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, (layer_name, attn) in enumerate(list(attention_weights.items())[:4]):
        ax = axes[idx // 2, idx % 2]
        # Moyenne sur les têtes d'attention
        attn_avg = tf.reduce_mean(attn, axis=1).numpy()[0]
        
        im = ax.imshow(attn_avg, cmap='viridis', aspect='auto')
        ax.set_title(f'{layer_name}')
        ax.set_xlabel('Position Encodeur')
        ax.set_ylabel('Position Décodeur')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Poids d\'Attention du Transformer Encodeur-Décodeur', fontsize=14)
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()


# =============================================================================
# Fonction Principale
# =============================================================================

def main():
    """Fonction principale pour l'entraînement et l'évaluation"""
    
    print("="*60)
    print("TRANSFORMER ENCODEUR-DÉCODEUR POUR DONNÉES CSI")
    print("="*60)
    
    # Configuration
    config = Config()
    
    # Afficher la configuration
    print("\n📋 Configuration:")
    print(f"  - Longueur séquence: {config.SEQUENCE_LENGTH}")
    print(f"  - Nombre de features: {config.NUM_FEATURES}")
    print(f"  - Dimension modèle: {config.D_MODEL}")
    print(f"  - Nombre de têtes: {config.NUM_HEADS}")
    print(f"  - Couches encodeur: {config.NUM_ENCODER_LAYERS}")
    print(f"  - Couches décodeur: {config.NUM_DECODER_LAYERS}")
    print(f"  - Classes: {config.ACTIVITIES}")
    
    # Chemin vers les données (à ajuster selon votre configuration)
    data_dir = "/Users/bealquentin/Documents/CSI-HAR-Dataset "  # Ajuster ce chemin
    
    # Vérifier si les données existent
    npz_files = glob.glob(os.path.join(data_dir, 'X_*.npz'))
    
    if npz_files:
        print(f"\n📂 Fichiers de données trouvés: {len(npz_files)}")
        
        # Charger les données
        print("\n📥 Chargement des données...")
        X, y, label_encoder = load_csi_data(data_dir, config.ACTIVITIES, config.SEQUENCE_LENGTH)
        
        # Prétraitement
        print("\n🔧 Prétraitement des données...")
        X = preprocess_csi_data(X)
        
        # Mettre à jour la configuration selon les données
        config.NUM_FEATURES = X.shape[2]
        config.SEQUENCE_LENGTH = X.shape[1]
        
        print(f"  - Shape des données: {X.shape}")
        print(f"  - Nombre de classes: {len(np.unique(y))}")
        
        # Division train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\n📊 Division des données:")
        print(f"  - Train: {X_train.shape[0]} échantillons")
        print(f"  - Validation: {X_val.shape[0]} échantillons")
        print(f"  - Test: {X_test.shape[0]} échantillons")
        
        # Créer le modèle
        print("\n🏗️ Création du modèle...")
        model = create_model(config)
        
        # Build le modèle avec un batch exemple
        _ = model(X_train[:1])
        model.summary()
        
        # Entraînement
        print("\n🚀 Début de l'entraînement...")
        history = train_model(model, X_train, y_train, X_val, y_val, config)
        
        # Évaluation
        print("\n📈 Évaluation sur le jeu de test...")
        evaluate_model(model, X_test, y_test, label_encoder)
        
        # Visualisation
        plot_training_history(history)
        visualize_attention(model, X_test)
        
    else:
        print("\n⚠️ Aucun fichier de données trouvé.")
        print("Création d'un exemple avec des données synthétiques...")
        
        # Données synthétiques pour démonstration
        np.random.seed(42)
        num_samples = 1000
        
        X = np.random.randn(num_samples, config.SEQUENCE_LENGTH, config.NUM_FEATURES).astype(np.float32)
        y = np.random.randint(0, config.NUM_CLASSES, num_samples)
        
        # Division
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Créer et entraîner le modèle
        model = create_model(config)
        _ = model(X_train[:1])
        model.summary()
        
        print("\n🚀 Entraînement sur données synthétiques (démo)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=config.BATCH_SIZE,
            verbose=1
        )
        
        # Évaluation
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\n📊 Résultats sur données test: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()
