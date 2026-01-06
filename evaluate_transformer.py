"""
Script pour charger le modèle Transformer sauvegardé et afficher les visualisations
Version autonome avec toutes les classes custom intégrées
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration du modèle"""
    SEQUENCE_LENGTH = 300
    NUM_FEATURES = 52
    NUM_CLASSES = 7
    D_MODEL = 128
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DFF = 512
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 32
    WARMUP_STEPS = 4000
    ACTIVITIES = ['bend', 'fall', 'lie down', 'run', 'sitdown', 'standup', 'walk']


# =============================================================================
# Classes Custom du Transformer (nécessaires pour charger le modèle)
# =============================================================================

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
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
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        return self.dense(concat_attention), attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads})
        return config


class FeedForwardNetwork(layers.Layer):
    def __init__(self, d_model, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dense1 = layers.Dense(dff, activation='gelu')
        self.dense2 = layers.Dense(d_model)
    
    def call(self, x):
        return self.dense2(self.dense1(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'dff': self.dff})
        return config


class EncoderLayer(layers.Layer):
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
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model, 'num_heads': self.num_heads,
            'dff': self.dff, 'dropout_rate': self.dropout_rate
        })
        return config


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3, attn_weights_block1, attn_weights_block2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model, 'num_heads': self.num_heads,
            'dff': self.dff, 'dropout_rate': self.dropout_rate
        })
        return config


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_dim = input_dim
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        
        self.input_projection = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers, 'd_model': self.d_model,
            'num_heads': self.num_heads, 'dff': self.dff,
            'input_dim': self.input_dim, 'max_len': self.max_len,
            'dropout_rate': self.dropout_rate
        })
        return config


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_dim, max_len, dropout_rate=0.1, **kwargs):
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
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        attention_weights = {}
        
        x = self.target_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
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
            'num_layers': self.num_layers, 'd_model': self.d_model,
            'num_heads': self.num_heads, 'dff': self.dff,
            'target_dim': self.target_dim, 'max_len': self.max_len,
            'dropout_rate': self.dropout_rate
        })
        return config


class TransformerEncoderDecoder(keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        self.encoder = Encoder(
            num_layers=config.NUM_ENCODER_LAYERS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dff=config.DFF,
            input_dim=config.NUM_FEATURES,
            max_len=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE
        )
        
        self.decoder = Decoder(
            num_layers=config.NUM_DECODER_LAYERS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dff=config.DFF,
            target_dim=config.D_MODEL,
            max_len=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE
        )
        
        self.query_token = self.add_weight(
            name='query_token',
            shape=(1, 1, config.D_MODEL),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = keras.Sequential([
            layers.Dense(config.DFF, activation='gelu'),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(config.DFF // 2, activation='gelu'),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(config.NUM_CLASSES, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        
        enc_output = self.encoder(inputs, training=training)
        query_tokens = tf.tile(self.query_token, [batch_size, 1, 1])
        
        dec_output, attention_weights = self.decoder(
            query_tokens, enc_output, training=training
        )
        
        enc_pooled = self.global_pool(enc_output)
        dec_squeezed = tf.squeeze(dec_output, axis=1)
        combined = enc_pooled + dec_squeezed
        
        return self.classifier(combined)
    
    def get_attention_weights(self, inputs):
        batch_size = tf.shape(inputs)[0]
        enc_output = self.encoder(inputs, training=False)
        query_tokens = tf.tile(self.query_token, [batch_size, 1, 1])
        _, attention_weights = self.decoder(query_tokens, enc_output, training=False)
        return attention_weights
    
    def get_config(self):
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


class TransformerLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
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
        return {'d_model': int(self.d_model.numpy()), 'warmup_steps': self.warmup_steps}


# =============================================================================
# Fonctions de chargement des données
# =============================================================================

def load_csi_data(data_dir, activities):
    """Charge les données CSI depuis les fichiers .npz"""
    X_list = []
    y_list = []
    
    for activity in activities:
        npz_file = os.path.join(data_dir, f'X_{activity}.npz')
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            X_activity = data['arr_0'] if 'arr_0' in data.files else data[data.files[0]]
            X_list.append(X_activity)
            y_list.extend([activity] * X_activity.shape[0])
            print(f"Chargé {activity}: {X_activity.shape}")
    
    X = np.concatenate(X_list, axis=0)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_list)
    
    return X, y, label_encoder


def preprocess_csi_data(X):
    """Prétraite les données CSI avec normalisation Z-score"""
    X_processed = X.astype(np.float32)
    mean = np.mean(X_processed, axis=(1, 2), keepdims=True)
    std = np.std(X_processed, axis=(1, 2), keepdims=True) + 1e-8
    X_processed = (X_processed - mean) / std
    return X_processed


# =============================================================================
# Visualisations
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_loaded.png'):
    """Affiche et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Matrice de Confusion (valeurs brutes)', fontsize=12)
    axes[0].set_xlabel('Prédiction')
    axes[0].set_ylabel('Vraie classe')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Matrice de Confusion (normalisée)', fontsize=12)
    axes[1].set_xlabel('Prédiction')
    axes[1].set_ylabel('Vraie classe')
    
    plt.suptitle('Transformer Encodeur-Décodeur CSI - Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Matrice de confusion sauvegardée: {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path='per_class_metrics.png'):
    """Affiche les métriques par classe"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Précision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Rappel', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')
    
    ax.set_xlabel('Classes d\'activités')
    ax.set_ylabel('Score')
    ax.set_title('Métriques par Classe - Transformer Encodeur-Décodeur CSI')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Métriques par classe sauvegardées: {save_path}")


def plot_prediction_distribution(y_true, y_pred, class_names, save_path='prediction_distribution.png'):
    """Affiche la distribution des prédictions vs vraies classes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    unique, counts = np.unique(y_true, return_counts=True)
    axes[0].bar(class_names, counts, color='#3498db', alpha=0.7)
    axes[0].set_title('Distribution des vraies classes')
    axes[0].set_xlabel('Classe')
    axes[0].set_ylabel('Nombre d\'échantillons')
    axes[0].tick_params(axis='x', rotation=45)
    
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    pred_counts = [0] * len(class_names)
    for u, c in zip(unique_pred, counts_pred):
        pred_counts[u] = c
    axes[1].bar(class_names, pred_counts, color='#e74c3c', alpha=0.7)
    axes[1].set_title('Distribution des prédictions')
    axes[1].set_xlabel('Classe')
    axes[1].set_ylabel('Nombre d\'échantillons')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Comparaison des distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Distribution des prédictions sauvegardée: {save_path}")


def plot_confidence_histogram(y_pred_proba, y_true, y_pred, save_path='confidence_histogram.png'):
    """Affiche l'histogramme des confiances"""
    max_proba = np.max(y_pred_proba, axis=1)
    correct = y_true == y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(max_proba, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=np.mean(max_proba), color='red', linestyle='--', label=f'Moyenne: {np.mean(max_proba):.3f}')
    axes[0].set_title('Distribution des confiances (max softmax)')
    axes[0].set_xlabel('Confiance')
    axes[0].set_ylabel('Fréquence')
    axes[0].legend()
    
    axes[1].hist(max_proba[correct], bins=30, alpha=0.7, label='Correctes', color='#2ecc71')
    axes[1].hist(max_proba[~correct], bins=30, alpha=0.7, label='Incorrectes', color='#e74c3c')
    axes[1].set_title('Confiance: Correctes vs Incorrectes')
    axes[1].set_xlabel('Confiance')
    axes[1].set_ylabel('Fréquence')
    axes[1].legend()
    
    plt.suptitle('Analyse des confiances du modèle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Histogramme des confiances sauvegardé: {save_path}")


def plot_model_architecture_summary(config, save_path='model_architecture.png'):
    """Affiche un résumé visuel de l'architecture"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    
    blocks = [
        ('Entrée CSI', f'({config.SEQUENCE_LENGTH}, {config.NUM_FEATURES})', '#3498db'),
        ('Encodeur', f'{config.NUM_ENCODER_LAYERS} couches\n{config.NUM_HEADS} têtes attention\nd_model={config.D_MODEL}', '#f39c12'),
        ('Décodeur', f'{config.NUM_DECODER_LAYERS} couches\nCross-Attention\nd_model={config.D_MODEL}', '#9b59b6'),
        ('Global Pooling', 'Average Pooling 1D', '#1abc9c'),
        ('Classifieur', f'Dense {config.DFF} → Dense {config.DFF//2}\nDropout {config.DROPOUT_RATE}', '#2ecc71'),
        ('Sortie', f'{config.NUM_CLASSES} classes\nSoftmax', '#e74c3c'),
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(blocks))
    
    for i, (name, desc, color) in enumerate(blocks):
        y = y_positions[i]
        rect = plt.Rectangle((0.2, y - 0.05), 0.6, 0.1, 
                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, f'{name}\n{desc}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(blocks) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + 0.06), xytext=(0.5, y - 0.06),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Architecture du Transformer Encodeur-Décodeur CSI', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Architecture du modèle sauvegardée: {save_path}")


# =============================================================================
# Fonction principale
# =============================================================================

def main():
    print("="*60)
    print("CHARGEMENT ET ÉVALUATION DU MODÈLE TRANSFORMER")
    print("="*60)
    
    config = Config()
    base_dir = "/Users/bealquentin/Documents"
    data_dir = "/Users/bealquentin/Documents/CSI-HAR-Dataset "  # Note: espace à la fin
    
    # Chercher le modèle sauvegardé
    model_path = os.path.join(base_dir, 'transformer_encoder_decoder_best.keras')
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print("\n📂 Fichiers disponibles:")
        for ext in ['*.keras', '*.h5']:
            for f in glob.glob(os.path.join(base_dir, ext)):
                print(f"   - {os.path.basename(f)}")
        return
    
    print(f"\n✅ Modèle trouvé: {model_path}")
    
    # Charger les données
    print("\n📥 Chargement des données...")
    X, y, label_encoder = load_csi_data(data_dir, config.ACTIVITIES)
    X = preprocess_csi_data(X)
    
    print(f"   - Shape des données: {X.shape}")
    print(f"   - Classes: {label_encoder.classes_}")
    
    # Division des données (même seed que l'entraînement)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Données de test: {X_test.shape[0]} échantillons")
    
    # Charger le modèle
    print("\n🔄 Chargement du modèle...")
    
    custom_objects = {
        'TransformerEncoderDecoder': TransformerEncoderDecoder,
        'PositionalEncoding': PositionalEncoding,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForwardNetwork': FeedForwardNetwork,
        'EncoderLayer': EncoderLayer,
        'DecoderLayer': DecoderLayer,
        'Encoder': Encoder,
        'Decoder': Decoder,
        'TransformerLRSchedule': TransformerLRSchedule,
    }
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"⚠️ Erreur avec load_model: {e}")
        print("\n🔄 Reconstruction du modèle et chargement des poids...")
        
        # Reconstruire le modèle
        config.SEQUENCE_LENGTH = X.shape[1]
        config.NUM_FEATURES = X.shape[2]
        
        model = TransformerEncoderDecoder(config)
        model(X_test[:1])  # Build
        model.load_weights(model_path)
        print("✅ Poids chargés!")
    
    # Prédictions
    print("\n🔮 Génération des prédictions...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Métriques globales
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    print(f"\n🎯 Accuracy sur le jeu de test: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Rapport de classification
    print("\n📋 Rapport de classification:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Visualisations
    print("\n📊 Génération des visualisations...")
    os.chdir(base_dir)
    
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
    plot_per_class_metrics(y_test, y_pred, label_encoder.classes_)
    plot_prediction_distribution(y_test, y_pred, label_encoder.classes_)
    plot_confidence_histogram(y_pred_proba, y_test, y_pred)
    plot_model_architecture_summary(config)
    
    print("\n" + "="*60)
    print("✅ ÉVALUATION TERMINÉE!")
    print("="*60)
    print("\n📁 Fichiers générés:")
    print("   - confusion_matrix_loaded.png")
    print("   - per_class_metrics.png")
    print("   - prediction_distribution.png")
    print("   - confidence_histogram.png")
    print("   - model_architecture.png")


if __name__ == "__main__":
    main()
