"""
Script de Comparaison d'Hyperparamètres pour Transformer Encodeur-Décodeur CSI
Effectue des apprentissages plus courts avec différentes configurations de paramètres
et compare les performances sur 20 exécutions pour chaque configuration.

Paramètres testés:
- Nombre de têtes d'attention (NUM_HEADS)
- Nombre de couches encodeur (NUM_ENCODER_LAYERS)
- Nombre de couches décodeur (NUM_DECODER_LAYERS)

Auteur: Assistant IA
Date: Janvier 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importer les classes du script principal
from Transformer_Encoder_Decoder_CSI import (
    Config, TransformerEncoderDecoder, load_csi_data, 
    preprocess_csi_data, TransformerLRSchedule
)


# =============================================================================
# Configuration des Expériences
# =============================================================================

class ExperimentConfig:
    """Configuration pour les expériences d'hyperparamètres"""
    
    # Paramètres fixes
    SEQUENCE_LENGTH = 300
    NUM_FEATURES = 52
    NUM_CLASSES = 7
    
    # Architecture Transformer (réduite pour accélérer)
    D_MODEL = 64  # Réduit de 128 à 64
    DFF = 256  # Réduit de 512 à 256
    DROPOUT_RATE = 0.1
    
    # Entraînement réduit pour accélérer les expériences (objectif: 1 heure totale)
    BATCH_SIZE = 64  # Augmenté pour accélérer
    EPOCHS = 15  # Réduit drastiquement
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 500  # Réduit drastiquement
    
    # Paramètres d'expérimentation
    NUM_RUNS_PER_CONFIG = 5  # Réduit de 20 à 5 pour gagner du temps
    
    # Configurations à tester (réduites)
    ATTENTION_HEADS = [4, 8]  # Seulement 2 configurations au lieu de 4
    ENCODER_LAYERS = [2, 4]   # Seulement 2 configurations au lieu de 4
    DECODER_LAYERS = [2, 4]   # Seulement 2 configurations au lieu de 4
    
    ACTIVITIES = ['bend', 'fall', 'lie down', 'run', 'sitdown', 'standup', 'walk']


# =============================================================================
# Fonctions Utilitaires
# =============================================================================

def create_config_variant(base_config, num_heads=None, num_encoder=None, num_decoder=None):
    """Crée une variante de configuration avec des paramètres modifiés"""
    config = Config()
    
    # Copier les paramètres de base
    config.SEQUENCE_LENGTH = base_config.SEQUENCE_LENGTH
    config.NUM_FEATURES = base_config.NUM_FEATURES
    config.NUM_CLASSES = base_config.NUM_CLASSES
    config.D_MODEL = base_config.D_MODEL
    config.DFF = base_config.DFF
    config.DROPOUT_RATE = base_config.DROPOUT_RATE
    config.BATCH_SIZE = base_config.BATCH_SIZE
    config.EPOCHS = base_config.EPOCHS
    config.LEARNING_RATE = base_config.LEARNING_RATE
    config.WARMUP_STEPS = base_config.WARMUP_STEPS
    config.ACTIVITIES = base_config.ACTIVITIES
    
    # Appliquer les modifications
    config.NUM_HEADS = num_heads if num_heads is not None else 8
    config.NUM_ENCODER_LAYERS = num_encoder if num_encoder is not None else 4
    config.NUM_DECODER_LAYERS = num_decoder if num_decoder is not None else 4
    
    return config


def create_and_compile_model(config):
    """Crée et compile un modèle avec la configuration donnée"""
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


def train_single_run(config, X_train, y_train, X_val, y_val, run_number, config_name, verbose=0):
    """
    Effectue une seule exécution d'entraînement
    
    Args:
        config: Configuration du modèle
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        run_number: Numéro de l'exécution
        config_name: Nom de la configuration
        verbose: Niveau de verbosité
    
    Returns:
        dict: Résultats de l'entraînement
    """
    print(f"  Run {run_number + 1}/5: ", end='', flush=True)
    
    # Créer le modèle
    model = create_and_compile_model(config)
    
    # Build le modèle
    _ = model(X_train[:1])
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Réduit pour accélérer (stop si pas d'amélioration en 5 epochs)
        restore_best_weights=True,
        verbose=0
    )
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    # Évaluation finale
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    # Nombre d'époques effectuées
    epochs_trained = len(history.history['loss'])
    
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Epochs: {epochs_trained}")
    
    return {
        'config_name': config_name,
        'run_number': run_number,
        'num_heads': config.NUM_HEADS,
        'num_encoder_layers': config.NUM_ENCODER_LAYERS,
        'num_decoder_layers': config.NUM_DECODER_LAYERS,
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'epochs_trained': epochs_trained,
        'final_train_history': history.history
    }


# =============================================================================
# Expérience 1: Variation du Nombre de Têtes d'Attention
# =============================================================================

def experiment_attention_heads(base_config, X_train, y_train, X_val, y_val, results_dir):
    """
    Teste différents nombres de têtes d'attention
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE 1: VARIATION DU NOMBRE DE TÊTES D'ATTENTION")
    print("="*80)
    
    results = []
    
    for num_heads in base_config.ATTENTION_HEADS:
        print(f"\n🔧 Configuration: {num_heads} têtes d'attention")
        
        # Vérifier que d_model est divisible par num_heads
        if base_config.D_MODEL % num_heads != 0:
            print(f"⚠️ Attention: D_MODEL ({base_config.D_MODEL}) n'est pas divisible par {num_heads}")
            print(f"   Ajustement de D_MODEL à {(base_config.D_MODEL // num_heads) * num_heads}")
            temp_config = ExperimentConfig()
            temp_config.D_MODEL = (base_config.D_MODEL // num_heads) * num_heads
            config = create_config_variant(temp_config, num_heads=num_heads)
        else:
            config = create_config_variant(base_config, num_heads=num_heads)
        
        config_name = f"heads_{num_heads}"
        
        # Effectuer 5 runs
        for run in range(base_config.NUM_RUNS_PER_CONFIG):
            result = train_single_run(
                config, X_train, y_train, X_val, y_val, 
                run, config_name, verbose=0
            )
            results.append(result)
    
    # Sauvegarder les résultats
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'experiment_1_attention_heads.csv'), index=False)
    
    return results


# =============================================================================
# Expérience 2: Variation du Nombre de Couches Encodeur
# =============================================================================

def experiment_encoder_layers(base_config, X_train, y_train, X_val, y_val, results_dir):
    """
    Teste différents nombres de couches encodeur
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE 2: VARIATION DU NOMBRE DE COUCHES ENCODEUR")
    print("="*80)
    
    results = []
    
    for num_layers in base_config.ENCODER_LAYERS:
        print(f"\n🔧 Configuration: {num_layers} couches encodeur")
        
        config = create_config_variant(base_config, num_encoder=num_layers)
        config_name = f"encoder_{num_layers}"
        
        # Effectuer 5 runs
        for run in range(base_config.NUM_RUNS_PER_CONFIG):
            result = train_single_run(
                config, X_train, y_train, X_val, y_val, 
                run, config_name, verbose=0
            )
            results.append(result)
    
    # Sauvegarder les résultats
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'experiment_2_encoder_layers.csv'), index=False)
    
    return results


# =============================================================================
# Expérience 3: Variation du Nombre de Couches Décodeur
# =============================================================================

def experiment_decoder_layers(base_config, X_train, y_train, X_val, y_val, results_dir):
    """
    Teste différents nombres de couches décodeur
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE 3: VARIATION DU NOMBRE DE COUCHES DÉCODEUR")
    print("="*80)
    
    results = []
    
    for num_layers in base_config.DECODER_LAYERS:
        print(f"\n🔧 Configuration: {num_layers} couches décodeur")
        
        config = create_config_variant(base_config, num_decoder=num_layers)
        config_name = f"decoder_{num_layers}"
        
        # Effectuer 5 runs
        for run in range(base_config.NUM_RUNS_PER_CONFIG):
            result = train_single_run(
                config, X_train, y_train, X_val, y_val, 
                run, config_name, verbose=0
            )
            results.append(result)
    
    # Sauvegarder les résultats
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'experiment_3_decoder_layers.csv'), index=False)
    
    return results


# =============================================================================
# Analyse et Visualisation des Résultats
# =============================================================================

def analyze_and_visualize_results(results, experiment_name, param_name, results_dir):
    """
    Analyse et visualise les résultats d'une expérience
    
    Args:
        results: Liste des résultats
        experiment_name: Nom de l'expérience
        param_name: Nom du paramètre testé
        results_dir: Répertoire pour sauvegarder les visualisations
    """
    df = pd.DataFrame(results)
    
    # Calculer les statistiques par configuration
    if 'heads' in experiment_name:
        groupby_col = 'num_heads'
    elif 'encoder' in experiment_name:
        groupby_col = 'num_encoder_layers'
    else:
        groupby_col = 'num_decoder_layers'
    
    stats = df.groupby(groupby_col).agg({
        'val_accuracy': ['mean', 'std', 'min', 'max'],
        'train_accuracy': ['mean', 'std'],
        'epochs_trained': ['mean', 'std']
    }).round(4)
    
    print(f"\n📊 Statistiques pour {experiment_name}:")
    print(stats)
    
    # Sauvegarder les statistiques
    stats.to_csv(os.path.join(results_dir, f'{experiment_name}_statistics.csv'))
    
    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Boxplot de la précision de validation
    ax1 = axes[0, 0]
    df.boxplot(column='val_accuracy', by=groupby_col, ax=ax1)
    ax1.set_title(f'Distribution de la Précision de Validation\n{param_name}')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Précision de Validation')
    ax1.get_figure().suptitle('')  # Supprimer le titre auto-généré
    
    # 2. Courbe de moyenne avec écart-type
    ax2 = axes[0, 1]
    grouped = df.groupby(groupby_col)['val_accuracy']
    means = grouped.mean()
    stds = grouped.std()
    param_values = means.index
    
    ax2.plot(param_values, means, 'o-', linewidth=2, markersize=8, label='Moyenne')
    ax2.fill_between(param_values, means - stds, means + stds, alpha=0.3, label='±1 écart-type')
    ax2.set_title(f'Précision Moyenne de Validation\n{param_name}')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Précision de Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparaison Train vs Validation
    ax3 = axes[1, 0]
    train_means = df.groupby(groupby_col)['train_accuracy'].mean()
    val_means = df.groupby(groupby_col)['val_accuracy'].mean()
    
    x = np.arange(len(param_values))
    width = 0.35
    
    ax3.bar(x - width/2, train_means, width, label='Train', alpha=0.8)
    ax3.bar(x + width/2, val_means, width, label='Validation', alpha=0.8)
    ax3.set_title(f'Comparaison Train vs Validation\n{param_name}')
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Précision')
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_values)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Nombre d'époques
    ax4 = axes[1, 1]
    epochs_means = df.groupby(groupby_col)['epochs_trained'].mean()
    epochs_stds = df.groupby(groupby_col)['epochs_trained'].std()
    
    ax4.bar(param_values, epochs_means, alpha=0.8, yerr=epochs_stds, capsize=5)
    ax4.set_title(f'Nombre Moyen d\'Époques\n{param_name}')
    ax4.set_xlabel(param_name)
    ax4.set_ylabel('Époques')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{experiment_name}_analysis.png'), dpi=150)
    plt.close()
    
    print(f"✅ Visualisation sauvegardée: {experiment_name}_analysis.png")


def create_comparison_summary(results_dir):
    """
    Crée un résumé comparatif de toutes les expériences
    """
    print("\n" + "="*80)
    print("CRÉATION DU RÉSUMÉ COMPARATIF")
    print("="*80)
    
    # Charger tous les résultats
    exp1 = pd.read_csv(os.path.join(results_dir, 'experiment_1_attention_heads.csv'))
    exp2 = pd.read_csv(os.path.join(results_dir, 'experiment_2_encoder_layers.csv'))
    exp3 = pd.read_csv(os.path.join(results_dir, 'experiment_3_decoder_layers.csv'))
    
    # Statistiques pour chaque expérience
    stats_exp1 = exp1.groupby('num_heads')['val_accuracy'].agg(['mean', 'std', 'max'])
    stats_exp2 = exp2.groupby('num_encoder_layers')['val_accuracy'].agg(['mean', 'std', 'max'])
    stats_exp3 = exp3.groupby('num_decoder_layers')['val_accuracy'].agg(['mean', 'std', 'max'])
    
    # Trouver les meilleures configurations
    best_heads = stats_exp1['mean'].idxmax()
    best_encoder = stats_exp2['mean'].idxmax()
    best_decoder = stats_exp3['mean'].idxmax()
    
    # Créer le rapport
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_configurations': {
            'attention_heads': int(best_heads),
            'encoder_layers': int(best_encoder),
            'decoder_layers': int(best_decoder)
        },
        'best_accuracies': {
            'attention_heads': float(stats_exp1.loc[best_heads, 'mean']),
            'encoder_layers': float(stats_exp2.loc[best_encoder, 'mean']),
            'decoder_layers': float(stats_exp3.loc[best_decoder, 'mean'])
        },
        'all_statistics': {
            'attention_heads': stats_exp1.to_dict(),
            'encoder_layers': stats_exp2.to_dict(),
            'decoder_layers': stats_exp3.to_dict()
        }
    }
    
    # Sauvegarder le résumé
    with open(os.path.join(results_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Afficher le résumé
    print("\n📋 MEILLEURES CONFIGURATIONS:")
    print(f"  ├─ Têtes d'attention: {best_heads} (accuracy: {summary['best_accuracies']['attention_heads']:.4f})")
    print(f"  ├─ Couches encodeur: {best_encoder} (accuracy: {summary['best_accuracies']['encoder_layers']:.4f})")
    print(f"  └─ Couches décodeur: {best_decoder} (accuracy: {summary['best_accuracies']['decoder_layers']:.4f})")
    
    # Créer une visualisation comparative
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Expérience 1
    stats_exp1['mean'].plot(kind='bar', ax=ax[0], yerr=stats_exp1['std'], capsize=5, alpha=0.8)
    ax[0].set_title('Têtes d\'Attention\n(Moyenne ± Écart-type)')
    ax[0].set_xlabel('Nombre de Têtes')
    ax[0].set_ylabel('Précision de Validation')
    ax[0].grid(True, alpha=0.3, axis='y')
    
    # Expérience 2
    stats_exp2['mean'].plot(kind='bar', ax=ax[1], yerr=stats_exp2['std'], capsize=5, alpha=0.8)
    ax[1].set_title('Couches Encodeur\n(Moyenne ± Écart-type)')
    ax[1].set_xlabel('Nombre de Couches')
    ax[1].set_ylabel('Précision de Validation')
    ax[1].grid(True, alpha=0.3, axis='y')
    
    # Expérience 3
    stats_exp3['mean'].plot(kind='bar', ax=ax[2], yerr=stats_exp3['std'], capsize=5, alpha=0.8)
    ax[2].set_title('Couches Décodeur\n(Moyenne ± Écart-type)')
    ax[2].set_xlabel('Nombre de Couches')
    ax[2].set_ylabel('Précision de Validation')
    ax[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_summary.png'), dpi=150)
    plt.close()
    
    print("\n✅ Résumé comparatif sauvegardé!")


# =============================================================================
# Fonction Principale
# =============================================================================

def main():
    """
    Fonction principale pour exécuter toutes les expériences
    """
    print("="*80)
    print("COMPARAISON D'HYPERPARAMÈTRES - TRANSFORMER ENCODEUR-DÉCODEUR CSI")
    print("="*80)
    print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    base_config = ExperimentConfig()
    
    # Créer le répertoire de résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_hyperparameter_comparison_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n📁 Résultats seront sauvegardés dans: {results_dir}")
    
    # Chemin vers les données
    data_dir = "/Users/bealquentin/Documents/CSI-HAR-Dataset "
    
    # Charger les données
    print("\n📥 Chargement des données...")
    X, y, label_encoder = load_csi_data(data_dir, base_config.ACTIVITIES, base_config.SEQUENCE_LENGTH)
    
    # Prétraitement
    print("🔧 Prétraitement des données...")
    X = preprocess_csi_data(X)
    
    # Mettre à jour la configuration selon les données
    base_config.NUM_FEATURES = X.shape[2]
    base_config.SEQUENCE_LENGTH = X.shape[1]
    
    print(f"  ├─ Shape des données: {X.shape}")
    print(f"  ├─ Nombre de classes: {len(np.unique(y))}")
    print(f"  └─ Features par séquence: {X.shape[2]}")
    
    # Division train/val/test (on garde seulement train et val pour les expériences)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Division des données:")
    print(f"  ├─ Train: {X_train.shape[0]} échantillons")
    print(f"  ├─ Validation: {X_val.shape[0]} échantillons")
    print(f"  └─ Test: {X_test.shape[0]} échantillons (non utilisé dans ces expériences)")
    
    # Lancer les expériences
    all_results = {}
    
    # Expérience 1: Têtes d'attention
    results_exp1 = experiment_attention_heads(base_config, X_train, y_train, X_val, y_val, results_dir)
    all_results['attention_heads'] = results_exp1
    analyze_and_visualize_results(
        results_exp1, 
        'experiment_1_attention_heads', 
        'Nombre de Têtes d\'Attention',
        results_dir
    )
    
    # Expérience 2: Couches encodeur
    results_exp2 = experiment_encoder_layers(base_config, X_train, y_train, X_val, y_val, results_dir)
    all_results['encoder_layers'] = results_exp2
    analyze_and_visualize_results(
        results_exp2, 
        'experiment_2_encoder_layers', 
        'Nombre de Couches Encodeur',
        results_dir
    )
    
    # Expérience 3: Couches décodeur
    results_exp3 = experiment_decoder_layers(base_config, X_train, y_train, X_val, y_val, results_dir)
    all_results['decoder_layers'] = results_exp3
    analyze_and_visualize_results(
        results_exp3, 
        'experiment_3_decoder_layers', 
        'Nombre de Couches Décodeur',
        results_dir
    )
    
    # Créer le résumé comparatif
    create_comparison_summary(results_dir)
    
    print("\n" + "="*80)
    print("EXPÉRIENCES TERMINÉES!")
    print("="*80)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Tous les résultats sont dans: {results_dir}")
    print("\nFichiers générés:")
    print("  ├─ experiment_1_attention_heads.csv")
    print("  ├─ experiment_1_attention_heads_statistics.csv")
    print("  ├─ experiment_1_attention_heads_analysis.png")
    print("  ├─ experiment_2_encoder_layers.csv")
    print("  ├─ experiment_2_encoder_layers_statistics.csv")
    print("  ├─ experiment_2_encoder_layers_analysis.png")
    print("  ├─ experiment_3_decoder_layers.csv")
    print("  ├─ experiment_3_decoder_layers_statistics.csv")
    print("  ├─ experiment_3_decoder_layers_analysis.png")
    print("  ├─ comparison_summary.png")
    print("  └─ summary_report.json")


if __name__ == "__main__":
    main()
