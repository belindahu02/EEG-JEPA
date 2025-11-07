# =============================================
# FIXED plot_results_musicid_time.py - Optimized for Imbalanced Data
# =============================================

from trainers_cosine_musicid import spectrogram_trainer_2d
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = "/app/data/grouped_embeddings_musicid_unstructured"
OUTPUT_PATH = "/app/data/experiments/scaling_musicid_unstructured/graph_data"
GRAPH_PATH = "/app/data/experiments/scaling_musicid_unstructured/graph"
CHECKPOINT_PATH = "/app/data/experiments/scaling_musicid_unstructured/graph_checkpoints"
MODEL_PATH = "/app/data/experiments/scaling_musicid_unstructured/model_checkpoints"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# =============================================
# EXPERIMENT PARAMETERS (FIXED FOR IMBALANCE)
# =============================================

TOTAL_USERS_AVAILABLE = 20
RUNS_PER_USER_COUNT = 1
NORMALIZATION_METHOD = 'none'
MODEL_TYPE = 'full'

variable_name = "number of users"
model_name = f"spectrogram_2d_musicid_FIXED_{NORMALIZATION_METHOD}_{MODEL_TYPE}_users"

user_counts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

checkpoint_file = os.path.join(CHECKPOINT_PATH, f"{model_name}_checkpoint.json")
results_backup_file = os.path.join(CHECKPOINT_PATH, f"{model_name}_results_backup.pkl")


def save_checkpoint(current_idx, current_itr, acc, kappa, experiment_start_time):
    checkpoint_data = {
        'current_user_count_idx': current_idx,
        'current_iteration': current_itr,
        'completed_user_counts': current_idx,
        'total_user_counts': len(user_counts),
        'experiment_start_time': experiment_start_time,
        'last_checkpoint_time': datetime.now().isoformat(),
        'user_counts_completed': user_counts[:current_idx],
        'results_shape': {
            'acc_lengths': [len(sublist) for sublist in acc],
            'kappa_lengths': [len(sublist) for sublist in kappa]
        }
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    if acc and kappa:
        import pickle
        backup_data = {
            'test_acc_list': acc,
            'kappa_score_list': kappa,
            'completed_user_counts': user_counts[:current_idx]
        }
        with open(results_backup_file, 'wb') as f:
            pickle.dump(backup_data, f)


def load_checkpoint():
    if not os.path.exists(checkpoint_file):
        return None, None, [], []

    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        start_idx = checkpoint_data.get('current_user_count_idx', 0)
        start_itr = checkpoint_data.get('current_iteration', 0)

        acc, kappa = [], []
        if os.path.exists(results_backup_file):
            try:
                import pickle
                with open(results_backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                acc = backup_data.get('test_acc_list', [])
                kappa = backup_data.get('kappa_score_list', [])
            except Exception as e:
                print(f"⚠️  Warning: Could not load backup results: {e}")

        print(f"✓ Resuming from checkpoint: {start_idx}/{len(user_counts)} user counts completed")
        return start_idx, start_itr, acc, kappa

    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        return None, None, [], []


def create_plots(acc, kappa, num_completed, suffix=""):
    if len(acc) == 0 and len(kappa) == 0:
        return

    current_user_counts = user_counts[:num_completed]
    
    acc_max = []
    acc_mean = []
    acc_std = []
    kappa_max = []
    kappa_mean = []
    kappa_std = []
    valid_user_counts = []

    for i in range(min(len(acc), len(kappa), len(current_user_counts))):
        if i < len(acc) and len(acc[i]) > 0:
            valid_user_counts.append(current_user_counts[i])
            acc_max.append(np.max(acc[i]))
            acc_mean.append(np.mean(acc[i]))
            acc_std.append(np.std(acc[i]))
            kappa_max.append(np.max(kappa[i]))
            kappa_mean.append(np.mean(kappa[i]))
            kappa_std.append(np.std(kappa[i]))

    if len(valid_user_counts) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.errorbar(valid_user_counts, acc_mean, yerr=acc_std, 
                 fmt='o-', capsize=5, label='Mean ± Std', linewidth=2, markersize=8)
    ax1.plot(valid_user_counts, acc_max, 's--', label='Max', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Number of Users', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'Test Accuracy vs {variable_name.title()} (FIXED)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.05])

    ax2.errorbar(valid_user_counts, kappa_mean, yerr=kappa_std, 
                 fmt='o-', capsize=5, label='Mean ± Std', linewidth=2, markersize=8, color='orange')
    ax2.plot(valid_user_counts, kappa_max, 's--', label='Max', linewidth=2, markersize=6, 
             alpha=0.7, color='darkorange')
    ax2.set_xlabel('Number of Users', fontsize=12)
    ax2.set_ylabel("Cohen's Kappa", fontsize=12)
    ax2.set_title(f"Cohen's Kappa vs {variable_name.title()} (FIXED)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    
    plot_filename = os.path.join(GRAPH_PATH, f"{model_name}{suffix}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()


print("\n" + "=" * 80)
print("MUSIC EEG - FIXED SCALING EXPERIMENT (Class-Balanced)")
print("=" * 80)
print(f"Using Class-Balanced Loss + Weighted Sampling")
print(f"Optimized hyperparameters for imbalanced data")
print("=" * 80 + "\n")

start_idx, start_itr, acc, kappa = load_checkpoint()

if start_idx is None:
    start_idx = 0
    start_itr = 0
    acc = []
    kappa = []

experiment_start_time = datetime.now().isoformat()
experiment_start = time.time()

try:
    for i, num_users in enumerate(user_counts[start_idx:], start=start_idx):
        print("\n" + "=" * 80)
        print(f"Testing with {num_users} users ({i + 1}/{len(user_counts)})")
        print("=" * 80)

        user_ids = list(range(1, num_users + 1))

        if i < len(acc):
            acc_temp = acc[i]
            kappa_temp = kappa[i]
        else:
            acc_temp = []
            kappa_temp = []

        for itr in range(start_itr if i == start_idx else 0, RUNS_PER_USER_COUNT):
            run_start_time = time.time()
            print(f"\n--- Run {itr + 1}/{RUNS_PER_USER_COUNT} for {num_users} users ---")

            try:
                # FIXED parameters for imbalanced data
                test_acc, kappa_score = spectrogram_trainer_2d(
                    data_path=DATA_PATH,
                    model_path=MODEL_PATH,
                    user_ids=user_ids,
                    normalization_method=NORMALIZATION_METHOD,
                    model_type=MODEL_TYPE,
                    epochs=100,
                    batch_size=8,          # Increased from 4
                    lr=0.0003,              # Increased from 0.0005
                    use_augmentation=True,
                    device='cuda',
                    save_model_checkpoints=True,
                    checkpoint_every=RUNS_PER_USER_COUNT,
                    max_cache_size=50,
                    use_cosine_classifier=True,
                    cosine_scale=30.0,     # Decreased from 30.0
                    label_smoothing=0.0,
                    warmup_epochs=3,       # Decreased from 10
                    dropout_rate=0.2,      # Decreased from 0.3
                    classifier_dropout=0.3, # Decreased from 0.5
                    lr_scheduler_type='cosine',
                    test_sessions_per_user=1,
                    val_sessions_per_user=1
                )
                
                acc_temp.append(test_acc)
                kappa_temp.append(kappa_score)

                run_time = time.time() - run_start_time
                print(f"✓ Run {itr + 1} completed in {run_time:.1f}s")
                print(f"   Test Accuracy: {test_acc:.4f}")
                print(f"   Kappa Score: {kappa_score:.4f}")

                if i < len(acc):
                    acc[i] = acc_temp
                    kappa[i] = kappa_temp
                else:
                    acc.append(acc_temp)
                    kappa.append(kappa_temp)

                save_checkpoint(i, itr + 1, acc, kappa, experiment_start_time)

            except Exception as e:
                print(f"⚠️  Error in run {itr + 1}: {e}")
                save_checkpoint(i, itr, acc, kappa, experiment_start_time)
                continue

        start_itr = 0

        if acc_temp:
            if i < len(acc):
                acc[i] = acc_temp
                kappa[i] = kappa_temp
            else:
                acc.append(acc_temp)
                kappa.append(kappa_temp)

            avg_acc = np.mean(acc_temp)
            std_acc = np.std(acc_temp)
            avg_kappa = np.mean(kappa_temp)
            std_kappa = np.std(kappa_temp)

            print(f"\n{'=' * 80}")
            print(f"Summary for {num_users} users:")
            print(f"  Accuracy: {avg_acc:.4f} ± {std_acc:.4f} (n={len(acc_temp)})")
            print(f"  Kappa:    {avg_kappa:.4f} ± {std_kappa:.4f} (n={len(kappa_temp)})")
            print(f"{'=' * 80}")

            create_plots(acc, kappa, i + 1, suffix="_intermediate")

        save_checkpoint(i + 1, 0, acc, kappa, experiment_start_time)

        if i > start_idx:
            elapsed_time = time.time() - experiment_start
            avg_time = elapsed_time / (i - start_idx + 1)
            remaining = len(user_counts) - i - 1
            est_remaining = remaining * avg_time
            print(f"\n⏱  Estimated remaining time: {est_remaining / 3600:.1f} hours")

except KeyboardInterrupt:
    print("\n\n⚠️  Experiment interrupted. Saving progress...")
    import pickle
    with open(os.path.join(OUTPUT_PATH, f"{model_name}_interrupted.pkl"), 'wb') as f:
        pickle.dump({'acc': acc, 'kappa': kappa}, f)

# Final results
if len(acc) > 0:
    print("\n" + "=" * 80)
    print("FINAL RESULTS (FIXED)")
    print("=" * 80)

    import pickle
    final_results = {
        'test_acc_list': acc,
        'kappa_score_list': kappa,
        'user_counts': user_counts[:len(acc)],
        'model_name': model_name,
        'fixed_version': True,
        'class_balanced_loss': True,
        'weighted_sampling': True
    }

    pickle_file = os.path.join(OUTPUT_PATH, f"{model_name}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(final_results, f)
    print(f"✓ Results saved to {pickle_file}")

    create_plots(acc, kappa, len(acc))
    print(f"✓ Final plots saved")

    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    
    for i, (a_list, k_list) in enumerate(zip(acc, kappa)):
        if len(a_list) > 0:
            print(f"Users {user_counts[i]:2d}: "
                  f"Acc={np.mean(a_list):.4f}±{np.std(a_list):.4f}, "
                  f"Kappa={np.mean(k_list):.4f}±{np.std(k_list):.4f} (n={len(a_list)})")

    all_acc = [v for sublist in acc for v in sublist]
    all_kappa = [v for sublist in kappa for v in sublist]
    if all_acc:
        print(f"\nOverall: Acc={np.mean(all_acc):.4f}±{np.std(all_acc):.4f}, "
              f"Kappa={np.mean(all_kappa):.4f}±{np.std(all_kappa):.4f}")
    
    print(f"{'=' * 80}\n")
