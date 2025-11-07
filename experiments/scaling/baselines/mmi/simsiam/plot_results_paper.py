from trainers_paper import *
from pre_trainers_paper import *
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gc


def save_checkpoint(checkpoint_path, data):
    """Save checkpoint data"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    np.savez_compressed(checkpoint_path, **data)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    """Load checkpoint data"""
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        return dict(data)
    return None


def save_progress(progress_path, completed_users, completed_iterations):
    """Save progress tracker"""
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, 'w') as f:
        json.dump({
            'completed_users': completed_users,
            'completed_iterations': completed_iterations
        }, f)
    print(f"Progress saved: {progress_path}")


def load_progress(progress_path):
    """Load progress tracker"""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {'completed_users': {}, 'completed_iterations': {}}


def plotspu(ft):
    scen = 1

    base_dir = "/app/data/experiments/scaling/baselines"

    graph_data_dir = os.path.join(base_dir, "simsiam/graph_data")
    graphs_dir = os.path.join(base_dir, "simsiam/graphs")
    checkpoint_dir = os.path.join(base_dir, "simsiam/checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_data_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    if ft == 0:
        model_name = "mmi_simsiam"
    else:
        model_name = "mmi_ft" + str(ft) + "_simsiam"

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.npz")
    progress_path = os.path.join(checkpoint_dir, f"{model_name}_progress.json")
    final_path = os.path.join(graph_data_dir, f"{model_name}.npz")

    if os.path.exists(final_path):
        print(f"Final results already exist at {final_path}. Loading and plotting...")
        data = np.load(final_path)
        acc = data['test_acc']
        kappa = data['kappa_score']
        variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]
        plot_results(variable, acc, kappa, model_name, scen, ft, graphs_dir)
        return

    encoder_path = os.path.join(checkpoint_dir, f"encoder_scen{scen}_fet6.h5")

    if os.path.exists(encoder_path):
        print(f"Loading pre-trained encoder from {encoder_path}")
        fet_extrct = tf.keras.models.load_model(encoder_path, compile=False)
    else:
        print("Training feature extractor...")
        fet_extrct = pre_trainer(scen=scen, fet=6, base_dir=base_dir)
        fet_extrct.save(encoder_path)
        print(f"Encoder saved to {encoder_path}")

    variable_name = "number of users"
    variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

    checkpoint_data = load_checkpoint(checkpoint_path)
    progress_data = load_progress(progress_path)

    if checkpoint_data is not None:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        acc = checkpoint_data['test_acc'].tolist()
        kappa = checkpoint_data['kappa_score'].tolist()
        completed_users = progress_data['completed_users']
        completed_iterations = progress_data['completed_iterations']
    else:
        print("Starting fresh experiment...")
        acc = []
        kappa = []
        completed_users = {}
        completed_iterations = {}

    for user_idx, num_users in enumerate(variable):
        user_key = str(num_users)

        if user_key in completed_users and completed_users[user_key]:
            print(f"\n{'=' * 50}")
            print(f"User count {num_users} already completed, skipping...")
            print(f"{'=' * 50}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Testing with {num_users} users")
        print(f"{'=' * 50}")

        if user_key in completed_iterations:
            acc_temp = completed_iterations[user_key]['acc']
            kappa_temp = completed_iterations[user_key]['kappa']
            start_iter = len(acc_temp)
            print(f"Resuming from iteration {start_iter + 1}/3")
        else:
            acc_temp = []
            kappa_temp = []
            start_iter = 0

        for itr in range(start_iter, 2):
            print(f"\nIteration {itr + 1}/2 for {num_users} users")

            try:
                gc.collect()
                tf.keras.backend.clear_session()
                
                test_acc, kappa_score = trainer(
                    num_users, fet_extrct, scen, ft=ft, checkpoint_dir=checkpoint_dir
                )
                acc_temp.append(test_acc)
                kappa_temp.append(kappa_score)

                completed_iterations[user_key] = {
                    'acc': acc_temp,
                    'kappa': kappa_temp
                }
                save_progress(progress_path, completed_users, completed_iterations)

                gc.collect()
                tf.keras.backend.clear_session()

            except Exception as e:
                print(f"Error in iteration {itr + 1}: {e}")
                import traceback
                traceback.print_exc()
                save_progress(progress_path, completed_users, completed_iterations)
                raise

        completed_users[user_key] = True

        if user_idx < len(acc):
            acc[user_idx] = acc_temp
            kappa[user_idx] = kappa_temp
        else:
            acc.append(acc_temp)
            kappa.append(kappa_temp)

        save_checkpoint(checkpoint_path, {
            'test_acc': np.array(acc, dtype=object),
            'kappa_score': np.array(kappa, dtype=object),
            'variable': np.array(variable[:len(acc)])
        })
        save_progress(progress_path, completed_users, completed_iterations)

        print(f"Completed {num_users} users. Progress: {len(acc)}/{len(variable)}")
        print(f"  Mean accuracy: {np.mean(acc_temp):.4f} ± {np.std(acc_temp):.4f}")
        print(f"  Mean kappa: {np.mean(kappa_temp):.4f} ± {np.std(kappa_temp):.4f}")

    acc = np.array(acc, dtype=object)
    kappa = np.array(kappa, dtype=object)

    np.savez(final_path, test_acc=acc, kappa_score=kappa, variable=np.array(variable))
    print(f"\nFinal results saved: {final_path}")

    plot_results(variable, acc, kappa, model_name, scen, ft, graphs_dir)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print("Checkpoints cleaned up.")


def plot_results(variable, acc, kappa, model_name, scen, ft, graphs_dir):
    """Generate and save plots"""

    acc_list = [np.array(x) for x in acc]
    kappa_list = [np.array(x) for x in kappa]
    
    acc_mean = np.array([np.mean(x) for x in acc_list])
    acc_std = np.array([np.std(x) for x in acc_list])
    kappa_mean = np.array([np.mean(x) for x in kappa_list])
    kappa_std = np.array([np.std(x) for x in kappa_list])

    plt.figure(figsize=(12, 8))
    plt.plot(variable, kappa_mean, 'm-o', label=model_name, linewidth=2, markersize=8)
    plt.fill_between(variable, kappa_mean - kappa_std, kappa_mean + kappa_std,
                     color='m', alpha=0.2)
    plt.title("Kappa Score vs Number of Users", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Users", fontsize=14)
    plt.ylabel("Kappa Score", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if ft == 0:
        kappa_path = os.path.join(graphs_dir, f'kappa.jpg')
    else:
        kappa_path = os.path.join(graphs_dir, f'kappa_ft{ft}.jpg')

    plt.savefig(kappa_path, dpi=150, bbox_inches='tight')
    print(f"Kappa plot saved: {kappa_path}")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(variable, acc_mean, 'm-o', label=model_name, linewidth=2, markersize=8)
    plt.fill_between(variable, acc_mean - acc_std, acc_mean + acc_std,
                     color='m', alpha=0.2)
    plt.title("Test Accuracy vs Number of Users", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Users", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if ft == 0:
        acc_path = os.path.join(graphs_dir, f'acc.jpg')
    else:
        acc_path = os.path.join(graphs_dir, f'acc_ft{ft}.jpg')

    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy plot saved: {acc_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for i, num_users in enumerate(variable):
        print(f"Users: {num_users:3d} | Acc: {acc_mean[i]:.4f} ± {acc_std[i]:.4f} | Kappa: {kappa_mean[i]:.4f} ± {kappa_std[i]:.4f}")
    print("="*60)
