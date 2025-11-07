from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt
import gc
import tensorflow as tf

# Base output directory on host
#base_dir = "test/"
base_dir = "/app/data/experiments/scaling/baselines"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "multi_task/graph_data")
graphs_dir = os.path.join(base_dir, "multi_task/graphs")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
print("Created or verified directory:", os.path.exists(graph_data_dir))

def plotspu(ft):
    """
    Plot results varying the number of users in classification task.

    Args:
        ft: Fine-tuning configuration (0-5)
    """
    scen = 1

    print("=" * 60)
    print("Starting pre-training...")
    print("=" * 60)
    fet_extrct, strategy = pre_trainer(scen=scen)  # Get both extractor AND strategy

    # Clear memory after pre-training
    tf.keras.backend.clear_session()
    gc.collect()

    if ft == 0:
        model_name = "musicid_scen" + str(scen) + "_multitask"
    else:
        model_name = "musicid_scen" + str(scen) + '_ft' + str(ft) + "_multitask"

    variable_name = "number of users"
    # Number of users to test
    # Start with smaller range, expand after verifying it works
    # Full range: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]
    variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

    acc = []
    kappa = []

    for num_users in variable:
        acc_temp = []
        kappa_temp = []
        
        # Run 3 iterations for each configuration
        for itr in range(3):
            print(f"\n{'=' * 60}")
            print(f"Iteration {itr + 1}/3 for {num_users} users")
            print(f"{'=' * 60}")

            test_acc, kappa_score = trainer(num_users, fet_extrct, scen, ft=ft, strategy=strategy)
            acc_temp.append(test_acc)
            kappa_temp.append(kappa_score)

            # Force garbage collection and clear session between iterations
            tf.keras.backend.clear_session()
            gc.collect()

        acc.append(acc_temp)
        kappa.append(kappa_temp)

        # Save intermediate results after each user count
        acc_array = np.array(acc)
        kappa_array = np.array(kappa)
        np.savez(os.path.join(graph_data_dir, model_name + "_partial.npz"),
                 test_acc=acc, kappa_score=kappa)
        print(f"Saved intermediate results for {num_users} users")
        
        # Additional cleanup after each user count
        gc.collect()

    acc = np.array(acc)
    kappa = np.array(kappa)

    # Save final results
    np.savez(os.path.join(graph_data_dir, model_name + ".npz"),
             test_acc=acc, kappa_score=kappa)
    print(f"\nFinal results shape:")
    print(f"  Accuracy: {acc.shape}")
    print(f"  Kappa: {kappa.shape}")

    # Plot kappa score
    kappa_mean = np.mean(kappa, axis=1)
    kappa_std = np.std(kappa, axis=1)
    kappa_max = np.max(kappa, axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(variable, kappa_mean, 'm-', linewidth=2, label=model_name + ' (mean)', marker='o')
    plt.fill_between(variable, kappa_mean - kappa_std, kappa_mean + kappa_std,
                     alpha=0.3, color='m')
    plt.plot(variable, kappa_max, 'm--', linewidth=1, label=model_name + ' (max)')
    plt.title("Kappa Score vs " + variable_name)
    plt.xlabel(variable_name)
    plt.ylabel("Kappa Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if ft == 0:
        plt.savefig(os.path.join(graphs_dir, 'kappa.jpg'), dpi=150, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(graphs_dir, 'kappa_ft' + str(ft)+'.jpg'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot accuracy
    acc_mean = np.mean(acc, axis=1)
    acc_std = np.std(acc, axis=1)
    acc_max = np.max(acc, axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(variable, acc_mean, 'm-', linewidth=2, label=model_name + ' (mean)', marker='o')
    plt.fill_between(variable, acc_mean - acc_std, acc_mean + acc_std,
                     alpha=0.3, color='m')
    plt.plot(variable, acc_max, 'm--', linewidth=1, label=model_name + ' (max)')
    plt.title("Test Accuracy vs " + variable_name)
    plt.xlabel(variable_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if ft == 0:
        plt.savefig(os.path.join(graphs_dir, 'acc.jpg'), dpi=150, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(graphs_dir, 'acc_ft' + str(ft)+'.jpg'), dpi=150, bbox_inches='tight')
    plt.close()

    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {graph_data_dir}")
    print(f"Plots saved to: {graphs_dir}")
    print(f"{'='*60}\n")

    return True
