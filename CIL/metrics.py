import numpy as np


class IncrementalMetrics:
    """
    Tracks incremental learning metrics for paper reporting
    
    Handles mapping from global class IDs to local indices internally
    """
    def __init__(self, total_tasks, total_classes, class_id_to_idx=None):
        """
        Args:
            total_tasks: Number of tasks (including base task)
            total_classes: Total number of classes
            class_id_to_idx: Optional mapping from global class ID to local index
        """
        self.total_tasks = total_tasks
        self.total_classes = total_classes
        self.class_id_to_idx = class_id_to_idx  # Global ID → Local Index mapping

        # Accuracy matrix A[t][k] using LOCAL indices
        self.acc_matrix = np.zeros((total_tasks, total_classes))

        # Task-wise statistics
        self.accuracies = []     # A_t
        self.param_counts = []
        self.task_times = []

        # Track when each class was introduced (using LOCAL indices)
        self.class_intro_task = np.full(total_classes, -1)

        self.current_task = -1

    def update(self, task_id, class_acc_dict, num_params, duration_seconds):
        """
        Update metrics after a task
        
        Args:
            task_id: Current task ID
            class_acc_dict: {global_class_id: accuracy} - uses GLOBAL class IDs
            num_params: Number of parameters
            duration_seconds: Training time
        """
        self.current_task = task_id

        # 1. Copy previous task accuracies (cumulative matrix)
        if task_id > 0:
            self.acc_matrix[task_id] = self.acc_matrix[task_id - 1]

        # 2. Update current task class accuracies
        # Convert global class IDs to local indices
        for global_cls_id, acc in class_acc_dict.items():
            # Map global class ID to local index
            if self.class_id_to_idx is not None:
                local_idx = self.class_id_to_idx.get(global_cls_id, global_cls_id)
            else:
                local_idx = global_cls_id
            
            # Ensure index is within bounds
            if 0 <= local_idx < self.total_classes:
                self.acc_matrix[task_id, local_idx] = acc
                if self.class_intro_task[local_idx] == -1:
                    self.class_intro_task[local_idx] = task_id
            else:
                print(f"Warning: Class index {local_idx} (global ID {global_cls_id}) out of bounds!")

        # 3. Compute A_t (average over ALL seen classes)
        seen_classes = np.where(self.class_intro_task != -1)[0]
        At = np.mean(self.acc_matrix[task_id, seen_classes]) if len(seen_classes) > 0 else 0.0

        self.accuracies.append(At)
        self.param_counts.append(num_params)
        self.task_times.append(duration_seconds)

    # ================= PAPER METRICS (Equations 9 & 10) =================

    def get_avg_acc(self):
        """Average Incremental Accuracy (Paper Equation 9)"""
        return float(np.mean(self.accuracies)) if self.accuracies else 0.0

    def get_avg_params(self):
        """Average number of trainable parameters"""
        return float(np.mean(self.param_counts)) if self.param_counts else 0.0

    def get_avg_time_hours(self):
        """Average training time in hours"""
        if not self.task_times:
            return 0.0
        return float(np.mean(self.task_times)) / 3600.0

    def get_forgetting(self):
        """
        Forgetting Measure (Paper Equation 10)
        
        F = mean over classes of (max accuracy before last task - final accuracy)
        """
        forgetting = []

        for c in range(self.total_classes):
            t0 = self.class_intro_task[c]
            if t0 == -1 or self.current_task <= t0:
                continue

            acc_hist = self.acc_matrix[t0:self.current_task + 1, c]
            if len(acc_hist) <= 1:
                continue
            
            peak = np.max(acc_hist[:-1])
            final = acc_hist[-1]
            forgetting.append(max(0.0, peak - final))

        return float(np.mean(forgetting)) if forgetting else 0.0

    # ================= PRINTING (Paper Table V Format) =================

    def print_paper_table(self):
        """Print results in Paper Table V format"""
        print("\n" + "=" * 80)
        print(f"{'Task':<6} | {'A_t':<10} | {'#Params':<15} | {'Time (s)':<10}")
        print("-" * 80)

        for t in range(len(self.accuracies)):
            print(f"{t:<6} | {self.accuracies[t]:<10.4f} | "
                  f"{self.param_counts[t]:<15,d} | {self.task_times[t]:<10.2f}")

        print("-" * 80)
        print(f"Avg Incremental Acc (A): {self.get_avg_acc():.4f}")
        print(f"Avg Trainable Params:    {int(self.get_avg_params()):,}")
        print(f"Avg Training Time:       {self.get_avg_time_hours():.6f} h")
        print(f"Forgetting Measure (F):  {self.get_forgetting():.4f}")
        print("=" * 80 + "\n")