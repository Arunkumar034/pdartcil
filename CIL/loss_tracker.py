"""
loss_tracker.py - Track and visualize training losses during incremental learning

Features:
- Track Stage 1 and Stage 2 losses per task
- Generate loss vs epoch graphs
- Generate task-wise comparison graphs
- Save all graphs to working directory
"""

import matplotlib
matplotlib.use('Agg')  # Don't display, only save
import matplotlib.pyplot as plt
import os
import numpy as np


class LossTracker:
    """
    Track losses across incremental learning tasks
    """
    
    def __init__(self, save_dir='graphs'):
        """
        Initialize loss tracker
        
        Args:
            save_dir: Directory to save graphs
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for losses
        self.task_losses = {}  # {task_id: {'stage1': [...], 'stage2': [...]}}
        self.task_info = {}    # {task_id: {'classes': [...], 'num_classes': X}}
        
        print(f"✅ Loss tracking initialized. Graphs will be saved to: {save_dir}/")
    
    def start_task(self, task_id, task_classes):
        """
        Start tracking a new task
        
        Args:
            task_id: Task index (0, 1, 2, ...)
            task_classes: List of class indices for this task
        """
        self.task_losses[task_id] = {
            'stage1': [],
            'stage2': []
        }
        self.task_info[task_id] = {
            'classes': task_classes,
            'num_classes': len(task_classes)
        }
    
    def add_stage1_loss(self, task_id, epoch, loss):
        """
        Record Stage 1 (Representation Learning) loss
        
        Args:
            task_id: Task index
            epoch: Epoch number
            loss: Loss value
        """
        if task_id not in self.task_losses:
            print(f"Warning: Task {task_id} not initialized. Call start_task() first.")
            return
        
        self.task_losses[task_id]['stage1'].append({
            'epoch': epoch,
            'loss': loss
        })
    
    def add_stage2_loss(self, task_id, epoch, loss):
        """
        Record Stage 2 (Classifier Tuning) loss
        
        Args:
            task_id: Task index
            epoch: Epoch number
            loss: Loss value
        """
        if task_id not in self.task_losses:
            print(f"Warning: Task {task_id} not initialized. Call start_task() first.")
            return
        
        self.task_losses[task_id]['stage2'].append({
            'epoch': epoch,
            'loss': loss
        })
    
    def plot_task_losses(self, task_id, scenario='T=2'):
        """
        Plot Stage 1 and Stage 2 losses for a single task
        
        Args:
            task_id: Task index
            scenario: Scenario name for filename
        """
        if task_id not in self.task_losses:
            print(f"Warning: No data for task {task_id}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Stage 1 losses
        stage1_data = self.task_losses[task_id]['stage1']
        if stage1_data:
            epochs1 = [d['epoch'] for d in stage1_data]
            losses1 = [d['loss'] for d in stage1_data]
            
            ax1.plot(epochs1, losses1, 'b-', linewidth=2, label='Stage 1 Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title(f'Task {task_id} - Stage 1: Representation Learning', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Add final loss annotation
            final_loss = losses1[-1]
            ax1.text(0.95, 0.95, f'Final Loss: {final_loss:.4f}', 
                    transform=ax1.transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        
        # Stage 2 losses
        stage2_data = self.task_losses[task_id]['stage2']
        if stage2_data:
            epochs2 = [d['epoch'] for d in stage2_data]
            losses2 = [d['loss'] for d in stage2_data]
            
            ax2.plot(epochs2, losses2, 'r-', linewidth=2, label='Stage 2 Loss')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Task {task_id} - Stage 2: Classifier Tuning', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Add final loss annotation
            final_loss = losses2[-1]
            ax2.text(0.95, 0.95, f'Final Loss: {final_loss:.4f}', 
                    transform=ax2.transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.save_dir}/task{task_id}_losses_{scenario.replace('=', '')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved: {filename}")
    
    def plot_all_tasks_comparison(self, scenario='T=2'):
        """
        Plot comparison of all tasks' losses
        
        Args:
            scenario: Scenario name for filename
        """
        if not self.task_losses:
            print("Warning: No task data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.task_losses)))
        
        # Stage 1 comparison
        for i, task_id in enumerate(sorted(self.task_losses.keys())):
            stage1_data = self.task_losses[task_id]['stage1']
            if stage1_data:
                epochs = [d['epoch'] for d in stage1_data]
                losses = [d['loss'] for d in stage1_data]
                
                task_type = "Base" if task_id == 0 else f"Inc {task_id}"
                ax1.plot(epochs, losses, '-', color=colors[i], linewidth=2, 
                        label=f'Task {task_id} ({task_type})', marker='o', markersize=3, markevery=10)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Stage 1: Representation Learning - All Tasks', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='best')
        
        # Stage 2 comparison
        for i, task_id in enumerate(sorted(self.task_losses.keys())):
            stage2_data = self.task_losses[task_id]['stage2']
            if stage2_data:
                epochs = [d['epoch'] for d in stage2_data]
                losses = [d['loss'] for d in stage2_data]
                
                task_type = "Base" if task_id == 0 else f"Inc {task_id}"
                ax2.plot(epochs, losses, '-', color=colors[i], linewidth=2, 
                        label=f'Task {task_id} ({task_type})', marker='s', markersize=3, markevery=5)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Stage 2: Classifier Tuning - All Tasks', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.save_dir}/all_tasks_comparison_{scenario.replace('=', '')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved: {filename}")
    
    def plot_final_loss_summary(self, scenario='T=2'):
        """
        Plot bar chart of final losses for each task
        
        Args:
            scenario: Scenario name for filename
        """
        if not self.task_losses:
            print("Warning: No task data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        tasks = sorted(self.task_losses.keys())
        stage1_final = []
        stage2_final = []
        task_labels = []
        
        for task_id in tasks:
            # Get final losses
            s1_data = self.task_losses[task_id]['stage1']
            s2_data = self.task_losses[task_id]['stage2']
            
            stage1_final.append(s1_data[-1]['loss'] if s1_data else 0)
            stage2_final.append(s2_data[-1]['loss'] if s2_data else 0)
            
            task_type = "Base" if task_id == 0 else f"Inc {task_id}"
            num_classes = self.task_info[task_id]['num_classes']
            task_labels.append(f"Task {task_id}\n({task_type})\n{num_classes} cls")
        
        x = np.arange(len(tasks))
        width = 0.35
        
        # Bar chart
        bars1 = ax1.bar(x - width/2, stage1_final, width, label='Stage 1', 
                       color='steelblue', edgecolor='black', linewidth=1.2)
        bars2 = ax1.bar(x + width/2, stage2_final, width, label='Stage 2', 
                       color='coral', edgecolor='black', linewidth=1.2)
        
        ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Final Loss per Task (Stage 1 vs Stage 2)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_labels, fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Line plot showing progression
        ax2.plot(tasks, stage1_final, 'o-', linewidth=2, markersize=8, 
                color='steelblue', label='Stage 1', markeredgecolor='black', markeredgewidth=1.5)
        ax2.plot(tasks, stage2_final, 's-', linewidth=2, markersize=8, 
                color='coral', label='Stage 2', markeredgecolor='black', markeredgewidth=1.5)
        
        ax2.set_xlabel('Task ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Loss Progression Across Tasks', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(tasks)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.save_dir}/final_loss_summary_{scenario.replace('=', '')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved: {filename}")
    
    def plot_classes_vs_accuracy(self, accuracies_dict, scenario='T=2'):
        """
        Plot Number of Classes Learned vs Accuracy
        
        Args:
            accuracies_dict: {task_id: accuracy_percentage}
            scenario: Scenario name for filename
        """
        if not accuracies_dict or len(accuracies_dict) == 0:
            print("⚠️  Warning: No accuracy data to plot. Make sure to track accuracies for all tasks!")
            print("   Add: task_accuracies[t] = overall_accuracy after evaluation")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate cumulative number of classes
        tasks = sorted(self.task_info.keys())  # Use all tracked tasks
        cumulative_classes = []
        accuracies = []
        task_labels = []
        
        total_classes = 0
        for task_id in tasks:
            if task_id in self.task_info:
                total_classes += self.task_info[task_id]['num_classes']
                cumulative_classes.append(total_classes)
                
                # Get accuracy for this task (if available)
                if task_id in accuracies_dict:
                    accuracies.append(accuracies_dict[task_id])
                    task_labels.append(f"Task {task_id}")
                else:
                    print(f"⚠️  Warning: No accuracy found for Task {task_id}. Skipping this point.")
        
        if len(cumulative_classes) == 0 or len(accuracies) == 0:
            print("⚠️  Warning: No valid data points to plot!")
            return
        
        # Plot with markers
        ax.plot(cumulative_classes, accuracies, 'o-', linewidth=3, markersize=12,
                color='darkgreen', markerfacecolor='lightgreen', 
                markeredgewidth=2, markeredgecolor='darkgreen', label='Test Accuracy')
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(cumulative_classes, accuracies)):
            ax.text(x, y + 1.5, f'{y:.2f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Labels and title
        ax.set_xlabel('Number of Classes Learned', fontsize=14, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Class Incremental Learning: Classes Learned vs Accuracy', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set y-axis limits with some padding
        if accuracies:
            y_min = min(accuracies) - 5
            y_max = max(accuracies) + 5
            ax.set_ylim([max(0, y_min), min(100, y_max)])
        
        # Set x-axis to show integer values
        ax.set_xticks(cumulative_classes)
        
        # Add legend
        ax.legend(fontsize=12, loc='best')
        
        # Add task annotations with class count
        for i, (x, task_id) in enumerate(zip(cumulative_classes, tasks)):
            if task_id in accuracies_dict:  # Only annotate tasks we have data for
                task_type = "Base" if task_id == 0 else f"Inc-{task_id}"
                num_classes = self.task_info[task_id]['num_classes']
                ax.axvline(x, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
                
                # Annotate below x-axis
                y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                ax.text(x, y_pos, f'{task_type}\n(+{num_classes})', 
                       ha='center', va='bottom',
                       fontsize=9, color='darkblue', style='italic',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.save_dir}/classes_vs_accuracy_{scenario.replace('=', '')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved: {filename}")
        print(f"     Data points: {len(accuracies)} tasks plotted")
    
    def plot_classes_vs_loss(self, scenario='T=2'):
        """
        Plot Number of Classes Learned vs Final Loss
        
        Args:
            scenario: Scenario name for filename
        """
        if not self.task_losses:
            print("Warning: No loss data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate cumulative number of classes and final losses
        tasks = sorted(self.task_losses.keys())
        cumulative_classes = []
        stage1_losses = []
        stage2_losses = []
        
        total_classes = 0
        for task_id in tasks:
            if task_id in self.task_info:
                total_classes += self.task_info[task_id]['num_classes']
                cumulative_classes.append(total_classes)
                
                # Get final losses
                s1_data = self.task_losses[task_id]['stage1']
                s2_data = self.task_losses[task_id]['stage2']
                
                stage1_losses.append(s1_data[-1]['loss'] if s1_data else 0)
                stage2_losses.append(s2_data[-1]['loss'] if s2_data else 0)
        
        # Plot Stage 1
        ax1.plot(cumulative_classes, stage1_losses, 'o-', linewidth=3, markersize=12,
                color='darkblue', markerfacecolor='lightblue', 
                markeredgewidth=2, markeredgecolor='darkblue', 
                label='Stage 1 Final Loss')
        
        # Add value labels
        for x, y in zip(cumulative_classes, stage1_losses):
            ax1.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlabel('Number of Classes Learned', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Final Loss (Stage 1)', fontsize=13, fontweight='bold')
        ax1.set_title('Stage 1: Representation Learning Loss', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='best')
        
        # Add task annotations for ax1
        for i, (x, task_id) in enumerate(zip(cumulative_classes, tasks)):
            task_type = "Base" if task_id == 0 else f"T{task_id}"
            ax1.axvline(x, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            y_pos = ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05
            ax1.text(x, y_pos, task_type, ha='center', 
                    fontsize=8, color='gray', style='italic')
        
        # Plot Stage 2
        ax2.plot(cumulative_classes, stage2_losses, 's-', linewidth=3, markersize=12,
                color='darkred', markerfacecolor='lightcoral', 
                markeredgewidth=2, markeredgecolor='darkred', 
                label='Stage 2 Final Loss')
        
        # Add value labels
        for x, y in zip(cumulative_classes, stage2_losses):
            ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlabel('Number of Classes Learned', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Final Loss (Stage 2)', fontsize=13, fontweight='bold')
        ax2.set_title('Stage 2: Classifier Tuning Loss', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='best')
        
        # Add task annotations for ax2
        for i, (x, task_id) in enumerate(zip(cumulative_classes, tasks)):
            task_type = "Base" if task_id == 0 else f"T{task_id}"
            ax2.axvline(x, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            y_pos = ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.05
            ax2.text(x, y_pos, task_type, ha='center', 
                    fontsize=8, color='gray', style='italic')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.save_dir}/classes_vs_loss_{scenario.replace('=', '')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved: {filename}")
    
    def save_all_graphs(self, scenario='T=2', accuracies_dict=None):
        """
        Generate and save all graphs
        
        Args:
            scenario: Scenario name for filenames
            accuracies_dict: {task_id: accuracy_percentage} for accuracy plot
        """
        print("\n" + "="*80)
        print("📊 GENERATING LOSS GRAPHS")
        print("="*80)
        
        # Individual task plots
        for task_id in sorted(self.task_losses.keys()):
            self.plot_task_losses(task_id, scenario)
        
        # Comparison plots
        self.plot_all_tasks_comparison(scenario)
        
        # Summary plots
        self.plot_final_loss_summary(scenario)
        
        # NEW: Classes vs Loss plot
        self.plot_classes_vs_loss(scenario)
        
        # NEW: Classes vs Accuracy plot (if accuracy data provided)
        if accuracies_dict:
            self.plot_classes_vs_accuracy(accuracies_dict, scenario)
        
        print(f"\n✅ All graphs saved to: {self.save_dir}/")
        print("="*80 + "\n")
    
    def get_summary_stats(self):
        """
        Get summary statistics of losses
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        for task_id in sorted(self.task_losses.keys()):
            stage1_losses = [d['loss'] for d in self.task_losses[task_id]['stage1']]
            stage2_losses = [d['loss'] for d in self.task_losses[task_id]['stage2']]
            
            stats[task_id] = {
                'stage1': {
                    'initial': stage1_losses[0] if stage1_losses else None,
                    'final': stage1_losses[-1] if stage1_losses else None,
                    'min': min(stage1_losses) if stage1_losses else None,
                    'mean': np.mean(stage1_losses) if stage1_losses else None
                },
                'stage2': {
                    'initial': stage2_losses[0] if stage2_losses else None,
                    'final': stage2_losses[-1] if stage2_losses else None,
                    'min': min(stage2_losses) if stage2_losses else None,
                    'mean': np.mean(stage2_losses) if stage2_losses else None
                }
            }
        
        return stats
    
    def print_summary(self):
        """Print summary of tracked losses"""
        stats = self.get_summary_stats()
        
        print("\n" + "="*80)
        print("📊 LOSS TRACKING SUMMARY")
        print("="*80)
        
        for task_id in sorted(stats.keys()):
            task_type = "BASE" if task_id == 0 else f"INCREMENTAL {task_id}"
            num_classes = self.task_info[task_id]['num_classes']
            
            print(f"\nTask {task_id} [{task_type}] ({num_classes} classes):")
            print(f"  Stage 1 (Representation Learning):")
            print(f"    Initial Loss: {stats[task_id]['stage1']['initial']:.4f}")
            print(f"    Final Loss:   {stats[task_id]['stage1']['final']:.4f}")
            print(f"    Min Loss:     {stats[task_id]['stage1']['min']:.4f}")
            print(f"    Mean Loss:    {stats[task_id]['stage1']['mean']:.4f}")
            
            print(f"  Stage 2 (Classifier Tuning):")
            print(f"    Initial Loss: {stats[task_id]['stage2']['initial']:.4f}")
            print(f"    Final Loss:   {stats[task_id]['stage2']['final']:.4f}")
            print(f"    Min Loss:     {stats[task_id]['stage2']['min']:.4f}")
            print(f"    Mean Loss:    {stats[task_id]['stage2']['mean']:.4f}")
        
        print("="*80 + "\n")
