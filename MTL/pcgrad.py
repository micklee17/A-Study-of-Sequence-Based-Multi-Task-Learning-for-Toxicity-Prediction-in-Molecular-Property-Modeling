import torch
import random
from typing import List, Dict, Any, Tuple, Optional

class PCGrad(torch.optim.Optimizer):
    """
    PCGrad optimizer wrapper that modifies gradients to reduce task interference.
    Implementation based on "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020).
    """
    def __init__(self, optimizer, reduction: str = 'mean'):
        """
        Args:
            optimizer: The base optimizer to wrap
            reduction: How to combine projected gradients ('mean' or 'sum')
        """
        self.optimizer = optimizer
        self.reduction = reduction
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._params = []
        for group in optimizer.param_groups:
            self._params.extend(group['params'])

    def zero_grad(self):
        """Clear gradients of all parameters."""
        self.optimizer.zero_grad()

    def step(self, per_task_losses: List[torch.Tensor], retain_graph: bool = False):
        """
        Performs a PCGrad update step.
        
        Args:
            per_task_losses: List of task-specific losses
            retain_graph: Whether to retain computation graph after backward
        """
        # Get gradients for each task
        task_grads = self._compute_task_gradients(per_task_losses, retain_graph)
        # Apply PCGrad: project conflicting gradients
        projected_grads = self._project_conflicting_gradients(task_grads)
        # Set the modified gradients
        self._set_modified_gradients(projected_grads)
        # Perform optimizer step
        self.optimizer.step()

    def _compute_task_gradients(self, per_task_losses: List[torch.Tensor], 
                                retain_graph: bool) -> List[Dict[torch.nn.Parameter, torch.Tensor]]:
        """Compute and store gradients per task."""
        task_grads = []
        num_tasks = len(per_task_losses)
        
        for i, loss in enumerate(per_task_losses):
            self.optimizer.zero_grad()
            
            # Critical fix: Always retain graph for all tasks except the last one
            # This ensures we can backprop through the same computation graph multiple times
            retain = True if i < num_tasks - 1 else retain_graph
            
            # Backward pass for this task
            loss.backward(retain_graph=retain)
            
            # Store the gradients for each parameter for this task
            task_grad = {}
            for param in self._params:
                if param.grad is not None:
                    # Clone the gradient to prevent it from being overwritten
                    task_grad[param] = param.grad.clone()
            
            task_grads.append(task_grad)
        
        return task_grads

    def _project_conflicting_gradients(self, 
                           task_grads: List[Dict[torch.nn.Parameter, torch.Tensor]]) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Project gradients to avoid conflicts."""
        # Shuffle task order to prevent bias 
        task_indices = list(range(len(task_grads)))
        random.shuffle(task_indices)
        
        # Initialize a dictionary to hold the modified gradients
        modified_grads = {}
        for param in self._params:
            modified_grads[param] = None
        
        # Apply PCGrad projection
        for i in task_indices:
            task_grad_i = task_grads[i]
            
            # Skip if this task doesn't have gradients
            if not task_grad_i:
                continue
                
            for param in self._params:
                if param not in task_grad_i or task_grad_i[param] is None:
                    continue
                
                grad_i = task_grad_i[param]
                
                # For each conflicting task gradient, project if dot product is negative
                for j in range(len(task_grads)):
                    if i == j:
                        continue
                        
                    task_grad_j = task_grads[j]
                    if param not in task_grad_j or task_grad_j[param] is None:
                        continue
                        
                    grad_j = task_grad_j[param]
                    
                    # Calculate dot product to determine conflict
                    dot_product = torch.sum(grad_i * grad_j)
                    
                    # Project if conflicting
                    if dot_product < 0:
                        # Calculate the projection
                        grad_j_norm_squared = torch.sum(grad_j * grad_j)
                        # Avoid division by zero
                        if grad_j_norm_squared > 0:
                            # Project grad_i onto grad_j
                            projection = dot_product / grad_j_norm_squared * grad_j
                            # Subtract the projection to avoid conflict
                            grad_i = grad_i - projection
                
                # Accumulate the projected gradient
                if modified_grads[param] is None:
                    modified_grads[param] = grad_i
                else:
                    if self.reduction == 'mean':
                        modified_grads[param] += grad_i / len(task_grads)
                    else:  # 'sum'
                        modified_grads[param] += grad_i
        
        # If we're using mean reduction and it wasn't applied during accumulation
        if self.reduction == 'mean':
            for param in modified_grads:
                if modified_grads[param] is not None:
                    # Scale by number of tasks with gradients for this param
                    count = sum(1 for task_grad in task_grads if param in task_grad and task_grad[param] is not None)
                    if count > 0:
                        modified_grads[param] /= count
        
        return modified_grads

    def _set_modified_gradients(self, modified_grads: Dict[torch.nn.Parameter, torch.Tensor]) -> None:
        """Set the modified gradients to parameters."""
        self.optimizer.zero_grad()
        
        for param in self._params:
            if param in modified_grads and modified_grads[param] is not None:
                param.grad = modified_grads[param]
