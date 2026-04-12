import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
# Commenting this out; more discussion later
# from torch.ao.quantization import fuse_modules
import onnxruntime as ort 
from tqdm import tqdm 
import scipy.stats as stats 

epsilon = 1e-7

# Simple fully connected network
class functionModel(nn.Module):
    def __init__(self):
        super(functionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.net(x)

# Save model in pth and onnx formats 
def save_model(model, optimizer, epoch, loss, device):
    model.cpu().eval()
    checkpoint={'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'epoch' : epoch,
            'loss' : loss} 
    # This actually does nothing for this current net since it uses Tanh not ReLU
    # Main optimization for nets with Conv/BN/ReLU patterns, but we include it here for demonstration and future extensibility  
    # fuse_modules(model.net, [['0', '1'], ['2', '3']], inplace=True)
    torch.save(checkpoint, '../model_files/function_model.pth')
    dummy_input = torch.randn(1, 1)
    torch.onnx.export(
        model, dummy_input,
        "../model_files/function_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        training=torch.onnx.TrainingMode.EVAL
    )
    session = ort.InferenceSession("../model_files/function_model.onnx")
    test_input = dummy_input.numpy()
    ort_output = session.run(None, {'input': test_input})[0]
    torch_output = model(dummy_input).detach().numpy()
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-4, atol=1e-5)
    model.to(device)
    print("ONNX round-trip verification passed")

def get_CAWR_next_reset(current_epoch, T_0, T_mult):
    cycle_end = T_0
    cycle_length = T_0
    while cycle_end <= current_epoch:
        cycle_length *= T_mult
        cycle_end += cycle_length
    return cycle_end

# Run the training 
def train_model(model, x_train, y_train, n_epochs=6200, lr=0.001, x_valid=None, y_valid=None):
    patience = 200
    epochs_no_improve = 0
    best_valid_loss = float('inf')
    best_state = None
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-86)
    train_losses = []
    valid_losses = []
    lr_per_epoch = []

    check_state = True
    return_epoch = -999

    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        lr_per_epoch.append(scheduler.get_last_lr()[0])
        scheduler.step()

        model.eval()
        if x_valid is not None and y_valid is not None:
            with torch.no_grad():
                valid_loss = criterion(model(x_valid), y_valid).item()
                valid_losses.append(valid_loss)
            if epoch % 100 == 0:
                tqdm.write(f"Epoch {epoch}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}")

            # Early stopping on validation loss
            if check_state:
                if valid_loss < best_valid_loss - epsilon:
                    best_valid_loss = valid_loss
                    epochs_no_improve = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    tqdm.write(f"Early stopping signaled at epoch {epoch}, best validation loss: {best_valid_loss:.6f}")
                    model.load_state_dict(best_state)
                    check_state = False
                    return_epoch = get_CAWR_next_reset(epoch, T_0=100, T_mult=2)
                    tqdm.write(f"Will check for CAWR reset at epoch {return_epoch}")
            elif epoch >= return_epoch:
                tqdm.write(f"Resetting early stopping at epoch {epoch}")
                break

    save_model(model=model, optimizer=optimizer, epoch=n_epochs, loss=train_losses[-1], device=device)
    return model, train_losses, (valid_losses if x_valid is not None else None), lr_per_epoch

# Generate data of function y = sin(x) * exp(-0.1 * x^2) 
def generate_data(n_samples=10000, x0=-10, x1=10, addNoise = False, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.linspace(x0, x1, n_samples)
    x2 = x*x
    y = np.sin(x) * np.exp(-0.1 * x2)
    # If we want to add noise to the data
    if addNoise:
        noise = np.random.normal(0, 0.1*np.abs(y), size=y.shape)
        y += noise
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
    return x_tensor, y_tensor

def quantify_model(y_truth, x_model, y_model, name=""):
    _, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10), constrained_layout=True)
    # Compute residuals 
    residuals = y_truth - y_model
    ax[0,0].hist(residuals, bins=50, color='blue', alpha=0.7)
    ax[0,0].set_title('Residuals Distribution')
    ax[0,0].set_xlabel('Residual')
    ax[0,0].set_ylabel('Frequency')
    mean_residuals = np.mean(residuals)
    sigma_residuals = np.std(residuals)
    ax[0,0].text(0.95, 0.95, f"Mean: {mean_residuals:.4f}\nStd: {sigma_residuals:.4f}", transform=ax[0,0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Plot truth and predicted 
    ax[0,1].scatter(x_model, y_truth, color='blue', label='True', alpha=0.5)
    ax[0,1].scatter(x_model, y_model, color='orange', label='Predicted', alpha=0.5)
    ax[0,1].set_title('True and Predicted')
    ax[0,1].set_xlabel('x Value')
    ax[0,1].set_ylabel('y Value')
    ax[0,1].legend()
    # Plot true vs predicted
    ax[1,0].hist2d(y_model, y_truth, bins=40, cmap='viridis', alpha=0.5, cmin=1)
    ax[1,0].set_title('True vs Predicted')
    ax[1,0].set_xlabel('Predicted')
    ax[1,0].set_ylabel('True')
    ax[1,0].plot([y_truth.min(), y_truth.max()], [y_truth.min(), y_truth.max()], color='red', linestyle='--')
    # Plot residuals vs predicted
    ax[1,1].scatter(x_model, residuals, color='purple', alpha=0.5)
    ax[1,1].set_title('Residuals vs x Value')
    ax[1,1].set_xlabel('x Value')
    ax[1,1].set_ylabel('Residual')
    plt.savefig(f"../python_figures/{name}_quantify.pdf")
    plt.close()

    # Generate R^2, chi^2 and MAE 
    r2 = 1 - np.sum(residuals**2) / np.sum((y_truth - np.mean(y_truth))**2)
    mae = np.mean(np.abs(residuals))
    max_abs_error = np.max(np.abs(residuals))
    # Estimate chi^2 using standard deviation of truth values
    sigma = 0.1 * np.std(y_truth)
    chi2 = np.sum((residuals / (sigma))**2)
    chi2_per_dof = chi2 / (len(y_truth) - 1) 
    print(f"{name.capitalize()} R^2: {r2:.4f}, MAE: {mae:.4f}, Max Abs Error: {max_abs_error:.4f}, Chi^2: {chi2_per_dof:.4f}")

    _, ax1 = plt.subplots(ncols=2, nrows=1, figsize=(10,10), constrained_layout=True)
    # QQ plot 
    stats.probplot(residuals, dist="norm", plot=ax1[0])
    ax1[0].set_title('QQ Plot of Residuals')
    # Residuals vs predicted
    ax1[1].scatter(y_model, residuals, color='purple', alpha=0.5)
    ax1[1].set_title('Residuals vs Predicted')
    ax1[1].set_xlabel('Predicted')
    ax1[1].set_ylabel('Residual')

    plt.savefig(f"../python_figures/{name}_further_quantification.pdf")
    plt.close()

def plot_losses_and_output(x_truth = None, y_truth = None, y_pred = None, x_train = None, y_train = None, train_loss = None, valid_loss = None, lr_per_epoch = None):
    plt.figure(figsize=(10, 6))
    x_truth_cpu = x_truth.cpu().numpy()
    plt.plot(x_truth_cpu, y_truth.cpu().numpy(), color='green', label='True Function', linewidth=2, linestyle='--')
    plt.plot(x_truth_cpu, y_pred, color='orange', label='Model Prediction', linewidth=2)
    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), color='red', label='Noisy Training Data', alpha=0.5)
    plt.title('Model Prediction vs True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../python_figures/model_prediction.pdf")
    plt.close()

    # Plot losses and lr on separate y axis
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(range(len(train_loss)), train_loss, color='red', label='Training Loss')
    ax1.plot(range(len(valid_loss)), valid_loss, color='blue', label='Validation Loss')
    ax2.set_ylabel('Learning Rate', color='green')
    ax2.tick_params(axis='y', colors='green')
    ax2.plot(range(len(lr_per_epoch)), lr_per_epoch, color='green', label='Learning Rate')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.tight_layout()
    plt.savefig("../python_figures/losses.pdf")
    plt.close()

if __name__ == "__main__":
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_truth, y_truth = generate_data(n_samples=1000, addNoise=False, device=device)
    x_train, y_train = generate_data(n_samples=10000, addNoise=True, device=device)
    x_valid, y_valid = generate_data(n_samples=2000, addNoise=True, device=device)
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), color='red', label='Noisy Training Data', alpha=0.5)
    plt.scatter(x_valid.cpu().numpy(), y_valid.cpu().numpy(), color='blue', label='Noisy Validation Data', alpha=0.5)
    plt.plot(x_truth.cpu().numpy(), y_truth.cpu().numpy(), color='green', label='True Function', linewidth=2)
    plt.title('Data for Function Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.savefig("../python_figures/training_validation_truth.pdf")
    plt.close()
    
    # Run the training 
    model = functionModel().to(device)
    model, train_loss, valid_loss, lr_per_epoch = train_model(model, x_train, y_train, n_epochs=6200, lr=0.001, x_valid=x_valid, y_valid=y_valid)
    print(f"Final Training Loss: {train_loss[-1]:.4f}, Final Validation Loss: {valid_loss[-1]:.4f}")
    
    # Plot the model predictions
    with torch.no_grad():
        y_pred = model(x_truth).cpu().numpy()
    plot_losses_and_output(x_truth=x_truth, y_truth=y_truth.cpu(), y_pred=y_pred, x_train=x_train, y_train=y_train, train_loss=train_loss, valid_loss=valid_loss, lr_per_epoch=lr_per_epoch)
    # Quantify model 
    with torch.no_grad():
        xy_per_model = {
            "training": (x_train, y_train),
            "validation": (x_valid, y_valid),
            "random": generate_data(n_samples=1000, addNoise=True, device=device),
            "no_noise": generate_data(n_samples=1000, addNoise=False, device=device)
        }
        for modelName, (xData, yData) in xy_per_model.items():
            print(f"Quantifying {modelName} data:")
            y_model = model(xy_per_model[modelName][0]).cpu().numpy().squeeze()
            quantify_model(
                y_truth=yData.cpu().numpy().squeeze(),
                x_model=xData.cpu().numpy().squeeze(),
                y_model=y_model,
                name=modelName
            )