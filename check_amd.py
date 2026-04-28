import torch_directml
import torch

# Crea il dispositivo DirectML
device = torch_directml.device()

print("--- VERIFICA HARDWARE ---")
print(f"Versione Torch: {torch.__version__}")
print(f"Dispositivo DirectML: {device}")
print(f"GPU Rilevata: {torch_directml.device_name(0)}")