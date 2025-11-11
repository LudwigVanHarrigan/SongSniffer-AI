import torch
from SimpleCNN1 import SimpleCNN

# Load the original checkpoint
original_checkpoint_path = "SimpleCNN1_T4.pth"
updated_checkpoint_path = "SimpleCNN1_T4_updated.pth"

# Define the new model architecture
num_classes = 2
model = SimpleCNN(num_classes=num_classes, img_height=224, img_width=224)

# Load the original state_dict
original_state_dict = torch.load(original_checkpoint_path, map_location="cpu")

# Update the state_dict to match the new architecture
new_state_dict = model.state_dict()
for key in new_state_dict.keys():
    if key in original_state_dict and new_state_dict[key].shape == original_state_dict[key].shape:
        new_state_dict[key] = original_state_dict[key]
    else:
        print(f"Skipping incompatible layer: {key}")

# Save the updated checkpoint
torch.save(new_state_dict, updated_checkpoint_path)
print(f"Updated checkpoint saved to {updated_checkpoint_path}")