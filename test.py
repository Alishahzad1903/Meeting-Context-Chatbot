# import torch
# print(torch.cuda.is_available())  # Should return True if CUDA is working
import torch
print(torch.backends.cudnn.is_available())  # Should return True if cuDNN is available
