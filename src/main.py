import torch

if __name__ == '__main__':
    print(f"Cuda {"is" if torch.cuda.is_available() else "isn't"} available!")
    