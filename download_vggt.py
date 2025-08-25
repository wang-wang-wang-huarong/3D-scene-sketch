import os
import torch

def download_vggt_pretrained(out_dir="weights", filename="vggt_model.pt"):
    """Download the VGGT pretrained weights from Hugging Face.

    Args:
        out_dir: Directory to save the weights.
        filename: Name of the saved weight file.
    Returns:
        Path to the downloaded weight file.
    """
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, filename)
    if not os.path.exists(output_path):
        torch.hub.download_url_to_file(url, output_path)
    return output_path


if __name__ == "__main__":
    path = download_vggt_pretrained()
    print(f"Saved VGGT pretrained model to {path}")