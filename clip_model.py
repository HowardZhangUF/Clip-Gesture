import torch
import open_clip
from PIL import Image

class ClipZeroShot:
    def __init__(self, backbone: str, weights: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=weights
        )
        self.tokenizer = open_clip.get_tokenizer(backbone)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def build_text_prototypes(self, classes: dict):
        """Average text embeddings per class over prompt templates."""
        out = {}
        for label, prompts in classes.items():
            toks = self.tokenizer(prompts).to(self.device)
            t = self.model.encode_text(toks)
            t = torch.nn.functional.normalize(t, dim=-1)
            out[label] = t.mean(dim=0, keepdim=True)  # [1, D]
        labels = list(out.keys())
        text_stack = torch.cat([out[k] for k in labels], dim=0)  # [C, D]
        return labels, text_stack

    @torch.no_grad()
    def encode_images(self, pil_images):
        """
        pil_images: list[PIL.Image.Image] or single PIL.Image.Image
        returns L2-normalized embeddings [N, D]
        """
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]
        ims = torch.cat([self.preprocess(im).unsqueeze(0) for im in pil_images], dim=0)
        ims = ims.to(self.device)
        img_emb = self.model.encode_image(ims)
        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
        return img_emb
