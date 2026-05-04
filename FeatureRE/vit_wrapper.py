import torch
import torch.nn as nn
import timm


class ViTWrapper(nn.Module):
    """
    Wrapper around timm's ViT that provides from_input_to_features / from_features_to_output
    for FeatureRE compatibility.

    State-dict keys exactly match the timm model (152 keys for vit_small_patch16_224).
    """

    def __init__(self, num_classes=10, in_channels=3, img_size=32, patch_size=4):
        super().__init__()

        # Create the timm model to copy submodules from
        vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
        )

        # Copy all named children as direct attributes so state_dict keys match
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.patch_embed = vit.patch_embed
        self.pos_drop = vit.pos_drop
        self.patch_drop = vit.patch_drop
        self.norm_pre = vit.norm_pre
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.fc_norm = vit.fc_norm
        self.head_drop = vit.head_drop
        self.head = vit.head

        # Copy config attributes needed for forward pass
        self.num_prefix_tokens = vit.num_prefix_tokens  # 1
        self.no_embed_class = vit.no_embed_class  # False
        self.global_pool = vit.global_pool  # 'token'
        self.grad_checkpointing = False

    def _pos_embed(self, x):
        """Prepend CLS token and add positional embedding."""
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        if not self.no_embed_class:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.no_embed_class:
            x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward_features(self, x):
        """patch_embed through norm (everything before the head)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def _pool(self, x):
        """Extract CLS token (global_pool='token')."""
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self._pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def from_input_to_features(self, x, index):
        """Extract features: everything through CLS token extraction. Returns (B, embed_dim)."""
        x = self.forward_features(x)
        x = self._pool(x)
        return x

    def from_features_to_output(self, x, index):
        """Features to logits: fc_norm -> head_drop -> head. Returns (B, num_classes)."""
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x
