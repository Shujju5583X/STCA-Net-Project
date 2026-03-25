import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.fft
import math


class SinusoidalPositionEmbedding(nn.Module):
    """
    Resolution-adaptive sinusoidal position embeddings.
    Unlike fixed learned embeddings, these can interpolate to any sequence length.
    """
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, seq_len):
        """Return position embeddings, interpolated if needed."""
        if seq_len <= self.pe.size(1):
            return self.pe[:, :seq_len, :]
        else:
            # Interpolate for longer sequences
            pe = self.pe.transpose(1, 2)  # (1, d_model, max_len)
            pe = F.interpolate(pe, size=seq_len, mode='linear', align_corners=False)
            return pe.transpose(1, 2)  # (1, seq_len, d_model)


class FrequencyGating(nn.Module):
    """
    Learnable gating mechanism for fusing frequency-domain features
    with the global context. More expressive than simple addition.
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, global_context, freq_features):
        """
        Args:
            global_context: (B, 1, d_model) from transformer CLS token
            freq_features: (B, 1, d_model) from frequency extractor
        Returns:
            Gated fusion of both features (B, 1, d_model)
        """
        combined = torch.cat([global_context, freq_features], dim=-1)  # (B, 1, 2*d_model)
        gate_values = self.gate(combined)  # (B, 1, d_model)
        return global_context * gate_values + freq_features * (1 - gate_values)


class TemporalAggregator(nn.Module):
    """
    Lightweight temporal modeling over per-frame embeddings using a 1-layer LSTM.
    Used during video prediction to capture temporal inconsistencies between frames.
    """
    def __init__(self, d_model=256, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Bidirectional LSTM outputs 2 * hidden_size
        self.proj = nn.Linear(hidden_size * 2, d_model)

    def forward(self, frame_embeddings):
        """
        Args:
            frame_embeddings: (B, T, d_model) where T = number of frames
        Returns:
            Temporally-aware summary (B, d_model)
        """
        lstm_out, _ = self.lstm(frame_embeddings)  # (B, T, 2*hidden)
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (B, 2*hidden)
        return self.proj(last_output)  # (B, d_model)


class STCANet(nn.Module):
    """
    Spatio-Temporal Cross-Attention Hybrid Network (STCA-Net)
    A novel, lightweight architecture for Deepfake Detection.
    
    Architecture:
    1. Local Spatial Extractor: MobileNetV3-Small (Pretrained)
    2. Global Context Encoder: Shallow Vision Transformer (ViT)
    3. Frequency Domain Extractor: Log-scaled FFT + CNN with gated fusion
    4. Fusion: Cross-Attention between Local and Global features
    5. Temporal Aggregator: Bidirectional LSTM for video-level prediction
    """
    def __init__(self, num_classes=2, d_model=256, nhead=8, num_encoder_layers=2):
        super(STCANet, self).__init__()
        
        # 1. Local Spatial Extractor (MobileNetV3-Small)
        # We use a lightweight CNN to extract fine-grained pixel artifacts.
        # Output shape from feature extractor is typically (Batch, 576, 7, 7) for 224x224 input
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.spatial_extractor = mobilenet.features
        
        # Reduce the channel dimension from 576 to our d_model (256)
        self.conv_proj = nn.Conv2d(576, d_model, kernel_size=1)
        
        # 2. Global Context Encoder (Shallow Transformer)
        # We flatten the spatial dimensions (7x7 = 49 patches) and treat them as a sequence
        # Resolution-adaptive sinusoidal position embeddings
        self.pos_embedding = SinusoidalPositionEmbedding(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, # Keep bottleneck small
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Cross-Attention Fusion
        # We use the Transformer's CLS token to attend back to the original CNN spatial maps
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 4. Frequency Domain Extractor (Enhanced)
        # Uses log-scaled FFT magnitude + deeper CNN + gated fusion
        self.freq_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, d_model)
        )
        
        # Learned gating for frequency fusion (replaces simple addition)
        self.freq_gate = FrequencyGating(d_model)
        
        # 5. Temporal Aggregator (for video-level prediction)
        self.temporal_aggregator = TemporalAggregator(d_model)
        
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def extract_embedding(self, x):
        """
        Extract the fused embedding for a single image (before classification).
        Used by the temporal aggregator for video-level prediction.
        
        Args:
            x: (B, 3, 224, 224) input image tensor
        Returns:
            fused_features: (B, d_model) embedding
            attn_weights: attention weights from cross-attention
        """
        B = x.size(0)
        
        # Step 1: Spatial Extraction (CNN)
        spatial_features = self.spatial_extractor(x)
        proj_features = self.conv_proj(spatial_features)
        
        # Flatten for Transformer
        seq_features = proj_features.flatten(2).transpose(1, 2)
        seq_len = seq_features.size(1)
        
        # Step 2: Global Context (Transformer)
        # Resolution-adaptive position embeddings
        pos_emb = self.pos_embedding(seq_len)
        seq_features = seq_features + pos_emb
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        transformer_input = torch.cat((cls_tokens, seq_features), dim=1)
        transformer_output = self.transformer_encoder(transformer_input)
        global_context = transformer_output[:, 0:1, :]  # (B, 1, d_model)
        
        # Step 2.5: Frequency Domain Injection (Log-scaled)
        freq_complex = torch.fft.fft2(x, norm='ortho')
        freq_mag = torch.log1p(torch.abs(freq_complex))  # Log-scale for better dynamic range
        freq_feat = self.freq_extractor(freq_mag).unsqueeze(1)  # (B, 1, d_model)
        
        # Gated fusion instead of simple addition
        global_context = self.freq_gate(global_context, freq_feat)
        
        # Step 3: Cross-Attention Fusion
        fused_features, attn_weights = self.cross_attention(
            query=global_context,
            key=seq_features,
            value=seq_features
        )
        
        fused_features = fused_features.squeeze(1)  # (B, d_model)
        return fused_features, attn_weights
        
    def forward(self, x):
        """
        Forward pass for single image classification.
        
        Args:
            x: (B, 3, 224, 224) input image tensor
        Returns:
            output: (B, num_classes) classification logits
            attn_weights: attention weights from cross-attention
        """
        fused_features, attn_weights = self.extract_embedding(x)
        output = self.classifier(fused_features)
        return output, attn_weights
    
    def forward_temporal(self, frame_embeddings):
        """
        Video-level prediction using temporal aggregation.
        
        Args:
            frame_embeddings: (B, T, d_model) per-frame embeddings from extract_embedding
        Returns:
            output: (B, num_classes) classification logits
        """
        temporal_feat = self.temporal_aggregator(frame_embeddings)  # (B, d_model)
        return self.classifier(temporal_feat)
        
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick test to verify shapes and parameters
    model = STCANet()
    print(f"STCA-Net Total Trainable Parameters: {model.get_parameter_count():,}")
    
    # Dummy input (1 image, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    output, attn = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # Test embedding extraction
    emb, _ = model.extract_embedding(dummy_input)
    print(f"Embedding shape: {emb.shape}")
    
    # Test temporal aggregation (simulate 10 frames)
    frame_embs = torch.randn(1, 10, 256)
    temporal_out = model.forward_temporal(frame_embs)
    print(f"Temporal output shape: {temporal_out.shape}")
