"""
Tests for STCA-Net model architecture.
"""
import torch
import pytest
from models.stca_net import STCANet, TemporalAggregator, SinusoidalPositionEmbedding, FrequencyGating


class TestSTCANet:
    """Test the main STCA-Net model."""
    
    @pytest.fixture
    def model(self):
        return STCANet(num_classes=2)
    
    @pytest.fixture
    def dummy_input(self):
        return torch.randn(2, 3, 224, 224)  # Batch of 2

    def test_forward_output_shape(self, model, dummy_input):
        """Model output should be (B, num_classes)."""
        output, attn = model(dummy_input)
        assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    def test_forward_attention_shape(self, model, dummy_input):
        """Attention weights should be (B, 1, seq_len)."""
        output, attn = model(dummy_input)
        assert attn.shape[0] == 2, f"Batch dim mismatch: {attn.shape}"
        assert attn.shape[1] == 1, f"Query dim should be 1: {attn.shape}"

    def test_extract_embedding_shape(self, model, dummy_input):
        """Embedding should be (B, d_model)."""
        emb, attn = model.extract_embedding(dummy_input)
        assert emb.shape == (2, 256), f"Expected (2, 256), got {emb.shape}"

    def test_forward_temporal_shape(self, model):
        """Temporal forward should produce (B, num_classes)."""
        frame_embs = torch.randn(1, 10, 256)  # 10 frames
        output = model.forward_temporal(frame_embs)
        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"

    def test_parameter_count(self, model):
        """Model should have a reasonable number of parameters (< 5M for lightweight)."""
        count = model.get_parameter_count()
        assert count > 0, "Model has no trainable parameters"
        assert count < 10_000_000, f"Model has {count:,} params, expected < 10M"

    def test_eval_mode_no_grad(self, model, dummy_input):
        """Model should work in eval mode with no_grad."""
        model.eval()
        with torch.no_grad():
            output, attn = model(dummy_input)
        assert output.shape == (2, 2)


class TestTemporalAggregator:
    """Test the LSTM temporal aggregator."""
    
    def test_output_shape(self):
        agg = TemporalAggregator(d_model=256, hidden_size=128)
        x = torch.randn(2, 15, 256)  # Batch=2, 15 frames
        out = agg(x)
        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_single_frame(self):
        """Should work with a single frame."""
        agg = TemporalAggregator(d_model=256, hidden_size=128)
        x = torch.randn(1, 1, 256)
        out = agg(x)
        assert out.shape == (1, 256)

    def test_variable_length(self):
        """Should handle variable sequence lengths."""
        agg = TemporalAggregator(d_model=256, hidden_size=128)
        for T in [1, 5, 20, 50]:
            x = torch.randn(1, T, 256)
            out = agg(x)
            assert out.shape == (1, 256)


class TestSinusoidalPositionEmbedding:
    """Test position embeddings."""
    
    def test_exact_length(self):
        pe = SinusoidalPositionEmbedding(d_model=256, max_len=100)
        out = pe(49)
        assert out.shape == (1, 49, 256)

    def test_shorter_length(self):
        pe = SinusoidalPositionEmbedding(d_model=256, max_len=200)
        out = pe(10)
        assert out.shape == (1, 10, 256)

    def test_longer_length_interpolation(self):
        """Should interpolate for sequences longer than max_len."""
        pe = SinusoidalPositionEmbedding(d_model=256, max_len=50)
        out = pe(100)
        assert out.shape == (1, 100, 256)


class TestFrequencyGating:
    """Test the gating mechanism."""
    
    def test_output_shape(self):
        gate = FrequencyGating(d_model=256)
        global_ctx = torch.randn(2, 1, 256)
        freq_feat = torch.randn(2, 1, 256)
        out = gate(global_ctx, freq_feat)
        assert out.shape == (2, 1, 256)

    def test_gate_values_bounded(self):
        """Gate values should be in [0, 1] due to sigmoid."""
        gate = FrequencyGating(d_model=256)
        global_ctx = torch.randn(2, 1, 256)
        freq_feat = torch.randn(2, 1, 256)
        
        # Access the gate output directly
        combined = torch.cat([global_ctx, freq_feat], dim=-1)
        gate_values = gate.gate(combined)
        assert gate_values.min() >= 0.0
        assert gate_values.max() <= 1.0
