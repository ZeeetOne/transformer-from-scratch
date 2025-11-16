"""
Visualization Data Extractor

Processes raw model outputs into visualization-friendly formats.
"""

import numpy as np
from typing import Dict, List, Any, Optional


class VisualizationExtractor:
    """
    Extract and format visualization data from transformer outputs.

    This class provides utilities to:
    - Format attention weights for heatmap visualization
    - Extract token flows through layers
    - Compute attention statistics
    - Prepare data for frontend consumption
    """

    @staticmethod
    def extract_attention_heatmaps(viz_data: Dict, layer_idx: int = 0) -> Dict:
        """
        Extract attention weights for heatmap visualization.

        Args:
            viz_data: Visualization data from model forward pass
            layer_idx: Which layer to extract (default: 0 for first layer)

        Returns:
            Dictionary with formatted attention data for each head
        """
        # Navigate to encoder attention weights
        encoder_layers = viz_data.get('encoder', {}).get('layer_wise_details', [])

        if layer_idx >= len(encoder_layers):
            return {"error": f"Layer {layer_idx} not found"}

        layer_data = encoder_layers[layer_idx]
        attention_data = layer_data.get('sublayer_1_self_attention', {})
        heads_data = attention_data.get('attention_per_head', [])

        # Format for frontend
        heatmaps = []
        for head in heads_data:
            heatmaps.append({
                "head_index": head['head_index'],
                "weights": head['attention_weights'],  # Shape: (batch, seq, seq)
                "entropy": head['avg_attention_entropy'],
                "focus_pattern": head['max_attention_positions']
            })

        return {
            "layer": layer_idx,
            "n_heads": len(heatmaps),
            "heatmaps": heatmaps
        }

    @staticmethod
    def extract_embedding_progression(viz_data: Dict) -> Dict:
        """
        Extract how embeddings transform through the network.

        Shows: Token Embedding → +Positional → Layer 1 → Layer 2 → ...

        Args:
            viz_data: Visualization data from model forward pass

        Returns:
            Dictionary with embedding progression data
        """
        encoder_data = viz_data.get('encoder', {})
        embedding_data = encoder_data.get('embedding', {})

        progression = {
            "step_1_token_embedding": embedding_data.get('step_1_token_embedding', {}),
            "step_2_with_position": embedding_data.get('step_2_positional_encoding', {}),
            "layer_outputs": encoder_data.get('layer_wise_outputs', [])
        }

        return progression

    @staticmethod
    def extract_positional_encoding_patterns(viz_data: Dict) -> Dict:
        """
        Extract positional encoding patterns for visualization.

        Shows the sinusoidal patterns across positions and dimensions.

        Args:
            viz_data: Visualization data from model forward pass

        Returns:
            Positional encoding pattern data
        """
        encoder_data = viz_data.get('encoder', {})
        embedding_data = encoder_data.get('embedding', {})
        pos_data = embedding_data.get('step_2_positional_encoding', {})

        return {
            "positional_encoding": pos_data.get('positional_encoding', []),
            "pattern": pos_data.get('encoding_pattern', {}),
            "sequence_length": pos_data.get('sequence_length', 0)
        }

    @staticmethod
    def extract_attention_flow(viz_data: Dict, layer_idx: int = 0, head_idx: int = 0) -> Dict:
        """
        Extract attention flow for a specific head to visualize token relationships.

        Args:
            viz_data: Visualization data from model forward pass
            layer_idx: Layer index
            head_idx: Attention head index

        Returns:
            Attention flow data showing which tokens attend to which
        """
        encoder_layers = viz_data.get('encoder', {}).get('layer_wise_details', [])

        if layer_idx >= len(encoder_layers):
            return {"error": f"Layer {layer_idx} not found"}

        layer_data = encoder_layers[layer_idx]
        attention_data = layer_data.get('sublayer_1_self_attention', {})
        heads_data = attention_data.get('attention_per_head', [])

        if head_idx >= len(heads_data):
            return {"error": f"Head {head_idx} not found"}

        head_data = heads_data[head_idx]
        weights = np.array(head_data['attention_weights'])  # Shape: (batch, seq, seq)

        # For first batch item
        if len(weights.shape) > 2:
            weights = weights[0]  # (seq, seq)

        # Extract top-k connections for each token
        seq_len = weights.shape[0]
        flows = []

        for src_idx in range(seq_len):
            attention_weights = weights[src_idx]  # Attention from token i to all tokens

            # Get top 5 attended tokens
            top_k = min(5, seq_len)
            top_indices = np.argsort(attention_weights)[-top_k:][::-1]

            flows.append({
                "source_token": src_idx,
                "targets": [
                    {
                        "token_idx": int(tgt_idx),
                        "weight": float(attention_weights[tgt_idx])
                    }
                    for tgt_idx in top_indices
                ]
            })

        return {
            "layer": layer_idx,
            "head": head_idx,
            "flows": flows
        }

    @staticmethod
    def extract_feed_forward_activations(viz_data: Dict, layer_idx: int = 0) -> Dict:
        """
        Extract feed-forward network activation patterns.

        Args:
            viz_data: Visualization data
            layer_idx: Layer index

        Returns:
            Feed-forward activation data
        """
        encoder_layers = viz_data.get('encoder', {}).get('layer_wise_details', [])

        if layer_idx >= len(encoder_layers):
            return {"error": f"Layer {layer_idx} not found"}

        layer_data = encoder_layers[layer_idx]
        ff_data = layer_data.get('sublayer_2_feed_forward', {})

        return {
            "layer": layer_idx,
            "hidden_dim": ff_data.get('hidden_dim', 0),
            "expansion_factor": ff_data.get('expansion_factor', 0),
            "activation_stats": ff_data.get('activation_stats', {}),
            "sparsity": ff_data.get('activation_stats', {}).get('sparsity', 0)
        }

    @staticmethod
    def create_architecture_summary(viz_data: Dict) -> Dict:
        """
        Create high-level architecture summary for visualization.

        Args:
            viz_data: Complete visualization data

        Returns:
            Architecture summary with layer counts, dimensions, etc.
        """
        arch_info = viz_data.get('architecture_info', {})
        encoder_data = viz_data.get('encoder', {})
        decoder_data = viz_data.get('decoder', {})

        return {
            "model_type": "Transformer (Encoder-Decoder)",
            "d_model": arch_info.get('d_model', 0),
            "n_encoder_layers": arch_info.get('n_encoder_layers', 0),
            "n_decoder_layers": arch_info.get('n_decoder_layers', 0),
            "encoder_layers": len(encoder_data.get('layer_wise_outputs', [])),
            "decoder_layers": len(decoder_data.get('layer_wise_outputs', [])),
            "has_cross_attention": True
        }

    @staticmethod
    def extract_layer_comparison(viz_data: Dict) -> Dict:
        """
        Compare representations across layers.

        Shows how representations evolve through the network.

        Args:
            viz_data: Visualization data

        Returns:
            Layer-wise comparison data
        """
        encoder_layers = viz_data.get('encoder', {}).get('layer_wise_outputs', [])

        layer_stats = []
        for idx, layer_output in enumerate(encoder_layers):
            output_array = np.array(layer_output)

            layer_stats.append({
                "layer": idx,
                "shape": list(output_array.shape),
                "mean": float(np.mean(output_array)),
                "std": float(np.std(output_array)),
                "min": float(np.min(output_array)),
                "max": float(np.max(output_array))
            })

        return {
            "n_layers": len(layer_stats),
            "layer_statistics": layer_stats
        }

    @staticmethod
    def format_for_frontend(
        viz_data: Dict,
        source_tokens: List[str],
        target_tokens: Optional[List[str]] = None
    ) -> Dict:
        """
        Format all visualization data for frontend consumption.

        Args:
            viz_data: Raw visualization data from model
            source_tokens: Source token strings (for labeling)
            target_tokens: Target token strings

        Returns:
            Complete formatted visualization data
        """
        extractor = VisualizationExtractor

        return {
            "architecture": extractor.create_architecture_summary(viz_data),
            "embeddings": {
                "progression": extractor.extract_embedding_progression(viz_data),
                "positional_encoding": extractor.extract_positional_encoding_patterns(viz_data)
            },
            "attention": {
                "all_layers": [
                    extractor.extract_attention_heatmaps(viz_data, layer_idx=i)
                    for i in range(viz_data.get('architecture_info', {}).get('n_encoder_layers', 0))
                ]
            },
            "feed_forward": {
                "all_layers": [
                    extractor.extract_feed_forward_activations(viz_data, layer_idx=i)
                    for i in range(viz_data.get('architecture_info', {}).get('n_encoder_layers', 0))
                ]
            },
            "layer_comparison": extractor.extract_layer_comparison(viz_data),
            "tokens": {
                "source": source_tokens,
                "target": target_tokens
            }
        }
