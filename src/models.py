from transformers import T5Config, T5ForConditionalGeneration

def get_t5_small_config(vocab_size):
    """Returns a T5-small configuration matching the project requirements."""
    return T5Config(
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        vocab_size=vocab_size,
        d_model=512,
        d_ff=2048,
        d_kv=64,
        num_heads=8,
        num_layers=6,
        num_decoder_layers=6,
        feed_forward_proj="gated-gelu"
    )

def init_model_from_scratch(vocab_size):
    config = get_t5_small_config(vocab_size)
    model = T5ForConditionalGeneration(config=config)
    model.resize_token_embeddings(vocab_size)
    return model
