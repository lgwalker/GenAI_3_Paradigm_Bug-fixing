import os
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from transformers import PreTrainedTokenizerFast

def train_java_tokenizer(corpus_path, vocab_size=16384, save_dir="./java_tokenizer"):
    """Trains a SentencePiece Unigram tokenizer and saves it as a HF Fast Tokenizer."""
    sentinel_tokens = [f"<extra_id_{i}>" for i in range(100)] #
    
    # Train SentencePiece
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix="sp_code",
        vocab_size=vocab_size,
        model_type="unigram",
        user_defined_symbols=",".join(sentinel_tokens),
        pad_id=0, 
        eos_id=1, 
        unk_id=2, 
        bos_id=-1,
        character_coverage=1.0,
        hard_vocab_limit=False
    )

    # Convert to HuggingFace format
    sp = spm.SentencePieceProcessor()
    sp.Load("sp_code.model")
    vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]
    
    tokenizer_obj = Tokenizer(Unigram(vocab, unk_id=2))
    tokenizer_obj.pre_tokenizer = Metaspace()
    tokenizer_obj.decoder = MetaspaceDecoder()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=sentinel_tokens,
    )
    
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    return tokenizer

def load_tokenizer(path="./java_tokenizer"):
    return PreTrainedTokenizerFast.from_pretrained(path)
