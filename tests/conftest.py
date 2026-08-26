"""Shared fixtures: synthetic preference data and a tiny offline Llama model.

The tiny model (~400k params, random weights, freshly-trained BPE tokenizer)
is built once per test session and saved to disk, so every offline integration
test — DPO, SFT, GRPO — exercises the real ``from_pretrained`` loading path
without any network access.
"""

import pytest

PREFERENCE_PAIRS = {
    "prompt": [
        "What is the capital of France?",
        "How many legs does a spider have?",
        "What color is the sky on a clear day?",
    ],
    "chosen": [
        "The capital of France is Paris.",
        "A spider has eight legs.",
        "The sky is blue on a clear day.",
    ],
    "rejected": [
        "Well, that is a really interesting question, and after this long preamble the "
        "answer is of course the world-famous city of Paris.",
        "Great question! Spiders are fascinating arachnids famously known throughout the "
        "animal kingdom for possessing a grand total of eight legs altogether.",
        "Ah, the sky! To human observers it appears to be a beautiful shade of blue due "
        "to a phenomenon called Rayleigh scattering.",
    ],
}


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Path to a tiny random Llama model + tokenizer saved on local disk."""
    torch = pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("transformers", reason="requires transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    save_dir = tmp_path_factory.mktemp("tiny-llama")
    corpus = sum(PREFERENCE_PAIRS.values(), [])
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.Whitespace()
    bpe.train_from_iterator(
        corpus, trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<s>", "</s>"])
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir
