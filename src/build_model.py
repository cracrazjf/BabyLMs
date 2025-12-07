from psychai.nn_builder import ModelSpec, Model, save_config, build_config_dict, save_pretrained, from_pretrained
from transformers import AutoTokenizer
from psychai.config import ModelConfig, update_config
import torch

def build_lstm(cfg: ModelConfig):
    spec = ModelSpec(vocab_size=cfg.vocab_size)

    spec.add_layer({
        "type": "embedding",
        "vocab_size": cfg.vocab_size,
        "embed_size": cfg.embed_size,
        "kind": "one_hot",
    }, name="embed")
    
    spec.add_layer({
        "type": "lstm",
        "input_size": cfg.vocab_size,
        "embed_size": embed_size,
    }, name="lstm")
    
    spec.add_layer({
        "type": "lm_head",
        "embed_size": cfg.embed_size,
        "vocab_size": cfg.vocab_size,
        "bias": True,
    }, name="lm_head")
    
    model = Model(spec)
    return model

def build_transformer(cfg: ModelConfig): 
    spec = ModelSpec(vocab_size=cfg.vocab_size)

    spec.add_layer({
        "type": "embedding",
        "vocab_size": cfg.vocab_size,
        "embed_size": cfg.embed_size,
        "kind": "learned",
    }, name="wte")

    spec.add_layer({
        "type": "position_embedding",
        "embed_size": cfg.embed_size,
        "block_size": cfg.block_size
    }, name="wpe")

    for i in range(cfg.num_layers):
        spec.add_layer({
            "type": "decoder_block",
            "embed_size": cfg.embed_size,
            "block_size": cfg.block_size,
            "num_heads": cfg.num_heads,
            "activation": "gelu"
        }, name=f"h_{i}")

    spec.add_layer({
        "type": "layer_norm",
        "normalized_shape": cfg.embed_size
    }, name="ln_f")

    spec.add_layer({
        "type": "lm_head",
        "embed_size": cfg.embed_size,
        "vocab_size": cfg.vocab_size,
        "bias": False,
    }, name="lm_head")

    model = Model(spec)
    return model

def copy_gpt2_weights(model):
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tokenizer_hf = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer_hf.save_pretrained("/root/autodl-tmp/tokenizer/gpt2_tokenizer")
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()
        

        def _map_key(k: str) -> str | None:
            if k.startswith("transformer.h."):
                parts = k.split(".")
                layer_idx = parts[2]
                rest = ".".join(parts[3:])
                return f"layers.h_{layer_idx}.{rest}"

            if k == "transformer.wte.weight":
                return "layers.wte.emb.weight"

            if k == "transformer.wpe.weight":
                return "layers.wpe.emb.weight"

            if k == "transformer.ln_f.weight":
                return "layers.ln_f.ln.weight"
            if k == "transformer.ln_f.bias":
                return "layers.ln_f.ln.bias"

            if k == "lm_head.weight":
                return "layers.lm_head.proj.weight"

            return None
        
        renamed_sd_hf = {}
        for k, v in sd_hf.items():
            new_k = _map_key(k)
            if new_k is not None:
                renamed_sd_hf[new_k] = v


        sd_keys_hf = renamed_sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert renamed_sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(renamed_sd_hf[k].t())
            else:
                assert renamed_sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(renamed_sd_hf[k])


def main():
    # tokenizer = AutoTokenizer.from_pretrained("./tokenizer/childes_tokenizer")
    cfg = ModelConfig()
    cfg = update_config(cfg, {
        "name": "gpt-2",
        "model_type": "transformer",
        "path": "/root/autodl-tmp/models/gpt-2",
        "vocab_size": 50257,
        "embed_size": 768,
        "block_size": 1024,
        "num_heads": 12,
        "num_layers": 12,
    })
    model = None

    if cfg.model_type == "lstm":
        model = build_lstm(cfg)
        config_dict = build_config_dict(model, model_type=cfg.name,)
        save_config(cfg.path, config_dict)
        
    elif cfg.model_type == "transformer":
        model = build_transformer(cfg)
        if cfg.name == "gpt-2":
            copy_gpt2_weights(model)
            save_pretrained(model, cfg.path, prefer_safetensors=True)
        else:
            config_dict = build_config_dict(model, model_type=cfg.name,)
            save_config(cfg.path, config_dict)

if __name__ == "__main__":
    main()

