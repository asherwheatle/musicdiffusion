from transformers import AutoModel
from miditok import MusicTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = MusicTokenizer.from_pretrained('shikhr/music_maker')
model = AutoModel.from_pretrained('shikhr/music_maker', trust_remote_code=True)
model.to(device)

# Generate some music
out = model.generate(
    torch.tensor([[1]]).to(device),
    max_new_tokens=400,
    temperature=1.0,
    top_k=None
)

# Save the generated MIDI
tokenizer(out[0].tolist()).dump_midi("generated.mid")
print("Done! Music saved to generated.mid")