import os
import torch
from transformers import AutoTokenizer

from clearnets.sparse_feedfwd_transformer.train_tinystories_transformers import (
    tiny_stories_8m_config,
    TinyStoriesModel
)

def generate_story(
    checkpoint_path: str,
    prompt: str = "",
    max_length: int = 100,
    temperature: float = 0.7,
    num_stories: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dense=False
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer")
    tiny_stories_8m_config["vocab_size"] = tokenizer.vocab_size  
    
    # Initialize model
    model = TinyStoriesModel.load_from_checkpoint(
        checkpoint_path,
        dense=dense, 
        tokenizer=tokenizer
    )
    model = model.model
    
    # Load checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Generate multiple stories
    stories = []
    for _ in range(num_stories):
        # Tokenize prompt if provided, otherwise start with BOS token
        if prompt:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)

        # Generate text
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and clean up the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        stories.append(generated_text)
    
    return stories

if __name__ == "__main__":
    # Find the latest checkpoint
    # checkpoint_dir = "data/tinystories-8/checkpoints"
    checkpoint_dir = "data/tinystories/dense-8m-max-e=200-esp=15-s=42/checkpoints"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    latest_checkpoint = max([os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints], key=os.path.getmtime)
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Generate stories with different prompts
    prompts = [
        "Once upon a time",
        "The little robot",
        "In a magical forest"
    ]
    
    for prompt in prompts:
        print(f"\nGenerating stories with prompt: '{prompt}'")
        stories = generate_story(
            checkpoint_path=latest_checkpoint,
            prompt=prompt,
            max_length=200,
            temperature=0.01,
            num_stories=2,
            dense = 'dense' in checkpoint_dir
        )
        
        for i, story in enumerate(stories, 1):
            print(f"\nStory {i}:")
            print(story)
            print("-" * 50)