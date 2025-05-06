import torch
from transformers import AutoModel, AutoTokenizer

def convert_to_torchscript():
    # Load model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save tokenizer
    tokenizer.save_pretrained("models")
    
    # Set model to eval mode
    model.eval()
    
    # Create example input
    example_input = tokenizer("This is a test sentence", return_tensors="pt")
    
    # Convert to TorchScript
    traced_model = torch.jit.trace(model, (example_input["input_ids"], example_input["attention_mask"]))
    
    # Save the model
    traced_model.save("models/embedding_model.pt")
    
    print("Model converted and saved successfully!")

if __name__ == "__main__":
    convert_to_torchscript() 