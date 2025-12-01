def export_model_for_inference():
    """
    Export the generator model for easy inference.
    """
    # Save only the generator
    inference_model_path = os.path.join(SAVE_DIR, "generator_inference.pth")
    torch.save(G.state_dict(), inference_model_path)
    print(f"Generator model exported to: {inference_model_path}")

    # Save model info
    model_info = {
        'input_size': (400, 600),
        'input_channels': 3,
        'output_channels': 3,
        'preprocessing': 'simple_retinex',
        'best_val_loss': best_val_loss,
    }

    import json
    info_path = os.path.join(SAVE_DIR, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model info saved to: {info_path}")

export_model_for_inference()