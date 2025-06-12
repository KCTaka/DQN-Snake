import torch
from helper import load_model
from snake_model import CNN

def export_image_model_to_onnx(episode=None):
    # Load the latest or specified image-mode model
    model_info = load_model('image', episode)
    input_shape, cnn_structure = model_info['model_args']
    model = CNN(input_shape, **cnn_structure)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()

    # Dummy input matching the model's input shape
    dummy_input = torch.randn(1, 4, 84, 84)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "web/snake_ai.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Exported snake_ai.onnx for image-mode AI.")

if __name__ == '__main__':
    export_image_model_to_onnx()
