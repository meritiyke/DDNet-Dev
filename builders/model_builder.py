from model.DDNet import DDNet  # Import the DDNet model

def build_model(model_name, num_channels):
    """
    Build the model based on the model name.
    """
    if model_name == 'DDNet':
        return DDNet(num_channels=num_channels)  # Initialize DDNet
    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")