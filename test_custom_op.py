import openvino as ov
import numpy as np
import os

# Paths
# Use absolute paths to be safe
base_dir = os.path.dirname(os.path.abspath(__file__))
model_xml = os.path.join(base_dir, "model.xml")
custom_layer_xml = os.path.join(base_dir, "custom_add_mul.xml")

# Initialize OpenVINO Core
core = ov.Core()

# Check if GPU is available
if "GPU" not in core.available_devices:
    print("GPU device not found. This tutorial requires an Intel GPU.")
    print("Available devices:", core.available_devices)

print(f"Loading custom layer config from {custom_layer_xml}...")

# Define Custom Op Class to allow read_model to succeed
class CustomAddMul(ov.Op):
    def __init__(self, inputs=None):
        # Follow the documentation pattern: super().__init__(self, inputs)
        # If inputs is a list of nodes, we might need to ensure they are Output objects
        if inputs is None:
            super().__init__(self)
        else:
            super().__init__(self, inputs)
            self.validate_and_infer_types()

    def validate_and_infer_types(self):
        # Output shape is same as input 0
        if self.get_input_size() > 0:
            self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return CustomAddMul(new_inputs)

    def visit_attributes(self, visitor):
        return True

# Register the custom op
try:
    core.add_extension(CustomAddMul)
    print("Registered CustomAddMul extension.")
except Exception as e:
    print(f"Warning: Could not register CustomAddMul extension: {e}")
    # Try with instance if class fails? No, usually it's class.
    # Or maybe ov.Extension?

# Config for GPU
config = {}
# Use the internal property key "CONFIG_FILE"
config["CONFIG_FILE"] = custom_layer_xml
# Force FP32 inference precision to match the OpenCL kernel (which uses float)
# If this is not set, the GPU plugin might use FP16, causing memory mismatch with the kernel.
config["INFERENCE_PRECISION_HINT"] = "f32"

# Load the model
print(f"Reading model from {model_xml}...")
try:
    model = core.read_model(model=model_xml)
except Exception as e:
    print(f"Error reading model: {e}")
    exit(1)

# Compile the model on GPU
print("Compiling model on GPU...")
try:
    compiled_model = core.compile_model(model, "GPU", config=config)
except Exception as e:
    print(f"Error compiling model: {e}")
    print("Ensure you have an Intel GPU and the OpenCL driver installed.")
    # If CONFIG_FILE is rejected, we might see it here.
    exit(1)

# Create inference request
request = compiled_model.create_infer_request()

# Prepare inputs
# We need 3 inputs for CustomAddMul
input0_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
input1_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
input2_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

inputs = [input0_data, input1_data, input2_data]

# Run inference
print("Running inference...")
results = request.infer(inputs)

# Get result
result = results[compiled_model.output(0)]

print("Inference successful!")
print(f"Result shape: {result.shape}")

# Verify result (A*B + C)
# The kernel implementation in custom_add_mul.cl is:
# output[i] = (input0[i] + input1[i]) * input2[i];
expected = (input0_data + input1_data) * input2_data

print("Verifying results...")
if np.allclose(result, expected, atol=1e-3):
    print("SUCCESS: Result matches expected output!")
else:
    print("FAILURE: Result does not match expected output.")
    max_diff = np.max(np.abs(result - expected))
    print(f"Max difference: {max_diff}")
    print("First few values:")
    print("Result:", result.flatten()[:5])
    print("Expected:", expected.flatten()[:5])
