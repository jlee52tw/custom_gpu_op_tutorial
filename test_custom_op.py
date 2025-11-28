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
    # For the sake of the agent running in a non-GPU environment, we might fail here.
    # But the user asked "how to add", so providing the code is the main goal.
    # I will print a warning but try to proceed if possible (it won't work without GPU).
    print("Available devices:", core.available_devices)
    # exit(1) 

print(f"Loading custom layer config from {custom_layer_xml}...")
# Set the custom layer config file
# The property key in C++ is ov::intel_gpu::config_file
# In Python, we can try passing it as a dict.
try:
    core.set_property("GPU", {"config_file": custom_layer_xml})
except Exception as e:
    print(f"Warning: Could not set property 'config_file': {e}")
    print("Trying 'cldnn_config_file'...")
    try:
        core.set_property("GPU", {"cldnn_config_file": custom_layer_xml})
    except Exception as e2:
        print(f"Warning: Could not set property 'cldnn_config_file': {e2}")

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
    compiled_model = core.compile_model(model=model, device_name="GPU")
except Exception as e:
    print(f"Error compiling model: {e}")
    print("This is expected if no Intel GPU is present or if the custom layer config is invalid.")
    exit(1)

# Create input data
shape = (1, 3, 224, 224)
in0 = np.random.rand(*shape).astype(np.float32)
in1 = np.random.rand(*shape).astype(np.float32)
in2 = np.random.rand(*shape).astype(np.float32)

# Run inference
print("Running inference...")
import time
start_time = time.time()
results = compiled_model([in0, in1, in2])
end_time = time.time()
print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

# Get output
out = results[compiled_model.output(0)]

# Verify result
# Expected: (in0 + in1) * in2
expected = (in0 + in1) * in2

if np.allclose(out, expected, atol=1e-5):
    print("Success! Output matches expected values.")
else:
    print("Failure! Output does not match.")
    print("Max difference:", np.max(np.abs(out - expected)))
