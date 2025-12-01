#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

// Define Custom Op Class to allow read_model to succeed
class CustomAddMul : public ov::op::Op {
public:
    OPENVINO_OP("CustomAddMul", "extension");

    CustomAddMul() = default;
    CustomAddMul(const ov::OutputVector& args) : Op(args) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // Output shape is same as input 0
        if (get_input_size() > 0) {
            set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
        }
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<CustomAddMul>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }
};

// Helper to fill tensor with random data
void fill_tensor(ov::Tensor& tensor) {
    float* data = tensor.data<float>();
    size_t size = tensor.get_size();
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

int main() {
    try {
        // Initialize Core
        ov::Core core;
        
        // Register the custom op
        core.add_extension<CustomAddMul>();
        std::cout << "Registered CustomAddMul extension." << std::endl;
        
        // Check for GPU
        std::vector<std::string> devices = core.get_available_devices();
        bool gpu_found = false;
        for (const auto& device : devices) {
            if (device.find("GPU") != std::string::npos) {
                gpu_found = true;
                break;
            }
        }
        if (!gpu_found) {
            std::cerr << "GPU not found! This tutorial requires an Intel GPU." << std::endl;
            std::cerr << "Available devices: ";
            for (const auto& d : devices) std::cerr << d << " ";
            std::cerr << std::endl;
            // return 1; // Proceed anyway, maybe it's hidden or named differently
        }

        // Config
        std::string custom_xml = "custom_add_mul.xml";
        std::string model_xml = "model.xml";
        
        std::cout << "Loading custom layer config from " << custom_xml << "..." << std::endl;

        // Prepare config map
        ov::AnyMap config;
        // Use the internal property key "CONFIG_FILE" to load the custom layer XML
        config["CONFIG_FILE"] = custom_xml;
        // Force FP32 inference precision to match the OpenCL kernel (which uses float)
        config[ov::hint::inference_precision.name()] = ov::element::f32;

        // Read model
        std::cout << "Reading model from " << model_xml << "..." << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_xml);

        // Compile model
        std::cout << "Compiling model on GPU..." << std::endl;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU", config);

        // Create request
        ov::InferRequest request = compiled_model.create_infer_request();

        // Prepare inputs
        // Model has 3 inputs: in0, in1, in2
        auto input0 = request.get_input_tensor(0);
        auto input1 = request.get_input_tensor(1);
        auto input2 = request.get_input_tensor(2);
        
        // Get shape from tensor
        ov::Shape shape = input0.get_shape();
        size_t H = shape[2];
        size_t W = shape[3];
        std::cout << "Generating input data (" << H << "x" << W << ")..." << std::endl;
        
        fill_tensor(input0);
        fill_tensor(input1);
        fill_tensor(input2);

        // Run inference
        std::cout << "Running inference..." << std::endl;
        request.infer();

        // Get result
        auto output = request.get_output_tensor(0);
        float* out_data = output.data<float>();
        float* in0_data = input0.data<float>();
        float* in1_data = input1.data<float>();
        float* in2_data = input2.data<float>();
        size_t size = output.get_size();

        // Verify
        std::cout << "Verifying results..." << std::endl;
        float max_diff = 0.0f;
        bool success = true;
        for (size_t i = 0; i < size; ++i) {
            float expected = (in0_data[i] + in1_data[i]) * in2_data[i];
            float diff = std::abs(out_data[i] - expected);
            if (diff > 1e-3) {
                success = false;
                max_diff = std::max(max_diff, diff);
                if (i < 5) {
                     std::cout << "Mismatch at " << i << ": got " << out_data[i] << ", expected " << expected << std::endl;
                }
            }
        }

        if (success) {
            std::cout << "SUCCESS: Result matches expected output!" << std::endl;
        } else {
            std::cout << "FAILURE: Max difference: " << max_diff << std::endl;
        }

        // Stress test
        std::cout << "\nStarting stress test for 30 seconds..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 0;
        while (true) {
            request.infer();
            iterations++;
            
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - start;
            if (elapsed.count() >= 30.0) break;
            
            if (iterations % 10 == 0) {
                std::cout << "Iterations: " << iterations << ", Time: " << elapsed.count() << "s" << std::endl;
            }
        }
        std::cout << "Stress test complete. Total iterations: " << iterations << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
