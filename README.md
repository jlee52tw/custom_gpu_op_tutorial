# OpenVINO Custom GPU Operator Tutorial

This tutorial demonstrates how to add a custom OpenCL operator to the OpenVINO GPU plugin without recompiling OpenVINO. It provides both **Python** and **C++** implementations.

## Overview

The process involves:
1.  **Kernel Implementation**: Writing the OpenCL C kernel code (`.cl`).
2.  **Configuration**: Creating an XML file (`.xml`) that describes the custom layer and maps it to the kernel.
3.  **Model Creation**: Creating an OpenVINO IR model (`.xml` + `.bin`) that uses the custom layer type.
4.  **Execution**: Loading the configuration and model in an OpenVINO application (Python or C++).

## Files

*   `custom_add_mul.cl`: The OpenCL kernel implementation.
*   `custom_add_mul.xml`: The configuration file for the custom layer.
*   `model.xml`: A manually created OpenVINO IR model using the `CustomAddMul` layer.
*   `create_model.py`: Python script to generate the IR model.
*   `test_custom_op.py`: Python implementation of the tutorial.
*   `main.cpp`: C++ implementation of the tutorial.
*   `CMakeLists.txt`: CMake build file for the C++ version.

## Prerequisites

*   **OpenVINO 2025.3.0** (or compatible version).
*   **Intel GPU** (Integrated or Discrete) with OpenCL drivers installed.
*   **Python 3.8+** (for Python version).
*   **CMake 3.10+** and a C++ Compiler (e.g., MSVC on Windows, GCC on Linux) (for C++ version).

## Environment Setup

Before running either version, you must initialize the OpenVINO environment variables.

**Windows (PowerShell):**
```powershell
& "C:\path\to\openvino\setupvars.ps1"
```
*(Example: `& "C:\working\gpt-oss-20b-custome-operator\openvino_genai_windows_2025.3.0.0_x86_64\setupvars.ps1"`)*

**Linux:**
```bash
source /path/to/openvino/setupvars.sh
```

## How to Run

### 1. Generate the Model
First, generate the OpenVINO IR model (`model.xml` and `model.bin`) with the correct topology and shapes (2048x2048).

```bash
python create_model.py
```

### 2. Run Python Version
The Python script loads the custom layer config, compiles the model, and runs a 30-second stress test.

```bash
python test_custom_op.py
```

### 3. Run C++ Version
The C++ version performs the same logic as the Python script.

**Build:**
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**Run:**
```bash
# Windows
.\Release\test_custom_op_cpp.exe

# Linux
./test_custom_op_cpp

```
<img width="1411" height="1204" alt="image" src="https://github.com/user-attachments/assets/876733c2-7d05-494b-837a-62979c3a5e59" />

## Interaction Flow

The following sequence diagram illustrates how the components interact when adding and using a custom GPU operator.

```mermaid
sequenceDiagram
    participant User as User App (Python)
    participant Core as OpenVINO Core
    participant GPU as GPU Plugin
    participant CL as OpenCL Driver
    participant XML as Config XML
    participant IR as IR Model

    Note over User, CL: Setup Phase
    User->>Core: set_property("GPU", {"config_file": "custom_add_mul.xml"})
    Core->>GPU: Set Config
    GPU->>XML: Parse custom_add_mul.xml
    XML-->>GPU: Kernel Source, Entry Point, Params
    GPU->>GPU: Register "CustomAddMul" Primitive

    Note over User, CL: Compilation Phase
    User->>Core: read_model("model.xml")
    Core->>IR: Parse Model
    IR-->>Core: ov::Model (Graph)
    User->>Core: compile_model(model, "GPU")
    Core->>GPU: Compile Graph
    GPU->>GPU: Match IR Node "CustomAddMul" to Registered Primitive
    GPU->>CL: Compile OpenCL Kernel (custom_add_mul.cl)
    CL-->>GPU: Kernel Object
    GPU-->>Core: CompiledModel

    Note over User, CL: Inference Phase
    User->>Core: infer(inputs)
    Core->>GPU: Execute Graph
    GPU->>CL: Enqueue Kernel Execution
    CL-->>GPU: Execution Complete
    GPU-->>Core: Results
    Core-->>User: Output Tensor
```

## Component Relationship

```mermaid
classDiagram
    class Application {
        +load_config()
        +read_model()
        +compile_model()
        +infer()
    }
    class OpenVINO_Core {
        +set_property()
        +compile_model()
    }
    class GPU_Plugin {
        +LoadFromFile()
        +CreateCustomOp()
        -CustomLayerMap
    }
    class CustomLayerConfig {
        +KernelSource
        +KernelEntry
        +Buffers
        +WorkSizes
    }
    class OpenCL_Kernel {
        +custom_add_mul()
    }
    class IR_Model {
        +Layer: CustomAddMul
    }

    Application --> OpenVINO_Core : Uses
    OpenVINO_Core --> GPU_Plugin : Loads
    GPU_Plugin --> CustomLayerConfig : Parses XML
    CustomLayerConfig --> OpenCL_Kernel : References .cl
    Application --> IR_Model : Loads
    GPU_Plugin ..> IR_Model : Matches Node Type
```

## Steps to Run

*(See "How to Run" section above for detailed instructions)*

## Implementation Details

### 1. OpenCL Kernel (`custom_add_mul.cl`)

The kernel performs `(A + B) * C` on float32 data.

```c
__kernel void custom_add_mul(
    const __global float* in0,
    const __global float* in1,
    const __global float* in2,
    __global float* out)
{
    const uint idx = get_global_id(0);
    out[idx] = (in0[idx] + in1[idx]) * in2[idx];
}
```

### 2. Configuration (`custom_add_mul.xml`)

Defines the mapping between the IR layer and the kernel. Note that `WorkSizes` calculates the global work size dynamically based on input dimensions.

```xml
<CustomLayer name="CustomAddMul" type="SimpleGPU" version="1">
    <Kernel entry="custom_add_mul">
        <Source filename="custom_add_mul.cl"/>
    </Kernel>
    <Buffers>
        <Tensor arg-index="0" port-index="0" type="input" format="BFYX"/>
        <Tensor arg-index="1" port-index="1" type="input" format="BFYX"/>
        <Tensor arg-index="2" port-index="2" type="input" format="BFYX"/>
        <Tensor arg-index="3" port-index="0" type="output" format="BFYX"/>
    </Buffers>
    <CompilerOptions options="-cl-mad-enable"/>
    <WorkSizes global="B*F*Y*X"/>
</CustomLayer>
```

### 3. IR Model (`model.xml`)

A standard OpenVINO XML where a layer has `type="CustomAddMul"`.

```xml
<layer id="3" name="custom_op" type="CustomAddMul" version="extension">
    ...
</layer>
```

## Important Notes

### Precision Mismatch
The OpenVINO GPU plugin defaults to **FP16** (half-precision) for performance. However, the custom OpenCL kernel is written for **FP32** (`float`).
If run without configuration, the GPU plugin allocates FP16 buffers while the kernel writes FP32 data, causing memory corruption (garbage output).

To fix this, we force the inference precision to FP32 in `test_custom_op.py`:
```python
config["INFERENCE_PRECISION_HINT"] = "f32"
```

### Configuration Property
The custom layer configuration file is loaded using the internal property key `CONFIG_FILE` (or `cldnn_config_file` in some contexts).
```python
config["CONFIG_FILE"] = "custom_add_mul.xml"
```
