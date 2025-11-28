# OpenVINO Custom GPU Operator Tutorial

This tutorial demonstrates how to add a custom OpenCL operator to the OpenVINO GPU plugin without recompiling OpenVINO.

## Overview

The process involves:
1.  **Kernel Implementation**: Writing the OpenCL C kernel code (`.cl`).
2.  **Configuration**: Creating an XML file (`.xml`) that describes the custom layer and maps it to the kernel.
3.  **Model Creation**: Creating an OpenVINO IR model (`.xml` + `.bin`) that uses the custom layer type.
4.  **Execution**: Loading the configuration and model in an OpenVINO application.

## Files

*   `custom_add_mul.cl`: The OpenCL kernel implementation.
*   `custom_add_mul.xml`: The configuration file for the custom layer.
*   `model.xml`: A manually created OpenVINO IR model using the `CustomAddMul` layer.
*   `test_custom_op.py`: A Python script to run the inference.

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

1.  Ensure you have OpenVINO installed and an Intel GPU available.
2.  Run the test script:
    ```bash
    python test_custom_op.py
    ```

## Implementation Details

### 1. OpenCL Kernel (`custom_add_mul.cl`)

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

Defines the mapping between the IR layer and the kernel.

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
    <WorkSizes global="B*F*Y*X" local="16"/>
</CustomLayer>
```

### 3. IR Model (`model.xml`)

A standard OpenVINO XML where a layer has `type="CustomAddMul"`.

```xml
<layer id="3" name="custom_op" type="CustomAddMul" version="extension">
    ...
</layer>
```
