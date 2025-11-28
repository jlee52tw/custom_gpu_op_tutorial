import os
import struct

def create_ir_model(model_path, model_name="CustomAddMul_Net"):
    xml_path = model_path + ".xml"
    bin_path = model_path + ".bin"
    
    # Define shapes
    N, C, H, W = 1, 3, 224, 224
    precision = "FP32"
    type_str = "f32"
    
    # XML Content
    xml_content = f"""<?xml version="1.0"?>
<net name="{model_name}" version="10">
	<layers>
		<layer id="0" name="in0" type="Parameter" version="opset1">
			<data shape="{N},{C},{H},{W}" element_type="{type_str}"/>
			<output>
				<port id="0" precision="{precision}" names="in0">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="in1" type="Parameter" version="opset1">
			<data shape="{N},{C},{H},{W}" element_type="{type_str}"/>
			<output>
				<port id="0" precision="{precision}" names="in1">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
			<data shape="{N},{C},{H},{W}" element_type="{type_str}"/>
			<output>
				<port id="0" precision="{precision}" names="in2">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="custom_op" type="CustomAddMul" version="extension">
			<input>
				<port id="0" precision="{precision}">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
				<port id="1" precision="{precision}">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
                <port id="2" precision="{precision}">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="{precision}" names="out">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="out" type="Result" version="opset1">
			<input>
				<port id="0" precision="{precision}">
					<dim>{N}</dim>
					<dim>{C}</dim>
					<dim>{H}</dim>
					<dim>{W}</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
		<edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
	</edges>
</net>
"""
    
    print(f"Writing XML to {xml_path}...")
    with open(xml_path, "w") as f:
        f.write(xml_content)
        
    # Create an empty BIN file (since we only have Parameters, no constants/weights)
    print(f"Writing BIN to {bin_path}...")
    with open(bin_path, "wb") as f:
        pass # Empty file

if __name__ == "__main__":
    create_ir_model("model")
    print("Model generation complete.")
