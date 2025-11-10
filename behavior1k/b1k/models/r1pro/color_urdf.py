import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_colored_urdf(input_urdf, output_urdf):
    """Create a URDF with embedded colors that PyBullet will respect"""

    # Parse the original URDF
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # Define a color palette
    colors = [
        ("gray", "0.7 0.7 0.7 1.0"),
        ("dark_gray", "0.4 0.4 0.4 1.0"),
        ("light_gray", "0.9 0.9 0.9 1.0"),
        ("metal", "0.6 0.6 0.7 1.0"),
        ("red", "0.8 0.3 0.3 1.0"),
        ("blue", "0.3 0.3 0.8 1.0"),
        ("green", "0.3 0.7 0.3 1.0"),
        ("orange", "0.9 0.5 0.1 1.0"),
        ("yellow", "0.9 0.9 0.2 1.0"),
        ("purple", "0.6 0.3 0.8 1.0"),
    ]

    # Add material definitions to the root
    for color_name, rgba in colors:
        material_elem = ET.Element("material", {"name": color_name})
        color_elem = ET.SubElement(material_elem, "color", {"rgba": rgba})
        root.insert(0, material_elem)

    # Assign colors to each link
    links = root.findall("link")
    for i, link in enumerate(links):
        visual = link.find("visual")
        if visual is not None:
            # Remove any existing material
            existing_material = visual.find("material")
            if existing_material is not None:
                visual.remove(existing_material)

            # Assign a color based on link index
            color_name, rgba = colors[i % len(colors)]
            material_elem = ET.Element("material", {"name": color_name})
            visual.append(material_elem)

    # Write the new URDF
    with open(output_urdf, "w") as f:
        f.write(prettify(root))

    print(f"✓ Created colored URDF: {output_urdf}")
    print(f"✓ Applied colors to {len(links)} links")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python color_urdf.py <input.urdf> <output.urdf>")
        print("Example: python color_urdf.py robot.urdf robot_colored.urdf")
        sys.exit(1)

    input_urdf = sys.argv[1]
    output_urdf = sys.argv[2]

    if not os.path.exists(input_urdf):
        print(f"Error: Input URDF '{input_urdf}' does not exist")
        sys.exit(1)

    create_colored_urdf(input_urdf, output_urdf)
