import os
import xml.etree.ElementTree as ET
from collections import Counter

def count_objects_in_xml(xml_path):
    """Count objects in a single XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Count objects by class
        objects = [obj.find('name').text for obj in root.findall('object')]
        return Counter(objects)
    except Exception as e:
        print(f"Error processing {xml_path}: {str(e)}")
        return Counter()

def main():
    # Path to annotations directory
    xml_dir = os.path.join('dataset', 'annotations', 'xmls')
    
    if not os.path.exists(xml_dir):
        print(f"Error: Directory not found: {xml_dir}")
        return
    
    # Get all XML files
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    if not xml_files:
        print("No XML files found in the directory")
        return
    
    # Initialize counters
    total_counts = Counter()
    images_with_cats = 0
    images_with_dogs = 0
    
    print(f"Processing {len(xml_files)} XML files...")
    
    # Process each XML file
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        counts = count_objects_in_xml(xml_path)
        
        # Update total counts
        total_counts.update(counts)
        
        # Count images with cats/dogs
        if counts.get('cat', 0) > 0:
            images_with_cats += 1
        if counts.get('dog', 0) > 0:
            images_with_dogs += 1
    
    # Print results
    print("\nAnnotation Statistics:")
    print("-" * 50)
    print(f"Total XML files processed: {len(xml_files)}")
    print("\nObject Counts:")
    for obj_type, count in total_counts.items():
        print(f"{obj_type.capitalize()}: {count}")
    
    print("\nImages containing:")
    print(f"Cats: {images_with_cats}")
    print(f"Dogs: {images_with_dogs}")
    
    # Calculate percentages
    total_images = len(xml_files)
    print("\nPercentages:")
    print(f"Images with cats: {(images_with_cats/total_images)*100:.1f}%")
    print(f"Images with dogs: {(images_with_dogs/total_images)*100:.1f}%")

if __name__ == '__main__':
    main() 