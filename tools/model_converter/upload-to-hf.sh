#!/bin/bash

# Directory containing the model folders
MODELAPI_DIR="modelapi"

# Check if the modelapi directory exists
if [ ! -d "$MODELAPI_DIR" ]; then
    echo "Error: Directory '$MODELAPI_DIR' not found"
    exit 1
fi

# Loop through all directories in the modelapi folder
for folder in "$MODELAPI_DIR"/*/ ; do
    # Remove trailing slash and get folder name
    folder_name=$(basename "$folder")
    
    # Skip if not a directory
    [ -d "$folder" ] || continue
    
    echo "Processing: $folder_name"
    
    # Create HuggingFace repo
    uvx hf repo create "modelapi/$folder_name" --private
    
    # Upload folder contents to HuggingFace
    uvx hf upload "modelapi/$folder_name" "modelapi/$folder_name" .
    
    echo "Completed: $folder_name"
    echo "---"
done

echo "All folders processed!"

