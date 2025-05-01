#!/bin/bash

# Define the paths to the modified files in the repository
MODIFIED_FILES_PATH="modified_llava_files"
GENERATION_UTILS_FILE="$MODIFIED_FILES_PATH/utils.py"
LLAMA_MODELING_FILE="$MODIFIED_FILES_PATH/modeling_llama.py"

SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
GENERATION_UTILS_DEST="$SITE_PACKAGES_PATH/transformers/generation/utils.py"
LLAMA_MODELING_DEST="$SITE_PACKAGES_PATH/transformers/models/llama/modeling_llama.py"

cp "$GENERATION_UTILS_FILE" "$GENERATION_UTILS_DEST"
cp "$LLAMA_MODELING_FILE" "$LLAMA_MODELING_DEST"

echo "✅ Patched core transformer files."

# === Verify everything is okay ===
echo "Verifying the setup..."
python -c "import transformers; print('Transformers library loaded successfully ✅')"

if [ $? -eq 0 ]; then
    echo "✅ Environment updated successfully and verified."
else
    echo "❌ Verification failed. Please check the setup."
fi
