#!/bin/bash

# Ollama Model Management Script
# Chapter 5: Model Management Examples

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}"
}

# Check if Ollama is installed and running
check_ollama() {
    print_header "Checking Ollama Installation"
    
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama is not installed or not in PATH"
        exit 1
    fi
    
    print_status "Ollama is installed"
    
    # Check if Ollama service is running
    if ! ollama list &> /dev/null; then
        print_warning "Ollama service might not be running"
        print_status "Starting Ollama service..."
        ollama serve &
        sleep 5
    fi
    
    print_status "Ollama service is running"
}

# List all available models
list_models() {
    print_header "Available Models"
    ollama list
}

# Pull a model with progress
pull_model() {
    local model_name="$1"
    
    if [ -z "$model_name" ]; then
        print_error "Model name is required"
        return 1
    fi
    
    print_header "Pulling Model: $model_name"
    print_status "This may take a while depending on model size..."
    
    ollama pull "$model_name"
    
    if [ $? -eq 0 ]; then
        print_status "Successfully pulled model: $model_name"
    else
        print_error "Failed to pull model: $model_name"
        return 1
    fi
}

# Remove a model
remove_model() {
    local model_name="$1"
    
    if [ -z "$model_name" ]; then
        print_error "Model name is required"
        return 1
    fi
    
    print_header "Removing Model: $model_name"
    
    # Confirm before removal
    read -p "Are you sure you want to remove $model_name? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        ollama rm "$model_name"
        
        if [ $? -eq 0 ]; then
            print_status "Successfully removed model: $model_name"
        else
            print_error "Failed to remove model: $model_name"
            return 1
        fi
    else
        print_status "Model removal cancelled"
    fi
}

# Show model information
show_model_info() {
    local model_name="$1"
    
    if [ -z "$model_name" ]; then
        print_error "Model name is required"
        return 1
    fi
    
    print_header "Model Information: $model_name"
    ollama show "$model_name"
}

# Copy a model
copy_model() {
    local source_model="$1"
    local destination_model="$2"
    
    if [ -z "$source_model" ] || [ -z "$destination_model" ]; then
        print_error "Both source and destination model names are required"
        return 1
    fi
    
    print_header "Copying Model"
    print_status "From: $source_model"
    print_status "To: $destination_model"
    
    ollama cp "$source_model" "$destination_model"
    
    if [ $? -eq 0 ]; then
        print_status "Successfully copied model"
    else
        print_error "Failed to copy model"
        return 1
    fi
}

# Batch pull multiple models
batch_pull() {
    local models=("$@")
    
    if [ ${#models[@]} -eq 0 ]; then
        print_error "No models specified"
        return 1
    fi
    
    print_header "Batch Pulling Models"
    
    for model in "${models[@]}"; do
        print_status "Pulling $model..."
        pull_model "$model"
        echo ""
    done
    
    print_status "Batch pull completed"
}

# Check disk space and model sizes
check_disk_space() {
    print_header "Disk Space Analysis"
    
    # Show available disk space
    print_status "Available disk space:"
    df -h ~/.ollama 2>/dev/null || df -h ~
    
    echo ""
    print_status "Model sizes:"
    
    # Get model information and sizes
    ollama list | tail -n +2 | while read -r line; do
        model_name=$(echo "$line" | awk '{print $1}')
        size=$(echo "$line" | awk '{print $2}')
        echo "  $model_name: $size"
    done
}

# Update all models
update_models() {
    print_header "Updating All Models"
    
    # Get list of installed models
    models=$(ollama list | tail -n +2 | awk '{print $1}')
    
    if [ -z "$models" ]; then
        print_warning "No models installed"
        return 0
    fi
    
    for model in $models; do
        print_status "Updating $model..."
        ollama pull "$model"
        echo ""
    done
    
    print_status "All models updated"
}

# Cleanup old model versions
cleanup_models() {
    print_header "Model Cleanup"
    
    print_warning "This will remove unused model layers"
    read -p "Continue? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        # This is a placeholder - Ollama doesn't have a built-in cleanup command yet
        print_status "Cleanup functionality would go here"
        print_status "You can manually remove unused models using 'ollama rm <model>'"
    else
        print_status "Cleanup cancelled"
    fi
}

# Export model list to file
export_model_list() {
    local output_file="${1:-model_list.txt}"
    
    print_header "Exporting Model List"
    
    {
        echo "# Ollama Models - $(date)"
        echo "# Generated by model-management.sh"
        echo ""
        ollama list
    } > "$output_file"
    
    print_status "Model list exported to: $output_file"
}

# Show usage information
show_usage() {
    echo "Ollama Model Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  check           Check Ollama installation and service"
    echo "  list            List all installed models"
    echo "  pull MODEL      Pull a specific model"
    echo "  remove MODEL    Remove a specific model"
    echo "  info MODEL      Show information about a model"
    echo "  copy SRC DEST   Copy a model"
    echo "  batch-pull M1 M2 ...  Pull multiple models"
    echo "  disk-space      Check disk space and model sizes"
    echo "  update          Update all installed models"
    echo "  cleanup         Cleanup unused model data"
    echo "  export [FILE]   Export model list to file"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 pull llama2"
    echo "  $0 batch-pull llama2 codellama mistral"
    echo "  $0 copy llama2 my-custom-llama2"
    echo "  $0 export my-models.txt"
}

# Main script logic
main() {
    case "${1:-}" in
        "check")
            check_ollama
            ;;
        "list")
            check_ollama
            list_models
            ;;
        "pull")
            check_ollama
            pull_model "$2"
            ;;
        "remove"|"rm")
            check_ollama
            remove_model "$2"
            ;;
        "info"|"show")
            check_ollama
            show_model_info "$2"
            ;;
        "copy"|"cp")
            check_ollama
            copy_model "$2" "$3"
            ;;
        "batch-pull")
            check_ollama
            shift
            batch_pull "$@"
            ;;
        "disk-space"|"space")
            check_ollama
            check_disk_space
            ;;
        "update")
            check_ollama
            update_models
            ;;
        "cleanup")
            check_ollama
            cleanup_models
            ;;
        "export")
            check_ollama
            export_model_list "$2"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
