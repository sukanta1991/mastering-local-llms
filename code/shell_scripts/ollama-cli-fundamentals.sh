#!/bin/bash
# ollama-cli-fundamentals.sh - CLI fundamentals and automation scripts
# 
# This script demonstrates basic Ollama CLI operations and provides
# wrapper functions for common tasks.
#
# Usage: ./ollama-cli-fundamentals.sh [command] [args...]
#
# Author: Book Example
# License: MIT

set -euo pipefail

# Configuration
DEFAULT_MODEL="llama3.2:3b"
API_BASE="http://localhost:11434/api"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/ollama-automation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $*${NC}" >&2
    log "ERROR: $*"
}

success() {
    echo -e "${GREEN}SUCCESS: $*${NC}"
    log "SUCCESS: $*"
}

warning() {
    echo -e "${YELLOW}WARNING: $*${NC}"
    log "WARNING: $*"
}

info() {
    echo -e "${BLUE}INFO: $*${NC}"
    log "INFO: $*"
}

# Check if Ollama is running
check_ollama() {
    if ! curl -s "$API_BASE/version" >/dev/null 2>&1; then
        error "Ollama server is not running"
        info "Start Ollama with: ollama serve"
        exit 1
    fi
    success "Ollama server is running"
}

# List available models
list_models() {
    info "Listing available models..."
    if ollama list | grep -q "No models"; then
        warning "No models installed"
        info "Pull a model with: ollama pull $DEFAULT_MODEL"
        return 1
    fi
    
    ollama list
    return 0
}

# Pull a model if not exists
ensure_model() {
    local model="${1:-$DEFAULT_MODEL}"
    
    if ! ollama list | grep -q "$model"; then
        info "Model $model not found. Pulling..."
        ollama pull "$model"
        success "Model $model pulled successfully"
    else
        info "Model $model already available"
    fi
}

# Simple text generation
generate_text() {
    local model="${1:-$DEFAULT_MODEL}"
    local prompt="$2"
    local temperature="${3:-0.7}"
    
    if [ -z "$prompt" ]; then
        error "Prompt is required"
        echo "Usage: generate_text <model> <prompt> [temperature]"
        return 1
    fi
    
    info "Generating text with model: $model"
    
    curl -s "$API_BASE/generate" -d "{
        \"model\": \"$model\",
        \"prompt\": \"$prompt\",
        \"options\": {\"temperature\": $temperature},
        \"stream\": false
    }" | jq -r '.response'
}

# Chat with model
chat() {
    local model="${1:-$DEFAULT_MODEL}"
    local message="$2"
    
    if [ -z "$message" ]; then
        error "Message is required"
        echo "Usage: chat <model> <message>"
        return 1
    fi
    
    info "Chatting with model: $model"
    
    curl -s "$API_BASE/chat" -d "{
        \"model\": \"$model\",
        \"messages\": [
            {\"role\": \"user\", \"content\": \"$message\"}
        ],
        \"stream\": false
    }" | jq -r '.message.content'
}

# Batch processing
batch_process() {
    local model="${1:-$DEFAULT_MODEL}"
    local input_file="$2"
    local output_file="$3"
    
    if [ ! -f "$input_file" ]; then
        error "Input file not found: $input_file"
        return 1
    fi
    
    info "Processing batch from $input_file to $output_file"
    
    > "$output_file"  # Clear output file
    
    local line_count=0
    while IFS= read -r line; do
        ((line_count++))
        
        if [ -n "$line" ]; then
            info "Processing line $line_count: ${line:0:50}..."
            
            response=$(generate_text "$model" "$line" 0.7)
            
            {
                echo "=== Input $line_count ==="
                echo "$line"
                echo "=== Output $line_count ==="
                echo "$response"
                echo ""
            } >> "$output_file"
            
            success "Completed line $line_count"
        fi
    done < "$input_file"
    
    success "Batch processing complete. Results in $output_file"
}

# Model comparison
compare_models() {
    local prompt="$1"
    local models=("${@:2}")
    
    if [ -z "$prompt" ]; then
        error "Prompt is required"
        echo "Usage: compare_models <prompt> <model1> [model2] [model3] ..."
        return 1
    fi
    
    if [ ${#models[@]} -eq 0 ]; then
        models=("llama3.2:3b" "llama3.2:1b")
    fi
    
    info "Comparing models with prompt: ${prompt:0:50}..."
    
    for model in "${models[@]}"; do
        echo ""
        echo "=== Model: $model ==="
        
        if ensure_model "$model"; then
            start_time=$(date +%s.%N)
            response=$(generate_text "$model" "$prompt" 0.7)
            end_time=$(date +%s.%N)
            
            duration=$(echo "$end_time - $start_time" | bc)
            word_count=$(echo "$response" | wc -w)
            
            echo "Response ($word_count words, ${duration}s):"
            echo "$response"
            echo ""
        else
            warning "Skipping model $model (not available)"
        fi
    done
}

# System information
system_info() {
    info "Gathering system information..."
    
    echo "=== Ollama Version ==="
    ollama --version
    
    echo ""
    echo "=== Available Models ==="
    list_models
    
    echo ""
    echo "=== Server Status ==="
    curl -s "$API_BASE/version" | jq '.'
    
    echo ""
    echo "=== System Resources ==="
    if command -v free >/dev/null; then
        free -h
    elif command -v vm_stat >/dev/null; then
        vm_stat
    fi
    
    echo ""
    echo "=== Disk Usage ==="
    df -h . 2>/dev/null || echo "Disk usage unavailable"
}

# Interactive chat session
interactive_chat() {
    local model="${1:-$DEFAULT_MODEL}"
    
    ensure_model "$model"
    
    info "Starting interactive chat with $model"
    info "Type 'quit' or 'exit' to end the session"
    echo ""
    
    while true; do
        echo -n "You: "
        read -r user_input
        
        if [[ "$user_input" =~ ^(quit|exit)$ ]]; then
            info "Ending chat session"
            break
        fi
        
        if [ -n "$user_input" ]; then
            echo -n "AI: "
            chat "$model" "$user_input"
            echo ""
        fi
    done
}

# Text summarization
summarize_text() {
    local input_file="$1"
    local model="${2:-$DEFAULT_MODEL}"
    local max_words="${3:-100}"
    
    if [ ! -f "$input_file" ]; then
        error "Input file not found: $input_file"
        return 1
    fi
    
    local content
    content=$(cat "$input_file")
    
    local prompt="Summarize the following text in $max_words words or less:

$content"
    
    info "Summarizing $input_file with $model..."
    generate_text "$model" "$prompt" 0.3
}

# Translation
translate_text() {
    local text="$1"
    local target_language="$2"
    local model="${3:-$DEFAULT_MODEL}"
    
    if [ -z "$text" ] || [ -z "$target_language" ]; then
        error "Text and target language are required"
        echo "Usage: translate_text <text> <target_language> [model]"
        return 1
    fi
    
    local prompt="Translate the following text to $target_language:

$text"
    
    info "Translating to $target_language with $model..."
    generate_text "$model" "$prompt" 0.1
}

# Code generation
generate_code() {
    local language="$1"
    local description="$2"
    local model="${3:-codellama:code}"
    
    if [ -z "$language" ] || [ -z "$description" ]; then
        error "Language and description are required"
        echo "Usage: generate_code <language> <description> [model]"
        return 1
    fi
    
    local prompt="Write $language code to $description. Include comments and follow best practices."
    
    info "Generating $language code with $model..."
    generate_text "$model" "$prompt" 0.3
}

# Benchmark model performance
benchmark_model() {
    local model="${1:-$DEFAULT_MODEL}"
    local iterations="${2:-5}"
    
    ensure_model "$model"
    
    info "Benchmarking model $model with $iterations iterations..."
    
    local prompt="Explain the concept of machine learning in simple terms."
    local total_time=0
    local total_words=0
    
    for ((i=1; i<=iterations; i++)); do
        info "Iteration $i/$iterations"
        
        start_time=$(date +%s.%N)
        response=$(generate_text "$model" "$prompt" 0.7)
        end_time=$(date +%s.%N)
        
        duration=$(echo "$end_time - $start_time" | bc)
        word_count=$(echo "$response" | wc -w)
        
        total_time=$(echo "$total_time + $duration" | bc)
        total_words=$((total_words + word_count))
        
        info "Iteration $i: ${duration}s, $word_count words"
    done
    
    avg_time=$(echo "scale=2; $total_time / $iterations" | bc)
    avg_words=$((total_words / iterations))
    words_per_second=$(echo "scale=2; $total_words / $total_time" | bc)
    
    echo ""
    echo "=== Benchmark Results for $model ==="
    echo "Iterations: $iterations"
    echo "Average time: ${avg_time}s"
    echo "Average words: $avg_words"
    echo "Words per second: $words_per_second"
    echo "Total time: ${total_time}s"
}

# Help function
show_help() {
    cat << EOF
Ollama CLI Fundamentals Script

Usage: $0 <command> [arguments...]

Commands:
  check              - Check if Ollama is running
  list               - List available models
  pull <model>       - Pull/download a model
  generate <prompt>  - Generate text with default model
  chat <message>     - Chat with default model
  batch <input> <output> - Process file line by line
  compare <prompt> <model1> [model2] ... - Compare models
  info               - Show system information
  interactive [model] - Start interactive chat
  summarize <file> [model] [max_words] - Summarize text file
  translate <text> <language> [model] - Translate text
  code <language> <description> [model] - Generate code
  benchmark [model] [iterations] - Benchmark model performance
  help               - Show this help message

Examples:
  $0 check
  $0 generate "What is AI?"
  $0 chat "Hello, how are you?"
  $0 compare "Explain Python" llama3.2:3b llama3.2:1b
  $0 interactive llama3.2:3b
  $0 summarize document.txt
  $0 translate "Hello world" Spanish
  $0 code python "sort a list of numbers"
  $0 benchmark llama3.2:3b 3

Configuration:
  Default model: $DEFAULT_MODEL
  API endpoint: $API_BASE
  Log file: $LOG_FILE

EOF
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        check)
            check_ollama
            ;;
        list)
            list_models
            ;;
        pull)
            ensure_model "$1"
            ;;
        generate)
            generate_text "$DEFAULT_MODEL" "$1"
            ;;
        chat)
            chat "$DEFAULT_MODEL" "$1"
            ;;
        batch)
            batch_process "$DEFAULT_MODEL" "$1" "$2"
            ;;
        compare)
            compare_models "$@"
            ;;
        info)
            system_info
            ;;
        interactive)
            interactive_chat "$1"
            ;;
        summarize)
            summarize_text "$1" "$2" "$3"
            ;;
        translate)
            translate_text "$1" "$2" "$3"
            ;;
        code)
            generate_code "$1" "$2" "$3"
            ;;
        benchmark)
            benchmark_model "$1" "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
