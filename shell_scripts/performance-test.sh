#!/bin/bash

# Ollama Performance Testing Script
# Chapter 17: Performance Optimization

set -e

# Configuration
DEFAULT_MODEL="llama2"
OUTPUT_DIR="performance_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Create output directory
setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    print_status "Output directory: $OUTPUT_DIR"
}

# Test response time for a single query
test_response_time() {
    local model="$1"
    local prompt="$2"
    local output_file="$3"
    
    print_status "Testing response time for model: $model"
    
    local start_time=$(date +%s.%N)
    
    local response=$(ollama run "$model" "$prompt" --verbose 2>&1)
    local exit_code=$?
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Count tokens (approximate)
    local token_count=$(echo "$response" | wc -w)
    
    # Calculate tokens per second
    local tokens_per_second=$(echo "scale=2; $token_count / $duration" | bc)
    
    # Log results
    {
        echo "Timestamp: $(date)"
        echo "Model: $model"
        echo "Prompt: $prompt"
        echo "Duration: ${duration}s"
        echo "Token Count: $token_count"
        echo "Tokens/Second: $tokens_per_second"
        echo "Exit Code: $exit_code"
        echo "Response:"
        echo "$response"
        echo "----------------------------------------"
    } >> "$output_file"
    
    echo "$duration,$token_count,$tokens_per_second,$exit_code"
}

# Run benchmark with multiple prompts
run_benchmark() {
    local model="${1:-$DEFAULT_MODEL}"
    local iterations="${2:-5}"
    
    print_header "Running Benchmark for $model"
    
    local results_file="$OUTPUT_DIR/benchmark_${model}_${TIMESTAMP}.txt"
    local csv_file="$OUTPUT_DIR/benchmark_${model}_${TIMESTAMP}.csv"
    
    # CSV header
    echo "iteration,prompt_type,duration,tokens,tokens_per_second,exit_code" > "$csv_file"
    
    # Test prompts
    local prompts=(
        "Hello, how are you?"
        "Explain quantum computing in simple terms."
        "Write a short story about a robot."
        "What are the benefits of renewable energy?"
        "Describe the process of photosynthesis."
    )
    
    local total_duration=0
    local total_tokens=0
    local successful_tests=0
    
    for i in $(seq 1 "$iterations"); do
        print_status "Iteration $i/$iterations"
        
        for j in "${!prompts[@]}"; do
            local prompt="${prompts[$j]}"
            local prompt_type="prompt_$((j+1))"
            
            print_status "Testing: $prompt_type"
            
            local result=$(test_response_time "$model" "$prompt" "$results_file")
            IFS=',' read -r duration tokens tokens_per_second exit_code <<< "$result"
            
            # Log to CSV
            echo "$i,$prompt_type,$duration,$tokens,$tokens_per_second,$exit_code" >> "$csv_file"
            
            if [ "$exit_code" -eq 0 ]; then
                total_duration=$(echo "$total_duration + $duration" | bc)
                total_tokens=$(echo "$total_tokens + $tokens" | bc)
                ((successful_tests++))
            fi
            
            sleep 1  # Brief pause between tests
        done
    done
    
    # Calculate averages
    if [ "$successful_tests" -gt 0 ]; then
        local avg_duration=$(echo "scale=2; $total_duration / $successful_tests" | bc)
        local avg_tokens_per_second=$(echo "scale=2; $total_tokens / $total_duration" | bc)
        
        print_header "Benchmark Results for $model"
        echo "Total Tests: $((iterations * ${#prompts[@]}))"
        echo "Successful Tests: $successful_tests"
        echo "Average Duration: ${avg_duration}s"
        echo "Average Tokens/Second: $avg_tokens_per_second"
        echo "Results saved to: $results_file"
        echo "CSV data saved to: $csv_file"
    else
        print_error "No successful tests completed"
    fi
}

# Test concurrent requests
test_concurrent_load() {
    local model="${1:-$DEFAULT_MODEL}"
    local concurrent_requests="${2:-3}"
    local prompt="${3:-"Explain machine learning in one paragraph."}"
    
    print_header "Testing Concurrent Load"
    print_status "Model: $model"
    print_status "Concurrent Requests: $concurrent_requests"
    
    local results_file="$OUTPUT_DIR/concurrent_test_${model}_${TIMESTAMP}.txt"
    
    # Start concurrent requests
    local pids=()
    local start_time=$(date +%s.%N)
    
    for i in $(seq 1 "$concurrent_requests"); do
        {
            echo "Request $i started at $(date)" >> "$results_file"
            local req_start=$(date +%s.%N)
            ollama run "$model" "$prompt" >> "$results_file" 2>&1
            local req_end=$(date +%s.%N)
            local req_duration=$(echo "$req_end - $req_start" | bc)
            echo "Request $i completed in ${req_duration}s" >> "$results_file"
        } &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    local end_time=$(date +%s.%N)
    local total_duration=$(echo "$end_time - $start_time" | bc)
    
    print_status "All concurrent requests completed in ${total_duration}s"
    print_status "Results saved to: $results_file"
}

# Memory usage monitoring
monitor_memory() {
    local model="${1:-$DEFAULT_MODEL}"
    local duration="${2:-60}"
    
    print_header "Monitoring Memory Usage"
    print_status "Model: $model"
    print_status "Duration: ${duration}s"
    
    local memory_log="$OUTPUT_DIR/memory_usage_${model}_${TIMESTAMP}.csv"
    
    # CSV header
    echo "timestamp,rss_mb,vsz_mb,cpu_percent" > "$memory_log"
    
    # Start Ollama with the model in background
    print_status "Starting model..."
    ollama run "$model" "Hello" > /dev/null 2>&1
    
    # Find Ollama process
    local ollama_pid=$(pgrep -f "ollama" | head -1)
    
    if [ -z "$ollama_pid" ]; then
        print_error "Could not find Ollama process"
        return 1
    fi
    
    print_status "Monitoring process: $ollama_pid"
    
    # Monitor memory usage
    local end_time=$(($(date +%s) + duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        if ps -p "$ollama_pid" > /dev/null 2>&1; then
            local memory_info=$(ps -o rss,vsz,pcpu -p "$ollama_pid" --no-headers)
            local rss=$(echo "$memory_info" | awk '{print $1}')
            local vsz=$(echo "$memory_info" | awk '{print $2}')
            local cpu=$(echo "$memory_info" | awk '{print $3}')
            
            # Convert KB to MB
            local rss_mb=$(echo "scale=2; $rss / 1024" | bc)
            local vsz_mb=$(echo "scale=2; $vsz / 1024" | bc)
            
            echo "$(date +%s),$rss_mb,$vsz_mb,$cpu" >> "$memory_log"
        else
            print_warning "Process $ollama_pid no longer running"
            break
        fi
        
        sleep 5
    done
    
    print_status "Memory monitoring completed"
    print_status "Data saved to: $memory_log"
}

# System information
collect_system_info() {
    local info_file="$OUTPUT_DIR/system_info_${TIMESTAMP}.txt"
    
    print_header "Collecting System Information"
    
    {
        echo "System Information - $(date)"
        echo "=================================="
        echo ""
        
        echo "Operating System:"
        uname -a
        echo ""
        
        echo "CPU Information:"
        if command -v lscpu &> /dev/null; then
            lscpu
        elif [ -f /proc/cpuinfo ]; then
            cat /proc/cpuinfo | grep -E "(model name|cpu cores|siblings)" | head -10
        else
            sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "CPU info not available"
        fi
        echo ""
        
        echo "Memory Information:"
        if command -v free &> /dev/null; then
            free -h
        else
            # macOS
            echo "Total Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
        fi
        echo ""
        
        echo "Disk Space:"
        df -h
        echo ""
        
        echo "Ollama Version:"
        ollama --version
        echo ""
        
        echo "Available Models:"
        ollama list
        
    } > "$info_file"
    
    print_status "System information saved to: $info_file"
}

# Generate performance report
generate_report() {
    local model="${1:-$DEFAULT_MODEL}"
    
    print_header "Generating Performance Report"
    
    local report_file="$OUTPUT_DIR/performance_report_${model}_${TIMESTAMP}.md"
    
    {
        echo "# Ollama Performance Report"
        echo ""
        echo "**Model:** $model"
        echo "**Date:** $(date)"
        echo "**Generated by:** performance-test.sh"
        echo ""
        
        echo "## System Information"
        echo ""
        echo "- **OS:** $(uname -s)"
        echo "- **Architecture:** $(uname -m)"
        echo "- **Ollama Version:** $(ollama --version 2>/dev/null || echo 'Unknown')"
        echo ""
        
        echo "## Test Results"
        echo ""
        
        # Include CSV data if available
        local latest_csv=$(ls -t "$OUTPUT_DIR"/benchmark_${model}_*.csv 2>/dev/null | head -1)
        if [ -n "$latest_csv" ]; then
            echo "### Benchmark Results"
            echo ""
            echo "CSV data available in: \`$(basename "$latest_csv")\`"
            echo ""
        fi
        
        echo "## Files Generated"
        echo ""
        for file in "$OUTPUT_DIR"/*"$TIMESTAMP"*; do
            if [ -f "$file" ]; then
                echo "- \`$(basename "$file")\`"
            fi
        done
        echo ""
        
        echo "## Recommendations"
        echo ""
        echo "- Monitor memory usage during extended sessions"
        echo "- Consider model size vs. performance trade-offs"
        echo "- Test with your specific use cases and prompts"
        echo "- Compare results across different hardware configurations"
        
    } > "$report_file"
    
    print_status "Performance report generated: $report_file"
}

# Show usage
show_usage() {
    echo "Ollama Performance Testing Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  benchmark [MODEL] [ITERATIONS]    Run benchmark tests"
    echo "  concurrent [MODEL] [REQUESTS]     Test concurrent load"
    echo "  memory [MODEL] [DURATION]         Monitor memory usage"
    echo "  system-info                       Collect system information"
    echo "  report [MODEL]                    Generate performance report"
    echo "  full-test [MODEL]                 Run all tests"
    echo "  help                              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 benchmark llama2 10"
    echo "  $0 concurrent llama2 5"
    echo "  $0 memory mistral 120"
    echo "  $0 full-test codellama"
}

# Run full test suite
run_full_test() {
    local model="${1:-$DEFAULT_MODEL}"
    
    print_header "Running Full Performance Test Suite"
    print_status "Model: $model"
    
    collect_system_info
    run_benchmark "$model" 3
    test_concurrent_load "$model" 2
    monitor_memory "$model" 30
    generate_report "$model"
    
    print_header "Full Test Suite Completed"
    print_status "Check the $OUTPUT_DIR directory for all results"
}

# Main script logic
main() {
    setup_output_dir
    
    case "${1:-}" in
        "benchmark")
            run_benchmark "$2" "$3"
            ;;
        "concurrent")
            test_concurrent_load "$2" "$3"
            ;;
        "memory")
            monitor_memory "$2" "$3"
            ;;
        "system-info")
            collect_system_info
            ;;
        "report")
            generate_report "$2"
            ;;
        "full-test")
            run_full_test "$2"
            ;;
        "help"|"-h"|"--help"|"")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Check dependencies
check_dependencies() {
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        print_warning "bc (calculator) not found - some calculations may not work"
    fi
}

# Run the script
check_dependencies
main "$@"
