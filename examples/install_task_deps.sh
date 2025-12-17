#!/bin/bash
# install_task_deps.sh
# Automatically install optional dependencies for specified tasks
# Usage: source install_task_deps.sh "task1,task2,task3" [eval_dir]

install_task_dependencies() {
    local tasks_string="$1"
    local eval_dir="${2:-$(pwd)}"

    if [ -z "$tasks_string" ]; then
        echo "No tasks specified, skipping optional dependency installation"
        return 0
    fi

    echo "========================================"
    echo "Installing task-specific dependencies"
    echo "========================================"
    echo "Tasks: ${tasks_string}"
    echo ""

    # Extract base task names from task variants
    # Converts: "ocrbench_v2,docvqa_val,chartqa_lite" -> "ocrbench_v2 docvqa chartqa"
    local base_tasks=$(echo "$tasks_string" | \
        tr ',' '\n' | \
        sed 's/_test$//' | \
        sed 's/_val$//' | \
        sed 's/_lite$//' | \
        sed 's/_pro$//' | \
        sed 's/_cot$//' | \
        sed 's/_solution$//' | \
        sed 's/_testmini$//' | \
        sort -u | \
        tr '\n' ' ')

    echo "Base tasks extracted: ${base_tasks}"
    echo ""

    # Dynamically extract available optional dependency groups from pyproject.toml
    echo "Extracting available extras from pyproject.toml..."
    cd "$eval_dir" || return 1

    local available_extras=$(python3 -c "
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import sys, subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tomli', '--quiet'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import tomli as tomllib

try:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
        extras = data.get('project', {}).get('optional-dependencies', {})
        print(' '.join(extras.keys()))
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

    if [ -z "$available_extras" ]; then
        echo "Warning: Could not extract extras from pyproject.toml"
        echo "Attempting to install all specified tasks..."
        available_extras="$base_tasks"
    else
        echo "Available extras: ${available_extras}"
    fi
    echo ""

    # Track which extras to install
    local extras_to_install=()

    # Check each base task against available extras
    for task in $base_tasks; do
        if echo "$available_extras" | grep -qw "$task"; then
            extras_to_install+=("$task")
        fi
    done

    # Remove duplicates
    extras_to_install=($(echo "${extras_to_install[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

    if [ ${#extras_to_install[@]} -eq 0 ]; then
        echo "No task-specific dependencies to install"
        echo "========================================"
        echo ""
        return 0
    fi

    echo "Installing optional dependencies for: ${extras_to_install[*]}"
    echo ""

    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        echo "Warning: pip not found, skipping optional dependency installation"
        echo "Please install dependencies manually"
        echo "========================================"
        echo ""
        return 1
    fi

    # Install dependencies using pip with extras
    cd "$eval_dir" || return 1

    for extra in "${extras_to_install[@]}"; do
        echo "Installing dependencies for: ${extra}"
        if pip install -e ".[$extra]"; then
            echo "  ✓ Successfully installed ${extra} dependencies"
        else
            echo "  ⚠ Warning: Failed to install ${extra} dependencies, continuing anyway"
        fi
        echo ""
    done

    echo "Task-specific dependency installation complete"
    echo "========================================"
    echo ""
}

# If script is sourced with arguments, run the function
if [ -n "$1" ]; then
    install_task_dependencies "$1" "$2"
fi
