#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run.sh [-n NPROCS] [--core-list LIST] --seed SEED [plm_data options...]

Wrapper options:
  -n NPROCS          Number of MPI ranks / physical cores to use (default: 1)
  --core-list LIST   Physical core ids to reserve for this run, e.g. 0-3 or 0,2,4,6
  -h, --help         Show this help

Examples:
  ./run.sh --seed 1234
  ./run.sh -n 4 --seed 1234
  ./run.sh -n 4 --core-list 0-3 --seed 1234
EOF
}

die() {
    echo "run.sh: $*" >&2
    exit 1
}

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

expand_int_list() {
    local spec="$1"
    local -a values=()
    local -a parts=()
    local part start end value

    IFS=',' read -r -a parts <<< "$spec"
    for part in "${parts[@]}"; do
        [[ -n "$part" ]] || die "invalid empty entry in core list '$spec'"
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            (( start <= end )) || die "invalid descending range '$part' in core list '$spec'"
            for ((value = start; value <= end; value++)); do
                values+=("$value")
            done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            values+=("$part")
        else
            die "invalid core list entry '$part' in '$spec'"
        fi
    done

    printf '%s\n' "${values[@]}" | awk '!seen[$0]++'
}

count_physical_cores() {
    lscpu -e=CORE | awk 'NR > 1 { print $1 }' | sort -u | wc -l
}

taskset_cpu_list_for_core_list() {
    local spec="$1"
    local -a requested_cores=()
    local -a logical_cpus=()
    local core cpu
    declare -A requested_map=()
    declare -A found_map=()

    while IFS= read -r core; do
        [[ -n "$core" ]] || continue
        requested_cores+=("$core")
        requested_map["$core"]=1
    done < <(expand_int_list "$spec")

    (( ${#requested_cores[@]} > 0 )) || die "empty core list '$spec'"

    while read -r cpu core; do
        if [[ -n "${requested_map[$core]+x}" ]]; then
            logical_cpus+=("$cpu")
            found_map["$core"]=1
        fi
    done < <(lscpu -e=CPU,CORE | awk 'NR > 1 { print $1, $2 }')

    for core in "${requested_cores[@]}"; do
        [[ -n "${found_map[$core]+x}" ]] || die "physical core '$core' is not available on this machine"
    done

    REQUESTED_CORE_COUNT=${#requested_cores[@]}
    TASKSET_CPU_LIST=$(printf '%s\n' "${logical_cpus[@]}" | paste -sd, -)
}

# Activate the project conda environment.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # Non-interactive shells used by automation often do not load conda onto PATH.
    # Source the local profile script without hard-coding the binary after that.
    # shellcheck source=/dev/null
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    die "conda is required but was not found on PATH or under ~/miniforge3 / ~/miniconda3"
fi
conda activate fenicsx-env

NPROCS=1
CORE_LIST_SPEC=""
REQUESTED_CORE_COUNT=0
TASKSET_CPU_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            [[ $# -ge 2 ]] || die "missing value for -n"
            NPROCS="$2"
            shift 2
            ;;
        --core-list)
            [[ $# -ge 2 ]] || die "missing value for --core-list"
            CORE_LIST_SPEC="$2"
            shift 2
            ;;
        --core-list=*)
            CORE_LIST_SPEC="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

is_positive_int "$NPROCS" || die "-n expects a positive integer, got '$NPROCS'"
command -v lscpu >/dev/null 2>&1 || die "lscpu is required for pinned-core launches"

TOTAL_PHYSICAL_CORES=$(count_physical_cores)
(( NPROCS <= TOTAL_PHYSICAL_CORES )) || die \
    "requested $NPROCS physical cores, but only $TOTAL_PHYSICAL_CORES are available"

if [[ -n "$CORE_LIST_SPEC" ]]; then
    taskset_cpu_list_for_core_list "$CORE_LIST_SPEC"
    (( NPROCS <= REQUESTED_CORE_COUNT )) || die \
        "requested $NPROCS ranks, but core list '$CORE_LIST_SPEC' only reserves $REQUESTED_CORE_COUNT physical cores"
fi

# Keep each MPI rank single-threaded so -n means "use N physical cores".
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

MPI_CMD=(
    mpirun
    -bind-to core
    -map-by core
    -n "$NPROCS"
    python -m plm_data
)

if [[ -n "$TASKSET_CPU_LIST" ]]; then
    exec taskset -c "$TASKSET_CPU_LIST" "${MPI_CMD[@]}" "$@"
else
    exec "${MPI_CMD[@]}" "$@"
fi
