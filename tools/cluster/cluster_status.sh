#!/bin/bash
# Cluster status summary script - reads precise GPU counts from SLURM

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                           CLUSTER STATUS SUMMARY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER-WIDE NODE STATES
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}CLUSTER NODE STATES${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"
printf "%-14s  %8s  %8s  %8s\n" "State" "Nodes" "GPUs" "% Total"
echo "───────────────────────────────────────────────────────────────────────────────"

sinfo -o "%T %D %G" --noheader | awk '
{
    state = $1
    nodes = $2
    # Parse GPU count from GRES field like "gpu:4(S:0-3)" or "gpu:4"
    gres = $3
    gpus_per_node = 4  # default
    if (match(gres, /gpu:([0-9]+)/, arr)) {
        gpus_per_node = arr[1]
    }

    state_nodes[state] += nodes
    state_gpus[state] += nodes * gpus_per_node
    total_nodes += nodes
    total_gpus += nodes * gpus_per_node
}
END {
    # Sort by node count descending
    n = asorti(state_nodes, sorted)

    # Collect and sort by GPU count
    for (i = 1; i <= n; i++) {
        s = sorted[i]
        count[i] = state_gpus[s]
        names[i] = s
    }

    # Simple bubble sort by count descending
    for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
            if (count[j] > count[i]) {
                tmp = count[i]; count[i] = count[j]; count[j] = tmp
                tmp = names[i]; names[i] = names[j]; names[j] = tmp
            }
        }
    }

    for (i = 1; i <= n; i++) {
        s = names[i]
        pct = (total_gpus > 0) ? state_gpus[s] / total_gpus * 100 : 0
        printf "%-14s  %8d  %8d  %7.1f%%\n", s, state_nodes[s], state_gpus[s], pct
    }
    print "───────────────────────────────────────────────────────────────────────────────"
    printf "%-14s  %8d  %8d  %7.1f%%\n", "TOTAL", total_nodes, total_gpus, 100.0
}'

# ─────────────────────────────────────────────────────────────────────────────
# MY ACTIVE JOBS (RUNNING ONLY)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}MY ACTIVE JOBS (RUNNING)${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"

# Get running jobs with precise GPU counts
running_jobs=$(squeue -u $USER -h -t RUNNING -o "%i|%j|%D|%b")

if [ -z "$running_jobs" ]; then
    echo "No running jobs"
    my_running_nodes=0
    my_running_gpus=0
else
    printf "%-12s  %-45s  %6s  %6s\n" "Job ID" "Name" "Nodes" "GPUs"
    echo "───────────────────────────────────────────────────────────────────────────────"

    my_running_nodes=0
    my_running_gpus=0

    echo "$running_jobs" | while IFS='|' read -r jobid name nodes gres; do
        # Parse GPU count from GRES field (e.g., "gres/gpu:4" -> 4)
        gpus_per_node=$(echo "$gres" | grep -oP 'gpu:\K[0-9]+' || echo "4")
        total_gpus=$((nodes * gpus_per_node))

        # Truncate long job names
        if [ ${#name} -gt 45 ]; then
            name="${name:0:42}..."
        fi

        printf "%-12s  %-45s  %6d  %6d\n" "$jobid" "$name" "$nodes" "$total_gpus"
    done

    # Calculate totals (need to do outside the pipe)
    my_running_nodes=$(echo "$running_jobs" | awk -F'|' '{sum+=$3} END {print sum+0}')
    my_running_gpus=$(echo "$running_jobs" | awk -F'|' '{
        nodes=$3
        gres=$4
        gpus_per_node=4
        if (match(gres, /gpu:([0-9]+)/, arr)) gpus_per_node=arr[1]
        sum += nodes * gpus_per_node
    } END {print sum+0}')

    echo "───────────────────────────────────────────────────────────────────────────────"
    printf "%-12s  %-45s  %6d  %6d\n" "TOTAL" "" "$my_running_nodes" "$my_running_gpus"
fi

# ─────────────────────────────────────────────────────────────────────────────
# MY PENDING JOBS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}MY PENDING JOBS${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"

pending_jobs=$(squeue -u $USER -h -t PENDING -o "%i|%j|%D|%b|%r")

if [ -z "$pending_jobs" ]; then
    echo "No pending jobs"
else
    printf "%-12s  %-35s  %6s  %6s  %s\n" "Job ID" "Name" "Nodes" "GPUs" "Reason"
    echo "───────────────────────────────────────────────────────────────────────────────"

    echo "$pending_jobs" | while IFS='|' read -r jobid name nodes gres reason; do
        gpus_per_node=$(echo "$gres" | grep -oP 'gpu:\K[0-9]+' || echo "4")
        total_gpus=$((nodes * gpus_per_node))

        # Truncate long names
        if [ ${#name} -gt 35 ]; then
            name="${name:0:32}..."
        fi
        if [ ${#reason} -gt 15 ]; then
            reason="${reason:0:12}..."
        fi

        printf "%-12s  %-35s  %6d  %6d  %s\n" "$jobid" "$name" "$nodes" "$total_gpus" "$reason"
    done

    pending_nodes=$(echo "$pending_jobs" | awk -F'|' '{sum+=$3} END {print sum+0}')
    pending_gpus=$(echo "$pending_jobs" | awk -F'|' '{
        nodes=$3; gres=$4; gpus=4
        if (match(gres, /gpu:([0-9]+)/, arr)) gpus=arr[1]
        sum += nodes * gpus
    } END {print sum+0}')

    echo "───────────────────────────────────────────────────────────────────────────────"
    printf "%-12s  %-35s  %6d  %6d\n" "TOTAL" "" "$pending_nodes" "$pending_gpus"
fi

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}SUMMARY${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"

# Get cluster totals
cluster_allocated=$(sinfo -h -o "%D %G" -t allocated,mixed,completing | awk '{
    nodes=$1; gres=$2; gpus=4
    if (match(gres, /gpu:([0-9]+)/, arr)) gpus=arr[1]
    sum += nodes * gpus
} END {print sum+0}')

cluster_total=$(sinfo -h -o "%D %G" | awk '{
    nodes=$1; gres=$2; gpus=4
    if (match(gres, /gpu:([0-9]+)/, arr)) gpus=arr[1]
    sum += nodes * gpus
} END {print sum+0}')

cluster_idle=$(sinfo -h -o "%D %G" -t idle | awk '{
    nodes=$1; gres=$2; gpus=4
    if (match(gres, /gpu:([0-9]+)/, arr)) gpus=arr[1]
    sum += nodes * gpus
} END {print sum+0}')

# Recalculate my running GPUs for summary
my_running_gpus=$(squeue -u $USER -h -t RUNNING -o "%D|%b" | awk -F'|' '{
    nodes=$1; gres=$2; gpus=4
    if (match(gres, /gpu:([0-9]+)/, arr)) gpus=arr[1]
    sum += nodes * gpus
} END {print sum+0}')

pct_of_cluster=$(echo "scale=1; $my_running_gpus * 100 / $cluster_total" | bc 2>/dev/null || echo "0")
pct_of_allocated=$(echo "scale=1; $my_running_gpus * 100 / $cluster_allocated" | bc 2>/dev/null || echo "0")

printf "My running GPUs:     ${GREEN}%6d${NC}  (%s%% of cluster, %s%% of allocated)\n" "$my_running_gpus" "$pct_of_cluster" "$pct_of_allocated"
printf "Cluster allocated:   %6d GPUs\n" "$cluster_allocated"
printf "Cluster idle:        %6d GPUs\n" "$cluster_idle"
printf "Cluster total:       %6d GPUs\n" "$cluster_total"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
