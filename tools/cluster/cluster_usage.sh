#!/bin/bash
# Cluster usage report - shows historical GPU/node hours by user and account

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Default time range
START_DATE="${1:-2025-01-01}"
END_DATE="${2:-now}"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                         CLUSTER USAGE REPORT${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "Period: ${CYAN}${START_DATE}${NC} to ${CYAN}${END_DATE}${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# TOP USERS BY GPU HOURS
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}TOP 20 USERS BY GPU HOURS${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"
printf "%-4s  %-20s  %-15s  %12s  %10s\n" "Rank" "User" "Account" "GPU Hours" "GPU Days"
echo "───────────────────────────────────────────────────────────────────────────────"

sreport user top TopCount=20 Start="$START_DATE" End="$END_DATE" -T gres/gpu -t hours -n -P 2>/dev/null | \
    grep -v "^$" | \
    awk -F'|' '
    NR>0 && $1 != "" {
        rank++
        user = $2
        account = $4
        gpu_hours = $6
        gpu_days = gpu_hours / 24

        # Truncate long names
        if (length(user) > 20) user = substr(user, 1, 17) "..."
        if (length(account) > 15) account = substr(account, 1, 12) "..."

        printf "%-4d  %-20s  %-15s  %12.0f  %10.1f\n", rank, user, account, gpu_hours, gpu_days
    }'

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# TOP ACCOUNTS BY GPU HOURS
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}TOP 15 ACCOUNTS BY GPU HOURS${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"
printf "%-4s  %-20s  %12s  %10s  %8s\n" "Rank" "Account" "GPU Hours" "GPU Days" "% Total"
echo "───────────────────────────────────────────────────────────────────────────────"

sreport account top TopCount=15 Start="$START_DATE" End="$END_DATE" -T gres/gpu -t hours -n -P 2>/dev/null | \
    grep -v "^$" | \
    awk -F'|' '
    NR>0 && $1 != "" {
        accounts[NR] = $2
        hours[NR] = $4
        total += $4
        count = NR
    }
    END {
        for (i = 1; i <= count; i++) {
            account = accounts[i]
            gpu_hours = hours[i]
            gpu_days = gpu_hours / 24
            pct = (total > 0) ? gpu_hours / total * 100 : 0

            if (length(account) > 20) account = substr(account, 1, 17) "..."

            printf "%-4d  %-20s  %12.0f  %10.1f  %7.1f%%\n", i, account, gpu_hours, gpu_days, pct
        }
    }'

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# MY USAGE
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}MY USAGE (${USER})${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"

my_gpu_hours=$(sreport user top Users=$USER Start="$START_DATE" End="$END_DATE" -T gres/gpu -t hours -n -P 2>/dev/null | \
    grep -v "^$" | awk -F'|' 'NR==1 {print $6}')

my_cpu_hours=$(sreport user top Users=$USER Start="$START_DATE" End="$END_DATE" -t hours -n -P 2>/dev/null | \
    grep -v "^$" | awk -F'|' 'NR==1 {print $5}')

# Get my rank
my_rank=$(sreport user top TopCount=100 Start="$START_DATE" End="$END_DATE" -T gres/gpu -t hours -n -P 2>/dev/null | \
    grep -v "^$" | awk -F'|' -v user="$USER" 'BEGIN {rank=0} {rank++; if ($2 == user) print rank}')

if [ -n "$my_gpu_hours" ] && [ "$my_gpu_hours" != "" ]; then
    gpu_days=$(echo "scale=1; $my_gpu_hours / 24" | bc)
    gpu_years=$(echo "scale=2; $my_gpu_hours / 8760" | bc)

    printf "GPU Hours:     %'12.0f  (%s days, %s GPU-years)\n" "$my_gpu_hours" "$gpu_days" "$gpu_years"
    printf "CPU Hours:     %'12.0f\n" "$my_cpu_hours"
    printf "Cluster Rank:  %12s\n" "#${my_rank:-N/A}"
else
    echo "No usage data found for $USER"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RECENT ACTIVITY (LAST 7 DAYS)
# ─────────────────────────────────────────────────────────────────────────────
WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d 2>/dev/null)

if [ -n "$WEEK_AGO" ]; then
    echo -e "${GREEN}TOP 10 USERS - LAST 7 DAYS${NC}"
    echo "───────────────────────────────────────────────────────────────────────────────"
    printf "%-4s  %-20s  %-15s  %12s\n" "Rank" "User" "Account" "GPU Hours"
    echo "───────────────────────────────────────────────────────────────────────────────"

    sreport user top TopCount=10 Start="$WEEK_AGO" End=now -T gres/gpu -t hours -n -P 2>/dev/null | \
        grep -v "^$" | \
        awk -F'|' '
        NR>0 && $1 != "" {
            rank++
            user = $2
            account = $4
            gpu_hours = $6

            if (length(user) > 20) user = substr(user, 1, 17) "..."
            if (length(account) > 15) account = substr(account, 1, 12) "..."

            printf "%-4d  %-20s  %-15s  %12.0f\n", rank, user, account, gpu_hours
        }'
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER UTILIZATION
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${BOLD}CLUSTER UTILIZATION${NC}"
echo "───────────────────────────────────────────────────────────────────────────────"

sreport cluster utilization Start="$START_DATE" End="$END_DATE" -t hours -n -P 2>/dev/null | \
    grep -v "^$" | \
    awk -F'|' '
    NR==1 {
        allocated = $2
        down = $3
        idle = $5
        total = $7

        alloc_pct = (total > 0) ? allocated / total * 100 : 0
        down_pct = (total > 0) ? down / total * 100 : 0
        idle_pct = (total > 0) ? idle / total * 100 : 0

        printf "Allocated:     %'\''12.0f CPU hours  (%5.1f%%)\n", allocated, alloc_pct
        printf "Idle:          %'\''12.0f CPU hours  (%5.1f%%)\n", idle, idle_pct
        printf "Down:          %'\''12.0f CPU hours  (%5.1f%%)\n", down, down_pct
        printf "Total:         %'\''12.0f CPU hours\n", total
    }'

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Usage: $0 [START_DATE] [END_DATE]"
echo "  Example: $0 2024-06-01 2024-12-31"
echo "  Example: $0 \$(date -d '30 days ago' +%Y-%m-%d) now"
echo ""
