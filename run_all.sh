#!/usr/bin/env bash
# Recomputes the results used by the notebooks, from the data (takes several days in total).
#
# Usage:  bash run_all.sh            # all steps, one after the other
#         bash run_all.sh 5 6        # only some steps (numbers below)
#
# The paths are taken from popy/config.py, or from the POPY_DATA_PATH / POPY_RESULTS_PATH environment variables.
# Logs are written to logs/. Each step uses all but a few CPU cores.

cd "$(dirname "$0")/notebooks" || exit 1
mkdir -p ../logs

STEPS=(
    "fig2 run_model_optimizing.py"                       # 1. Fig. 2a
    "fig2 run_model_fitting.py"                          # 2. Fig. 2b-c
    "fig2 run_model_fitting.py --no-cv"                  # 3. Fig. 2e-h (and the parameters in popy/config.py)
    "fig2 run_bootstrapping.py"                          # 4. Fig. 2d
    "fig3 run_glm.py"                                    # 5. Fig. 3c-e
    "fig3 run_decoding.py fb_target"                     # 6. Fig. 3f
    "fig3 run_decoding.py stay_value"                    # 7. Fig. 3f
    "fig3 run_decoding.py multiple_alphas_ridge"         # 8. Fig. 4g
    "fig4 run_build_composite_session.py meta_rates_history"  # 9. Fig. 4a-b
    "fig4 run_neural_value_all_sessions.py"              # 10. Fig. 4c-f
    "fig5 run_neural_value_across_target.py"             # 11. Fig. 5c-e, Supp. Fig. 13-14 (all six time windows)
    "fig4 run_build_composite_session.py meta_rates_value_4x20"  # 12. Fig. 6
    "fig2 run_model_recovery.py"                         # 13. Supp. Fig. 3-4
    "fig2 run_model_fitting_first_trials.py 10"          # 14. Supp. Fig. 5
    "fig2 run_model_fitting_first_trials.py 15"          # 15. Supp. Fig. 5
    "fig3 run_decoding.py alternative_model_values"      # 16. Supp. Fig. 6
    "fig3 run_decoding.py stay_value_current_trial"      # 17. Supp. Fig. 7c
)

if [ $# -gt 0 ]; then SELECTED=("$@"); else SELECTED=($(seq 1 ${#STEPS[@]})); fi

for i in "${SELECTED[@]}"; do
    step=${STEPS[$((i - 1))]}
    folder=${step%% *}
    cmd=${step#* }
    log="../logs/step${i}_${folder}_$(echo "$cmd" | tr ' ./' '___').log"
    echo "$(date '+%F %T')  step $i started:  $folder/$cmd  (log: logs/$(basename "$log"))"
    (cd "$folder" && python -u $cmd) > "$log" 2>&1
    status=$?
    echo "$(date '+%F %T')  step $i finished: $folder/$cmd  (exit code $status)"
done
