# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

# Activate the project environment in the current Ubuntu terminal.
# Run from the project root:   source start_session.sh
# or from anywhere:            source /mnt/c/Users/Nyiko/fl-iot-ids/start_session.sh
# It finds the project root (the folder that contains .venv) from its own
# location, changes into it and activates .venv.
# Note: use "source", not "bash": only "source" changes the folder and the
# active environment of the current terminal.
#
# 1. start in the folder that contains this file
_fl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 2. go up one folder at a time until a folder with .venv is found (or / is reached)
while [ "$_fl_dir" != "/" ] && [ ! -d "$_fl_dir/.venv" ]; do
    _fl_dir="$(dirname "$_fl_dir")"
done
# 3. no .venv found anywhere above: print a message and stop
if [ ! -d "$_fl_dir/.venv" ]; then
    echo "Could not find the project's .venv above $(dirname "${BASH_SOURCE[0]}")"
    return 1
fi
# 4. go to the project root and activate the Python environment (.venv)
cd "$_fl_dir" || return 1
source .venv/bin/activate || return 1
# 5. settings for this terminal: Flower does not check online for new versions,
#    and Python prints its output immediately (so logs show progress as it happens)
export FLWR_DISABLE_UPDATE_CHECK=1
export PYTHONUNBUFFERED=1
# 6. print the project root and the active Python version (should be 3.12)
echo "Project root: $(pwd)"
python --version
# remove the helper variable again
unset _fl_dir
