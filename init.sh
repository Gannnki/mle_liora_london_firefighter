#!/bin/bash
cd "/mnt/e/Users/KilianAT/Documents/Estudios/Weiterbildung/Data Science/Proyecto/mle_liora_london_firefighter/notebooks" || exit
source ~/venvs/mle_liora_london_firefighter/bin/activate

# Abre una nueva terminal split con el venv activado
code --reuse-window --new-window
bash --init-file <(echo "source ~/venvs/mle_liora_london_firefighter/bin/activate")

#abre jupyter
jupyter notebook
