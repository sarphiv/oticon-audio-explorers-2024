fish_add_path --path /home/non-root/.local/bin
alias mamba='micromamba' 
alias conda='micromamba'


# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
set -gx MAMBA_EXE "/home/non-root/.local/bin/micromamba"
set -gx MAMBA_ROOT_PREFIX "/home/non-root/micromamba"
$MAMBA_EXE shell hook --shell fish --root-prefix $MAMBA_ROOT_PREFIX | source
# <<< mamba initialize <<<

mamba activate env


if status is-interactive

end
