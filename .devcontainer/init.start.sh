#!/bin/fish
# NOTE: Must run this in start to avoid creating a .gitconfig
#  after container creation, which prevents copy of local .gitconfig
echo 'Setting up git...'
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1
    git init
    git add -A
    git commit -m "Initial template commit"
end
git config --global --add safe.directory /workspace
git config devcontainers-theme.show-dirty 1
pre-commit install
