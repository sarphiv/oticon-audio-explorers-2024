#!/usr/bin/fish
echo 'Setting up terminal...'
cp .devcontainer/config.fish ~/.config/fish/config.fish
echo 'y' | fish_config theme save 'ayu Dark'
echo 'y' | fish_config prompt save astronaut
sudo rm -rdf /tmp/fish.user

eval "$(micromamba shell hook -s fish)"
micromamba shell init -s fish -p ~/micromamba
micromamba activate env


echo 'Setting up permissions...'
sudo chown -R $(id -u):$(id -u) /workspace


echo 'Setting up packages...'
pip install -e .[dev]


exit 0
