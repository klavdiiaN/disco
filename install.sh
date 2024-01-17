#!/bin/bash
set -e # Exit script on error
DIR="$( cd "$( dirname "$0" )" ; pwd -P )" # Fetch current directory

echo '\n>>> Installing disco.js dependencies and building the library\n'
cd $DIR/discojs
npm ci
npm run build

echo '\n>>> Installing the web client dependencies\n' 
cd $DIR/web-client
npm ci
npm link $DIR/discojs/discojs-web

echo '\n>>> Installing the server dependencies\n'
cd $DIR/server
npm ci
npm link $DIR/discojs/discojs-node

echo '\n>>> Installing the CLI dependencies\n'
cd $DIR/cli
npm ci

echo '\n>>> Installing ts-node\n'
npm install -g ts-node

echo '\n>>> Downloading and extracting sample training data\n'
sh get_training_data

echo '\n>>> Installation sucessful!\n'
