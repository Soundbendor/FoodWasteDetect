#!/bin/bash

PORT=11434

# Start ollama server as daemon
./bin/ollama serve >~/ollama.log 2>&1 &
