#!/bin/bash

PORT=11434

# Start ollama server as daemon
 ../../deps/ollama/bin/ollama serve >~/ollama.log 2>&1 &
