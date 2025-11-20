#!/bin/bash
set -xe

# if HOSTS_FILE_LINE is not set, error out
if [ -z "$HOSTS_FILE_LINE" ]; then
  echo "Error: HOSTS_FILE_LINE is not set"
  exit 1
else
  echo "$HOSTS_FILE_LINE" >> /etc/hosts
  echo "Added '$HOSTS_FILE_LINE' to /etc/hosts"
  cat /etc/hosts
fi

python inference_perf/main.py --config_file config.yml