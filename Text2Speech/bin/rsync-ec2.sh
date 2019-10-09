#!/usr/bin/env bash

cd $(dirname "$0")/..
rsync -avzr \
--exclude "search-terms.txt" \
--exclude ".idea" \
--exclude "static" \
--exclude "AvalanchioConnector.iml" \
--exclude "__pycache__" \
. einext03:/home/abasar/workspace/python/Text2Speech