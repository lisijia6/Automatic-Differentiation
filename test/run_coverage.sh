#!/usr/bin/env bash
python3 -m pytest --cov-fail-under=90 --cov-report term-missing --cov=AutoDiff ./