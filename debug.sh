#!/bin/bash
srun -Q --immediate=10 --account=cvcs2026 --partition=all_serial --mem=24G --gres=gpu:1 -w ailb-login-03 --time 04:00:00 --pty bash